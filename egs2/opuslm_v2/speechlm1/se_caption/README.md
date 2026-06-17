# Generative Speech Enhancement with SpeechLM (+ caption conditioning)

Generative speech enhancement (SE) built on the SpeechLM / OpusLM-v2 model: a noisy
recording is fed through the model's **continuous audio encoder**, and the model **re-generates**
clean speech as **discrete Xcodec tokens** (16 kHz) — i.e. it resynthesizes clean speech rather
than masking/filtering the input. On top of plain SE we add a **caption-conditioned** variant: we
first caption the *noisy* audio with the same SpeechLM (a short scene/transcript description), and
feed that caption as an extra text prompt so the generator has an explicit semantic anchor.

This directory is self-contained: task definitions live in `espnet2/speechlm/` (see below);
everything else — configs, data prep, captioning, training/inference launchers, and the
URGENT-2026 evaluation harness — is here.

---

## 1. What's implemented

Two tasks are registered (in **both** `espnet2/speechlm/model/speechlm/task_conf_speechlm.py`
`SPEECHLM_TASK_CONFIGS` and `espnet2/speechlm/dataloader/task_conf.py` `TASK_CONFIGS` — they
cross-validate at import):

| task | sequence template | meaning |
|------|-------------------|---------|
| `speech_enhancement` | `[(user, audio1), (assistant, audio2)]` | noisy → clean |
| `speech_enhancement_captioned` | `[(user, audio1), (user, text1), (assistant, audio2)]` | noisy + caption → clean |

`_apply_chat_template` maps `audio`+user → the continuous encoder (Qwen3-Omni), `audio`+assistant →
discrete Xcodec output, and `text` → text tokens. The training loss is on the assistant audio only;
at inference the sequence stops before the assistant turn, so the prompt is `[noisy (+ caption)]`
and the model generates the clean audio.

Fine-tuning modes (all resume from the SpeechLM pretrain checkpoint):
- **LoRA** (rank 16, α 32) on the backbone q/k/v/o + gate/up/down, plus `lm_head`/`adaptor`/
  `stream_emb`; everything else frozen (~0.7 B trainable). `merge_weights: false` (never fold
  `BA` into the base weight — the trainer's `train→eval→save` loop would otherwise double-count).
- **Full-FT** (all ~8.4 B LM params; encoder + codec frozen), 4-GPU ZeRO-2.
- **PEFT** (partial fine-tune of the top layers) — an earlier baseline, kept for reference.

### Library changes (in `espnet2/`, outside this dir)
- `layers/create_adapter_fn.py` — threads a `merge_weights` flag through the in-place LoRA builder.
- `speechlm/model/speechlm/speechlm_job.py` — builds LoRA adapters from a `model.lora` config block
  (shared by train **and** inference, so the structure matches on load).
- `speechlm/model/speechlm/task_conf_speechlm.py`, `speechlm/dataloader/task_conf.py` — the two tasks.
- `speechlm/dataloader/multimodal_loader/{audio_loader,text_loader}.py` — SE audio I/O + a JSONL
  caption reader (`{"id","text"}` or `<id> <text>`).
- `speechlm/model/speechlm/multimodal_io/audio.py` — Xcodec decode hardening (clamp out-of-range
  stream ids; inference-only).
- `speechlm/bin/inference.py`, `speechlm/trainer/deepspeed_trainer.py`,
  `speechlm/dataloader/iterator.py`, `speechlm/model/speechlm/lm/parallel.py` — strict-load guard,
  complete-checkpoint detection for resume, spawn dataloader context, multi-input routing.

---

## 2. Layout

```
se_caption/
├── README.md                 ← this file
├── conf/                      training / inference / deepspeed configs (see §5)
├── data/
│   ├── build_se_manifest.py           noisy+clean scp → Lhotse + dataset.json
│   └── build_captioned_manifest.py    + caption text → captioned dataset.json
├── captioning/
│   ├── run_bagpiper_server.sh         launch the vLLM SpeechLM captioner
│   └── caption_worker.py              concurrent, resumable captioner (→ JSONL)
├── eval/                      URGENT-2026 metric harness (run_urgent_score.sh + aggregators)
├── eval_val_loss.py           offline teacher-forced val-loss probe
├── launchers/                 portable SLURM templates (+ env.example.sh)
├── REPORT.md                  full URGENT-2026 results (noisy / Full-FT / LoRA)
└── REPORT_caption.md          caption-conditioned results + analysis
```

---

## 3. Prerequisites

- A Python env with `torch`, `deepspeed`, `loralib`, `lhotse`, `soundfile`, and this espnet checkout
  installed (`pip install -e .` at the repo root).
- The SpeechLM pretrain checkpoint ("step_260000"): the **universal** checkpoint dir for full-FT,
  or its `mp_rank_00_model_states.pt` for LoRA/PEFT. Point `PRETRAIN_CKPT_*` at it.
- Models are pulled by HF tag (`Qwen/Qwen3-8B-Base`, the Qwen3-Omni encoder, Xcodec) into `HF_HOME`.
- For captioning: a vLLM SpeechLM build + the base model exported for vLLM (`BAGPIPER_MODEL_DIR`).
- For evaluation: the URGENT-2026 challenge repo (`evaluation_metrics/`), the metric env, and the
  DNSMOS onnx models. See `launchers/env.example.sh` for every variable.

**Setup:** `cd launchers && cp env.example.sh env.sh && $EDITOR env.sh` (env.sh is gitignored).

---

## 4. End-to-end pipeline

All launchers are thin SLURM wrappers; each one's header shows the bare `torchrun`/`python` command
if you don't use SLURM. Run them as e.g.
`sbatch --account=$ACC --partition=$PART launchers/train_se.sbatch lora`.

**(1) Build manifests** from URGENT scp pairs (`wav.scp` = noisy, `spk1.scp` = clean):
```bash
python data/build_se_manifest.py --noisy_scp .../train/wav.scp \
    --clean_scp .../train/spk1.scp --out_dir $SE_ROOT/data/se_train
# repeat for the validation split → $SE_ROOT/data/se_valid
```

**(2) (caption variant only) Caption the noisy audio, then build the captioned manifest:**
```bash
# noisy.scp: "<uid> <abs_noisy_wav>"  (e.g. awk over se_train's dataset.json)
sbatch launchers/caption.sbatch noisy_train.scp captions_train.jsonl 48
python data/build_captioned_manifest.py \
    <noisy_lhotse_dir> <clean_lhotse_dir> captions_train.jsonl $SE_ROOT/data/se_cap_train
```
The captioner is the SpeechLM **base** (weight-identical to the SE pretrain ckpt). Captions are
derived from the **noisy** audio only, with the **same** captioner at train and test — no clean-target
leakage, matched train/test distribution.

**(3) Train** (`mode` = `lora` | `full` | `peft` | `caption_lora` | `caption_full`; add `SMOKE` for a
20-step sanity run):
```bash
sbatch --gpus-per-node=1 launchers/train_se.sbatch lora
sbatch --gpus-per-node=4 launchers/train_se.sbatch caption_full
```

**(4) Inference** (audio in [+ caption] → enhanced audio out; builds `inf.scp`):
```bash
sbatch launchers/infer_se.sbatch <checkpoint> speech_enhancement se_valid out_lora lora
sbatch launchers/infer_se.sbatch <checkpoint> speech_enhancement_captioned se_cap_valid out_caplora lora
```

**(5) Score** with the URGENT-2026 suite and aggregate:
```bash
sbatch launchers/score_urgent.sbatch $SE_ROOT/eval/enhanced/out_lora/inf.scp $SE_ROOT/eval/scores/lora
SE_ROOT=$SE_ROOT python eval/aggregate.py          # mean table over systems noisy/fullft/lora
SE_ROOT=$SE_ROOT URGENT_DATA=$URGENT_DATA python eval/breakdown.py   # per-category
# compare any set of systems on a fixed uid set:
python eval/cap_compare.py <inf.scp> noisy:.../noisy/score lora:.../lora/score cap:.../cap/score
```

---

## 5. Configs (`conf/`)

| file | purpose |
|------|---------|
| `train_se_bagpiper_lora.yaml` (+`_smoke`) | LoRA SE training (also reused for `caption_lora` via the task specifier) |
| `train_se_bagpiper_full.yaml` (+`_smoke`) | full-FT SE, 4-GPU ZeRO-2 |
| `train_se_bagpiper_full_cap.yaml` | full-FT **caption-conditioned** SE |
| `train_se_bagpiper_peft.yaml` (+`_smoke`) | partial-FT baseline |
| `deepspeed_se_{lora,full,peft}.json` | matching DeepSpeed configs (lr / warmup / ZeRO stage) |
| `infer_se_bagpiper_train.yaml` | inference build config (full-FT/PEFT checkpoints) |
| `infer_se_bagpiper_train_lora.yaml` | inference build config (LoRA checkpoints — rebuilds adapters) |
| `infer_se_bagpiper_demo_cfg3t03.yaml` | decoding config: classifier-free guidance **cfg≈3**, temp 0.3 |

---

## 6. Results

### URGENT-2026 `simulation_validation` (2,200 clips; means; ↑/↓ = better). Full table in `REPORT.md`.

| metric | dir | Noisy | Full-FT | LoRA |
|--------|-----|-------|---------|------|
| DNSMOS-OVRL | ↑ | 1.79 | 3.28 | **3.30** |
| NISQA | ↑ | 1.46 | 4.23 | **4.30** |
| UTMOS | ↑ | 1.52 | 2.99 | **3.12** |
| SCOREQ | ↑ | 1.68 | 3.75 | **3.84** |
| SpeechBERTScore | ↑ | 0.71 | **0.82** | 0.79 |
| Phoneme-sim | ↑ | 0.58 | **0.76** | 0.71 |
| CER % | ↓ | **17.0** | 29.7 | 33.2 |
| WER % | ↓ | **28.2** | 42.7 | 46.5 |
| SpkSim | ↑ | **0.62** | 0.36 | 0.33 |

Two findings: **(1) LoRA ≈ Full-FT** on perceptual quality at ~4× fewer trainable params (1 GPU vs 4)
— LoRA is the efficient choice. **(2) Quality-vs-fidelity tradeoff:** both enhancers roughly double
perceptual quality but, being *generative*, are less faithful than the unprocessed noisy input on
WER and speaker similarity. Intrusive metrics (PESQ/ESTOI/SI-SDR) are **not meaningful** here — they
require sample-aligned output, which a resynthesis model does not produce.

### Caption conditioning (300-clip matched subset; full analysis in `REPORT_caption.md`)
Caption conditioning is a **net positive**: after a phase transition between step 4k→6k, the
caption-conditioned LoRA at step 8k **matches-or-beats** the best no-caption LoRA on 8/10 metrics
(e.g. WER 53.1% vs 53.5%, DNSMOS 3.357 vs 3.345) and keeps improving where the no-caption model had
plateaued. Early (2k/4k) checkpoints look bad purely because the `audio+text→audio` format takes
longer to learn — don't judge it before ~6k steps.

---

## 7. Gotchas (read before running)

- **Inference needs classifier-free guidance** (cfg≈3, temp 0.3 — `infer_se_bagpiper_demo_cfg3t03.yaml`).
  The model trains with `audio_cfg: 0.05`; decoding at cfg 1 gives degenerate/silent output.
- **LoRA `merge_weights: false`** is mandatory (see §1) — otherwise eval-time folding double-counts on reload.
- **Dataloader uses the `spawn` start method** when `num_workers>0` (CUDA + fork deadlocks otherwise).
- **Caption-source parity:** caption train and test with the *same* model on *noisy* audio; freeze the prompt.
- **vLLM captioning:** re-encode flac→wav (the worker does this); after shutdown, kill any orphan
  `EngineCore` process holding GPU memory.
- **Resume:** full-FT uses the *universal* checkpoint (optimizer state for all params); LoRA/PEFT
  resume weights-only from the base `mp_rank_00_model_states.pt` (adapters start fresh, base names match).

---

*Heavy artifacts (`data/`, `exp/`, `ckpt/`, `enhanced/`, `eval/scores/`, refs) are gitignored — they
are regenerated by the pipeline above. `launchers/env.sh` is gitignored; commit nothing machine-specific.*
