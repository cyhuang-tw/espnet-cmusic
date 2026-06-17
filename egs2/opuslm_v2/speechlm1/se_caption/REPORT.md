# Bagpiper Generative Speech Enhancement — URGENT 2026 Comprehensive Evaluation

**Set:** URGENT 2026 `simulation_validation` (held-out), **2,200 clips** (WER/CER over the
1,955 with reference text). **Systems:** noisy input (floor) · **Full-FT** (all 8.27 B,
step 13000) · **LoRA** (rank-16, 0.71 B trainable, step 6000). Decoding cfg=3, temp=0.3,
seed=0. Output is 16 kHz (Xcodec ceiling); references resampled to 16 kHz for reference-based
metrics. Official URGENT `eval_all.sh` suite minus emotion-similarity (fairseq unbuildable on arm).

## Headline table (mean over 2,200; ↑/↓ = better direction)

| Metric | dir | Noisy | Full-FT | LoRA | Family |
|--------|-----|-------|---------|------|--------|
| DNSMOS-OVRL | ↑ | 1.79 | 3.28 | **3.30** | non-intrusive quality |
| NISQA | ↑ | 1.46 | 4.23 | **4.30** | non-intrusive quality |
| UTMOS | ↑ | 1.52 | 2.99 | **3.12** | non-intrusive quality |
| SCOREQ | ↑ | 1.68 | 3.75 | **3.84** | non-intrusive quality |
| SpeechBERTScore | ↑ | 0.71 | **0.82** | 0.79 | content fidelity |
| Phoneme-sim | ↑ | 0.58 | **0.76** | 0.71 | content fidelity |
| **CER** | ↓ | **16.96%** | 29.73% | 33.20% | intelligibility |
| **WER** | ↓ | **28.19%** | 42.68% | 46.54% | intelligibility |
| SpkSim | ↑ | **0.62** | 0.36 | 0.33 | speaker identity |
| LID-Acc | ↑ | 0.95 | **0.97** | 0.95 | language |

**Intrusive metrics (PESQ/ESTOI/SI-SDR) are excluded as inapplicable.** They require
sample/frame-level alignment between output and reference, which a *generative* (resynthesis)
model fundamentally breaks — output is not sample-synchronous and differs in duration/timing.
ESTOI collapsed to ≈0 despite high perceptual quality, confirming the metric is failing on
misalignment, not measuring quality. (URGENT's intrusive script also hard-asserts equal length,
which only discriminative/mask-based SE satisfies.) Fidelity-to-reference is instead captured by
the alignment-robust metrics above: SpeechBERT, phoneme-sim, speaker-sim, and WER/CER.

## Two findings

**1. LoRA ≈ Full-FT — at 4× fewer trainable params and 1 GPU vs 4.**
On every perceptual-quality metric LoRA (0.71 B) matches or marginally beats Full-FT (8.27 B):
DNSMOS 3.30 vs 3.28, NISQA 4.30 vs 4.23, UTMOS 3.12 vs 2.99, SCOREQ 3.84 vs 3.75. Full-FT is
marginally ahead on content fidelity (SpeechBERT 0.82 vs 0.79, phoneme 0.76 vs 0.71) and WER
(42.7% vs 46.5%). Net: the two are near-equivalent; **LoRA is the cost-efficient choice.**

**2. The generative quality-vs-fidelity tradeoff: both enhancers sound far cleaner but are
LESS faithful than the unprocessed noisy input.**
- **Quality** roughly doubles (DNSMOS 1.79→~3.3, NISQA 1.46→~4.3).
- **Intelligibility degrades**: WER rises 28%→43–47%, CER 17%→30–33%. The enhancers
  **make ASR worse than the noisy input on the majority of clips** (Full-FT 53%, LoRA 58%;
  they help on only 15%/12%).
- **Speaker identity degrades**: SpkSim 0.62→~0.34.

Why: the model **resynthesizes** speech (noisy→encoder→LM→Xcodec tokens→vocoder) rather than
filtering it. The codec+LM produce clean, natural audio (high DNSMOS) but a re-generated voice
(low SpkSim) with fluent-but-altered words (high WER). The noisy input, being the *original*
recording, retains the true speaker and words — and the speaker/ASR models are robust to noise —
so it is a strong baseline on fidelity metrics. Example: ref "I really enjoyed dinner tonight,
it was quite nice" → Full-FT ASR'd as "how isn't he enjoyed dinners tonight man quite nights".

## Per-category (DNSMOS ↑ | WER% ↓), by sampling rate

| Category | Noisy | Full-FT | LoRA |
|----------|-------|---------|------|
| 1ch_16000Hz | 1.82 \| 29.4 | 3.22 \| 36.3 | 3.26 \| 39.8 |
| 1ch_22050Hz | 1.77 \| 26.8 | 3.25 \| 37.5 | 3.28 \| 39.5 |
| 1ch_24000Hz | 1.76 \| 17.6 | 3.26 \| 33.2 | 3.28 \| 34.7 |
| 1ch_32000Hz | 1.76 \| 33.0 | 3.31 \| 47.0 | 3.32 \| 50.2 |
| 1ch_44100Hz | 1.88 \| 29.0 | 3.37 \| 46.1 | 3.36 \| 50.6 |
| 1ch_48000Hz | 1.77 \| 28.3 | 3.22 \| 43.2 | 3.27 \| 50.2 |

- DNSMOS gains are **uniform** across sample rates (~1.8 → ~3.3).
- WER degradation is **worst at high sample rates** (32–48 kHz: WER up to ~50%) — the 16 kHz
  output ceiling discards bandwidth *and* the generative process alters content, compounding the
  loss for originally-wideband clips.

## Verdict
- For **perceptual quality**, both models are strong; **LoRA is the efficient winner** (matches
  Full-FT at a fraction of the training cost).
- For **fidelity-critical use** (downstream ASR, speaker preservation), this generative SE is a
  **net regression vs. doing nothing** — it should not be used upstream of ASR as-is.
- Direction for improvement: higher codec rate / wideband output, a fidelity term in training,
  or a discriminative/hybrid path; and the 16 kHz ceiling specifically hurts ≥32 kHz inputs.

Artifacts: scores in `eval/scores/{noisy,fullft,lora}/score/`, aggregators `eval/aggregate.py`
+ `eval/breakdown.py`, harness `eval/run_urgent_score*.{sh,sbatch}`.
