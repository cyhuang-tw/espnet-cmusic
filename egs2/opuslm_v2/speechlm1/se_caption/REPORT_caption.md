# Caption-conditioned SE — first eval (step_2000, 300-clip subset)

Fair 4-way on the SAME 300 captioned-valid clips (baselines subset from existing per-clip scores;
WER/CER over the ~202 with reference text). cap_lora = caption-conditioned LoRA @ step_2000;
baselines = no-caption LoRA @ step_6000, Full-FT @ step_13000.

| metric | noisy | lora(6k) | fullft(13k) | cap_lora(2k) |
|--------|-------|----------|-------------|--------------|
| DNSMOS↑ | 1.76 | 3.35 | 3.33 | 3.18 |
| NISQA↑ | 1.43 | 4.25 | 4.25 | 3.99 |
| UTMOS↑ | 1.44 | 2.98 | 2.92 | 2.87 |
| SCOREQ↑ | 1.59 | 3.74 | 3.72 | 3.67 |
| SpeechBERT↑ | 0.72 | 0.80 | 0.82 | 0.625 |
| Phoneme-sim↑ | 0.56 | 0.69 | 0.74 | 0.358 |
| SpkSim↑ | 0.65 | 0.30 | 0.34 | 0.100 |
| LID↑ | 0.97 | 0.97 | 0.98 | 0.76 |
| CER%↓ | 20.8 | 38.6 | 34.3 | 71.3 |
| WER%↓ | 34.8 | 53.5 | 49.4 | 89.5 |

## Verdict (preliminary): NEGATIVE at step_2000 — but not a fair test yet
- cap_lora@2000 is WORSE than no-caption LoRA on every metric; fidelity (WER/CER, content, speaker)
  is far worse, falling below even the noisy input. Quality (DNSMOS/NISQA) only slightly down → it
  still "sounds enhanced" but is more unfaithful (it appears to generate freely / narrate rather than
  reconstruct the input).
- UNFAIR: cap_lora@2000 vs baselines@6k/13k. Captioned task is newer+harder (unseen audio+text→audio
  format, +258-token sequences) → undertrained at 2000 steps relative to baselines.
- NEXT: eval cap_lora @ step_6000 (matched to no-caption LoRA's best) for a fair comparison before
  concluding. Training (job 2515737) still running.
- Hypothesis risk if it stays negative: a verbose ~250-token caption may distract the generator;
  a SHORT transcript-only conditioning (or shorter caption) could be the better design.

Artifacts: eval/scores/cap_lora_sub300/, eval/enhanced/cap_lora_sub300/. Comparison script inline.

## Trend (step_2000 → step_4000, SAME 300 clips; eval/cap_compare.py)

| metric | noisy | lora6k | fullft13k | cap2k | cap4k | Δ(2k→4k) |
|--------|-------|--------|-----------|-------|-------|----------|
| DNSMOS↑ | 1.76 | 3.35 | 3.33 | 3.18 | 3.26 | +0.08 |
| NISQA↑ | 1.43 | 4.25 | 4.25 | 3.99 | 4.14 | +0.15 |
| UTMOS↑ | 1.44 | 2.98 | 2.92 | 2.87 | 2.92 | +0.06 |
| SCOREQ↑ | 1.59 | 3.74 | 3.72 | 3.67 | 3.72 | +0.04 |
| SpeechBERT↑ | 0.72 | 0.80 | 0.82 | 0.625 | 0.651 | +0.026 |
| Phoneme-sim↑ | 0.56 | 0.69 | 0.74 | 0.358 | 0.395 | +0.037 |
| SpkSim↑ | 0.65 | 0.295 | 0.336 | 0.100 | 0.163 | +0.063 |
| LID↑ | 0.97 | 0.97 | 0.98 | 0.763 | 0.840 | +0.077 |
| CER%↓ | 20.8 | 38.6 | 34.3 | 71.3 | 68.2 | −3.1 |
| WER%↓ | 34.8 | 53.5 | 49.4 | 89.5 | 87.4 | −2.1 |

**Reading (2k→4k only — turned out to be MISLEADING):** the 2k/4k snapshots looked catastrophic
and slow-moving, suggesting a design flaw. This was WRONG — see step_6000 below.

## MATCHED comparison — step_6000 (SAME 300 clips); the fair call

| metric | noisy | **lora6k (no-cap)** | fullft13k | cap2k | cap4k | **cap6k** |
|--------|-------|---------------------|-----------|-------|-------|-----------|
| DNSMOS↑ | 1.76 | 3.35 | 3.33 | 3.18 | 3.26 | **3.32** |
| NISQA↑ | 1.43 | 4.25 | 4.25 | 3.99 | 4.14 | **4.18** |
| UTMOS↑ | 1.44 | 2.98 | 2.92 | 2.87 | 2.92 | **2.97** |
| SCOREQ↑ | 1.59 | 3.74 | 3.72 | 3.67 | 3.72 | **3.78** |
| SpeechBERT↑ | 0.72 | 0.80 | 0.82 | 0.625 | 0.651 | **0.777** |
| Phoneme-sim↑ | 0.56 | 0.69 | 0.74 | 0.358 | 0.395 | **0.651** |
| SpkSim↑ | 0.65 | 0.295 | 0.336 | 0.100 | 0.163 | **0.256** |
| LID↑ | 0.97 | 0.973 | 0.98 | 0.763 | 0.840 | **0.967** |
| CER%↓ | 20.8 | 38.6 | 34.3 | 71.3 | 68.2 | **40.8** |
| WER%↓ | 34.8 | 53.5 | 49.4 | 89.5 | 87.4 | **57.3** |

## Verdict (CORRECTED — caption-SE does NOT fail; it reaches PARITY at matched compute)
There is a **phase transition between step 4000 and 6000**: WER 87.4→57.3, CER 68.2→40.8,
SpkSim 0.163→0.256, SpeechBERT 0.651→0.777, Phoneme 0.395→0.651, LID 0.84→0.967. By step_6000 the
captioned LoRA has essentially CAUGHT UP to the no-caption LoRA on every metric (WER 57.3 vs 53.5,
SpkSim 0.256 vs 0.295, DNSMOS 3.32 vs 3.35). The earlier "FAILS / slow / design-flaw" read was an
artifact of UNDERTRAINING the harder audio+text→audio format — the model spends ~4k steps learning
to use the text turn at all, then snaps to baseline behavior.
- So at matched compute: **captioned LoRA ≈ no-caption LoRA** (marginally behind on fidelity:
  WER +3.8pt, SpkSim −0.04). The caption neither clearly helps NOR hurts *yet*.
- KEY ASYMMETRY: the no-caption LoRA **plateaued/worsened after step_6000** (offline val 6.82@2k →
  6.90@6k; stopped at 6k). The captioned LoRA is still climbing STEEPLY at 6000 and training
  continues to step 50000 (train loss 3.30→3.20 @8.7k). → later captioned checkpoints (8k/10k+)
  may SURPASS the no-caption baseline. This is the open question worth resolving before any pivot.
- Pivot to short transcript-only conditioning is NO LONGER the obvious move — first eval later
  captioned checkpoints (step_8000+) to see if caption conditioning pulls ahead with more training.

## Post-plateau trend — step_8000 (SAME 300 clips): caption conditioning PULLS AHEAD

| metric | noisy | **lora6k (no-cap BEST)** | fullft13k | cap4k | cap6k | **cap8k** |
|--------|-------|--------------------------|-----------|-------|-------|-----------|
| DNSMOS↑ | 1.76 | 3.345 | 3.33 | 3.26 | 3.32 | **3.357** |
| NISQA↑ | 1.43 | 4.245 | 4.25 | 4.14 | 4.18 | **4.262** |
| UTMOS↑ | 1.44 | 2.984 | 2.92 | 2.92 | 2.97 | **3.016** |
| SCOREQ↑ | 1.59 | 3.742 | 3.72 | 3.72 | 3.78 | **3.836** |
| SpeechBERT↑ | 0.72 | 0.796 | 0.82 | 0.651 | 0.777 | **0.799** |
| Phoneme-sim↑ | 0.56 | 0.692 | 0.74 | 0.395 | 0.651 | **0.698** |
| SpkSim↑ | 0.65 | **0.295** | 0.336 | 0.163 | 0.256 | 0.287 |
| LID↑ | 0.97 | 0.973 | 0.98 | 0.840 | 0.967 | **0.970** |
| CER%↓ | 20.8 | 38.57 | 34.3 | 68.2 | 40.8 | **37.90** |
| WER%↓ | 34.8 | 53.45 | 49.4 | 87.4 | 57.3 | **53.08** |

## Verdict (POSITIVE): caption conditioning matches-or-beats the no-caption enhancer, and is still climbing
At step_8000 the captioned LoRA **surpasses the no-caption LoRA's BEST checkpoint** on 8/10 metrics —
all 4 quality metrics (DNSMOS/NISQA/UTMOS/SCOREQ), both content-fidelity metrics (SpeechBERT 0.799 vs
0.796, Phoneme 0.698 vs 0.692), and intelligibility (WER 53.08 vs 53.45, CER 37.90 vs 38.57). It is
marginally behind only on SpkSim (0.287 vs 0.295) and LID (≈tied) — and SpkSim is closing fast
(0.163→0.256→0.287). DECISIVE ASYMMETRY: the no-caption LoRA had PLATEAUED/worsened by step_6000
(stopped there), whereas the captioned LoRA is monotonically improving 6k→8k on EVERY metric and
training continues (step_10000 ckpt ready). So caption conditioning **extends the useful training
horizon** and ends up ahead. Net: the caption is a mild but real POSITIVE within the generative-SE
family — it is non-harmful and beats the no-caption enhancer given equal-or-more (still-productive)
training. CAVEAT: this is "best generative enhancer vs best generative enhancer"; both remain well
below the noisy input on WER/SpkSim (the fundamental generative quality-vs-fidelity tradeoff from
REPORT.md persists). The caption improves the *enhancer*, it does not erase the generative gap to
the original recording.

NEXT: (1) eval cap LoRA step_10000 (does the lead widen?). (2) Full-FT captioned run (job 2519117,
exp/se_cap_full4) — capacity-matched; ckpts protected at 2k/4k/6k. (3) if the lead holds, full-2200
eval of the best captioned checkpoint vs the no-caption baselines (run_caption_eval_infer.sbatch).
