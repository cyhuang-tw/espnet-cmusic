#!/usr/bin/env python3
"""Visualize cross-attention weights during teacher-forced inference.

Extracts decoder cross-attention heatmaps and computes sharpness metrics
to diagnose whether the model can localize onset events in encoder time.

Usage:
    python local/visualize_cross_attention.py \
        --config $EXP_DIR/config.yaml \
        --checkpoint $EXP_DIR/51epoch.pth \
        --data-dir dump/raw/maestro_train_segments \
        --output-dir cross_attn_plots
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from espnet2.legacy.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet2.tasks.asr import ASRTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter

SEGMENTS = [
    "maestro-v3.0.0_TRAIN_AMT_MULTIINS_ONSETS_0321.flac_w3",  # sparse
    "maestro-v3.0.0_TRAIN_AMT_MULTIINS_ONSETS_0021.flac_w1",  # medium
    "maestro-v3.0.0_TRAIN_AMT_MULTIINS_ONSETS_0581.flac_w1",  # dense
]

MS_PER_FRAME = 8.0  # hop_length=128 / sr=16000


def shorten_token(tok, max_len=20):
    """Shorten token for y-axis labels."""
    tok = tok.replace("NOTE_ON_INS_000_PITCH_", "P")
    tok = tok.replace("NOTE_ON_", "N")
    tok = tok.replace("AMT_MULTIINS_ONSETS", "TASK")
    return tok[:max_len]


def compute_entropy(attn_row):
    """Compute entropy of an attention distribution (in bits)."""
    attn_row = attn_row.clamp(min=1e-10)
    return -(attn_row * attn_row.log2()).sum().item()


def plot_heatmap(attn, tokens, gt_timestamps, utt_id, layer_idx, output_dir):
    """Plot cross-attention heatmap."""
    # attn: (decoder_steps, encoder_frames) — head-averaged
    n_dec, n_enc = attn.shape
    labels = [shorten_token(t) for t in tokens]

    fig, ax = plt.subplots(figsize=(14, max(4, n_dec * 0.15)))
    im = ax.imshow(attn, aspect="auto", origin="lower", interpolation="nearest",
                   cmap="hot")
    plt.colorbar(im, ax=ax, fraction=0.02)

    # Y-axis: decoder tokens
    ax.set_yticks(range(n_dec))
    ax.set_yticklabels(labels, fontsize=6)

    # X-axis: encoder frames with time labels
    frame_ticks = np.arange(0, n_enc, 50)
    ax.set_xticks(frame_ticks)
    ax.set_xticklabels([f"{f * MS_PER_FRAME / 1000:.1f}s" for f in frame_ticks],
                       fontsize=7)

    # Vertical lines at ground truth onset positions
    for ts in gt_timestamps:
        frame = ts / (MS_PER_FRAME / 1000)
        ax.axvline(x=frame, color="cyan", linewidth=0.5, alpha=0.7)

    ax.set_xlabel("Encoder time")
    ax.set_ylabel("Decoder tokens")
    ax.set_title(f"{utt_id}\nLayer {layer_idx} cross-attention (head-averaged)")
    plt.tight_layout()

    fname = f"{utt_id}_layer{layer_idx}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    return fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="cross_attn_plots")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segments", nargs="*", default=SEGMENTS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    asr_model, asr_train_args = ASRTask.build_model_from_file(
        args.config, args.checkpoint, device=args.device
    )
    asr_model.eval()
    token_list = asr_train_args.token_list
    tokenizer = build_tokenizer(token_type="word")
    converter = TokenIDConverter(token_list=token_list)
    n_layers = len(asr_model.decoder.decoders)

    # Load data
    wav_scp, ref_text = {}, {}
    with open(f"{args.data_dir}/wav.scp") as f:
        for line in f:
            p = line.strip().split(maxsplit=1)
            wav_scp[p[0]] = p[1]
    with open(f"{args.data_dir}/text") as f:
        for line in f:
            p = line.strip().split(maxsplit=1)
            ref_text[p[0]] = p[1] if len(p) > 1 else ""

    # Per-layer sharpness stats
    layer_ts_entropies = {i: [] for i in range(n_layers)}
    layer_pitch_entropies = {i: [] for i in range(n_layers)}

    for utt_id in args.segments:
        print(f"\n{'='*70}")
        print(f"Processing: {utt_id}")
        print(f"{'='*70}")

        # Load and encode
        speech_np, sr = sf.read(wav_scp[utt_id])
        speech = torch.tensor(speech_np, dtype=torch.float32).unsqueeze(0).to(args.device)
        speech_lengths = torch.tensor([speech.size(1)], dtype=torch.long).to(args.device)

        tokens = tokenizer.text2tokens(ref_text[utt_id])
        token_ids = converter.tokens2ids(tokens)
        text_tensor = torch.tensor([token_ids], dtype=torch.long).to(args.device)
        text_lengths = torch.tensor([len(token_ids)], dtype=torch.long).to(args.device)

        with torch.no_grad():
            encoder_out, encoder_out_lens = asr_model.encode(speech, speech_lengths)
            ys_in_pad, _ = add_sos_eos(
                text_tensor, asr_model.sos, asr_model.eos, asr_model.ignore_id
            )
            ys_in_lens = text_lengths + 1
            decoder_out, _ = asr_model.decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
            )

        n_enc = encoder_out.size(1)
        ys_in_tokens = [token_list[i] for i in ys_in_pad[0].cpu().tolist()]

        # ys_in_tokens = [SOS, tok0, tok1, ..., tokN]
        # decoder position i predicts the token AFTER ys_in_tokens[i]
        # So position 0 (input=SOS) predicts tok0, position 1 (input=tok0) predicts tok1, etc.
        # The predicted token at position i is tokens[i] (the reference)
        pred_tokens = tokens  # what each decoder position should predict

        # Extract ground truth timestamps for vertical lines
        gt_timestamps = []
        for t in tokens:
            if t.startswith("T") and "." in t:
                try:
                    gt_timestamps.append(float(t[1:]))
                except ValueError:
                    pass

        # Extract attention from selected layers
        for layer_idx in [0, n_layers // 2, n_layers - 1]:
            attn = asr_model.decoder.decoders[layer_idx].src_attn.attn[0]  # (heads, dec, enc)
            attn_avg = attn.mean(dim=0).cpu()  # (dec, enc) — head-averaged

            # Plot — skip SOS position (position 0), show positions 1..N which predict tokens[0..N-1]
            attn_to_plot = attn_avg[1:len(tokens)+1]  # (N_tokens, enc)
            fname = plot_heatmap(
                attn_to_plot.numpy(), pred_tokens, gt_timestamps,
                utt_id, layer_idx, args.output_dir
            )
            print(f"  Saved: {fname}")

        # Sharpness analysis — use last layer
        print(f"\n  Sharpness analysis (layer {n_layers-1}, head-averaged):")
        print(f"  {'POS':>3s} {'PRED_TOKEN':>25s} | {'ENTROPY':>8s} | {'PEAK_FRAME':>10s} | {'PEAK_TIME':>9s} | {'GT_TIME':>7s} | {'ERROR':>7s}")
        print(f"  {'-'*90}")

        last_attn = asr_model.decoder.decoders[n_layers-1].src_attn.attn[0]
        last_attn_avg = last_attn.mean(dim=0).cpu()  # (dec, enc)

        ts_entropies, pitch_entropies = [], []
        ts_peak_errors = []

        for i, tok in enumerate(pred_tokens):
            dec_pos = i + 1  # skip SOS
            attn_row = last_attn_avg[dec_pos]
            ent = compute_entropy(attn_row)
            peak_frame = attn_row.argmax().item()
            peak_time = peak_frame * MS_PER_FRAME / 1000

            is_timestamp = tok.startswith("T") and "." in tok
            if is_timestamp:
                gt_time = float(tok[1:])
                error = abs(peak_time - gt_time)
                ts_entropies.append(ent)
                ts_peak_errors.append(error)
                print(f"  {i:3d} {tok:>25s} | {ent:8.2f} | {peak_frame:10d} | {peak_time:8.3f}s | {gt_time:6.2f}s | {error*1000:5.0f}ms")
            else:
                pitch_entropies.append(ent)
                print(f"  {i:3d} {tok:>25s} | {ent:8.2f} | {peak_frame:10d} | {peak_time:8.3f}s |         |")

        # Collect for per-layer analysis
        for li in range(n_layers):
            attn_li = asr_model.decoder.decoders[li].src_attn.attn[0].mean(dim=0).cpu()
            for i, tok in enumerate(pred_tokens):
                ent = compute_entropy(attn_li[i + 1])
                is_ts = tok.startswith("T") and "." in tok
                if is_ts:
                    layer_ts_entropies[li].append(ent)
                else:
                    layer_pitch_entropies[li].append(ent)

        print(f"\n  Summary (last layer):")
        if ts_entropies:
            print(f"    Timestamp tokens: mean entropy={np.mean(ts_entropies):.2f} bits, mean peak error={np.mean(ts_peak_errors)*1000:.0f}ms")
        if pitch_entropies:
            print(f"    Pitch tokens:     mean entropy={np.mean(pitch_entropies):.2f} bits")

    # Per-layer summary
    print(f"\n{'='*70}")
    print(f"PER-LAYER ENTROPY (across all segments)")
    print(f"{'='*70}")
    print(f"{'LAYER':>5s} | {'TS_ENTROPY':>12s} | {'PITCH_ENTROPY':>14s}")
    print(f"{'-'*40}")
    for li in range(n_layers):
        ts_e = np.mean(layer_ts_entropies[li]) if layer_ts_entropies[li] else 0
        p_e = np.mean(layer_pitch_entropies[li]) if layer_pitch_entropies[li] else 0
        print(f"  {li:3d}  | {ts_e:11.2f}  | {p_e:13.2f}")

    print(f"\nDone! Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
