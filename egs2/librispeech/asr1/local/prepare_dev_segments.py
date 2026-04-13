#!/usr/bin/env python3
"""Prepare 4-second non-overlapping segments from selected dev clips."""

import os

import numpy as np
import soundfile as sf

SRC_DIR = "/work/nvme/bbjs/chuang14/espnet-wc4/egs2/librispeech/asr1/dump/raw/maestro_dev_onsets"
SRC_BASE = "/work/nvme/bbjs/chen26/espnet_music/egs2/librispeech/asr1"
OUT_DIR = "/work/nvme/bbjs/chuang14/espnet-wc4/egs2/librispeech/asr1/dump/raw/maestro_dev_segments"

# Selected clips: sparse / medium / dense
SELECTED_UTTS = [
    "maestro-v3.0.0_VALIDATION_AMT_MULTIINS_ONSETS_0611.flac",  # sparse 509 events
    "maestro-v3.0.0_VALIDATION_AMT_MULTIINS_ONSETS_0252.flac",  # sparse 629 events
    "maestro-v3.0.0_VALIDATION_AMT_MULTIINS_ONSETS_0618.flac",  # medium 2907 events
    "maestro-v3.0.0_VALIDATION_AMT_MULTIINS_ONSETS_0625.flac",  # medium 2911 events
    "maestro-v3.0.0_VALIDATION_AMT_MULTIINS_ONSETS_1009.flac",  # dense 23469 events
]

NUM_WINDOWS = 5
MAX_CONTEXT_SEC = 4.0
SR = 16000


def process_text(text_tokens, start_time, end_time):
    task_token = text_tokens[0]
    tokens = text_tokens[1:]

    if len(tokens) < 2:
        return [task_token], 0

    ts_tokens = tokens[::2]
    timestamps = np.array([float(t[1:]) for t in ts_tokens], dtype=np.float32)

    in_window = (timestamps >= start_time) & (timestamps < end_time)

    if not in_window.any():
        return [task_token], 0

    start_index = int(in_window.argmax())
    end_index = int(np.where(in_window)[0][-1])

    sliced = tokens[start_index * 2 : (end_index + 1) * 2]
    sliced_ts = timestamps[start_index : end_index + 1]
    normalized = sliced_ts - start_time

    new_tokens = list(sliced)
    for i, t in enumerate(normalized):
        new_tokens[i * 2] = f"T{t:.2f}"

    n_events = end_index - start_index + 1
    return [task_token] + new_tokens, n_events


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wav_dir = os.path.join(OUT_DIR, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    wav_scp = {}
    with open(os.path.join(SRC_DIR, "wav.scp")) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            wav_scp[parts[0]] = parts[1]

    text_data = {}
    with open(os.path.join(SRC_DIR, "text")) as f:
        for line in f:
            parts = line.strip().split()
            text_data[parts[0]] = parts[1:]

    out_wav_scp = []
    out_text = []
    out_utt2spk = []

    for utt_id in SELECTED_UTTS:
        wav_path = wav_scp[utt_id]
        if not os.path.isabs(wav_path):
            wav_path = os.path.join(SRC_BASE, wav_path)

        speech, sr = sf.read(wav_path)
        assert sr == SR, f"Expected {SR}Hz, got {sr}Hz"
        tokens = text_data[utt_id]

        print(f"\n=== {utt_id} ({len(speech)/sr:.1f}s total) ===")

        for w in range(NUM_WINDOWS):
            start_time = w * MAX_CONTEXT_SEC
            start_sample = int(start_time * SR)
            end_sample = int((start_time + MAX_CONTEXT_SEC) * SR)

            if end_sample > len(speech):
                print(f"  w{w}: skipped (beyond audio length)")
                continue

            cropped = speech[start_sample:end_sample]
            actual_start = start_sample / SR
            actual_end = end_sample / SR

            cropped_tokens, n_events = process_text(tokens, actual_start, actual_end)

            seg_id = f"{utt_id}_w{w}"
            out_wav_path = os.path.join(wav_dir, f"{seg_id}.wav")
            sf.write(out_wav_path, cropped, SR)

            out_wav_scp.append(f"{seg_id} {out_wav_path}")
            out_text.append(f"{seg_id} {' '.join(cropped_tokens)}")
            out_utt2spk.append(f"{seg_id} {seg_id}")

            print(f"  w{w}: [{actual_start:.1f}s, {actual_end:.1f}s) -> {n_events} events, {len(cropped_tokens)} tokens")

    with open(os.path.join(OUT_DIR, "wav.scp"), "w") as f:
        f.write("\n".join(out_wav_scp) + "\n")
    with open(os.path.join(OUT_DIR, "text"), "w") as f:
        f.write("\n".join(out_text) + "\n")
    with open(os.path.join(OUT_DIR, "utt2spk"), "w") as f:
        f.write("\n".join(out_utt2spk) + "\n")
    with open(os.path.join(OUT_DIR, "feats_type"), "w") as f:
        f.write("raw\n")

    spk2utt = {}
    for line in out_utt2spk:
        utt, spk = line.split()
        spk2utt.setdefault(spk, []).append(utt)
    with open(os.path.join(OUT_DIR, "spk2utt"), "w") as f:
        for spk in sorted(spk2utt):
            f.write(f"{spk} {' '.join(spk2utt[spk])}\n")

    print(f"\nDone! Wrote {len(out_wav_scp)} segments to {OUT_DIR}")


if __name__ == "__main__":
    main()
