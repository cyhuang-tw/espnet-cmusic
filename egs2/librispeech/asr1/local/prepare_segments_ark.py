#!/usr/bin/env python3
"""Segment audio into fixed-length chunks and store in Kaldi ark format.

Usage:
    python local/prepare_segments_ark.py \
        --src-dir dump/raw/maestro_dev_onsets \
        --src-base /work/nvme/bbjs/chen26/espnet_music/egs2/librispeech/asr1 \
        --out-dir dump/raw/maestro_dev_seg_4s \
        --segment-sec 4.0
"""

import argparse
import os

import kaldiio
import numpy as np
import soundfile as sf


def process_text(text_tokens, start_time, end_time):
    """Crop and normalize text tokens to a time window."""
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
    parser = argparse.ArgumentParser(
        description="Segment audio and store in Kaldi ark format"
    )
    parser.add_argument("--src-dir", required=True, help="Source data directory")
    parser.add_argument(
        "--src-base",
        default=".",
        help="Base path for resolving relative paths in wav.scp",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--segment-sec", type=float, default=4.0, help="Segment length in seconds"
    )
    parser.add_argument("--sr", type=int, default=16000, help="Expected sample rate")
    parser.add_argument(
        "--overlap-sec",
        type=float,
        default=0.0,
        help="Overlap between consecutive segments",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read source data
    wav_scp = {}
    with open(os.path.join(args.src_dir, "wav.scp")) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            wav_scp[parts[0]] = parts[1]

    text_data = {}
    with open(os.path.join(args.src_dir, "text")) as f:
        for line in f:
            parts = line.strip().split()
            text_data[parts[0]] = parts[1:]

    ark_path = os.path.join(args.out_dir, "feats.ark")
    scp_path = os.path.join(args.out_dir, "feats.scp")

    out_text = []
    out_utt2spk = []
    total_segments = 0
    step_sec = args.segment_sec - args.overlap_sec

    with kaldiio.WriteHelper(f"ark,scp:{ark_path},{scp_path}") as writer:
        for utt_id in sorted(wav_scp.keys()):
            wav_path = wav_scp[utt_id]
            if not os.path.isabs(wav_path):
                wav_path = os.path.join(args.src_base, wav_path)

            speech, sr = sf.read(wav_path)
            assert sr == args.sr, f"Expected {args.sr}Hz, got {sr}Hz for {utt_id}"
            tokens = text_data[utt_id]

            total_duration = len(speech) / args.sr
            num_windows = int((total_duration - args.overlap_sec) // step_sec)
            if num_windows == 0:
                num_windows = 1

            for w in range(num_windows):
                start_time = w * step_sec
                start_sample = int(start_time * args.sr)
                end_sample = int((start_time + args.segment_sec) * args.sr)

                if end_sample > len(speech):
                    break

                cropped = speech[start_sample:end_sample].astype(np.float32)
                actual_start = start_sample / args.sr
                actual_end = end_sample / args.sr

                cropped_tokens, n_events = process_text(tokens, actual_start, actual_end)

                seg_id = f"{utt_id}_w{w}"

                # Write audio to ark as 1D float32 array
                writer[seg_id] = cropped

                out_text.append(f"{seg_id} {' '.join(cropped_tokens)}")
                out_utt2spk.append(f"{seg_id} {seg_id}")
                total_segments += 1

    # Write text
    with open(os.path.join(args.out_dir, "text"), "w") as f:
        f.write("\n".join(out_text) + "\n")

    # Write utt2spk
    with open(os.path.join(args.out_dir, "utt2spk"), "w") as f:
        f.write("\n".join(out_utt2spk) + "\n")

    # Write spk2utt
    spk2utt = {}
    for line in out_utt2spk:
        utt, spk = line.split()
        spk2utt.setdefault(spk, []).append(utt)
    with open(os.path.join(args.out_dir, "spk2utt"), "w") as f:
        for spk in sorted(spk2utt):
            f.write(f"{spk} {' '.join(spk2utt[spk])}\n")

    # Write feats_type
    with open(os.path.join(args.out_dir, "feats_type"), "w") as f:
        f.write("raw\n")

    print(f"Done! {len(wav_scp)} utterances -> {total_segments} segments in {args.out_dir}")
    print(f"  ark: {ark_path}")
    print(f"  scp: {scp_path}")


if __name__ == "__main__":
    main()
