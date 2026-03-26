#!/usr/bin/env python3
"""Prepare SOT (Serialized Output Training) data from Lhotse CutSets.

Reads Lhotse CutSet .jsonl.gz files, applies a speaker ordering strategy,
and writes Kaldi-format data directories for ESPnet2 training.

Output files:
    wav.scp   — utt_id /path/to/audio.wav
    text      — utt_id <|startoftranscript|> spk1_text <sc> spk2_text <|endoftext|>
    utt2spk   — utt_id utt_id
    spk2utt   — utt_id utt_id

Usage:
    python local/prepare_sot.py \
        --cutset_paths manifest1.jsonl.gz manifest2.jsonl.gz \
        --output_dir data/train \
        --sot_strategy speaker_longest_first \
        --use_timestamps true \
        --max_timestamp_pause 2.0
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

from lhotse import CutSet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SOT_STRATEGIES = [
    "speaker_longest_first",
    "speaker_shortest_first",
    "speaker_start_time",
    "speaker_loudest_first",
    "speaker_quietest_first",
    "utterance_start_time",
]

SPEAKER_CHANGE_TOKEN = " <sc> "


def round_nearest(value: float, resolution: float) -> float:
    return round(value / resolution) * resolution


def get_cut_speakers(cut) -> list:
    """Get sorted unique speakers from a cut."""
    spks = set()
    for sup in cut.supervisions:
        spks.add(sup.speaker)
    return sorted(spks)


def merge_supervisions(supervisions, max_timestamp_pause: float):
    """Merge adjacent supervisions for the same speaker within pause threshold."""
    merged = []
    for sup in sorted(supervisions, key=lambda x: x.start):
        if len(merged) == 0:
            merged.append({
                "start": sup.start,
                "end": sup.start + sup.duration,
                "text": sup.text,
                "speaker": sup.speaker,
            })
        else:
            last = merged[-1]
            gap = sup.start - last["end"]
            if gap <= max_timestamp_pause:
                last["end"] = sup.start + sup.duration
                last["text"] = last["text"] + " " + sup.text
            else:
                merged.append({
                    "start": sup.start,
                    "end": sup.start + sup.duration,
                    "text": sup.text,
                    "speaker": sup.speaker,
                })
    return merged


def get_transcript_units(cut, use_timestamps: bool, max_timestamp_pause: float):
    """Extract per-speaker transcript units from a cut.

    Returns list of dicts with keys: speaker, text, start.
    """
    units = []
    speakers = get_cut_speakers(cut)

    for speaker_id in speakers:
        spk_supervisions = [s for s in cut.supervisions if s.speaker == speaker_id]
        merged = merge_supervisions(spk_supervisions, max_timestamp_pause)

        # Check if last segment is unfinished (custom attribute)
        last_segment_unfinished = False
        if hasattr(cut, "per_spk_flags"):
            last_segment_unfinished = cut.per_spk_flags.get(speaker_id, False)

        for idx, seg in enumerate(merged):
            text = seg["text"].strip()
            if not text:
                continue

            if use_timestamps:
                start_ts = f"<|{round_nearest(seg['start'], 0.02):.2f}|>"
                skip_end = (idx == len(merged) - 1) and last_segment_unfinished
                if skip_end:
                    text = f"{start_ts} {text}"
                else:
                    end_ts = f"<|{round_nearest(seg['end'], 0.02):.2f}|>"
                    text = f"{start_ts} {text}{end_ts}"

            units.append({
                "speaker": speaker_id,
                "text": text,
                "start": seg["start"],
            })

    return units


def merge_speaker_units(units, use_timestamps: bool):
    """Merge all utterance units per speaker into one entry."""
    spk_map = {}
    for u in units:
        spk_map.setdefault(u["speaker"], []).append(u)

    items = []
    joiner = "" if use_timestamps else " "
    for spk, utts in spk_map.items():
        items.append({
            "speaker": spk,
            "text": joiner.join(u["text"] for u in utts),
            "start": min(u["start"] for u in utts),
        })
    return items


def sort_units(items, sot_strategy: str, energy_per_spk=None):
    """Sort transcript items according to the SOT strategy."""
    if sot_strategy.endswith("start_time"):
        return sorted(items, key=lambda x: x["start"])
    elif sot_strategy.endswith("longest_first"):
        return sorted(items, key=lambda x: len(x["text"]), reverse=True)
    elif sot_strategy.endswith("shortest_first"):
        return sorted(items, key=lambda x: len(x["text"]))
    elif sot_strategy.endswith("loudest_first"):
        if energy_per_spk is None:
            return items
        return sorted(
            items,
            key=lambda x: energy_per_spk.get(x["speaker"], 0),
            reverse=True,
        )
    elif sot_strategy.endswith("quietest_first"):
        if energy_per_spk is None:
            return items
        return sorted(
            items,
            key=lambda x: energy_per_spk.get(x["speaker"], 0),
        )
    return items


def build_sot_text(
    cut,
    sot_strategy: str,
    use_timestamps: bool,
    max_timestamp_pause: float,
    use_spk_count_tokens: bool,
    use_spk_rem_tokens: bool = False,
    use_spk_id_tokens: bool = False,
):
    """Build the serialized SOT text for a single cut.

    Speaker token formats (mutually exclusive except spk_count can combine):
      - spk_count: prepend <|Nspk|> once at start (total speaker count)
      - spk_rem:   prepend <|Nspk_rem|> before each speaker (remaining count)
      - spk_id:    prepend <|spkN|> before each item (sequential speaker ID)

    Returns the full text string with speaker change tokens.
    """
    units = get_transcript_units(cut, use_timestamps, max_timestamp_pause)

    if sot_strategy.startswith("utterance"):
        items = units
    elif sot_strategy.startswith("speaker"):
        items = merge_speaker_units(units, use_timestamps)
    else:
        raise ValueError(f"Unknown SOT strategy: {sot_strategy}")

    energy_per_spk = getattr(cut, "per_spk_energy", None)
    items = sort_units(items, sot_strategy, energy_per_spk=energy_per_spk)

    # Apply per-item speaker tokens if requested
    if use_spk_id_tokens:
        spk_to_id = {}
        next_id = 1
        parts = []
        for item in items:
            spk = item["speaker"]
            if spk not in spk_to_id:
                spk_to_id[spk] = min(next_id, 5)
                next_id += 1
            parts.append(f"<|spk{spk_to_id[spk]}|>" + item["text"])
        text = SPEAKER_CHANGE_TOKEN.join(parts)
    elif use_spk_rem_tokens:
        parts = []
        for i, item in enumerate(items):
            remaining = min(len(items) - i, 5)
            parts.append(f"<|{remaining}spk_rem|>" + item["text"])
        text = SPEAKER_CHANGE_TOKEN.join(parts)
    else:
        text = SPEAKER_CHANGE_TOKEN.join(x["text"] for x in items)

    # spk_count is orthogonal — prepend total speaker count at the very start
    if use_spk_count_tokens:
        n = min(len(get_cut_speakers(cut)), 5)
        text = f"<|{n}spk|>" + text

    return text


def get_audio_entry(cut, segments_dir: Optional[str] = None) -> Optional[str]:
    """Get wav.scp entry for a cut's audio.

    If the cut is a segment of a longer recording and segments_dir is
    provided, extracts the audio segment to a WAV file and returns its
    path.  Otherwise returns the plain recording file path.
    """
    if not hasattr(cut, "recording") or cut.recording is None:
        return None

    audio_path = None
    for source in cut.recording.sources:
        if source.type == "file":
            audio_path = source.source
            break
    if audio_path is None:
        return None

    # Check if the cut is a segment of a longer recording
    is_segment = (
        cut.start > 0.01
        or abs(cut.duration - cut.recording.duration) > 0.01
    )
    if is_segment and segments_dir is not None:
        import soundfile as sf

        seg_path = os.path.join(segments_dir, f"{cut.id}.wav")
        if not os.path.exists(seg_path):
            audio = cut.load_audio()  # shape: (channels, samples)
            sf.write(seg_path, audio[0], cut.sampling_rate)
        return seg_path
    else:
        return audio_path


def process_cutset(
    cutset: CutSet,
    sot_strategy: str,
    use_timestamps: bool,
    max_timestamp_pause: float,
    use_spk_count_tokens: bool,
    use_spk_rem_tokens: bool = False,
    use_spk_id_tokens: bool = False,
    segments_dir: Optional[str] = None,
) -> tuple:
    """Process a CutSet and return lists of (utt_id, wav_path, text).

    Returns:
        entries: list of (utt_id, wav_path, text) tuples
        stats: dict with processing statistics
    """
    entries = []
    stats = {"total": 0, "skipped_no_audio": 0, "skipped_no_text": 0,
             "skipped_too_long": 0, "processed": 0}

    for cut in cutset:
        stats["total"] += 1
        utt_id = cut.id

        # Filter out cuts >= 30s.  Whisper timestamp tokens only cover
        # 0-30 seconds, so supervisions in longer cuts would produce
        # out-of-vocabulary token IDs.  This matches the
        # LhotseLongFormDataset filter in TS-ASR-Whisper.
        if cut.duration >= 30.0:
            stats["skipped_too_long"] += 1
            logger.debug(f"Skipping {utt_id}: duration {cut.duration:.2f}s >= 30s")
            continue

        # Get audio wav.scp entry (extract segment if needed)
        audio_path = get_audio_entry(cut, segments_dir=segments_dir)
        if audio_path is None:
            stats["skipped_no_audio"] += 1
            logger.debug(f"Skipping {utt_id}: no audio file path")
            continue

        # Build SOT text
        text = build_sot_text(
            cut,
            sot_strategy=sot_strategy,
            use_timestamps=use_timestamps,
            max_timestamp_pause=max_timestamp_pause,
            use_spk_count_tokens=use_spk_count_tokens,
            use_spk_rem_tokens=use_spk_rem_tokens,
            use_spk_id_tokens=use_spk_id_tokens,
        )

        if not text.strip():
            stats["skipped_no_text"] += 1
            logger.debug(f"Skipping {utt_id}: empty text")
            continue

        # Append <|endoftext|> so the model learns to generate it.
        # Do NOT prepend <|startoftranscript|> — the ESPnet preprocessor's
        # tokens2ids adds the SOT prefix [<|en|>, <|transcribe|>, <|notimestamps|>],
        # and the model's _shift_tokens_right adds <|startoftranscript|> as
        # decoder_start_token_id.
        full_text = f"{text} <|endoftext|>"

        entries.append((utt_id, audio_path, full_text))
        stats["processed"] += 1

    return entries, stats


def write_kaldi_data(entries: list, output_dir: str):
    """Write Kaldi-format data files."""
    os.makedirs(output_dir, exist_ok=True)

    wav_scp_path = os.path.join(output_dir, "wav.scp")
    text_path = os.path.join(output_dir, "text")
    utt2spk_path = os.path.join(output_dir, "utt2spk")
    spk2utt_path = os.path.join(output_dir, "spk2utt")

    # Sort by utt_id for Kaldi compatibility
    entries.sort(key=lambda x: x[0])

    with open(wav_scp_path, "w") as f_wav, \
         open(text_path, "w") as f_text, \
         open(utt2spk_path, "w") as f_utt2spk, \
         open(spk2utt_path, "w") as f_spk2utt:
        for utt_id, wav_path, text in entries:
            f_wav.write(f"{utt_id} {wav_path}\n")
            f_text.write(f"{utt_id} {text}\n")
            # For SOT, each utterance is its own "speaker"
            f_utt2spk.write(f"{utt_id} {utt_id}\n")
            f_spk2utt.write(f"{utt_id} {utt_id}\n")

    logger.info(f"Wrote {len(entries)} entries to {output_dir}")
    logger.info(f"  wav.scp:  {wav_scp_path}")
    logger.info(f"  text:     {text_path}")
    logger.info(f"  utt2spk:  {utt2spk_path}")
    logger.info(f"  spk2utt:  {spk2utt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SOT data from Lhotse CutSets for ESPnet2 training.",
    )
    parser.add_argument(
        "--cutset_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to Lhotse CutSet .jsonl.gz files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for Kaldi-format data",
    )
    parser.add_argument(
        "--sot_strategy",
        type=str,
        default="speaker_longest_first",
        choices=SOT_STRATEGIES,
        help="SOT speaker ordering strategy",
    )
    parser.add_argument(
        "--use_timestamps",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Include timestamps in text output",
    )
    parser.add_argument(
        "--max_timestamp_pause",
        type=float,
        default=2.0,
        help="Max pause (seconds) for merging adjacent segments",
    )
    parser.add_argument(
        "--use_spk_count_tokens",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Prepend speaker count token <|Nspk|> before transcript",
    )
    parser.add_argument(
        "--use_spk_rem_tokens",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Prepend remaining speaker count <|Nspk_rem|> before each speaker chunk",
    )
    parser.add_argument(
        "--use_spk_id_tokens",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Prepend speaker ID <|spkN|> before each utterance/chunk",
    )
    parser.add_argument(
        "--added_tokens_file",
        type=str,
        default=None,
        help="Path to added_tokens.txt for token validation (optional)",
    )
    args = parser.parse_args()

    # Mutual exclusivity check (matches TS-ASR-Whisper)
    if args.use_spk_id_tokens and args.use_spk_rem_tokens:
        parser.error("--use_spk_id_tokens and --use_spk_rem_tokens are mutually exclusive")

    # Validate tokens against added_tokens_file if provided
    if args.added_tokens_file is not None:
        try:
            with open(args.added_tokens_file) as f:
                registered_tokens = {line.strip() for line in f if line.strip()}
            logger.info(
                f"Validating against {args.added_tokens_file}: {registered_tokens}"
            )
            if "<sc>" not in registered_tokens:
                logger.warning(
                    f"<sc> not found in {args.added_tokens_file} — "
                    f"speaker change tokens will be split into subwords by the tokenizer"
                )
            if args.use_spk_count_tokens:
                expected = {f"<|{n}spk|>" for n in range(1, 6)}
                missing = expected - registered_tokens
                if missing:
                    logger.warning(
                        f"--use_spk_count_tokens is enabled but these tokens "
                        f"are missing from {args.added_tokens_file}: {sorted(missing)} — "
                        f"they will be split into subwords by the tokenizer"
                    )
            if args.use_spk_rem_tokens:
                expected = {f"<|{n}spk_rem|>" for n in range(1, 6)}
                missing = expected - registered_tokens
                if missing:
                    logger.warning(
                        f"--use_spk_rem_tokens is enabled but these tokens "
                        f"are missing from {args.added_tokens_file}: {sorted(missing)} — "
                        f"they will be split into subwords by the tokenizer"
                    )
            if args.use_spk_id_tokens:
                expected = {f"<|spk{n}|>" for n in range(1, 6)}
                missing = expected - registered_tokens
                if missing:
                    logger.warning(
                        f"--use_spk_id_tokens is enabled but these tokens "
                        f"are missing from {args.added_tokens_file}: {sorted(missing)} — "
                        f"they will be split into subwords by the tokenizer"
                    )
        except FileNotFoundError:
            logger.warning(
                f"added_tokens_file not found: {args.added_tokens_file}, "
                f"skipping validation"
            )

    logger.info(f"SOT strategy: {args.sot_strategy}")
    logger.info(f"Use timestamps: {args.use_timestamps}")
    logger.info(f"Max timestamp pause: {args.max_timestamp_pause}")
    logger.info(f"Use speaker count tokens: {args.use_spk_count_tokens}")
    logger.info(f"Use speaker remaining tokens: {args.use_spk_rem_tokens}")
    logger.info(f"Use speaker ID tokens: {args.use_spk_id_tokens}")

    # Create segments_wav directory for pre-extracted audio segments
    segments_dir = os.path.join(args.output_dir, "segments_wav")
    os.makedirs(segments_dir, exist_ok=True)

    # Load and concatenate cutsets
    all_entries = []
    total_stats = {"total": 0, "skipped_no_audio": 0, "skipped_no_text": 0, "processed": 0}

    for cutset_path in args.cutset_paths:
        logger.info(f"Loading cutset: {cutset_path}")
        cutset = CutSet.from_file(cutset_path)

        entries, stats = process_cutset(
            cutset,
            sot_strategy=args.sot_strategy,
            use_timestamps=args.use_timestamps,
            max_timestamp_pause=args.max_timestamp_pause,
            use_spk_count_tokens=args.use_spk_count_tokens,
            use_spk_rem_tokens=args.use_spk_rem_tokens,
            use_spk_id_tokens=args.use_spk_id_tokens,
            segments_dir=segments_dir,
        )
        all_entries.extend(entries)
        for k in total_stats:
            total_stats[k] += stats[k]

    logger.info(f"Total stats: {total_stats}")

    # Check for duplicate utt_ids
    utt_ids = [e[0] for e in all_entries]
    if len(utt_ids) != len(set(utt_ids)):
        dupes = len(utt_ids) - len(set(utt_ids))
        logger.warning(f"Found {dupes} duplicate utt_ids — deduplicating by keeping first")
        seen = set()
        deduped = []
        for entry in all_entries:
            if entry[0] not in seen:
                seen.add(entry[0])
                deduped.append(entry)
        all_entries = deduped

    write_kaldi_data(all_entries, args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
