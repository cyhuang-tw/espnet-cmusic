#!/usr/bin/env python3
"""Stage 1 filtering: Filter by caption token count and audio duration."""

import argparse
import json
import os
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

from nltk import ngrams

DISCARD_REASONS = [
    "finish_reason",
    "missing_fields",
    "non_latin",
    "char_repetition",
    "word_repetition",
    "ngram_repetition",
    "unique_word_ratio_low",
    "avg_word_length_low",
    "avg_word_length_high",
    "uppercase_ratio_high",
    "tokens_low",
    "tokens_high",
    "duration_low",
    "duration_high",
]

# Unicode ranges for non-Latin scripts (to be filtered out)
NON_LATIN_RANGES = [
    (0x0400, 0x04FF),    # Cyrillic
    (0x0500, 0x052F),    # Cyrillic Supplement
    (0x0600, 0x06FF),    # Arabic
    (0x0750, 0x077F),    # Arabic Supplement
    (0x0590, 0x05FF),    # Hebrew
    (0x0900, 0x097F),    # Devanagari
    (0x0980, 0x09FF),    # Bengali
    (0x0E00, 0x0E7F),    # Thai
    (0x1100, 0x11FF),    # Hangul Jamo (Korean)
    (0x3000, 0x303F),    # CJK Symbols and Punctuation
    (0x3040, 0x309F),    # Hiragana (Japanese)
    (0x30A0, 0x30FF),    # Katakana (Japanese)
    (0x3100, 0x312F),    # Bopomofo
    (0x3130, 0x318F),    # Hangul Compatibility Jamo
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs (Chinese/Japanese/Korean)
    (0xAC00, 0xD7AF),    # Hangul Syllables (Korean)
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),    # Halfwidth and Fullwidth Forms
]


def contains_non_latin(text: str) -> bool:
    """Check if text contains non-Latin script characters."""
    for char in text:
        code = ord(char)
        for start, end in NON_LATIN_RANGES:
            if start <= code <= end:
                return True
    return False


def get_max_char_repetition(text: str) -> int:
    """Get the maximum consecutive repeated character count."""
    if not text:
        return 0
    max_rep = 1
    for match in re.finditer(r'(.)\1+', text):
        rep_len = len(match.group())
        if rep_len > max_rep:
            max_rep = rep_len
    return max_rep


def get_max_word_repetition(words: list) -> int:
    """Get the maximum consecutive repeated word count."""
    if not words:
        return 0
    max_rep = 1
    current_rep = 1
    for i in range(1, len(words)):
        if words[i].lower() == words[i - 1].lower():
            current_rep += 1
            max_rep = max(max_rep, current_rep)
        else:
            current_rep = 1
    return max_rep


def get_ngram_repetition_ratio(words: list, n: int) -> float:
    """Get the ratio of duplicate n-grams to total n-grams."""
    if len(words) < n:
        return 0.0
    ngram_list = list(ngrams(words, n))
    if not ngram_list:
        return 0.0
    ngram_counts = Counter(ngram_list)
    total = len(ngram_list)
    unique = len(ngram_counts)
    # Ratio of duplicates: 1 - (unique / total)
    return 1.0 - (unique / total)


def get_unique_word_ratio(words: list) -> float:
    """Get the ratio of unique words to total words."""
    if not words:
        return 0.0
    unique = len(set(w.lower() for w in words))
    return unique / len(words)


def get_avg_word_length(words: list) -> float:
    """Get the average word length."""
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def get_uppercase_ratio(text: str) -> float:
    """Get the ratio of uppercase letters to total letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    uppercase = sum(1 for c in letters if c.isupper())
    return uppercase / len(letters)


def filter_single_file(args: tuple) -> Tuple[List[str], dict, Dict[str, List[dict]]]:
    """Filter a single JSONL file based on criteria.

    Args:
        args: Tuple of (file_path, min_tokens, max_tokens, min_dur, max_dur,
              max_char_rep, max_word_rep, max_ngram_rep_ratio, ngram_n,
              min_unique_word_ratio, min_avg_word_len, max_avg_word_len,
              max_uppercase_ratio).

    Returns:
        Tuple of (survived_ids, stats, discarded).
    """
    (
        file_path, min_tokens, max_tokens, min_dur, max_dur,
        max_char_rep, max_word_rep, max_ngram_rep_ratio, ngram_n,
        min_unique_word_ratio, min_avg_word_len, max_avg_word_len,
        max_uppercase_ratio
    ) = args

    survived_ids = []
    discarded = {reason: [] for reason in DISCARD_REASONS}
    stats = {
        "total": 0,
        "survived": 0,
        "filtered_finish_reason": 0,
        "filtered_missing_fields": 0,
        "filtered_non_latin": 0,
        "filtered_char_repetition": 0,
        "filtered_word_repetition": 0,
        "filtered_ngram_repetition": 0,
        "filtered_unique_word_ratio_low": 0,
        "filtered_avg_word_length_low": 0,
        "filtered_avg_word_length_high": 0,
        "filtered_uppercase_ratio_high": 0,
        "filtered_tokens_low": 0,
        "filtered_tokens_high": 0,
        "filtered_duration_low": 0,
        "filtered_duration_high": 0,
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                stats["total"] += 1

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    stats["filtered_missing_fields"] += 1
                    continue

                # Get utt_id early for tracking
                utt_id = data.get("utt_id")

                # Check finish_reason is "stop"
                if data.get("finish_reason") != "stop":
                    stats["filtered_finish_reason"] += 1
                    if utt_id:
                        discarded["finish_reason"].append({"utt_id": utt_id})
                    continue

                # Get required fields
                duration = data.get("duration")
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens")

                caption = data.get("caption", "")

                if utt_id is None or duration is None or completion_tokens is None:
                    stats["filtered_missing_fields"] += 1
                    if utt_id:
                        discarded["missing_fields"].append({"utt_id": utt_id})
                    continue

                # Filter by non-Latin script characters in caption
                if contains_non_latin(caption):
                    stats["filtered_non_latin"] += 1
                    discarded["non_latin"].append({"utt_id": utt_id})
                    continue

                # Tokenize caption for repetition checks
                words = caption.split()

                # Filter by character repetition
                char_rep = get_max_char_repetition(caption)
                if char_rep > max_char_rep:
                    stats["filtered_char_repetition"] += 1
                    discarded["char_repetition"].append({
                        "utt_id": utt_id, "max_char_rep": char_rep
                    })
                    continue

                # Filter by word repetition
                word_rep = get_max_word_repetition(words)
                if word_rep > max_word_rep:
                    stats["filtered_word_repetition"] += 1
                    discarded["word_repetition"].append({
                        "utt_id": utt_id, "max_word_rep": word_rep
                    })
                    continue

                # Filter by n-gram repetition ratio
                ngram_ratio = get_ngram_repetition_ratio(words, ngram_n)
                if ngram_ratio > max_ngram_rep_ratio:
                    stats["filtered_ngram_repetition"] += 1
                    discarded["ngram_repetition"].append({
                        "utt_id": utt_id, "ngram_ratio": round(ngram_ratio, 4)
                    })
                    continue

                # Filter by unique word ratio (lexical diversity)
                unique_ratio = get_unique_word_ratio(words)
                if unique_ratio < min_unique_word_ratio:
                    stats["filtered_unique_word_ratio_low"] += 1
                    discarded["unique_word_ratio_low"].append({
                        "utt_id": utt_id, "unique_ratio": round(unique_ratio, 4)
                    })
                    continue

                # Filter by average word length
                avg_word_len = get_avg_word_length(words)
                if avg_word_len < min_avg_word_len:
                    stats["filtered_avg_word_length_low"] += 1
                    discarded["avg_word_length_low"].append({
                        "utt_id": utt_id, "avg_word_len": round(avg_word_len, 2)
                    })
                    continue
                if avg_word_len > max_avg_word_len:
                    stats["filtered_avg_word_length_high"] += 1
                    discarded["avg_word_length_high"].append({
                        "utt_id": utt_id, "avg_word_len": round(avg_word_len, 2)
                    })
                    continue

                # Filter by uppercase ratio
                upper_ratio = get_uppercase_ratio(caption)
                if upper_ratio > max_uppercase_ratio:
                    stats["filtered_uppercase_ratio_high"] += 1
                    discarded["uppercase_ratio_high"].append({
                        "utt_id": utt_id, "upper_ratio": round(upper_ratio, 4)
                    })
                    continue

                # Filter by caption tokens
                if completion_tokens < min_tokens:
                    stats["filtered_tokens_low"] += 1
                    discarded["tokens_low"].append({
                        "utt_id": utt_id, "tokens": completion_tokens
                    })
                    continue
                if completion_tokens > max_tokens:
                    stats["filtered_tokens_high"] += 1
                    discarded["tokens_high"].append({
                        "utt_id": utt_id, "tokens": completion_tokens
                    })
                    continue

                # Filter by audio duration
                if duration < min_dur:
                    stats["filtered_duration_low"] += 1
                    discarded["duration_low"].append({
                        "utt_id": utt_id, "duration": duration
                    })
                    continue
                if duration > max_dur:
                    stats["filtered_duration_high"] += 1
                    discarded["duration_high"].append({
                        "utt_id": utt_id, "duration": duration
                    })
                    continue

                # All checks passed
                survived_ids.append(utt_id)
                stats["survived"] += 1

    except Exception as e:
        print(f"Error processing {file_path}: {e}", flush=True)
        return [], stats, discarded

    print(
        f"Processed {file_path}: {stats['survived']}/{stats['total']} survived",
        flush=True
    )
    return survived_ids, stats, discarded


def find_jsonl_files(input_dirs: List[str]) -> List[str]:
    """Find all JSONL files in the given directories.

    Args:
        input_dirs: List of input directory paths.

    Returns:
        Sorted list of JSONL file paths.
    """
    jsonl_files = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Warning: Directory {input_dir} does not exist", flush=True)
            continue

        # Find all .jsonl files
        for f in input_path.glob("**/*.jsonl"):
            if f.is_file():
                jsonl_files.append(str(f))

    return sorted(set(jsonl_files))


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 filtering: Filter by caption tokens and audio duration."
    )
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Input directories containing JSONL files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file path for survived example IDs.",
    )
    parser.add_argument(
        "--min_caption_tokens",
        type=int,
        default=100,
        help="Minimum caption tokens (inclusive, default: 100).",
    )
    parser.add_argument(
        "--max_caption_tokens",
        type=int,
        default=1024,
        help="Maximum caption tokens (inclusive, default: 1024).",
    )
    parser.add_argument(
        "--min_audio_duration",
        type=float,
        default=2.0,
        help="Minimum audio duration in seconds (inclusive, default: 2.0).",
    )
    parser.add_argument(
        "--max_audio_duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (inclusive, default: 30.0).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Number of worker processes (default: 64).",
    )
    parser.add_argument(
        "--max_char_repetition",
        type=int,
        default=5,
        help="Max consecutive repeated characters allowed (default: 5).",
    )
    parser.add_argument(
        "--max_word_repetition",
        type=int,
        default=3,
        help="Max consecutive repeated words allowed (default: 3).",
    )
    parser.add_argument(
        "--max_ngram_repetition_ratio",
        type=float,
        default=0.1,
        help="Max n-gram repetition ratio allowed (default: 0.1).",
    )
    parser.add_argument(
        "--ngram_n",
        type=int,
        default=5,
        help="N-gram size for repetition check (default: 5).",
    )
    parser.add_argument(
        "--min_unique_word_ratio",
        type=float,
        default=0.3,
        help="Min unique word ratio (default: 0.3).",
    )
    parser.add_argument(
        "--min_avg_word_length",
        type=float,
        default=2.0,
        help="Min average word length (default: 2.0).",
    )
    parser.add_argument(
        "--max_avg_word_length",
        type=float,
        default=15.0,
        help="Max average word length (default: 15.0).",
    )
    parser.add_argument(
        "--max_uppercase_ratio",
        type=float,
        default=0.5,
        help="Max uppercase letter ratio (default: 0.5).",
    )
    args = parser.parse_args()

    # Find all JSONL files
    jsonl_files = find_jsonl_files(args.input_dirs)
    if not jsonl_files:
        print("No JSONL files found in the input directories.")
        return

    print(f"Found {len(jsonl_files)} JSONL files to process", flush=True)

    # Prepare arguments for multiprocessing
    process_args = [
        (
            f,
            args.min_caption_tokens,
            args.max_caption_tokens,
            args.min_audio_duration,
            args.max_audio_duration,
            args.max_char_repetition,
            args.max_word_repetition,
            args.max_ngram_repetition_ratio,
            args.ngram_n,
            args.min_unique_word_ratio,
            args.min_avg_word_length,
            args.max_avg_word_length,
            args.max_uppercase_ratio,
        )
        for f in jsonl_files
    ]

    # Process files in parallel
    all_survived_ids = []
    all_discarded = {reason: [] for reason in DISCARD_REASONS}
    total_stats = {
        "total": 0,
        "survived": 0,
        "filtered_finish_reason": 0,
        "filtered_missing_fields": 0,
        "filtered_non_latin": 0,
        "filtered_char_repetition": 0,
        "filtered_word_repetition": 0,
        "filtered_ngram_repetition": 0,
        "filtered_unique_word_ratio_low": 0,
        "filtered_avg_word_length_low": 0,
        "filtered_avg_word_length_high": 0,
        "filtered_uppercase_ratio_high": 0,
        "filtered_tokens_low": 0,
        "filtered_tokens_high": 0,
        "filtered_duration_low": 0,
        "filtered_duration_high": 0,
    }

    with Pool(processes=args.num_workers) as pool:
        results = pool.map(filter_single_file, process_args)

    # Aggregate results
    for survived_ids, stats, discarded in results:
        all_survived_ids.extend(survived_ids)
        for key in total_stats:
            total_stats[key] += stats[key]
        for reason in DISCARD_REASONS:
            all_discarded[reason].extend(discarded[reason])

    # Write survived IDs to output file
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for utt_id in all_survived_ids:
            f.write(f"{utt_id}\n")

    # Write discarded samples to separate JSONL files
    discard_dir = os.path.join(output_dir, "discarded") if output_dir else "discarded"
    os.makedirs(discard_dir, exist_ok=True)
    for reason in DISCARD_REASONS:
        if all_discarded[reason]:
            discard_file = os.path.join(discard_dir, f"discarded_{reason}.jsonl")
            with open(discard_file, "w", encoding="utf-8") as f:
                for item in all_discarded[reason]:
                    f.write(json.dumps(item) + "\n")
            print(f"Wrote {len(all_discarded[reason]):,} to {discard_file}", flush=True)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("Stage 1 Filtering Summary", flush=True)
    print("=" * 60, flush=True)
    print(f"Total samples processed:    {total_stats['total']:,}", flush=True)
    print(f"Samples survived:           {total_stats['survived']:,}", flush=True)
    print(f"Survival rate:              {100*total_stats['survived']/max(1, total_stats['total']):.2f}%", flush=True)
    print("-" * 60, flush=True)
    print("Filtered out by reason:", flush=True)
    print(f"  - finish_reason != stop:  {total_stats['filtered_finish_reason']:,}", flush=True)
    print(f"  - missing fields:         {total_stats['filtered_missing_fields']:,}", flush=True)
    print(f"  - non-Latin script:       {total_stats['filtered_non_latin']:,}", flush=True)
    print(f"  - char repetition (>{args.max_char_repetition}): {total_stats['filtered_char_repetition']:,}", flush=True)
    print(f"  - word repetition (>{args.max_word_repetition}): {total_stats['filtered_word_repetition']:,}", flush=True)
    print(f"  - {args.ngram_n}-gram rep ratio (>{args.max_ngram_repetition_ratio}): {total_stats['filtered_ngram_repetition']:,}", flush=True)
    print(f"  - unique word ratio (<{args.min_unique_word_ratio}): {total_stats['filtered_unique_word_ratio_low']:,}", flush=True)
    print(f"  - avg word len (<{args.min_avg_word_length}): {total_stats['filtered_avg_word_length_low']:,}", flush=True)
    print(f"  - avg word len (>{args.max_avg_word_length}): {total_stats['filtered_avg_word_length_high']:,}", flush=True)
    print(f"  - uppercase ratio (>{args.max_uppercase_ratio}): {total_stats['filtered_uppercase_ratio_high']:,}", flush=True)
    print(f"  - tokens < {args.min_caption_tokens}: {total_stats['filtered_tokens_low']:,}", flush=True)
    print(f"  - tokens > {args.max_caption_tokens}: {total_stats['filtered_tokens_high']:,}", flush=True)
    print(f"  - duration < {args.min_audio_duration}s: {total_stats['filtered_duration_low']:,}", flush=True)
    print(f"  - duration > {args.max_audio_duration}s: {total_stats['filtered_duration_high']:,}", flush=True)
    print("=" * 60, flush=True)
    print(f"Output written to: {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
