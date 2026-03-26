"""SOT post-processing utilities.

Ported from the original DiCoW codebase:
  - truncate_at_repeating_ngram: src/data/postprocess.py
  - process_sot_output: adapted from src/utils/evaluation.py:process_session_sot
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


def count_ngrams(text: str, min_n: int = 2, max_n: int = 5) -> Dict[str, int]:
    """Count n-gram occurrences, skipping n-grams where all words are identical."""
    words = text.split()
    counts: Dict[str, int] = defaultdict(int)
    for n in range(min_n, max_n + 1):
        for i in range(len(words) - n + 1):
            ngram_words = words[i : i + n]
            if all(
                word.lower() == ngram_words[0].lower() for word in ngram_words
            ):
                continue
            ngram = " ".join(ngram_words)
            counts[ngram] += 1
    return counts


def truncate_at_repeating_ngram(
    text: str,
    ngram_length: int = 10,
    min_n: int = 1,
    max_n: int = None,
    min_word_threshold: int = 30,
    unigram_min_repeat: int = 10,
    repeat_threshold: int = 10,
) -> str:
    """Truncate text at the first repeating n-gram above threshold.

    Args:
        text: Input text to process.
        ngram_length: Target n-gram length to check for.
        min_n: Minimum n-gram size to check.
        max_n: Maximum n-gram size to check (defaults to ngram_length).
        min_word_threshold: Minimum words required before processing.
        unigram_min_repeat: Minimum consecutive repeats for unigrams.
        repeat_threshold: Minimum total occurrences to consider repeating.

    Returns:
        Truncated text, or original text if no repetition found.
    """
    if max_n is None:
        max_n = ngram_length

    words = text.split()
    if len(words) < min_word_threshold:
        return text

    earliest_truncation_idx = len(words)

    # Handle unigrams with consecutive repetition
    if min_n == 1:
        for i in range(len(words) - unigram_min_repeat + 1):
            current_word = words[i].lower()
            consecutive_count = 1
            for j in range(i + 1, len(words)):
                if words[j].lower() == current_word:
                    consecutive_count += 1
                else:
                    break
            if consecutive_count >= unigram_min_repeat:
                earliest_truncation_idx = min(
                    earliest_truncation_idx, i + 1
                )
                break

    # Count all n-grams
    all_ngram_counts = count_ngrams(text, min_n=max(2, min_n), max_n=max_n)

    # Find earliest occurrence of any repeated n-gram above threshold
    lengths_to_check = [ngram_length] + [
        n for n in range(min_n, max_n + 1) if n != ngram_length and n > 1
    ]

    for n in lengths_to_check:
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            if all_ngram_counts[ngram] > repeat_threshold:
                earliest_truncation_idx = min(
                    earliest_truncation_idx, i + n
                )

    if earliest_truncation_idx < len(words):
        return " ".join(words[:earliest_truncation_idx])
    return text


def process_sot_output(
    token_int: List[int],
    hf_tokenizer,
    separator_token_id: int,
    spk_count_token_ids: Optional[List[int]] = None,
    spk_rem_token_ids: Optional[List[int]] = None,
    spk_id_token_ids: Optional[List[int]] = None,
    task_token_ids: Optional[List[int]] = None,
    separator_str: str = "<sc>",
) -> Tuple[List[str], str, Optional[int], List[Optional[int]]]:
    """Process SOT output token IDs into per-speaker transcripts.

    Splits token IDs by separator BEFORE decoding each speaker block to
    avoid the Whisper tokenizer adding spurious 30s offsets when timestamps
    restart after a speaker change.

    Adapted from src/utils/evaluation.py:process_session_sot.

    Args:
        token_int: List of output token IDs (without SOS/EOS).
        hf_tokenizer: HuggingFace WhisperTokenizer instance.
        separator_token_id: Token ID of the separator (<sc>).
        spk_count_token_ids: Ordered list of speaker count token IDs.
        spk_rem_token_ids: Ordered list of speaker remaining token IDs.
        spk_id_token_ids: Ordered list of speaker ID token IDs.
        task_token_ids: Task-specific token IDs to filter out.
        separator_str: String representation of separator for raw transcript.

    Returns:
        Tuple of:
          - per_speaker_texts: List of decoded text per speaker block.
          - raw_transcript: Full transcript with separator_str between blocks.
          - predicted_n_spk: Predicted speaker count (or None).
          - block_speaker_ids: Per-block speaker ID (1-indexed, or None).
    """
    spk_count_set: Set[int] = (
        set(spk_count_token_ids) if spk_count_token_ids else set()
    )
    spk_id_set: Set[int] = (
        set(spk_id_token_ids) if spk_id_token_ids else set()
    )
    spk_rem_set: Set[int] = (
        set(spk_rem_token_ids) if spk_rem_token_ids else set()
    )
    task_set: Set[int] = set(task_token_ids) if task_token_ids else set()
    filter_set: Set[int] = spk_count_set | spk_rem_set | spk_id_set | task_set

    # Extract predicted speaker count before filtering
    predicted_n_spk: Optional[int] = None
    if spk_count_token_ids:
        for tok_id in token_int:
            if tok_id in spk_count_set:
                predicted_n_spk = spk_count_token_ids.index(tok_id) + 1
                break

    # Split by separator, filter specials, track speaker IDs
    blocks: List[List[int]] = []
    block_speaker_ids: List[Optional[int]] = []
    current: List[int] = []
    current_spk_id: Optional[int] = None

    pad_token_id = getattr(hf_tokenizer, "pad_token_id", None)

    for tok_id in token_int:
        if tok_id == separator_token_id:
            blocks.append(current)
            block_speaker_ids.append(current_spk_id)
            current = []
            current_spk_id = None
        elif spk_id_set and tok_id in spk_id_set:
            current_spk_id = spk_id_token_ids.index(tok_id) + 1
        elif tok_id != pad_token_id and tok_id not in filter_set:
            current.append(tok_id)

    if current:
        blocks.append(current)
        block_speaker_ids.append(current_spk_id)

    # Decode each block and apply hallucination truncation
    per_speaker_texts: List[str] = []
    raw_parts: List[str] = []
    for block_ids in blocks:
        if not block_ids:
            raw_parts.append("")
            per_speaker_texts.append("")
            continue
        # Use convert_ids_to_tokens + convert_tokens_to_string to preserve
        # timestamp tokens (decode() silently drops them with added tokens).
        raw_tokens = hf_tokenizer.convert_ids_to_tokens(block_ids)
        decoded = hf_tokenizer.convert_tokens_to_string(raw_tokens)
        raw_parts.append(decoded)
        per_speaker_texts.append(truncate_at_repeating_ngram(decoded))

    raw_transcript = separator_str.join(raw_parts)
    return per_speaker_texts, raw_transcript, predicted_n_spk, block_speaker_ids
