#!/usr/bin/env python3

# Copyright 2025 Chien-yu Huang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import uuid
import logging
import random
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output data folder for training"
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        help="Root dir of SIFT50M-style data"
    )
    parser.add_argument(
        "--vctk_dir",
        type=Path,
        help="Path to VCTK corpus"
    )
    parser.add_argument(
        "--mls_dir",
        type=Path,
        help="Path to Multilingual LibriSpeech corpus"
    )
    parser.add_argument(
        "--cv_dir",
        type=Path,
        help="Path to CommonVoice corpus"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for writing files (default: 1000)"
    )

    return parser


# Pre-build CV map globally to avoid rebuilding in each worker
CV_MAP = None

def build_cv_map(cv_dir: Path) -> dict:
    """Build CommonVoice file mapping once"""
    cv_map = {}
    for sub_dir in cv_dir.iterdir():
        if not sub_dir.is_dir():
            continue
        for file in sub_dir.iterdir():
            cv_map[file.stem] = file.absolute()
    return cv_map


def init_worker(cv_dir: Optional[Path]):
    """Initialize worker process with shared data"""
    global CV_MAP
    if cv_dir is not None:
        CV_MAP = build_cv_map(cv_dir)


def get_mls_path(filename: str, audio_dir: Path) -> Optional[Path]:
    # /path/to/xxx_yyy_zzz.wav -> xxx_yyy_zzz
    filename = Path(filename).stem
    tokens = filename.split("_")
    if len(tokens) != 3:
        return None
    speaker_id, book_id, chapter_id = tokens
    path = audio_dir / speaker_id / book_id / f"{speaker_id}_{book_id}_{chapter_id}.flac"
    return path if path.exists() else None


def get_vctk_path(filename: str, audio_dir: Path) -> Optional[Path]:
    # /path/to/pxxx_yyy.wav -> pxxx_yyy
    filename = Path(filename).stem
    tokens = filename.split("_")
    if len(tokens) != 2:
        return None
    speaker_id, utterance_id = tokens
    path = audio_dir / speaker_id / f"{speaker_id}_{utterance_id}.wav"
    return path if path.exists() else None


def get_cv_path(filename: str, audio_dir: Path) -> Optional[Path]:
    global CV_MAP
    filename = Path(filename).stem
    path = CV_MAP.get(filename, None)
    return path if path is not None and path.exists() else None


def get_path(filename: str, source: str, audio_dir: dict) -> Optional[Path]:
    if source == "multilingual_librispeech_en":
        return get_mls_path(filename, audio_dir[source])
    elif source == "vctk_en":
        return get_vctk_path(filename, audio_dir[source])
    elif source == "common_voice_en":
        return get_cv_path(filename, audio_dir[source])
    else:
        return None


def process_single_file(
    file: Path,
    subset: str,
    output_dir: Path,
    audio_dir: dict,
    batch_size: int = 1000
) -> Tuple[int, int]:
    """Process a single JSONL file and write outputs"""

    # Create separate output directory for each file
    writer_dir = output_dir / f"{subset.replace('/', '_')}_{file.stem}"
    if writer_dir.exists():
        logging.info(f"Skipping existing directory: {writer_dir}")
        return 0, 0

    writer_dir.mkdir(exist_ok=True, parents=True)

    # Batch buffers for writing
    wav_buffer = []
    instr_buffer = []
    output_buffer = []

    file_examples = 0
    skipped = 0

    # Process file line by line (lazy reading)
    with file.open(mode="r") as f:
        for line in f:
            try:
                data = json.loads(line)

                idx = data["id"]
                source = data["data_source"]
                question = data["messages"][0]["content"][1]["text"]
                audio = data["messages"][0]["content"][0]["audio_path"]
                answer = data["messages"][1]["content"][0]["text"]

                # Validate data
                if not all([question, answer, audio]):
                    skipped += 1
                    continue

                # Get audio path
                audio_path = get_path(audio, source, audio_dir)
                if audio_path is None:
                    skipped += 1
                    continue

                audio_path_str = audio_path.as_posix()

                # Create unique dialogue ID
                random_uuid = uuid.uuid4()
                dialogue_id = f"{subset.replace('/', '_')}_{file.stem}_{idx}_{random_uuid}"

                # Buffer writes
                wav_buffer.append(f"{dialogue_id} {audio_path_str}\n")
                instr_buffer.append(f"{dialogue_id} {question}\n")
                output_buffer.append(f"{dialogue_id} {answer}\n")

                file_examples += 1

                # Flush buffers when batch size is reached
                if len(wav_buffer) >= batch_size:
                    with (writer_dir / "wav.scp").open(mode="a") as wav_writer:
                        wav_writer.writelines(wav_buffer)
                    with (writer_dir / "prompt").open(mode="a") as instr_writer:
                        instr_writer.writelines(instr_buffer)
                    with (writer_dir / "text").open(mode="a") as output_writer:
                        output_writer.writelines(output_buffer)

                    wav_buffer.clear()
                    instr_buffer.clear()
                    output_buffer.clear()

            except Exception as e:
                logging.error(f"Error processing line in {file}: {e}")
                skipped += 1
                continue

    # Flush remaining buffers
    if wav_buffer:
        with (writer_dir / "wav.scp").open(mode="a") as wav_writer:
            wav_writer.writelines(wav_buffer)
        with (writer_dir / "prompt").open(mode="a") as instr_writer:
            instr_writer.writelines(instr_buffer)
        with (writer_dir / "text").open(mode="a") as output_writer:
            output_writer.writelines(output_buffer)

    logging.info(f"Processed {file.name}: {file_examples} examples, {skipped} skipped")
    return file_examples, skipped


def process_subset(
    subset: str,
    root_dir: Path,
    output_dir: Path,
    audio_dir: dict,
    batch_size: int,
    num_workers: int
) -> Tuple[int, int]:
    """Process all files in a subset"""

    curr_dir = root_dir / subset / "en"
    if not curr_dir.exists():
        logging.warning(f"{curr_dir} does not exist. Skipping...")
        return 0, 0

    files = [f for f in curr_dir.iterdir() if f.is_file() and f.suffix == '.jsonl']
    random.shuffle(files)
    files = files[:min(50, len(files))]  # Limit to 50 files per subset

    logging.info(f"Processing subset: {subset} with {len(files)} files")

    if not files:
        return 0, 0

    # Process files in parallel
    process_func = partial(
        process_single_file,
        subset=subset,
        output_dir=output_dir,
        audio_dir=audio_dir,
        batch_size=batch_size
    )

    total_examples = 0
    total_skipped = 0

    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_func, files)

        for examples, skipped in results:
            total_examples += examples
            total_skipped += skipped
    else:
        # Single-threaded processing
        for file in files:
            examples, skipped = process_func(file)
            total_examples += examples
            total_skipped += skipped

    return total_examples, total_skipped


def main():
    random.seed(42)
    parser = get_parser()
    args = parser.parse_args()

    output_dir = args.output_dir
    root_dir = args.root_dir
    vctk_dir = args.vctk_dir
    mls_dir = args.mls_dir
    cv_dir = args.cv_dir
    num_workers = args.num_workers or cpu_count()
    batch_size = args.batch_size

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Using {num_workers} workers, batch size {batch_size}")

    audio_dir = {
        "vctk_en": vctk_dir,
        "multilingual_librispeech_en": mls_dir,
        "common_voice_en": cv_dir,
    }

    # Pre-build CV map for multiprocessing
    if cv_dir:
        logging.info("Building CommonVoice file map...")
        global CV_MAP
        CV_MAP = build_cv_map(cv_dir)
        logging.info(f"CV map built with {len(CV_MAP)} entries")

        # Initialize worker processes with CV map
        init_worker(cv_dir)

    subsets = [
        "closed_ended/acoustic_level",
        "closed_ended/comparison",
        "closed_ended/content_level",
        "closed_ended/word_align",
        "open_ended",
    ]

    grand_total_examples = 0
    grand_total_skipped = 0

    for subset in subsets:
        examples, skipped = process_subset(
            subset=subset,
            root_dir=root_dir,
            output_dir=output_dir,
            audio_dir=audio_dir,
            batch_size=batch_size,
            num_workers=num_workers
        )
        grand_total_examples += examples
        grand_total_skipped += skipped
        logging.info(f"Subset {subset}: {examples} examples, {skipped} skipped")

    logging.info(f"Data preparation complete. Total: {grand_total_examples} examples, {grand_total_skipped} skipped")


if __name__ == "__main__":
    main()
