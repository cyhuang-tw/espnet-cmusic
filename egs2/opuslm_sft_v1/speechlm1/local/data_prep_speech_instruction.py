#!/usr/bin/env python3

# Copyright 2025 Chien-yu Huang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import uuid
import logging

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
        "--split_ratio",
        type=float,
        default=0.01,
        help="Ratio for validation split (default: 0.01)"
    )

    return parser


def get_mls_path(filename: str, audio_dir: Path) -> Path:
    # /path/to/xxx_yyy_zzz.wav -> xxx_yyy_zzz
    filename = Path(filename).stem
    tokens = filename.split("_")
    assert len(tokens) == 3
    speaker_id, book_id, chapter_id = tokens
    path = audio_dir / speaker_id / book_id / f"{speaker_id}_{book_id}_{chapter_id}.flac"
    if not path.exists():
        logging.warning(f"{path} does not exist.")
        return None
    return path


def get_vctk_path(filename: str, audio_dir: Path) -> Path:
    # /path/to/pxxx_yyy.wav -> pxxx_yyy
    filename = Path(filename).stem
    tokens = filename.split("_")
    assert len(tokens) == 2
    speaker_id, utterance_id = tokens
    path = audio_dir / speaker_id / f"{speaker_id}_{utterance_id}.wav"
    if not path.exists():
        logging.warning(f"{path} does not exist.")
        return None
    return path


def get_cv_path(filename: str, audio_dir: Path) -> Path:
    # CommonVoice path resolution (implement based on your CV structure)
    filename = Path(filename).stem
    path = audio_dir / f"{filename}.wav"
    if not path.exists():
        logging.warning(f"{path} does not exist.")
        return None
    return path


def get_path(filename: str, source: str, audio_dir: dict) -> Path:
    if source == "multilingual_librispeech_en":
        return get_mls_path(filename, audio_dir[source])
    elif source == "vctk_en":
        return get_vctk_path(filename, audio_dir[source])
    elif source == "common_voice_en":
        return get_cv_path(filename, audio_dir[source])
    else:
        raise ValueError(f"Not recognized data source: {source}")


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    output_dir = args.output_dir
    root_dir = args.root_dir
    vctk_dir = args.vctk_dir
    mls_dir = args.mls_dir
    cv_dir = args.cv_dir
    split_ratio = args.split_ratio

    logging.basicConfig(level=logging.INFO)
    
    audio_dir = {
        "vctk_en": vctk_dir,
        "multilingual_librispeech_en": mls_dir,
        "common_voice_en": cv_dir,
    }

    # Create dataset objects for speech instruction task
    train_dataset = DialogueDataset(task="audio_text_dialogue")
    valid_dataset = DialogueDataset(task="audio_text_dialogue")

    subsets = [
        "closed_ended/acoustic_level",
        "closed_ended/comparison", 
        "closed_ended/content_level",
        "closed_ended/word_align",
        "open_ended",
    ]
    lang = "en"  # We only use English data

    total_examples = 0
    valid_examples = 0

    for subset in subsets:
        curr_dir = root_dir / subset / lang
        if not curr_dir.exists():
            logging.warning(f"{curr_dir} does not exist. Skipping...")
            continue
            
        files = list(curr_dir.iterdir())
        logging.info(f"Processing subset: {subset} with {len(files)} files")

        for file in files:
            if not file.is_file() or file.suffix != '.json':
                continue
                
            logging.info(f"Processing file: {file}")
            
            lines = file.open(mode="r").readlines()
            metadata = [json.loads(line) for line in lines]
            
            for data in metadata:
                try:
                    idx = data["id"]
                    task = data["task"]
                    source = data["data_source"]
                    question = data["messages"][0]["content"][1]["text"]
                    audio = data["messages"][0]["content"][0]["audio_path"]
                    answer = data["messages"][1]["content"][0]["text"]
                    
                    # Validate data
                    if not all([question, answer, audio]):
                        logging.warning(f"Missing data for example {idx}. Skipping...")
                        continue
                    
                    # Get audio path
                    audio_path = get_path(audio, source, audio_dir)
                    if audio_path is None:
                        logging.warning(f"Audio file not found for {idx}. Skipping...")
                        continue
                    
                    audio_path_str = audio_path.as_posix()
                    
                    # Create unique dialogue ID
                    random_uuid = uuid.uuid4()
                    dialogue_id = f"{subset.replace('/', '_')}_{file.stem}_{idx}_{random_uuid}"
                    
                    # Create dialogue with speech input + text instruction + text output
                    dialogue = Dialogue(task="audio_text_dialogue")
                    
                    # User provides speech segment (as condition)
                    dialogue.add_segment("user", "codec", False, audio_path_str)
                    
                    # User provides text instruction (as condition)  
                    dialogue.add_segment("user", "text_bpe", False, question)
                    
                    # Assistant provides text response (as target)
                    dialogue.add_segment("assistant", "text_bpe", True, answer)
                    
                    # Decide train/valid split
                    total_examples += 1
                    if total_examples % int(1 / split_ratio) == 0:
                        valid_dataset.add_dialogue(dialogue_id, dialogue)
                        valid_examples += 1
                    else:
                        train_dataset.add_dialogue(dialogue_id, dialogue)
                        
                except Exception as e:
                    logging.error(f"Error processing example {idx}: {e}")
                    continue

    # Save datasets
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"
    
    logging.info(f"Saving {len(train_dataset.dialogues)} training examples to {train_dir}")
    train_dataset.dump_dataset(train_dir)
    
    logging.info(f"Saving {len(valid_dataset.dialogues)} validation examples to {valid_dir}")
    valid_dataset.dump_dataset(valid_dir)
    
    logging.info(f"Data preparation complete. Total: {total_examples}, Train: {len(train_dataset.dialogues)}, Valid: {len(valid_dataset.dialogues)}")


if __name__ == "__main__":
    main()