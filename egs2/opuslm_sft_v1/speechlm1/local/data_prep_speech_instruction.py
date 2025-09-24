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
    def build_cv_map(cv_dir: Path) -> dict:
        cv_map = {}
        for sub_dir in cv_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            for file in sub_dir.iterdir():
                if file.is_file():
                    cv_map[file.stem] = file.absolute()
        return cv_map

    if not hasattr(get_cv_path, "cv_map"):
        get_cv_path.cv_map = build_cv_map(audio_dir)

    filename = Path(filename).stem
    path = get_cv_path.cv_map.get(filename, None)
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

    logging.basicConfig(level=logging.INFO)
    
    audio_dir = {
        "vctk_en": vctk_dir,
        "multilingual_librispeech_en": mls_dir,
        "common_voice_en": cv_dir,
    }

    subsets = [
        "closed_ended/acoustic_level",
        "closed_ended/comparison", 
        "closed_ended/content_level",
        "closed_ended/word_align",
        "open_ended",
    ]
    lang = "en"  # We only use English data

    total_examples = 0

    for subset in subsets:
        curr_dir = root_dir / subset / lang
        if not curr_dir.exists():
            logging.warning(f"{curr_dir} does not exist. Skipping...")
            continue
            
        files = list(curr_dir.iterdir())
        logging.info(f"Processing subset: {subset} with {len(files)} files")

        for file in files:
            if not file.is_file() or file.suffix != '.jsonl':
                continue
                
            logging.info(f"Processing file: {file}")
            
            # Create separate output directory for each file to avoid OOM
            writer_dir = output_dir / f"{subset.replace('/', '_')}_{file.stem}"
            writer_dir.mkdir(exist_ok=True, parents=True)
            
            # Create separate dataset for each file to avoid storing all data in memory
            dataset = DialogueDataset(task="audio_text_dialogue")
            wav_writer = (writer_dir / "wav.scp").open(mode="w")
            
            lines = file.open(mode="r").readlines()
            metadata = [json.loads(line) for line in lines]
            
            file_examples = 0
            for data in metadata:
                try:
                    idx = data["id"]
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
                    
                    # User provides speech segment (as condition) - use codec_ssl for tokenized speech
                    dialogue.add_segment("user", "codec_ssl", False, audio_path_str)
                    
                    # User provides text instruction (as condition)  
                    dialogue.add_segment("user", "text_bpe", False, question)
                    
                    # Assistant provides text response (as target)
                    dialogue.add_segment("assistant", "text_bpe", True, answer)
                    
                    # Add to dataset
                    dataset.add_dialogue(dialogue_id, dialogue)
                    
                    # Write wav.scp entry
                    wav_writer.write(f"{dialogue_id}_turn0_speech {audio_path_str}\n")
                    
                    file_examples += 1
                        
                except Exception as e:
                    logging.error(f"Error processing example {idx}: {e}")
                    continue
            
            wav_writer.close()
            
            # Save dataset for this file
            if len(dataset.dialogues) > 0:
                logging.info(f"Saving {len(dataset.dialogues)} examples to {writer_dir}")
                dataset.dump_dataset(writer_dir)
                total_examples += file_examples
            else:
                logging.warning(f"No valid examples found in {file}")

    logging.info(f"Data preparation complete. Total: {total_examples} examples")


if __name__ == "__main__":
    main()