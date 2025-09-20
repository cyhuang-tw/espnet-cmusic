#!/usr/bin/env python3

# Copyright 2025 Chien-yu Huang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json

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
        help="Root dir of SIFT50M"
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
    assert path.exists(), f"{path} does not exist."
    return path

def get_vctk_path(filename: str, audio_dir: Path) -> Path:
    # /path/to/pxxx_yyy.wav -> pxxx_yyy
    filename = Path(filename).stem
    tokens = filename.split("_")
    assert len(tokens) == 2
    speaker_id, utterance_id = tokens
    path = audio_dir / speaker_id / f"{speaker_id}_{utterance_id}.wav"
    assert path.exists(), f"{path} does not exist."
    return path

def get_cv_path(filename: str, audio_dir: Path) -> Path:
    return None

def get_path(filename: str, source: str, audio_dir: dict) -> Path:
    if source == "multilingual_librispeech_en":
        return get_mls_path(filename, audio_dir[source])
    elif source == "vctk_en":
        return get_vctk_path(filename, audio_dir[source])
    elif source == "common_voice_en":
        return get_cv_path(filename, audio_dir[source])
    else:
        raise ValueError(f"Not recognized data source: {source}")

def main(output_dir: Path, root_dir: Path, vctk_dir: Path, mls_dir: Path, cv_dir: Path):
    audio_dir = {
                    "vctk_en": vctk_dir,
                    "multilingual_librispeech_en": mls_dir,
                    "common_voice_en": cv_dir,
                 }

    # (1) create dataset objects

    subsets = [
                "closed_ended/acoustic_level",
                "closed_ended/comparison",
                "closed_ended/content_level",
                "closed_ended/word_align",
                "open_ended",
               ]
    lang = "en" # We only use English data.

    for subset in subsets:
        curr_dir = root_dir / subset / lang
        if not curr_dir.exists():
            raise FileNotFoundError(f"{curr_dir} does not exist.")
        files = list(curr_dir.iterdir())

        writer_dir = output_dir / subset.replace("/", "_")
        writer_dir.mkdir(exist_ok=True, parents=True)
        dataset = DialogueDataset(task="text_dialogue")
        data_dict = {}
        wav_writer = (writer_dir / "wav.scp").open(mode="w")

        for file in files:
            lines = file.open(mode="r").readlines()
            metadata = [json.loads(line) for line in lines]
            for data in metadata:
                idx = data["id"]
                task = data["task"]
                source = data["data_source"]
                question = data["messages"][0]["content"][1]["text"]
                audio = data["messages"][0]["content"][0]["audio_path"]
                answer = data["messages"][1]["content"][0]["text"]
                assert question is not None
                assert answer is not None
                assert audio is not None
                audio = get_path(audio, source, audio_dir).as_posix()
                if audio is None:
                    continue
                data_key = f"{subset.replace('/', '_')}:{idx}"
                data_dict[data_key] = {
                    "task": task,
                    "source": source,
                    "question": question,
                    "answer": answer,
                    "audio": audio,
                }
                dialogue = Dialogue(task="text_dialogue")
                dialogue.add_segment("user", "text_bpe", False, question)
                dialogue.add_segment("assistant", "text_bpe", True, answer)
                dataset.add_dialogue(data_key, dialogue)
                wav_writer.write(f"{data_key}_turn0_speech {audio}\n")
            dataset.dump_dataset(writer_dir)
            with (writer_dir / "qa.json").open(mode="wb") as f:
                f.write(
                    json.dumps(
                            data_dict,
                            indent=4,
                            ensure_ascii=False,
                            sort_keys=False
                            ).encode("utf_8")
                        )

if __name__ == "__main__":
    parser = get_parser()
    main(**vars(parser.parse_args()))
