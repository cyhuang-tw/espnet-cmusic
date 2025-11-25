#!/usr/bin/env python3
"""Prepare text SFT datasets from HuggingFace for ESPnet SpeechLM training."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and parse HuggingFace text SFT datasets."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated list of HuggingFace dataset names "
        "(e.g., 'tatsu-lab/alpaca,Open-Orca/OpenOrca').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save the parsed datasets.",
    )
    return parser.parse_args()


def process_olmo3(example):
    example_id = example['id']
    messages = example['messages']

    new_messages = list()
    for msg in messages:
        role = msg['role']
        modality = "text"
        content = msg['content']

        if content is None:
            return None
        
        if role not in ['assistant', 'system', 'user']:
            return None

        msg = (role, modality, content)
        new_messages.append(msg)
    
    return {"example_id": example_id, "messages": new_messages}

process_methods = {
    "allenai/Dolci-Instruct-SFT": process_olmo3,
    "allenai/Dolci-Think-SFT": process_olmo3,
    "nvidia/AceMath-Instruct-Training-Data": process_olmo3,
}

def load_and_parse_dataset(dataset_name: str, output_dir: Path):
    """Load all splits of a HuggingFace dataset and save to output directory.

    Args:
        dataset_name: Name of the HuggingFace dataset.
        output_dir: Directory to save the parsed data.
    """
    print(f"Loading dataset: {dataset_name}")

    # Create dataset-specific output directory
    dataset_slug = dataset_name.replace("/", "_")
    dataset_output_dir = output_dir / dataset_slug
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Load all available splits
    dataset = load_dataset(dataset_name, num_proc=16)
    porcess_method = process_methods[dataset_name]

    for split_name, split_data in dataset.items():
        print(f"  Processing split: {split_name} ({len(split_data)} examples)")

        output_file = dataset_output_dir / f"{split_name}.jsonl"
        writer = open(output_file, 'w')

        with open(output_file, "w", encoding="utf-8") as f:
            for example in split_data:
                example = porcess_method(example)

                if example is None:
                    continue

                example = json.dumps(example)
                writer.write(example + "\n")



def main():
    args = parse_args()

    # Parse dataset names
    dataset_names = [name.strip() for name in args.datasets.split(",")]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Datasets to process: {dataset_names}")
    print("-" * 50)

    # Process each dataset
    for dataset_name in dataset_names:
        if dataset_name:
            load_and_parse_dataset(dataset_name, output_dir)
            print("-" * 50)

    print("Done!")


if __name__ == "__main__":
    main()
