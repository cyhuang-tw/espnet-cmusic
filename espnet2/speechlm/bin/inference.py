#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-processing inference script for SpeechLM with data sharding."""

import argparse
import json
import logging
import multiprocessing as mp
import sys
import threading
import time
from pathlib import Path
from queue import Empty

import torch
import torch.multiprocessing as torch_mp
import yaml

from espnet2.speechlm.dataloader.iterator import DataIteratorFactory
from espnet2.speechlm.model import _all_job_types
from espnet2.speechlm.utils.data import to_device


def get_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="SpeechLM Multi-Processing Inference Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--train-config",
        type=Path,
        required=True,
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        required=True,
        help="Path to inference configuration file",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint to load",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/inference_mp"),
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--test-unregistered-specifier",
        type=str,
        default=None,
        help="Unregistered test data specifier " "(e.g., 'asr:librispeech:test.json')",
    )
    parser.add_argument(
        "--test-registered-specifier",
        type=str,
        default=None,
        help="Registered test data specifier " "(e.g., 'asr:librispeech')",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for inference",
    )

    return parser


def setup_worker_logger(rank: int, output_dir: Path) -> logging.Logger:
    """Set up logger for worker process.

    Args:
        rank: Worker rank/ID
        output_dir: Directory to save log files

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"inference_worker_{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        f"[Worker-{rank}] [%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["module"]
    model.load_state_dict(state_dict, strict=True)
    return model


def inference_worker(
    rank: int,
    world_size: int,
    train_config_path: Path,
    inference_config_path: Path,
    model_checkpoint_path: Path,
    unregistered_specifier: str,
    registered_specifier: str,
    output_dir: Path,
):
    """Worker process for inference with data sharding."""
    # Set up logger for this worker
    logger = setup_worker_logger(rank, output_dir)
    logger.info(f"Starting inference worker (rank {rank}/{world_size})")

    # Load configs in worker
    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)

    with open(inference_config_path, "r") as f:
        inference_config = yaml.safe_load(f)

    job_template_class = _all_job_types[train_config["job_type"]]
    job_template = job_template_class(train_config, is_train=True)

    # Build model and preprocessor in worker
    model = job_template.build_model()
    model = load_checkpoint(model, model_checkpoint_path)
    model.prepare_inference()
    dtype = inference_config.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype)
    model = model.to(device="cuda", dtype=dtype).eval()

    preprocessor = job_template.build_preprocessor()

    # Build data iterator with sharding
    iterator_factory = DataIteratorFactory(
        unregistered_specifier=unregistered_specifier,
        registered_specifier=registered_specifier,
        collate_fn=preprocessor.collate_fn,
        num_workers=0,
        rank=rank,
        world_size=world_size,
        sequential_load=True,
    )

    # Process this worker's shard
    test_iterator = iterator_factory.build_iter()
    logger.info("Starting inference on data shard")

    for idx, sample in enumerate(test_iterator):
        sample = to_device(sample, "cuda", dtype=dtype)
        task, data_name, example_id = sample.pop("keys")[0]

        logger.info(f"Processing sample {idx}: {task}/{data_name}/{example_id}")
        messages = model.inference(inference_config, **sample)
        assert 1 == 2

    logger.info(f"Worker {rank} completed successfully")


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Enforce GPU availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires GPU.")
        sys.exit(1)

    # Validate that exactly one specifier is provided
    if not args.test_registered_specifier and not args.test_unregistered_specifier:
        parser.error(
            "Provide either --test-registered-specifier or "
            "--test-unregistered-specifier"
        )
    if args.test_registered_specifier and args.test_unregistered_specifier:
        parser.error(
            "Provide only one of --test-registered-specifier or "
            "--test-unregistered-specifier"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Workers: {args.num_workers}")
    print(
        f"Specifier: {args.test_unregistered_specifier or args.test_registered_specifier}"
    )

    # Enable spawn for CUDA
    torch_mp.set_start_method("spawn", force=True)

    # Start worker processes (all configs will be loaded in workers)
    processes = []
    for rank in range(args.num_workers):
        p = mp.Process(
            target=inference_worker,
            args=(
                rank,
                args.num_workers,
                args.train_config,
                args.inference_config,
                args.model_checkpoint,
                args.test_unregistered_specifier or "",
                args.test_registered_specifier or "",
                args.output_dir,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all workers
    for p in processes:
        p.join()

    print("All workers completed!")


if __name__ == "__main__":
    main()
