#!/usr/bin/env python3
"""Stage 2 client: Summarize captions using vLLM service."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Summarization prompt template
SUMMARIZATION_PROMPT = """Summarize the following audio caption into a short description of 20 words or fewer. Focus only on describing the audio content (sounds, music, speech characteristics). Ignore any ASR transcription or spoken words content.

Caption:
{caption}

Short summary (20 words or fewer):"""


def find_jsonl_files(text_dir: str) -> List[str]:
    """Find all JSONL files in the given directory.

    Args:
        text_dir: Input directory path.

    Returns:
        Sorted list of JSONL file paths.
    """
    jsonl_files = []
    input_path = Path(text_dir)
    if not input_path.exists():
        print(f"Warning: Directory {text_dir} does not exist", flush=True)
        return jsonl_files

    for f in input_path.glob("**/*.jsonl"):
        if f.is_file():
            jsonl_files.append(str(f))

    return sorted(set(jsonl_files))


def load_jsonl_files(files: List[str]) -> List[Dict]:
    """Load all JSONL files into memory.

    Args:
        files: List of JSONL file paths.

    Returns:
        List of all records from all files.
    """
    records = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        data["_source_file"] = file_path
                        records.append(data)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error loading {file_path}: {e}", flush=True)
    return records


def query_vllm(
    caption: str,
    vllm_host: str,
    vllm_port: int,
    max_tokens: int = 50,
    temperature: float = 0.7,
) -> Optional[str]:
    """Query vLLM service for caption summarization.

    Args:
        caption: The caption text to summarize.
        vllm_host: vLLM server host.
        vllm_port: vLLM server port.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.

    Returns:
        Summarized caption or None if failed.
    """
    url = f"http://{vllm_host}:{vllm_port}/v1/completions"
    prompt = SUMMARIZATION_PROMPT.format(caption=caption)

    payload = {
        "model": "Qwen/Qwen3-32B-FP8",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["\n\n", "\n"],
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["text"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}", flush=True)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Parse error: {e}", flush=True)

    return None


def process_single_record(
    record: Dict,
    vllm_host: str,
    vllm_port: int,
    max_tokens: int,
    temperature: float,
) -> Dict:
    """Process a single record: query vLLM for summarization.

    Args:
        record: The record containing caption.
        vllm_host: vLLM server host.
        vllm_port: vLLM server port.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.

    Returns:
        Record with added summary field.
    """
    caption = record.get("caption", "")
    utt_id = record.get("utt_id", "unknown")

    if not caption:
        record["summary"] = ""
        record["summary_status"] = "empty_caption"
        return record

    summary = query_vllm(
        caption=caption,
        vllm_host=vllm_host,
        vllm_port=vllm_port,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if summary is not None:
        record["summary"] = summary
        record["summary_status"] = "success"
    else:
        record["summary"] = ""
        record["summary_status"] = "failed"

    return record


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 client: Summarize captions using vLLM service."
    )
    parser.add_argument(
        "--text_dir",
        type=str,
        required=True,
        help="Input directory containing JSONL files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for summarized JSONL files.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Current process rank (default: 0).",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of processes (default: 1).",
    )
    parser.add_argument(
        "--vllm_host",
        type=str,
        default="localhost",
        help="vLLM server host (default: localhost).",
    )
    parser.add_argument(
        "--vllm_port",
        type=int,
        default=8001,
        help="vLLM server port (default: 8001).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of worker threads (default: 32).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum tokens in vLLM response (default: 50).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    args = parser.parse_args()

    # Find all JSONL files
    all_files = find_jsonl_files(args.text_dir)
    if not all_files:
        print("No JSONL files found in the input directory.")
        return

    print(f"Found {len(all_files)} total JSONL files", flush=True)

    # Select files for this rank
    my_files = all_files[args.rank::args.world_size]
    print(
        f"Rank {args.rank}/{args.world_size}: processing {len(my_files)} files",
        flush=True
    )

    if not my_files:
        print(f"Rank {args.rank}: No files to process.")
        return

    # Load all records into memory
    print("Loading JSONL files into memory...", flush=True)
    records = load_jsonl_files(my_files)
    print(f"Loaded {len(records)} records from {len(my_files)} files", flush=True)

    if not records:
        print("No records to process.")
        return

    # Process records using thread pool
    print(
        f"Processing {len(records)} records with {args.num_workers} workers...",
        flush=True
    )

    processed_records = []
    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_single_record,
                record,
                args.vllm_host,
                args.vllm_port,
                args.max_tokens,
                args.temperature,
            ): record
            for record in records
        }

        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                processed_records.append(result)
                if result.get("summary_status") == "success":
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"Error processing record: {e}", flush=True)
                failed_count += 1

            # Progress update every 1000 records
            if (i + 1) % 1000 == 0:
                print(
                    f"Progress: {i + 1}/{len(records)} "
                    f"(success: {success_count}, failed: {failed_count})",
                    flush=True
                )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Write output JSONL file
    output_file = os.path.join(
        args.output_dir, f"summarized_rank{args.rank}.jsonl"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        for record in processed_records:
            # Remove internal tracking field
            record.pop("_source_file", None)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print(f"Stage 2 Summarization Summary (Rank {args.rank})", flush=True)
    print("=" * 60, flush=True)
    print(f"Total records processed:  {len(processed_records)}", flush=True)
    print(f"Successful summaries:     {success_count}", flush=True)
    print(f"Failed summaries:         {failed_count}", flush=True)
    print(
        f"Success rate:             "
        f"{100 * success_count / max(1, len(processed_records)):.2f}%",
        flush=True
    )
    print(f"Output written to:        {output_file}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
