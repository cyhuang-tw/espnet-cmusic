#!/usr/bin/env python3
"""Query vLLM API with arkive audio using Threading (High Concurrency)."""

import argparse
import base64
import io
import json
import sys
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
import numpy as np
import pandas as pd
from openai import OpenAI
import httpx

try:
    from arkive import audio_read
except ImportError:
    raise ImportError(
        "arkive is not installed. Install at https://github.com/wanchichen/arkive"
    )

try:
    import duckdb
except ImportError:
    raise ImportError(
        "duckdb is not installed. Please install it with: pip install duckdb"
    )


class ArkiveAudioReader:
    """Dict-like lazy audio reader using arkive parquets."""
    def __init__(
        self,
        query: str,
        parquet_dir: str = None,
        valid_ids: list = None,
        worker_id: int = None,
        world_size: int = None,
    ):
        self.parquet_dir = parquet_dir
        # result = duckdb.query(query)
        # print('parquet loaded', flush=True)

        # if worker_id is not None:
        #     assert (
        #         world_size is not None
        #     ), f"filtering by worker_id requires world_size, got {world_size}"
        #     result = duckdb.query(
        #         f"""
        #         SELECT * FROM result
        #         QUALIFY (row_number() OVER (ORDER BY utt_id) - 1)
        #         % {world_size} = {worker_id}
        #     """
        #     )

        # df = result.df()
        # print('df converted', flush=True)

        result = duckdb.query(query)

        # if valid_ids is not None:
        #     # Properly escape and quote string IDs for SQL safety
        #     quoted_ids = [f"'{id.replace('\'', '\'\'')}'" for id in valid_ids]
        #     result = duckdb.query(
        #         f"""
        #         SELECT * FROM result
        #         WHERE utt_id IN ({','.join(quoted_ids)})
        #          """
        #     )

        # filter query result before loading to df
        # avoids loading the whole query result into memory
        # if worker_id is not None:
        #     assert (
        #         world_size is not None
        #     ), f"filtering by worker_id requires world_size, got {world_size}"
        #     result = duckdb.query(
        #         f"""
        #         SELECT * FROM result
        #         QUALIFY (row_number() OVER (ORDER BY utt_id) - 1) 
        #         % {world_size} = {worker_id}
        #     """
        #     )

        df = result.df()

        has_start_time = "start_time" in df.columns
        has_end_time = "end_time" in df.columns

        if has_start_time and has_end_time:
            data = dict(
                zip(
                    df["utt_id"],
                    zip(
                        df["path"],
                        df["start_byte_offset"],
                        df["file_size_bytes"],
                        df["start_time"],
                        df["end_time"],
                    ),
                )
            )
        else:
            data = dict(
                zip(
                    df["utt_id"],
                    zip(
                        df["path"],
                        df["start_byte_offset"],
                        df["file_size_bytes"],
                        [None] * len(df),
                        [None] * len(df),
                    ),
                )
            )

        # print('start filtering')
        if valid_ids:
            data = {k: data[k] for k in set(valid_ids) if k in data}

        self.data = data

    def __getitem__(self, key: str):
        path, start_byte, file_size, start_time, end_time = self.data[key]

        # # Reconstruct path: parquet's directory + bin_filename
        # if self.parquet_dir:
        #     parquet_directory = str(Path(self.parquet_dir).parent)
        #     bin_filename = Path(path).name
        #     path = parquet_directory + '/' + bin_filename

        if pd.isna(start_time): start_time = None
        if pd.isna(end_time): end_time = None

        data = audio_read(
            path,
            start_offset=start_byte,
            file_size=file_size,
            start_time=start_time,
            end_time=end_time,
        )
        return data.array.T, data.sample_rate

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def keys(self):
        return self.data.keys()


def audio_to_base64_wav(audio: np.ndarray, sample_rate: int) -> str:
    """Convert audio array to base64 encoded WAV."""
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    audio = audio.T
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode('utf-8')
    return b64


def load_processed_ids(output_path):
    """Load already processed IDs from output file."""
    processed = set()
    if Path(output_path).exists():
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed.add(data['utt_id'])
                    except json.JSONDecodeError:
                        continue
    return processed


def writer_thread_func(q, output_path, save_freq):
    """Dedicated writer thread."""
    processed_count = 0
    with open(output_path, 'a') as f:
        while True:
            item = q.get()
            if item is None:  # Poison pill
                break

            f.write(json.dumps(item) + '\n')
            f.flush()
            processed_count += 1

            if processed_count % save_freq == 0:
                print(f"Processed {processed_count} examples", file=sys.stderr)
            
            q.task_done()


# Global variables - shared by all threads automatically
_worker_queue = None
_reader = None
_client = None


def query_audio(utt_id: str):
    """Query the audio captioning service for a single audio sample."""
    # Threads access globals directly
    max_tokens = 1024
    try:
        # Load audio (Thread-safe because dict access is atomic-ish and file read is OS level)
        audio, sample_rate = _reader[utt_id]

        if audio.ndim > 1:
            num_samples = audio.shape[1]
        else:
            num_samples = len(audio)
        duration = num_samples / sample_rate
        # --------------------------

        # --- Check audio length (skip if > 60s) ---
        if duration > 60.0:
            output_data = {
                "utt_id": utt_id,
                "caption": "skip_audio_too_long",
                "duration": duration,
                "finish_reason": "audio_length"
            }
            _worker_queue.put(output_data)
            return
        # -----------------------------------------

        # Convert to base64 WAV
        b64_audio = audio_to_base64_wav(audio, sample_rate)
        data_url = f"data:audio/wav;base64,{b64_audio}"

        # Query API (httpx client is thread-safe)
        response = _client.chat.completions.create(
            model="Qwen/Qwen3-Omni-30B-A3B-Captioner",
            messages=[{
                "role": "user",
                "content": [{"type": "audio_url", "audio_url": {"url": data_url}}]
            }],
            temperature=0.6,
            top_p=0.95,
            max_tokens=max_tokens
        )

        result = response.choices[0].message.content

        # --- Extract usage and status information ---
        usage_info = None
        if hasattr(response, 'usage') and response.usage:
            usage_info = {
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', None),
                "completion_tokens": getattr(response.usage, 'completion_tokens', None),
                "total_tokens": getattr(response.usage, 'total_tokens', None),
                "prompt_tokens_details": getattr(response.usage, 'prompt_tokens_details', None)
            }

        # Get finish_reason (status)
        finish_reason = None
        if response.choices and len(response.choices) > 0:
            finish_reason = getattr(response.choices[0], 'finish_reason', None)

        # --- Check if caption is too long (tokens >= max_tokens) ---
        if usage_info and usage_info.get('completion_tokens', 0) >= max_tokens:
            result = "skip_caption_too_long"
        # ----------------------------------------------------------
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing {utt_id}: {e}", file=sys.stderr)
        print(f"Fatal error detected, exiting with code 1", file=sys.stderr)
        import os
        os._exit(1)  # Exit with code 1 to trigger server restart

    output_data = {
        "utt_id": utt_id,
        "caption": result,
        "duration": duration
    }

    # Add usage info if available
    if usage_info:
        output_data["usage"] = usage_info

    # Add finish_reason if available
    if finish_reason:
        output_data["finish_reason"] = finish_reason

    _worker_queue.put(output_data)


def main():
    parser = argparse.ArgumentParser(description="Query vLLM API with arkive audio")
    parser.add_argument("--parquet", required=True, help="Path to parquet file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-jobs", type=int, required=True, help="Total number of jobs")
    parser.add_argument("--rank", type=int, required=True, help="Current job rank (0-indexed)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="Base URL")
    parser.add_argument("--workers", type=int, default=32, help="Number of worker threads")
    parser.add_argument("--save-freq", type=int, default=10, help="Save frequency")
    parser.add_argument("--timeout", type=float, default=600.0, help="Request timeout in seconds")

    args = parser.parse_args()

    # Construct output paths
    output_dir = Path(args.output)
    output_file = output_dir / f"captions_rank{args.rank}.jsonl"
    done_dir = output_dir / "done"
    done_file = done_dir / f".done.{args.rank}"

    # Load metadata
    query = f"SELECT utt_id FROM read_parquet('{args.parquet}')"
    result = duckdb.query(query)
    all_ids = result.df()['utt_id'].tolist()

    print(f"Found {len(all_ids)} total examples", file=sys.stderr)
    rank = args.rank - 1
    my_ids = [uid for idx, uid in enumerate(all_ids) if idx % args.n_jobs == rank]
    print(f"Job {rank}/{args.n_jobs} responsible for {len(my_ids)} examples", file=sys.stderr)

    processed = load_processed_ids(str(output_file))
    print(f'processed: {len(processed)}')
    utt_ids = [uid for uid in my_ids if uid not in processed]
    print(f"Processing {len(utt_ids)} new examples with {args.workers} threads", file=sys.stderr)

    if not utt_ids:
        print("All examples already processed!", file=sys.stderr)
        # Create done file
        done_dir.mkdir(parents=True, exist_ok=True)
        done_file.touch()
        print(f"Created done marker: {done_file}", file=sys.stderr)
        return

    global _reader, _client, _worker_queue
    
    print("Initializing global reader...", file=sys.stderr)
    reader_query = f"SELECT * FROM read_parquet('{args.parquet}')"
    _reader = ArkiveAudioReader(
        reader_query, parquet_dir=args.parquet,
        worker_id=rank, world_size=args.n_jobs,
    )
    print('launch the reader')

    http_client = httpx.Client(
        limits=httpx.Limits(max_keepalive_connections=args.workers, max_connections=args.workers),
        timeout=args.timeout
    )

    _client = OpenAI(
        base_url=args.base_url,
        api_key="dummy",
        http_client=http_client
    )

    _worker_queue = queue.Queue()

    writer = threading.Thread(
        target=writer_thread_func,
        args=(_worker_queue, str(output_file), args.save_freq)
    )
    writer.start()

    print('start the query', flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(query_audio, utt_ids))
    finally:
        _worker_queue.put(None)
        writer.join()

    print(f"Completed! Total processed: {len(utt_ids) + len(processed)}", file=sys.stderr)

    # Create done file after successful completion
    done_dir.mkdir(parents=True, exist_ok=True)
    done_file.touch()
    print(f"Created done marker: {done_file}", file=sys.stderr)

    # Check if all jobs are done and merge if needed
    merge_all_outputs(output_dir, done_dir, args.n_jobs)


def merge_all_outputs(output_dir, done_dir, n_jobs):
    """Merge all output files if all jobs are completed."""
    # Check if all done files exist
    all_done = all(
        (done_dir / f".done.{i}").exists()
        for i in range(1, n_jobs + 1)
    )

    if not all_done:
        return  # Not all jobs completed yet

    merged_file = output_dir / "captions_merged.jsonl"

    # Check if merged file already exists
    if merged_file.exists():
        print(f"Merged file already exists: {merged_file}", file=sys.stderr)
        return

    print(f"All {n_jobs} jobs completed. Merging outputs...", file=sys.stderr)

    # Merge all caption files
    with open(merged_file, 'w') as outf:
        for i in range(1, n_jobs + 1):
            caption_file = output_dir / f"captions_rank{i}.jsonl"
            if caption_file.exists():
                print(f"Merging {caption_file}...", file=sys.stderr)
                with open(caption_file, 'r') as inf:
                    for line in inf:
                        outf.write(line)
            else:
                print(f"Warning: {caption_file} does not exist", file=sys.stderr)

    print(f"Merged all outputs to: {merged_file}", file=sys.stderr)


if __name__ == "__main__":
    main()