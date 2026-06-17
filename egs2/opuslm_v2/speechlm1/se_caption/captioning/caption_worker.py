#!/usr/bin/env python3
"""Concurrent, resumable captioning client for the Bagpiper vLLM server.
Usage: caption_worker.py <scp> <out.jsonl> [n_workers] [port]
scp lines: '<uid> <abs_noisy_path>'. Writes JSONL {"id","text"} incrementally.
Resumable: skips uids already present in out.jsonl. flac->wav (vLLM gotcha)."""
import base64, io, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import soundfile as sf, requests

scp, out_path = sys.argv[1], sys.argv[2]
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 48
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 9011
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

items = [ln.split(maxsplit=1) for ln in open(scp) if ln.strip()]
items = [(u, p.strip()) for u, p in items]
done = set()
if Path(out_path).exists():
    for ln in open(out_path):
        try: done.add(json.loads(ln)["id"])
        except: pass
todo = [(u, p) for u, p in items if u not in done]
print(f"[cap] {len(items)} total, {len(done)} done, {len(todo)} to do, {NW} workers", flush=True)

lock = threading.Lock()
fout = open(out_path, "a", encoding="utf-8")
cnt = [0]; t0 = time.time()

def wav_b64(p):
    a, sr = sf.read(p)
    if a.ndim > 1: a = a[:, 0]
    buf = io.BytesIO(); sf.write(buf, a, sr, format="WAV", subtype="PCM_16"); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def work(item):
    uid, p = item
    try:
        msgs = [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": wav_b64(p), "format": "wav"}}]}]
        payload = {"model": "speechlm-qwen3-8b", "messages": msgs,
                   "max_tokens": 256, "stop_token_ids": [3]}
        r = requests.post(URL, json=payload, timeout=300).json()
        cap = r["choices"][0]["message"].get("content", "")
        if not cap: return (uid, False)
        with lock:
            fout.write(json.dumps({"id": uid, "text": cap}, ensure_ascii=False) + "\n"); fout.flush()
            cnt[0] += 1
            if cnt[0] % 200 == 0:
                print(f"[cap] {cnt[0]}/{len(todo)} ({cnt[0]/(time.time()-t0):.1f}/s)", flush=True)
        return (uid, True)
    except Exception as e:
        return (uid, f"ERR {type(e).__name__}")

with ThreadPoolExecutor(max_workers=NW) as ex:
    fails = [r for r in ex.map(work, todo) if r[1] is not True]
fout.close()
print(f"[cap] DONE {cnt[0]} ok, {len(fails)} fail in {time.time()-t0:.0f}s", flush=True)
if fails[:10]: print("fails sample:", fails[:10], flush=True)
