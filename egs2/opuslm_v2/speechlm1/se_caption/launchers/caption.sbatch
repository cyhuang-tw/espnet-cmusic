#!/bin/bash
# Caption noisy clips with the vLLM SpeechLM server (Stage 1 of caption-conditioned SE).
# Starts the server, runs the concurrent captioner over an scp, then tears the server down.
#
#   sbatch --account=$ACC --partition=$PART caption.sbatch <noisy.scp> <out.jsonl> [n_workers]
#     noisy.scp lines: "<uid> <abs_noisy_wav_path>"
#     out.jsonl: written as {"id","text"} per line; resumable (re-run to fill gaps).
#
# Requires (set in env.sh): BAGPIPER_MODEL_DIR, VLLM_PY (vLLM-SpeechLM env), optionally VLLM_REPO.
#SBATCH --job-name=se_caption
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
set -u
cd "$(dirname "$0")"; source ./env.sh
SCP=${1:?noisy scp}; OUTJSONL=${2:?output jsonl}; NW=${3:-48}
PORT=${CAPTION_PORT:-9011}
HERE="$ESPNET_ROOT/egs2/opuslm_v2/speechlm1/se_caption/captioning"

# 1) launch the vLLM captioning server in the background
bash "$HERE/run_bagpiper_server.sh" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; pkill -f EngineCore 2>/dev/null' EXIT

# 2) wait until it answers
echo "waiting for vLLM server on :$PORT ..."
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 10
done

# 3) caption (concurrent, resumable). caption_worker re-encodes flac->wav internally.
python "$HERE/caption_worker.py" "$SCP" "$OUTJSONL" "$NW" "$PORT"
echo "captioning done -> $OUTJSONL"
