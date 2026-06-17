#!/usr/bin/env bash
# Launch the Bagpiper (SpeechLM) vLLM server used to caption the noisy audio.
# The captioner is the SpeechLM *base* model exported for vLLM (weight-identical
# to the SE pretrain checkpoint). Captioning uses a vLLM SpeechLM build; the
# stock espnet audio->text path is unreliable for this model.
#
# Set before running (see se_caption/README.md):
#   BAGPIPER_MODEL_DIR  vLLM model dir for the captioner.                 [required]
#   VLLM_PY             python in the vLLM-SpeechLM env.                  [default: `python`]
#   VLLM_REPO           vLLM fork checkout to run from (cd'd into).       [optional]
#   CUDA_HOME           CUDA toolkit (only if your build needs it).      [optional]
#   CAPTION_PORT        server port.                                      [default: 9011]
set -euo pipefail

HOST=127.0.0.1
PORT=${CAPTION_PORT:-9011}
MODEL_DIR=${BAGPIPER_MODEL_DIR:?set BAGPIPER_MODEL_DIR to the vLLM captioner model dir}
PY=${VLLM_PY:-python}

[ -n "${CUDA_HOME:-}" ] && export PATH=$CUDA_HOME/bin:$PATH
export VLLM_USE_PRECOMPILED=0
[ -n "${VLLM_REPO:-}" ] && cd "$VLLM_REPO"

exec "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "speechlm-qwen3-8b" \
    --trust-remote-code \
    --max-model-len 16384 \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --tensor-parallel-size 1 \
    --limit-mm-per-prompt '{"audio": 1}' \
    --enable-prefix-caching
