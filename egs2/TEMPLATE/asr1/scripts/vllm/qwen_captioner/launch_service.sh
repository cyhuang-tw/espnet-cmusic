echo "Loading Qwen3-captioner on 1 GPU as API service..."

PORT=${1}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



apptainer exec --cleanenv --nv \
  -B /work \
  "/work/nvme/bbjs/chuang14/Haoran/vllm_arm.sif" \
  bash -lc "set -e; export VLLM_USE_V1=0; vllm serve Qwen/Qwen3-Omni-30B-A3B-Captioner \
    --host 0.0.0.0 --port ${PORT} \
    --max-model-len 1800 \
    --max-num-seqs 1024 \
    --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests \
    --dtype bfloat16 "
