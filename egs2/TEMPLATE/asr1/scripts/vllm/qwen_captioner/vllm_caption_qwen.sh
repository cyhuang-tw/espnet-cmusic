#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

parquet_path=
output_dir=
base_port=8000
nj=1

log "$0 $*"
. utils/parse_options.sh

# . ./path.sh
. ./cmd.sh

mkdir -p ${output_dir}

${cuda_cmd} --gpu 1 JOB=1:${nj} ${output_dir}/logs/vllm_qwen_captioner_inference.JOB.log \
  scripts/vllm/qwen_captioner/vllm_caption_qwen_worker.sh \
    --nj ${nj} \
    --rank JOB \
    --parquet-path ${parquet_path} \
    --output-dir ${output_dir} \
    --base-port ${base_port}