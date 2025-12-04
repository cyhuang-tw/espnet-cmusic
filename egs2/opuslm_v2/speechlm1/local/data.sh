#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=100000

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Dump datasets for LAION-Audio-300M"
    for part in 1 2 3 4; do
        dir=/mnt/home/jinchuat-andr-d6b58f/jinchuat/data/rich_caption/laion_audio_300m_part${part}
        output_dir=${dir}/dump; mkdir -p ${output_dir}

        python3 local/dump_text.py \
          --input_dir ${dir} \
          --output_dir ${output_dir} \
          --mode qwen_caption \
          --file_regex '^captions_rank.+\.jsonl$' \
          --num_workers 128
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Prepare OWSM v4"
    audio_dir=/mnt/home/jinchuat-andr-d6b58f/jinchuat/data/audio/owsm_v4
    dataset_dir=/mnt/home/jinchuat-andr-d6b58f/jinchuat/data/data_jsons/owsm_v4

    # Build audio-to-rich-caption dataset json
    text_dir=/mnt/home/jinchuat-andr-d6b58f/jinchuat/data/rich_caption/owsm_v4
    output_dir=${text_dir}/dump; mkdir -p ${output_dir}

    python3 local/dump_text.py \
        --input_dir ${dir} \
        --output_dir ${output_dir} \
        --mode qwen_caption \
        --file_regex '^captions_rank.+\.jsonl$' \
        --num_workers 128

    python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
        --triplets audio1,${audio_dir}/metadata.parquet,arkive_audio \
                   text1,${text_dir}/dump/metadata.parquet,arkive_text \
        --output_json ${dataset_dir}/caption.json

    # Build audio-to-text dataset json
    text_dir=/mnt/home/jinchuat-andr-d6b58f/jinchuat/data/text/owsm_v4
    output_dir=${text_dir}/dump; mkdir -p ${output_dir}
    python3 local/dump_text.py \
        --input_dir ${text_dir} \
        --output_dir ${output_dir} \
        --mode kaldi \
        --file_regex '^text$' \
        --num_workers 128
    
    python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
        --triplets audio1,${audio_dir}/metadata.parquet,arkive_audio \
                   text1,${text_dir}/dump/metadata.parquet,arkive_text \
        --output_json ${dataset_dir}/asr.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Prepare dolma3"
    dir=/mnt/home/jinchuat-andr-d6b58f/jinchuat/data/text/dolma3_dolmino_mix-100B-1125
    output_dir=${dir}/dump; mkdir -p ${output_dir}

    python3 local/dump_text.py \
        --input_dir ${dir} \
        --output_dir ${output_dir} \
        --mode dolma3 \
        --file_regex '.*\.jsonl$' \
        --num_workers 128
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Prepare llama nemotron"
    dir=/mnt/home/jinchuat-andr-d6b58f/jinchuat/data/text/llama_nemotron
    output_dir=${dir}/dump; mkdir -p ${output_dir}

    python3 local/dump_text.py \
        --input_dir ${dir} \
        --output_dir ${output_dir} \
        --mode llama_nemotron \
        --file_regex '.*\.jsonl$' \
        --num_workers 8
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
