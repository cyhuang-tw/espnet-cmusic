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
stop_stage=100
output_dir=/work/nvme/bbjs/shared/opuslm_v2_data/text

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Processing AI2 data"
    # for tag in allenai/Dolci-Think-SFT allenai/Dolci-Instruct-SFT; do
    for tag in allenai/Dolci-Think-SFT ; do
        python local/prepare_text_sft.py \
          --datasets ${tag} \
          --output_dir ${output_dir}

        tag=${tag//\//_}
        python ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
          --triplets dialogue,${output_dir}/${tag}/train.jsonl,dialogue \
          --output_json ${output_dir}/${tag}/train.json
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
