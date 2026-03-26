#!/usr/bin/env bash
# SOT multi-talker ASR recipe for AMI dataset.
#
# Uses native OpenAI Whisper encoder/decoder (not HuggingFace Transformers).
# Tokenization via tiktoken (SOTWhisperPreprocessor).
#
# Prerequisites:
#   - Lhotse CutSet .jsonl.gz files for train/dev/test
#   - OR pre-prepared Kaldi-format data directories
#
# Usage:
#   # Full pipeline:
#   ./run.sh --stage 0 --sot_strategy speaker_longest_first
#
#   # Train only:
#   ./run.sh --stage 1 --stop_stage 1
#
#   # Decode + evaluate:
#   ./run.sh --stage 2 --stop_stage 3

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# General
stage=0
stop_stage=100
ngpu=1
nj=8
expdir=exp
python=python3

# Data
train_set=train
valid_set=dev
test_sets="dev test"

# SOT data prep
sot_strategy=speaker_longest_first
use_timestamps=false
max_timestamp_pause=2.0
use_spk_count_tokens=false
use_spk_rem_tokens=false
use_spk_id_tokens=false
train_cutset=   # path to train cutset .jsonl.gz (required for stage 0)
valid_cutset=   # path to valid cutset .jsonl.gz (required for stage 0)
test_cutsets=   # space-separated paths to test cutset .jsonl.gz files

# Config
asr_config=conf/tuning/train_sot_tiny.yaml
decode_config=conf/tuning/decode_sot.yaml
token_list=  # path to token_list (auto-generated if empty)
added_tokens_file=local/added_tokens.txt

log "$0 $*"

# Inline parse_options (no dependency on utils/ symlink)
while true; do
    [ -z "${1:-}" ] && break
    case "$1" in
        --*) name=$(echo "$1" | sed 's/^--//' | sed 's/-/_/g')
             eval "${name}=\"$2\""
             shift 2 ;;
        *)   break ;;
    esac
done

# ================================
# Token list generation
# ================================
if [ -z "${token_list}" ]; then
    token_list="${expdir}/token_list.txt"
    if [ ! -f "${token_list}" ]; then
        log "Generating token list from tiktoken"
        mkdir -p "${expdir}"
        ${python} local/generate_token_list.py \
            --output "${token_list}" \
            --added_tokens_txt "${added_tokens_file}"
    fi
fi

# ================================
# Stage 0: Prepare SOT data
# ================================
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Prepare SOT data from Lhotse CutSets"

    _sot_opts="--sot_strategy ${sot_strategy} \
        --use_timestamps ${use_timestamps} \
        --max_timestamp_pause ${max_timestamp_pause} \
        --use_spk_count_tokens ${use_spk_count_tokens} \
        --use_spk_rem_tokens ${use_spk_rem_tokens} \
        --use_spk_id_tokens ${use_spk_id_tokens} \
        --added_tokens_file ${added_tokens_file}"

    if [ -n "${train_cutset}" ]; then
        log "Preparing train set: ${train_set}"
        ${python} local/prepare_sot.py \
            --cutset_paths ${train_cutset} \
            --output_dir data/${train_set} \
            ${_sot_opts}
    fi

    if [ -n "${valid_cutset}" ]; then
        log "Preparing valid set: ${valid_set}"
        ${python} local/prepare_sot.py \
            --cutset_paths ${valid_cutset} \
            --output_dir data/${valid_set} \
            ${_sot_opts}
    fi

    if [ -n "${test_cutsets}" ]; then
        idx=0
        for dset in ${test_sets}; do
            cutset=$(echo ${test_cutsets} | cut -d' ' -f$((idx+1)))
            if [ -n "${cutset}" ]; then
                log "Preparing test set: ${dset}"
                ${python} local/prepare_sot.py \
                    --cutset_paths ${cutset} \
                    --output_dir data/${dset} \
                    ${_sot_opts}
            fi
            idx=$((idx+1))
        done
    fi

    # Validate data directories
    for dset in ${train_set} ${valid_set} ${test_sets}; do
        dir=data/${dset}
        if [ ! -d "${dir}" ]; then
            log "WARNING: ${dir} does not exist"
            continue
        fi
        for f in wav.scp text utt2spk; do
            if [ ! -f "${dir}/${f}" ]; then
                log "ERROR: Missing required file ${dir}/${f}"
                exit 1
            fi
        done
        log "Data directory ${dir}: $(wc -l < "${dir}/wav.scp") utterances"
    done
fi

# ================================
# Stage 1: Training
# ================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: SOT Training (native Whisper)"

    _train_dir=data/${train_set}
    _valid_dir=data/${valid_set}

    _opts=""
    _opts+="--train_data_path_and_name_and_type ${_train_dir}/wav.scp,speech,sound "
    _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
    _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/wav.scp,speech,sound "
    _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "

    _tag=$(basename "${asr_config}" .yaml)
    _expdir="${expdir}/sot_${_tag}"

    ${python} -m espnet2.bin.sot_train \
        --config "${asr_config}" \
        --token_list "${token_list}" \
        --token_type whisper_multilingual \
        --output_dir "${_expdir}" \
        --ngpu ${ngpu} \
        --num_workers ${nj} \
        ${_opts} \
        "$@"
fi

# ================================
# Stage 2: Decoding
# ================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: SOT Decoding (native Whisper)"

    _tag=$(basename "${asr_config}" .yaml)
    _expdir="${expdir}/sot_${_tag}"
    _model_file="${_expdir}/valid.loss.best.pth"

    if [ ! -f "${_model_file}" ]; then
        log "ERROR: Model file not found: ${_model_file}"
        exit 1
    fi

    for dset in ${test_sets}; do
        _data_dir=data/${dset}
        _decode_dir="${_expdir}/decode_${dset}"

        _decode_opts=""
        _decode_opts+="--data_path_and_name_and_type ${_data_dir}/wav.scp,speech,sound "

        ${python} -m espnet2.bin.sot_inference \
            --config "${decode_config}" \
            --asr_train_config "${_expdir}/config.yaml" \
            --asr_model_file "${_model_file}" \
            --output_dir "${_decode_dir}" \
            --ngpu ${ngpu} \
            ${_decode_opts} \
            "$@"

        log "Decoding results: ${_decode_dir}"
    done
fi

# ================================
# Stage 3: Evaluate cpWER
# ================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Evaluate cpWER"

    _tag=$(basename "${asr_config}" .yaml)
    _expdir="${expdir}/sot_${_tag}"

    for dset in ${test_sets}; do
        _decode_dir="${_expdir}/decode_${dset}"
        _hyp_text="${_decode_dir}/1best_recog/text"
        _ref_text="data/${dset}/text"
        _eval_dir="${_decode_dir}/eval"

        if [ ! -f "${_hyp_text}" ]; then
            log "WARNING: Hypothesis file not found: ${_hyp_text}, skipping ${dset}"
            continue
        fi

        ${python} local/evaluate_sot.py \
            --hyp_text "${_hyp_text}" \
            --ref_text "${_ref_text}" \
            --output_dir "${_eval_dir}" \
            --speaker_change_token "<sc>"

        log "Evaluation results for ${dset}: ${_eval_dir}/cpwer.json"
        if [ -f "${_eval_dir}/cpwer.json" ]; then
            cat "${_eval_dir}/cpwer.json"
        fi
    done
fi

log "Done."
