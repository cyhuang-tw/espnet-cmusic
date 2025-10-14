#!/usr/bin/env bash

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script tokenizes a single subdirectory for parallel SLURM job submission

set -e
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

# Parse arguments
if [ $# -ne 10 ]; then
    echo "Usage: $0 <subdirectory> <nj> <fs> <codec_choice> <codec_hf_model_tag> <ssl_choice> <ssl_checkpoint_path> <ssl_kmeans_path> <ssl_nlayer> <ssl_batch_bins>"
    exit 1
fi

subdirectory=$1
nj=$2
fs=$3
codec_choice=$4
codec_hf_model_tag=$5
ssl_choice=$6
ssl_checkpoint_path=$7
ssl_kmeans_path=$8
ssl_nlayer=$9
ssl_batch_bins=${10}

subdir_name=$(basename ${subdirectory})

# Source environment
set +u

# CRITICAL: Override cmd_backend BEFORE sourcing cmd.sh
# We're already inside a SLURM job, so don't spawn more SLURM jobs!
# This prevents the "job explosion" where each tokenization task
# would spawn additional SLURM jobs (8 format + 8 codec + 8 ssl)
export cmd_backend='local'

. ./path.sh
. ./cmd.sh

# Double-check that commands are set correctly
export train_cmd="run.pl"
export cuda_cmd="run.pl"
export decode_cmd="run.pl"
set -u

log "Starting tokenization for subdirectory: ${subdir_name}"

# Check if wav.scp exists and has content
if [ ! -f "${subdirectory}/wav.scp" ] || [ ! -s "${subdirectory}/wav.scp" ]; then
    log "No audio files found in ${subdir_name}, skipping tokenization"
    exit 0
fi

# Check if already tokenized (skip if complete)
if [ -f "${subdirectory}/audio/wav.scp" ] && \
   [ -f "${subdirectory}/audio/codec_wav.scp" ] && \
   [ -f "${subdirectory}/audio/ssl_wav.scp" ]; then
    log "Already tokenized: ${subdir_name}, skipping"
    exit 0
fi

log "Tokenizing audio for ${subdir_name}"

# Step 1: Audio format conversion
log "Step 1/3: Format wav.scp for ${subdir_name}"
scripts/audio/format_wav_scp.sh \
    --nj "${nj}" \
    --cmd "run.pl" \
    --audio-format "flac.ark" \
    --fs "${fs}" \
    --out_filename wav.scp \
    ${subdirectory}/wav.scp \
    ${subdirectory}/audio_raw

# Create audio directory and copy metadata
mkdir -p ${subdirectory}/audio
cp ${subdirectory}/audio_raw/utt2num_samples ${subdirectory}/audio

# Step 2: Codec and SSL tokenization
log "Step 2/3: Codec and SSL tokenization for ${subdir_name}"

scripts/feats/codec_ssl_tokenization.sh \
    --src_dir ${subdirectory}/audio_raw \
    --tgt_dir ${subdirectory}/audio \
    --file_name wav.scp \
    --fs ${fs} \
    --nj ${nj} \
    --codec_choice ${codec_choice} \
    --codec_hf_model_tag ${codec_hf_model_tag} \
    --codec_dump_audio false \
    --ssl_choice ${ssl_choice} \
    --ssl_checkpoint_path ${ssl_checkpoint_path} \
    --ssl_kmeans_path ${ssl_kmeans_path} \
    --ssl_nlayer ${ssl_nlayer} \
    --ssl_batch_bins ${ssl_batch_bins}

# Step 3: Generate data.json
log "Step 3/3: Generate data.json for ${subdir_name}"
python3 pyscripts/utils/make_speechlm_json.py \
    --task instruction_tuning \
    --output_json ${subdirectory}/data.json \
    --file_modality_type ${subdirectory}/audio/wav.scp,codec_ssl,kaldi_ark \
    --file_modality_type ${subdirectory}/prompt,text_bpe,text \
    --file_modality_type ${subdirectory}/text,text_bpe,text

# Count examples
if [ -f "${subdirectory}/data.json" ]; then
    num_examples=$(wc -l < ${subdirectory}/data.json)
    log "Successfully tokenized ${subdir_name} with ${num_examples} examples [elapsed=${SECONDS}s]"
else
    log "Warning: data.json not created for ${subdir_name}"
    exit 1
fi

log "Tokenization complete for ${subdir_name} [elapsed=${SECONDS}s]"
