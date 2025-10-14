#!/usr/bin/env bash

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Efficient parallel tokenization using SLURM array jobs
# This script should be run on a login/CPU node - it orchestrates GPU jobs but doesn't need GPU itself

set -e
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=7
stop_stage=7

# Tokenization parameters
nj=8                # Number of parallel jobs within each tokenization task
fs=16000
codec_choice=ESPnet
codec_hf_model_tag=ftshijt/espnet_codec_dac_large_v1.4_360epoch
ssl_choice=espnet_hubert
ssl_nlayer=18
ssl_checkpoint_path=exp/kmeans/38epoch.pth
ssl_kmeans_path=exp/kmeans/xeus_18_5000clusters/km_5000.mdl
ssl_batch_bins=5000000

# SLURM array job settings
max_parallel_jobs=20      # Maximum concurrent GPU jobs
job_time="48:00:00"       # Time limit per subdirectory job
job_mem="240G"            # Memory per job
job_partition="ghx4"      # SLURM partition
job_account="bbjs-dtai-gh"  # SLURM account
job_cpus=16               # CPUs per task (for parallel processing within each job)

# Data directories
sift_dir=
vctk_dir=
cv_dir=
mls_dir=

. utils/parse_options.sh

set +u
. ./db.sh
. ./path.sh
. ./cmd.sh
set -u

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Parallel tokenization of speech instruction data"

    dir=/work/hdd/bbjs/chuang14/dump/raw_audio_text_dialogue_speech_instruction

    # Step 1: Data preparation (runs on login node - CPU only)
    log "Step 1: Preparing speech instruction data (running on login node)"

    # Check if data prep is already done
    if [ ! -f "${dir}/.data_prep_done" ]; then
        log "Running data preparation script..."

        python local/data_prep_instr_tuning_fast.py \
            --output_dir ${dir} \
            --root_dir ${sift_dir} \
            --vctk_dir ${vctk_dir} \
            --mls_dir ${mls_dir} \
            --cv_dir ${cv_dir}

        # Mark as done
        touch ${dir}/.data_prep_done
        log "Data preparation completed successfully"
    else
        log "Data preparation already done (found ${dir}/.data_prep_done)"
    fi

    # Step 2: Create list of subdirectories to tokenize
    log "Step 2: Scanning subdirectories for tokenization"

    subdir_list=${dir}/subdirs_to_process.txt
    > ${subdir_list}  # Clear file

    for subdirectory in ${dir}/*; do
        if [ -d "${subdirectory}" ]; then
            subdir_name=$(basename ${subdirectory})

            # Check if wav.scp exists and has content
            if [ -f "${subdirectory}/wav.scp" ] && [ -s "${subdirectory}/wav.scp" ]; then
                # Check if already tokenized
                if [ -f "${subdirectory}/audio/wav.scp" ] && \
                   [ -f "${subdirectory}/audio/codec_wav.scp" ] && \
                   [ -f "${subdirectory}/audio/ssl_wav.scp" ]; then
                    log "Already tokenized: ${subdir_name}, skipping"
                else
                    echo "${subdirectory}" >> ${subdir_list}
                fi
            else
                log "No audio files in ${subdir_name}, skipping"
            fi
        fi
    done

    # Count subdirectories to process
    num_subdirs=$(wc -l < ${subdir_list} 2>/dev/null || echo 0)
    log "Found ${num_subdirs} subdirectories to tokenize"

    if [ ${num_subdirs} -eq 0 ]; then
        log "No subdirectories to process, exiting"
        exit 0
    fi

    # Step 3: Submit SLURM array job for parallel tokenization
    log "Step 3: Submitting SLURM array job for ${num_subdirs} subdirectories"

    # Create log directory
    mkdir -p ${dir}/logs

    # Create array job script
    array_job_script=${dir}/tokenize_array_job.sh
    cat > ${array_job_script} << 'EOFSCRIPT'
#!/usr/bin/env bash
#SBATCH --job-name=tokenize_array
#SBATCH --output=OUTPUT_DIR/logs/tokenize_%A_%a.log
#SBATCH --error=OUTPUT_DIR/logs/tokenize_%A_%a.err
#SBATCH --time=JOB_TIME
#SBATCH --mem=JOB_MEM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=JOB_CPUS
#SBATCH --partition=JOB_PARTITION
#SBATCH --account=JOB_ACCOUNT

set -e
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

# Get subdirectory for this array task
subdirectory=$(sed -n "${SLURM_ARRAY_TASK_ID}p" SUBDIR_LIST)

if [ -z "${subdirectory}" ]; then
    echo "Error: Could not get subdirectory for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

subdir_name=$(basename ${subdirectory})
log "Processing subdirectory: ${subdir_name} (task ${SLURM_ARRAY_TASK_ID})"

# Change to working directory
cd WORK_DIR

# Source environment - critical to set cmd_backend to 'local'
# to prevent spawning more SLURM jobs from within this job
set +u
. ./path.sh
# CRITICAL: Export cmd_backend BEFORE sourcing cmd.sh
# This ensures child scripts also see cmd_backend='local'
export cmd_backend='local'
. ./cmd.sh
# Verify commands are set correctly (should be run.pl from cmd.sh)
log "Verifying command backend: train_cmd=${train_cmd}, cuda_cmd=${cuda_cmd}"
if [[ "${cuda_cmd}" != "run.pl" ]]; then
    log "ERROR: cuda_cmd=${cuda_cmd} (expected run.pl). Forcing override."
    export train_cmd="run.pl"
    export cuda_cmd="run.pl"
    export decode_cmd="run.pl"
fi
set -u

# Parse tokenization parameters
nj=NJ_VALUE
fs=FS_VALUE
codec_choice="CODEC_CHOICE_VALUE"
codec_hf_model_tag="CODEC_MODEL_VALUE"
ssl_choice="SSL_CHOICE_VALUE"
ssl_checkpoint_path="SSL_CKPT_VALUE"
ssl_kmeans_path="SSL_KMEANS_VALUE"
ssl_nlayer=SSL_NLAYER_VALUE
ssl_batch_bins=SSL_BATCH_BINS_VALUE

# Check if already tokenized (race condition protection)
if [ -f "${subdirectory}/audio/wav.scp" ] && \
   [ -f "${subdirectory}/audio/codec_wav.scp" ] && \
   [ -f "${subdirectory}/audio/ssl_wav.scp" ]; then
    log "Already tokenized: ${subdir_name}, skipping"
    exit 0
fi

log "Step 1/3: Audio format conversion for ${subdir_name}"
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

log "Step 2/3: Codec and SSL tokenization for ${subdir_name}"
# Run with explicitly set cmd variables in the environment
train_cmd="run.pl" cuda_cmd="run.pl" decode_cmd="run.pl" \
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

log "Step 3/3: Generate data.json for ${subdir_name}"
python3 pyscripts/utils/make_speechlm_json.py \
    --task instruction_tuning \
    --output_json ${subdirectory}/data.json \
    --file_modality_type ${subdirectory}/audio/wav.scp,codec_ssl,kaldi_ark \
    --file_modality_type ${subdirectory}/prompt,text_bpe,text \
    --file_modality_type ${subdirectory}/text,text_bpe,text

# Verify success
if [ -f "${subdirectory}/data.json" ]; then
    num_examples=$(wc -l < ${subdirectory}/data.json)
    log "Successfully tokenized ${subdir_name} with ${num_examples} examples [elapsed=${SECONDS}s]"
else
    log "Error: data.json not created for ${subdir_name}"
    exit 1
fi

log "Tokenization complete for ${subdir_name} [elapsed=${SECONDS}s]"
EOFSCRIPT

    # Replace placeholders in the script
    sed -i "s|OUTPUT_DIR|${dir}|g" ${array_job_script}
    sed -i "s|JOB_TIME|${job_time}|g" ${array_job_script}
    sed -i "s|JOB_MEM|${job_mem}|g" ${array_job_script}
    sed -i "s|JOB_CPUS|${job_cpus}|g" ${array_job_script}
    sed -i "s|JOB_PARTITION|${job_partition}|g" ${array_job_script}
    sed -i "s|JOB_ACCOUNT|${job_account}|g" ${array_job_script}
    sed -i "s|SUBDIR_LIST|${subdir_list}|g" ${array_job_script}
    sed -i "s|WORK_DIR|${PWD}|g" ${array_job_script}
    sed -i "s|NJ_VALUE|${nj}|g" ${array_job_script}
    sed -i "s|FS_VALUE|${fs}|g" ${array_job_script}
    sed -i "s|CODEC_CHOICE_VALUE|${codec_choice}|g" ${array_job_script}
    sed -i "s|CODEC_MODEL_VALUE|${codec_hf_model_tag}|g" ${array_job_script}
    sed -i "s|SSL_CHOICE_VALUE|${ssl_choice}|g" ${array_job_script}
    sed -i "s|SSL_CKPT_VALUE|${ssl_checkpoint_path}|g" ${array_job_script}
    sed -i "s|SSL_KMEANS_VALUE|${ssl_kmeans_path}|g" ${array_job_script}
    sed -i "s|SSL_NLAYER_VALUE|${ssl_nlayer}|g" ${array_job_script}
    sed -i "s|SSL_BATCH_BINS_VALUE|${ssl_batch_bins}|g" ${array_job_script}

    # Submit array job with proper array syntax
    job_id=$(sbatch --parsable --array=1-${num_subdirs}%${max_parallel_jobs} ${array_job_script})

    log "=========================================="
    log "Submitted SLURM array job: ${job_id}"
    log "Number of tasks: ${num_subdirs}"
    log "Max parallel jobs: ${max_parallel_jobs}"
    log "=========================================="
    log "Monitor progress with:"
    log "  squeue -j ${job_id}"
    log "  squeue -u \$USER"
    log "View logs in: ${dir}/logs/"
    log "Check completion: ls ${dir}/*/data.json | wc -l"
    log "=========================================="

    log "Total elapsed time: ${SECONDS}s"
fi

log "Script completed [elapsed=${SECONDS}s]"
