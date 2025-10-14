#!/usr/bin/env bash

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script prepare the SFT data for OpusLM-V1.

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
# set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=5
stop_stage=5

TULU3=data/local/tulu3
OpenAudioBench=data/local/openaudiobench
OLMO2_SFT=/work/hdd/bbjs/jtian1/tools/olmo2_dpo
OLMO2_DPO=data/local/olmo2_dpo
sift_dir=
vctk_dir=
mls_dir=
cv_dir=

user_prompt_list=dump/raw_codec_ssl_tts_yodas/train_yodas/index_files/wav.scp       # prompt to generate user speech
assistant_prompt_list=data/assistant_prompt.scp                                     # prompt to generate assistant speech
user_prompt_list=$assistant_prompt_list

. utils/parse_options.sh

set +u
. ./db.sh
. ./path.sh
. ./cmd.sh
set -u

if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli command not found. Please install it first." >&2
    exit 1
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Convert TULU3 dataset to ESPnet-SpeechLM data format"
    python3 local/data_prep_tulu3.py \
      --download_dir ${TULU3}/data \
      --output_dir dump/raw_text_dialogue_tulu3

    for dset in train valid; do
        dir=dump/raw_text_dialogue_tulu3/${dset}
        cp ${dir}/data/dialogue.1 ${dir}/dialogue
        python3 pyscripts/utils/make_speechlm_json.py \
          --task text_dialogue \
          --output_json ${dir}/data.json \
          --file_modality_type ${dir}/dialogue,dialogue,dialogue_json
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Convert OpenAudioBench dataset to ESPnet-SpeechLM data format"
    # huggingface-cli download --repo-type dataset --local-dir ${OpenAudioBench} baichuan-inc/OpenAudioBench
    python3 local/data_prep_openaudiobench.py \
      --download_dir ${OpenAudioBench}/eval_datas \
      --output_dir dump/raw_text_dialogue_openaudiobench
    
    for dset in alpaca_eval llama_questions trivia_qa web_questions; do
        dir=dump/raw_text_dialogue_openaudiobench/${dset}
        cp ${dir}/data/dialogue.1 ${dir}/dialogue
        python3 pyscripts/utils/make_speechlm_json.py \
          --task text_dialogue \
          --output_json ${dir}/data.json \
          --file_modality_type ${dir}/dialogue,dialogue,dialogue_json
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Convert OLMo2 DPO dataset to ESPnet-SpeechLM data format"
    python local/data_prep_olmo2_7b_dpo.py \
      --download_dir ${OLMO2_DPO} \
      --output_dir dump/raw_text_dialogue_olmo2_dpo 
    for dset in train valid; do
        dir=dump/raw_text_dialogue_olmo2_dpo/${dset}
        cp ${dir}/data/dialogue.1 ${dir}/dialogue
        python3 pyscripts/utils/make_speechlm_json.py \
          --task text_dialogue \
          --output_json ${dir}/data.json \
          --file_modality_type ${dir}/dialogue,dialogue,dialogue_json
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Convert OLMO2_SFT dataset to ESPnet-SpeechLM data format"
    python3 local/data_prep_olmo2_7b_sft.py \
      --download_dir ${OLMO2_SFT}/data \
      --output_dir dump/raw_text_dialogue_olmo2_sft

    for dset in train valid; do
        dir=dump/raw_text_dialogue_olmo2_sft/${dset}
        cp ${dir}/data/dialogue.1 ${dir}/dialogue
        python3 pyscripts/utils/make_speechlm_json.py \
          --task text_dialogue \
          --output_json ${dir}/data.json \
          --file_modality_type ${dir}/dialogue,dialogue,dialogue_json
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Convert OpenAudioBench dataset to Speech-to-Speech QA"
    huggingface-cli download --repo-type dataset --local-dir ${OpenAudioBench} baichuan-inc/OpenAudioBench
    dir=dump/raw_text_dialogue_openaudiobench
    python3 local/data_prep_openaudiobench.py \
      --download_dir ${OpenAudioBench}/eval_datas --output_dir ${dir}
    
    # for dset in alpaca_eval llama_questions trivia_qa web_questions; do

    # Audio-Text dialogues
    # tgt_dir=dump/raw_audio_text_dialogue_openaudiobench
    # for dset in llama_questions; do
    #     cp ${dir}/${dset}/data/dialogue.1 ${dir}/${dset}/dialogue

    #     bash scripts/utils/speechlm_text_dialogue_to_speech_dialogue.sh \
    #       --input_dir ${dir}/${dset} \
    #       --output_dir ${tgt_dir}/${dset} \
    #       --task audio_text_dialogue \
    #       --ready_audio_list ${dir}/${dset}/wav.scp \
    #       --user_prompt_list ${user_prompt_list} \
    #       --assistant_prompt_list ${assistant_prompt_list} 
    # done

    # Audio-Audio dialogues
    tgt_dir=dump/raw_audio_dialogue_openaudiobench
    for dset in alpaca_eval llama_questions trivia_qa web_questions; do

        cp ${dir}/${dset}/data/dialogue.1 ${dir}/${dset}/dialogue
        bash scripts/utils/speechlm_text_dialogue_to_speech_dialogue.sh \
          --input_dir ${dir}/${dset} \
          --output_dir ${tgt_dir}/${dset} \
          --task audio_dialogue \
          --ready_audio_list ${dir}/${dset}/wav.scp \
          --user_prompt_list ${user_prompt_list} \
          --assistant_prompt_list ${assistant_prompt_list} 

        cp ${tgt_dir}/${dset}/data/dialogue.1 ${tgt_dir}/${dset}/dialogue
        python3 pyscripts/utils/make_speechlm_json.py \
          --task audio_dialogue \
          --output_json ${tgt_dir}/${dset}/data.json \
          --file_modality_type ${tgt_dir}/${dset}/dialogue,dialogue,dialogue_json
    done
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Convert SIFT-50M dataset for Speech Instruction Tuning"
    
    # Check required directories
    if [ -z "${sift_dir}" ] || [ ! -d "${sift_dir}" ]; then
        log "Error: sift_dir is not set or does not exist. Please set --sift_dir"
        exit 1
    fi
    if [ -z "${vctk_dir}" ] || [ ! -d "${vctk_dir}" ]; then
        log "Error: vctk_dir is not set or does not exist. Please set --vctk_dir"
        exit 1
    fi
    if [ -z "${mls_dir}" ] || [ ! -d "${mls_dir}" ]; then
        log "Error: mls_dir is not set or does not exist. Please set --mls_dir"
        exit 1
    fi
    if [ -z "${cv_dir}" ] || [ ! -d "${cv_dir}" ]; then
        log "Error: cv_dir is not set or does not exist. Please set --cv_dir"
        exit 1
    fi
    
    # Prepare speech instruction data (now saves data file by file to avoid OOM)
    dir=/work/hdd/bbjs/chuang14/dump/raw_audio_text_dialogue_speech_instruction
    python local/data_prep_instr_tuning.py \
      --output_dir ${dir} \
      --root_dir ${sift_dir} \
      --vctk_dir ${vctk_dir} \
      --mls_dir ${mls_dir} \
      --cv_dir ${cv_dir}
    
    # Tokenization parameters (consistent with speechlm_text_dialogue_to_speech_dialogue.sh)
    nj=8
    fs=16000
    codec_choice=ESPnet
    codec_hf_model_tag=ftshijt/espnet_codec_dac_large_v1.4_360epoch
    ssl_choice=espnet_hubert
    ssl_nlayer=18
    ssl_checkpoint_path=exp/kmeans/38epoch.pth
    ssl_kmeans_path=exp/kmeans/xeus_18_5000clusters/km_5000.mdl
    ssl_batch_bins=5000000
    
    # Process each subdirectory and tokenize audio
    total_examples=0
    for subdirectory in ${dir}/*; do
        if [ -d "${subdirectory}" ]; then
            subdir_name=$(basename ${subdirectory})
            log "Processing subdirectory: ${subdir_name}"
            
            # Check if wav.scp exists and has content
            if [ -f "${subdirectory}/wav.scp" ] && [ -s "${subdirectory}/wav.scp" ]; then
                log "Tokenizing audio for ${subdir_name}"
                
                # Audio format conversion and tokenization
                scripts/audio/format_wav_scp.sh \
                    --nj "${nj}" \
                    --cmd "${train_cmd}" \
                    --audio-format "flac.ark" \
                    --fs "${fs}" \
                    --out_filename wav.scp \
                    ${subdirectory}/wav.scp \
                    ${subdirectory}/audio_raw
                
                # Create audio directory and copy metadata
                mkdir -p ${subdirectory}/audio
                cp ${subdirectory}/audio_raw/utt2num_samples ${subdirectory}/audio
                
                # Perform codec and SSL tokenization
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
            else
                log "No audio files found in ${subdir_name}, skipping tokenization"
            fi
            
            # Generate data.json with proper file references
            # cp ${subdirectory}/data/dialogue.1 ${subdirectory}/dialogue
            python3 pyscripts/utils/make_speechlm_json.py \
              --task instruction_tuning \
              --output_json ${subdirectory}/data.json \
              --file_modality_type ${subdirectory}/audio/wav.scp,codec_ssl,kaldi_ark \
              --file_modality_type ${subdirectory}/prompt,text_bpe,text \
              --file_modality_type ${subdirectory}/text,text_bpe,text

            
            # subdir_examples=$(wc -l < ${subdirectory}/dialogue)
            # total_examples=$((total_examples + subdir_examples))
            # log "Generated ${subdirectory}/data.json with ${subdir_examples} examples"
        fi
    done
    log "Total examples across all subdirectories: ${total_examples}"
fi