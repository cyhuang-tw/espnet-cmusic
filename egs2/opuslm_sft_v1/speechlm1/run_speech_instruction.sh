#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Data configuration for speech instruction tuning
# Note: Multiple data.json files from subdirectories (file-by-file structure to avoid OOM)
train_jsons=""
valid_jsons=""

# Auto-discover all data.json files from subdirectories
base_dir="dump/raw_audio_text_dialogue_speech_instruction"
if [ -d "${base_dir}" ]; then
    for subdir in ${base_dir}/*/; do
        if [ -f "${subdir}data.json" ]; then
            train_jsons="${train_jsons} ${subdir}data.json"
            valid_jsons="${valid_jsons} ${subdir}data.json"
        fi
    done
    # Remove leading spaces
    train_jsons=$(echo ${train_jsons} | sed 's/^ *//')
    valid_jsons=$(echo ${valid_jsons} | sed 's/^ *//')
    echo "Found training datasets: ${train_jsons}"
else
    echo "Warning: ${base_dir} not found. Please run data preparation first."
    exit 1
fi

# Training and inference configuration
train_config=conf/train_speech_instruction.yaml
inference_config=conf/decode_general.yaml

# Token list configuration
token_list_dir=data/token_list/llm_vocab_olmo  # Use LLM vocab
bpe_opts="--subword_choice huggingface --subword_model allenai/OLMo-2-1124-7B"

# Run speechlm training pipeline
./speechlm.sh \
    --skip_data_prep true \
    --data_combo_name speech_instruction \
    --fs 16000 \
    --ngpu 4 \
    --nj 16 \
    --inference_nj 16 \
    --nbest 10 \
    --gpu_inference true \
    --token_list_dir ${token_list_dir} \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --audio_format "flac.ark" \
    --train_jsons "${train_jsons}" \
    --valid_jsons "${valid_jsons}" \
    --dumpdir dump \
    ${bpe_opts} \
    "$@"