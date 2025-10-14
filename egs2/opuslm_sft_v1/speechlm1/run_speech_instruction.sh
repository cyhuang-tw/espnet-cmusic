#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
# set -u
set -o pipefail

# Data configuration for speech instruction tuning
# Note: Multiple data.json files from subdirectories (file-by-file structure to avoid OOM)
train_jsons=""
valid_jsons=""

# Auto-discover all data.json files from subdirectories
base_dir="/work/hdd/bbjs/chuang14/dump/raw_audio_text_dialogue_speech_instruction"
if [ -d "${base_dir}" ]; then
    # Create temporary files to store the JSON paths
    train_jsons_file=$(mktemp)
    valid_jsons_file=$(mktemp)
    
    # Find all data.json files but exclude those in stats directories
    find "${base_dir}" -name "data.json" -type f -not -path "*/stats/*" | sort > "${train_jsons_file}"
    cp "${train_jsons_file}" "${valid_jsons_file}"
    
    
    echo "Found $(wc -l < "${train_jsons_file}") training datasets"
    echo "First few datasets:"
    head -3 "${train_jsons_file}"
    
    # Convert to space-separated strings for speechlm.sh
    train_jsons=$(cat "${train_jsons_file}" | tr '\n' ' ' | sed 's/ $//')
    valid_jsons=$(cat "${valid_jsons_file}" | tr '\n' ' ' | sed 's/ $//')
    
    # Create a temporary config file to pass these variables
    config_file=$(mktemp --suffix=.conf)
    cat > "${config_file}" << EOF
train_jsons="${train_jsons}"
valid_jsons="${valid_jsons}"
EOF
    
    # Set cleanup trap
    trap "rm -f '${train_jsons_file}' '${valid_jsons_file}' '${config_file}'" EXIT
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
# Use a temporary config file to pass train_jsons and valid_jsons to avoid argument length limits
./speechlm.sh \
    --config "${config_file}" \
    --skip_data_prep true \
    --data_combo_name speech_instruction \
    --fs 16000 \
    --ngpu 2 \
    --nj 16 \
    --inference_nj 16 \
    --nbest 10 \
    --gpu_inference true \
    --token_list_dir ${token_list_dir} \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --audio_format "flac.ark" \
    --dumpdir /work/hdd/bbjs/chuang14/dump \
    ${bpe_opts} \
    "$@"

    # --init_param "/work/hdd/bbjs/chuang14/model.pth" \
