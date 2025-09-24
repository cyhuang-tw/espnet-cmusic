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
    # Create temporary files to store the JSON paths
    train_jsons_file=$(mktemp)
    valid_jsons_file=$(mktemp)
    
    # Find all data.json files and write them to temporary files
    find "${base_dir}" -name "data.json" -type f | sort > "${train_jsons_file}"
    cp "${train_jsons_file}" "${valid_jsons_file}"
    
    # Export the file paths for speechlm.sh to use
    export TRAIN_JSONS_FILE="${train_jsons_file}"
    export VALID_JSONS_FILE="${valid_jsons_file}"
    
    echo "Found $(wc -l < "${train_jsons_file}") training datasets"
    echo "First few datasets:"
    head -3 "${train_jsons_file}"
    
    # Create wrapper script for speechlm.sh to handle the file list
    wrapper_script=$(mktemp)
    cat > "${wrapper_script}" << 'EOF'
#!/bin/bash
# Read the JSON file paths from the temporary files
if [ -f "${TRAIN_JSONS_FILE}" ] && [ -f "${VALID_JSONS_FILE}" ]; then
    train_jsons=$(cat "${TRAIN_JSONS_FILE}" | tr '\n' ' ' | sed 's/ $//')
    valid_jsons=$(cat "${VALID_JSONS_FILE}" | tr '\n' ' ' | sed 's/ $//')
    export train_jsons
    export valid_jsons
fi

# Execute the original speechlm.sh with all arguments
exec ./speechlm.sh "$@"
EOF
    chmod +x "${wrapper_script}"
    
    # Set cleanup trap
    trap "rm -f '${train_jsons_file}' '${valid_jsons_file}' '${wrapper_script}'" EXIT
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

# Run speechlm training pipeline using wrapper script
${wrapper_script} \
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
    --dumpdir dump \
    ${bpe_opts} \
    "$@"