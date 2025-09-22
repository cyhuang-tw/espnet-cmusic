#!/usr/bin/env bash

# Example script to run speech instruction data preparation
# Modify the paths below according to your data locations

set -e
set -u
set -o pipefail

# Set your data paths here
SIFT_DIR="/path/to/sift50m/data"          # Root directory of SIFT-50M dataset
VCTK_DIR="/path/to/vctk"                  # Path to VCTK corpus
MLS_DIR="/path/to/multilingual_librispeech" # Path to Multilingual LibriSpeech
CV_DIR="/path/to/common_voice"            # Path to CommonVoice

# Run the data preparation stage for speech instruction tuning
./local/data.sh \
  --stage 7 \
  --stop_stage 7 \
  --sift_dir "${SIFT_DIR}" \
  --vctk_dir "${VCTK_DIR}" \
  --mls_dir "${MLS_DIR}" \
  --cv_dir "${CV_DIR}"

echo "Speech instruction data preparation completed!"
echo "Data saved to: dump/raw_audio_text_dialogue_speech_instruction/"
echo ""
echo "To run training, execute:"
echo "./run_speech_instruction.sh"