#!/usr/bin/env bash
# Train with positional encoding added to encoder (input_layer: linear)
set -e
set -u
set -o pipefail

train_set="maestro_train_onsets"
valid_set="maestro_dev_onsets"
test_sets="maestro_dev_onsets"

asr_config=conf/tuning/train_asr_transformer_linear_4_nonorm_righttime_large_xxl_posenc.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --token_type word \
    --use_lm false \
    --stage 11 \
    --stop_stage 11 \
    --max_wav_duration 9000 \
    --feats_normalize utt_mvn \
    --nj 1 \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
