#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=2
stop_stage=100

num_nodes=1
num_proc_per_node=2
node_rank=0
master_addr=localhost
master_port=12346

# ASR
train_registered_specifier="audio_to_text:librispeech_train_960"
valid_registered_specifier="audio_to_text:librispeech_dev_clean"
test_registered_specifier="audio_to_text:librispeech_test_clean"

# TTS
train_registered_specifier="text_to_audio:librispeech_train_960"
valid_registered_specifier="text_to_audio:librispeech_dev_clean"
test_registered_specifier="text_to_audio:librispeech_test_clean"

train_config=conf/train.yaml
inference_config=conf/inference.yaml
stats_dir=exp/stats
exp_dir=exp/librispeech_asr

. utils/parse_options.sh

. ./db.sh
# source ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  python ../../../espnet2/speechlm/bin/prepare_length_stats.py \
    --train-registered-specifier "${train_registered_specifier}" \
    --valid-registered-specifier "${valid_registered_specifier}" \
    --train-config ${train_config} \
    --output-dir ${stats_dir} \
    --num-workers 128
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  deepspeed \
    --num_nodes ${num_nodes} \
    --num_gpus ${num_proc_per_node} \
    --node_rank ${node_rank} \
    --master_addr ${master_addr} \
    --master_port ${master_port} \
      ../../../espnet2/speechlm/bin/train.py \
      --train-registered-specifier "${train_registered_specifier}" \
      --valid-registered-specifier "${valid_registered_specifier}" \
      --train-config ${train_config} \
      --stats-dir ${stats_dir} \
      --output-dir ${exp_dir} 
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  inference_step=60000
  inference_tag=$(basename "${inference_config%.*}")
  python ../../../espnet2/speechlm/bin/inference.py \
    --train-config ${train_config} \
    --inference-config ${inference_config} \
    --model-checkpoint ${exp_dir}/checkpoints/step_${inference_step}/global_step60000/mp_rank_00_model_states.pt \
    --output-dir ${exp_dir}/inference/step_${inference_step} \
    --test-registered-specifier ${test_registered_specifier} \
    --num-worker 1
fi