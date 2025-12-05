#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=100

num_nodes=1
num_proc_per_node=8
node_rank=0
master_addr=localhost
master_port=12346

# ASR
train_registered_specifier="audio_to_text:owsm_v4"
valid_registered_specifier="audio_to_text:librispeech_dev"
test_registered_specifier="audio_to_text:librispeech_dev"

# Rich caption - Audio-to-text
train_registered_specifier="audio_to_text:owsm_v4_caption"
valid_registered_specifier="audio_to_text:librispeech_dev"

# Rich caption - Text-to-Audio
# train_registered_specifier="text_to_audio:owsm_v4_caption"
# valid_registered_specifier="text_to_audio:librispeech_dev"

# Text-only
# train_registered_specifier="text_only:dolma3"

train_config=conf/train.yaml

stats_dir=exp/stats
exp_dir=exp/owsm_audio_to_text_caption_v2

inference_config=conf/inference.yaml
inference_step=10000
inference_nj=1

. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  python ../../../espnet2/speechlm/bin/prepare_length_stats.py \
    --train-registered-specifier "${train_registered_specifier}" \
    --valid-registered-specifier "${valid_registered_specifier}" \
    --train-config ${train_config} \
    --output-dir ${stats_dir} \
    --num-workers 64
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then  
  torchrun \
    --nnodes=${num_nodes} \
    --node_rank=${node_rank} \
    --nproc_per_node=${num_proc_per_node} \
    --master_addr=${master_addr} \
    --master_port=${master_port} \
      ../../../espnet2/speechlm/bin/train.py \
      --train-registered-specifier "${train_registered_specifier}" \
      --valid-registered-specifier "${valid_registered_specifier}" \
      --train-config ${train_config} \
      --stats-dir ${stats_dir} \
      --output-dir ${exp_dir} \
      --save-loader-state
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  inference_tag=$(basename "${inference_config%.*}")

  inference_dir=${exp_dir}/inference/${inference_tag}_step_${inference_step}
  mkdir -p ${inference_dir}

  inference_ckpt=${exp_dir}/checkpoints/step_${inference_step}/global_step${inference_step}/mp_rank_00_model_states.pt

  echo "Start model inference. Log at ${inference_dir}/logs/inference.*.log"
  ${cuda_cmd} JOB=1:${inference_nj} ${inference_dir}/logs/inference.JOB.log \
    ../../../espnet2/speechlm/bin/inference.py \
      --rank JOB --world-size ${inference_nj} \
      --train-config ${exp_dir}/train.yaml \
      --inference-config ${inference_config} \
      --model-checkpoint ${inference_ckpt} \
      --output-dir ${inference_dir} \
      --test-registered-specifier "${test_registered_specifier}" \
      --num-worker 1
fi