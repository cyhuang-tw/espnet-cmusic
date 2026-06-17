#!/bin/bash
# Train generative SE (plain or caption-conditioned; LoRA / full-FT / peft).
#
#   sbatch --account=$ACC --partition=$PART train_se.sbatch <mode> [SMOKE]
#     mode = lora | full | peft | caption_lora | caption_full
#
# Bare-metal equivalent (no SLURM) is just the `torchrun ...` block below.
# Configure paths in env.sh first (cp env.example.sh env.sh).
#SBATCH --job-name=se_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
# NOTE: set --account / --partition on the sbatch CLI, and --gpus-per-node below.
#SBATCH --gpus-per-node=1            # full / caption_full want 4; override: sbatch --gpus-per-node=4
#SBATCH --output=%x_%j.log
set -eu
cd "$(dirname "$0")"; source ./env.sh
MODE=${1:?usage: train_se.sbatch <lora|full|peft|caption_lora|caption_full> [SMOKE]}
SMOKE=${2:-}
RECIPE="$ESPNET_ROOT/egs2/opuslm_v2/speechlm1"
NGPU=${SLURM_GPUS_ON_NODE:-1}

case "$MODE" in
  lora)         TASK=speech_enhancement;           CFG=train_se_bagpiper_lora.yaml;     RESUME="--resume-path $PRETRAIN_CKPT_FILE" ;;
  peft)         TASK=speech_enhancement;           CFG=train_se_bagpiper_peft.yaml;     RESUME="--resume-path $PRETRAIN_CKPT_FILE" ;;
  full)         TASK=speech_enhancement;           CFG=train_se_bagpiper_full.yaml;     RESUME="--resume-path $PRETRAIN_CKPT_DIR" ;;
  caption_lora) TASK=speech_enhancement_captioned; CFG=train_se_bagpiper_lora.yaml;     RESUME="--resume-path $PRETRAIN_CKPT_FILE" ;;
  caption_full) TASK=speech_enhancement_captioned; CFG=train_se_bagpiper_full_cap.yaml; RESUME="--resume-path $PRETRAIN_CKPT_DIR" ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
[ -n "$SMOKE" ] && CFG="${CFG%.yaml}_smoke.yaml"

# Which manifests to use: plain SE uses se_train/se_valid; captioned uses the captioned ones.
case "$TASK" in
  speech_enhancement)            TRAIN=se_train;     VALID=se_valid ;;
  speech_enhancement_captioned)  TRAIN=se_cap_train; VALID=se_cap_valid ;;
esac
TRAIN_SPEC="$TASK:$TRAIN:$SE_ROOT/data/$TRAIN/dataset.json"
VALID_SPEC="$TASK:$VALID:$SE_ROOT/data/$VALID/dataset.json"
OUT="$SE_ROOT/exp/se_${MODE}${SMOKE:+_smoke}"
STATS="$SE_ROOT/exp/stats_${TASK}"

cd "$RECIPE"
# 1) length stats (once per task+data): the trainer needs them for length batching.
if [ ! -s "$STATS/stats_${TASK}_${TRAIN}.jsonl" ]; then
  python "$ESPNET_ROOT/espnet2/speechlm/bin/prepare_length_stats.py" \
    --train-config "se_caption/conf/$CFG" \
    --train-unregistered-specifier "$TRAIN_SPEC" \
    --valid-unregistered-specifier "$VALID_SPEC" \
    --output-dir "$STATS" --num-workers 16
fi
# 2) resume policy: a COMPLETE local checkpoint (has inner "latest") auto-resumes;
#    otherwise resume from the pretrain checkpoint ($RESUME).
for d in "$OUT"/checkpoints/step_*; do [ -f "$d/latest" ] && RESUME=""; done

torchrun --nproc_per_node="$NGPU" --master_port=29611 \
  "$ESPNET_ROOT/espnet2/speechlm/bin/train.py" \
  --train-config "se_caption/conf/$CFG" \
  --train-unregistered-specifier "$TRAIN_SPEC" \
  --valid-unregistered-specifier "$VALID_SPEC" \
  --stats-dir "$STATS" --output-dir "$OUT" $RESUME --wandb-mode disabled
