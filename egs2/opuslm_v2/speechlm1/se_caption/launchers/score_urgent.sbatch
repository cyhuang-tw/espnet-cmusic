#!/bin/bash
# Score enhanced audio with the URGENT-2026 metric suite, then aggregate.
#
#   sbatch --account=$ACC --partition=$PART score_urgent.sbatch <inf.scp> <out_dir>
#
# Requires the metric env vars (set in env.sh or the shell) — see se_caption/README.md:
#   URGENT_DATA (challenge dir), URGENT_PY, SCOREQ_PY, DNSMOS_MODELS, [ESPEAK_DIR], [ORTPATCH]
# References (16 kHz) are built once from the URGENT validation set.
#SBATCH --job-name=se_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log
set -eu
cd "$(dirname "$0")"; source ./env.sh
INF=${1:?inf.scp}; OUT=${2:?output dir}
EVAL="$ESPNET_ROOT/egs2/opuslm_v2/speechlm1/se_caption/eval"
: "${URGENT_DATA:?set URGENT_DATA}"

# 1) build 16 kHz references once (clean + noisy) for the reference-based metrics.
if [ ! -s "$SE_ROOT/eval/refs/ref16k.scp" ]; then
  SE_ROOT="$SE_ROOT" URGENT_DATA="$URGENT_DATA" python "$EVAL/build_refs16k.py"
fi
REF="$SE_ROOT/eval/refs/ref16k.scp"
TEXT="$URGENT_DATA/data/validation/text"
UTT2LANG="$URGENT_DATA/data/validation/utt2lang"

# 2) run the official metric suite (intrusive metrics are reported but are not
#    meaningful for a generative model — see README/REPORT.md).
bash "$EVAL/run_urgent_score.sh" "$INF" "$REF" "$TEXT" "$UTT2LANG" "$OUT" cuda 8
echo "scores in $OUT/score/"
