#!/bin/bash
# Run generative-SE inference (audio in [+ caption] -> enhanced audio out) and
# build an inf.scp for scoring.
#
#   sbatch --account=$ACC --partition=$PART infer_se.sbatch \
#       <checkpoint> <task> <manifest_name> <out_name> [build=lora|full]
#     task         = speech_enhancement | speech_enhancement_captioned
#     manifest_name= a dir under $SE_ROOT/data/ holding dataset.json
#     build        = lora (default) loads the LoRA-aware build config; full for full-FT/peft
#
# The bare command is the `python .../inference.py ...` block below.
#SBATCH --job-name=se_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log
set -eu
cd "$(dirname "$0")"; source ./env.sh
CKPT=${1:?checkpoint path}; TASK=${2:?task}; MANIFEST=${3:?manifest name}
OUTNAME=${4:?output name}; BUILD=${5:-lora}
RECIPE="$ESPNET_ROOT/egs2/opuslm_v2/speechlm1"
case "$BUILD" in
  lora) TRAIN_CFG=se_caption/conf/infer_se_bagpiper_train_lora.yaml ;;
  full) TRAIN_CFG=se_caption/conf/infer_se_bagpiper_train.yaml ;;
  *) echo "build must be lora|full"; exit 1 ;;
esac
OUT="$SE_ROOT/eval/enhanced/$OUTNAME"; mkdir -p "$OUT"
cd "$RECIPE"
# cfg≈3 + temp 0.3 (the inference-config) is REQUIRED — see README "Gotchas".
python "$ESPNET_ROOT/espnet2/speechlm/bin/inference.py" \
  --train-config "$TRAIN_CFG" \
  --inference-config se_caption/conf/infer_se_bagpiper_demo_cfg3t03.yaml \
  --model-checkpoint "$CKPT" \
  --test-unregistered-specifier "$TASK:$MANIFEST:$SE_ROOT/data/$MANIFEST/dataset.json" \
  --output-dir "$OUT" --rank 1 --world-size 1 --num-workers 1 --seed 0
# collapse per-rank results.json into a single inf.scp (uid -> enhanced wav)
python - "$OUT" <<'PY'
import json, glob, sys
out = sys.argv[1]; rows = {}
for rj in glob.glob(out + "/inference_rank*/results.json"):
    for u, segs in json.load(open(rj)).items():
        for r, m, c in segs:
            if m == "audio": rows[u] = c
open(out + "/inf.scp", "w").writelines(f"{u} {rows[u]}\n" for u in sorted(rows))
print("inf.scp:", len(rows))
PY
