#!/usr/bin/env bash
# Run segmentation + parallel inference + evaluation pipeline.
#
# Usage:
#   bash local/run_inference.sh \
#       --segment-sec 4 \
#       --num-splits 4 \
#       --exp-dir exp/asr_train_asr_transformer_linear_4_nonorm_righttime_large_xxl_posenc_raw_en_word \
#       --ckpt 58epoch.pth \
#       --src-dir dump/raw/maestro_dev_onsets \
#       --decode-config conf/decode_asr.yaml

set -euo pipefail

# Defaults
SEGMENT_SEC=4
NUM_SPLITS=4
EXP_DIR=""
CKPT=""
SRC_DIR="dump/raw/maestro_dev_onsets"
SRC_BASE="/work/nvme/bbjs/chen26/espnet_music/egs2/librispeech/asr1"
DECODE_CONFIG="conf/decode_asr.yaml"
OVERLAP_SEC=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --segment-sec) SEGMENT_SEC="$2"; shift 2 ;;
        --num-splits) NUM_SPLITS="$2"; shift 2 ;;
        --exp-dir) EXP_DIR="$2"; shift 2 ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        --src-dir) SRC_DIR="$2"; shift 2 ;;
        --src-base) SRC_BASE="$2"; shift 2 ;;
        --decode-config) DECODE_CONFIG="$2"; shift 2 ;;
        --overlap-sec) OVERLAP_SEC="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXP_DIR" || -z "$CKPT" ]]; then
    echo "Error: --exp-dir and --ckpt are required"
    exit 1
fi

SEG_DIR="dump/raw/maestro_dev_seg_${SEGMENT_SEC}s"
CKPT_NAME="${CKPT%.pth}"
OUT_DIR="${EXP_DIR}/decode_dev_seg_${SEGMENT_SEC}s_${CKPT_NAME}"

echo "=== Pipeline Config ==="
echo "  Segment:  ${SEGMENT_SEC}s"
echo "  Splits:   ${NUM_SPLITS}"
echo "  Model:    ${EXP_DIR}/${CKPT}"
echo "  Decode:   ${DECODE_CONFIG}"
echo "  Output:   ${OUT_DIR}"
echo ""

# Stage 1: Segment and write ark
if [[ ! -f "${SEG_DIR}/feats.scp" ]]; then
    echo "=== Stage 1: Segmentation ==="
    python local/prepare_segments_ark.py \
        --src-dir "${SRC_DIR}" \
        --src-base "${SRC_BASE}" \
        --out-dir "${SEG_DIR}" \
        --segment-sec "${SEGMENT_SEC}" \
        --overlap-sec "${OVERLAP_SEC}"
else
    echo "=== Stage 1: Skipped (${SEG_DIR}/feats.scp exists) ==="
fi

TOTAL=$(wc -l < "${SEG_DIR}/feats.scp")
echo "  Total segments: ${TOTAL}"

# Stage 2: Split key file
echo ""
echo "=== Stage 2: Split key file ==="
SPLIT_DIR="/tmp/ark_split_${SEGMENT_SEC}s_$$"
mkdir -p "${SPLIT_DIR}"
split -n "l/${NUM_SPLITS}" -d "${SEG_DIR}/feats.scp" "${SPLIT_DIR}/split_"

for i in $(seq 0 $((NUM_SPLITS-1))); do
    f="${SPLIT_DIR}/split_$(printf '%02d' $i)"
    echo "  Split $i: $(wc -l < "$f") segments"
done

# Stage 3: Parallel inference
echo ""
echo "=== Stage 3: Inference (${NUM_SPLITS} parallel jobs) ==="
mkdir -p "${OUT_DIR}"

pids=()
for i in $(seq 0 $((NUM_SPLITS-1))); do
    python -m espnet2.bin.asr_inference \
        --asr_train_config "${EXP_DIR}/config.yaml" \
        --asr_model_file "${EXP_DIR}/${CKPT}" \
        --data_path_and_name_and_type "${SEG_DIR}/feats.scp,speech,kaldi_ark" \
        --key_file "${SPLIT_DIR}/split_$(printf '%02d' $i)" \
        --output_dir "${OUT_DIR}/split${i}" \
        --config "${DECODE_CONFIG}" \
        --ngpu 1 \
        --batch_size 1 2>/dev/null &
    pids+=($!)
    echo "  Started split $i (PID $!)"
done

echo "  Waiting for all jobs..."
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "  All jobs done."

# Stage 4: Merge results
echo ""
echo "=== Stage 4: Merge results ==="
mkdir -p "${OUT_DIR}/1best_recog"
cat "${OUT_DIR}"/split*/1best_recog/text | sort > "${OUT_DIR}/1best_recog/text"
DECODED=$(wc -l < "${OUT_DIR}/1best_recog/text")
echo "  Merged: ${DECODED} predictions"

# Stage 5: Evaluate
echo ""
echo "=== Stage 5: Evaluation ==="
python local/evaluate_onsets.py \
    --pred "${OUT_DIR}/1best_recog/text" \
    --ref "${SEG_DIR}/text"

# Cleanup
rm -rf "${SPLIT_DIR}"

echo ""
echo "=== Done ==="
echo "  Results: ${OUT_DIR}/1best_recog/text"
