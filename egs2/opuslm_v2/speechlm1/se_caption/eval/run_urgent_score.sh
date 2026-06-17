#!/bin/bash
# Official URGENT-2026 metric suite (minus emotion2vec).
# Usage: run_urgent_score.sh <inf_scp> <ref_scp> <text> <utt2lang> <out_dir> <device> <nj>
#
# Cluster-agnostic: set these env vars before running (see se_caption/README.md):
#   URGENT_DATA    URGENT challenge dir (contains evaluation_metrics/).            [required]
#   URGENT_PY      python with the URGENT metric deps installed.                   [default: `python`]
#   SCOREQ_PY      python for SCOREQ (torchaudio-compatible env).                  [default: $URGENT_PY]
#   DNSMOS_MODELS  dir with sig_bak_ovr.onnx + model_v8.onnx (DNSMOS).             [required for dnsmos]
#   ESPEAK_DIR     espeak-ng install prefix (phoneme similarity).                  [optional]
#   ORTPATCH       PYTHONPATH shim dir forcing onnxruntime intra-op threads        [optional; only
#                  (works around pthread_setaffinity failures under slurm cpuset).  needed if SCOREQ fails]
set -u
R=${URGENT_DATA:?set URGENT_DATA to the URGENT challenge dir}
EM=$R/evaluation_metrics
UPY=${URGENT_PY:-python}
UENV=${SCOREQ_PY:-$UPY}
DNS=${DNSMOS_MODELS:?set DNSMOS_MODELS to the dir with the DNSMOS .onnx files}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -n "${ESPEAK_DIR:-}" ]; then
  export PHONEMIZER_ESPEAK_LIBRARY=$ESPEAK_DIR/lib/libespeak-ng.so
  export PATH=$ESPEAK_DIR/bin:$PATH LD_LIBRARY_PATH=$ESPEAK_DIR/lib:${LD_LIBRARY_PATH:-}
fi
export OMP_NUM_THREADS=4   # avoid onnxruntime pthread_setaffinity failure under slurm cpuset
inf=$(readlink -f "$1"); ref=$(readlink -f "$2"); text=$3; utt2lang=$4; OUT=$5; dev=${6:-cuda}; nj=${7:-8}
mkdir -p $OUT/score
# length-aligned copies for the equal-shape-asserting metrics (generative output != ref length)
$UPY $HERE/align_pairs.py $inf $ref $OUT/aligned
infA=$OUT/aligned/inf_al.scp; refA=$OUT/aligned/ref_al.scp
cd $R
run(){ echo "### $1 $(date +%H:%M:%S)"; eval "$2" && echo "  [$1] OK" || echo "  [$1] FAIL rc=$?"; }

run intrusive   "$UPY $EM/calculate_intrusive_se_metrics.py --ref_scp $refA --inf_scp $infA --output_dir $OUT/score/se --nj $nj"
run dnsmos      "$UPY $EM/calculate_nonintrusive_dnsmos.py --inf_scp $inf --output_dir $OUT/score/dnsmos --device $dev --primary_model $DNS/sig_bak_ovr.onnx --p808_model $DNS/model_v8.onnx"
run nisqa       "$UPY $EM/calculate_nonintrusive_nisqa.py --inf_scp $inf --output_dir $OUT/score/nisqa --device $dev"
run utmos       "$UPY $EM/calculate_nonintrusive_utmos.py --inf_scp $inf --output_dir $OUT/score/utmos --device $dev"
run scoreq      "PYTHONPATH=${ORTPATCH:-} $UENV $EM/calculate_nonintrusive_scoreq.py --inf_scp $inf --output_dir $OUT/score/scoreq"
run speechbert  "$UPY $EM/calculate_speechbert_score.py --ref_scp $refA --inf_scp $infA --output_dir $OUT/score/speechbert --device $dev"
run phoneme     "$UPY $EM/calculate_phoneme_similarity.py --ref_scp $refA --inf_scp $infA --output_dir $OUT/score/lps --device $dev"
run spk_sim     "$UPY $EM/calculate_speaker_similarity.py --ref_scp $refA --inf_scp $infA --output_dir $OUT/score/spk_sim --device $dev"
run lid         "$UPY $EM/calculate_lid_accuracy.py --meta_tsv $utt2lang --inf_scp $inf --output_dir $OUT/score/lid --device $dev"
run wer         "$UPY $EM/calculate_wer.py --meta_tsv $text --utt2lang $utt2lang --inf_scp $inf --output_dir $OUT/score/wer --device $dev"
echo "ALL METRICS DONE $(date)"
