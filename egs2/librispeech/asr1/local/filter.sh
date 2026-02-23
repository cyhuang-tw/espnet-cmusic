#!/usr/bin/env bash
# Usage: bash filter_maestro.sh <input_dir> <output_dir>

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    exit 1
fi

input_dir=$1
output_dir=$2

mkdir -p "$output_dir"

for f in wav.scp text utt2spk feats.scp utt2dur utt2num_frames cmvn.scp; do
    src="$input_dir/$f"
    [ -f "$src" ] || continue
    grep -v 'OFFSETS' "$src" > "$output_dir/$f"
    echo "Filtered $f: $(wc -l < "$output_dir/$f") utterances"
done

# spk2utt is derived from utt2spk
if [ -f "$output_dir/utt2spk" ]; then
    utils/utt2spk_to_spk2utt.pl "$output_dir/utt2spk" > "$output_dir/spk2utt"
    echo "Regenerated spk2utt"
fi