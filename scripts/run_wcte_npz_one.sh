#!/bin/bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 RUN INPUT_ROOT OUTPUT_DIR REPO_DIR EVENTS_PER_FILE" >&2
    exit 2
fi

run=$1
input_root=$2
output_dir=$3
repo_dir=$4
events_per_file=$5

if [[ "$input_root" != /eos/* || "$output_dir" != /eos/* || "$repo_dir" != /eos/* ]]; then
    echo "Input, output, and repository must be on /eos" >&2
    exit 2
fi
if [[ -f "$output_dir/conversion.done" ]]; then
    echo "Already complete: run $run"
    exit 0
fi
mkdir -p "$output_dir"
if compgen -G "$output_dir/*.npz" > /dev/null; then
    echo "Unmarked NPZ files already exist in $output_dir; refusing to overwrite" >&2
    exit 1
fi
stage="$output_dir/.conversion_in_progress"
if ! mkdir "$stage"; then
    echo "Conversion staging directory already exists: $stage" >&2
    exit 1
fi

set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_swan/x86_64-el9-gcc13-opt/setup.sh
set -u
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"
python3 -c 'import numpy, awkward, uproot; from analysis_tools import DataLoader, WCSimPMTMapping'

python3 "$repo_dir/scripts/export_wcte_npz.py" \
    "$input_root" \
    "$stage/WCTE_merged_production_R${run}.npz" \
    --events-per-file "$events_per_file"

shopt -s nullglob
parts=("$stage"/*.npz)
if (( ${#parts[@]} == 0 )); then
    echo "Conversion wrote no NPZ parts for run $run" >&2
    exit 1
fi
mv -- "${parts[@]}" "$output_dir/"
manifests=("$stage"/*_conversion_manifest.json)
if (( ${#manifests[@]} != 1 )); then
    echo "Expected exactly one conversion manifest in $stage" >&2
    exit 1
fi
mv -- "${manifests[@]}" "$output_dir/"
rmdir "$stage"
touch "$output_dir/conversion.done"
echo "Finished run $run: ${#parts[@]} NPZ part(s) in $output_dir"
