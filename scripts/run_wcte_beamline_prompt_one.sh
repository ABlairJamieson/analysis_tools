#!/bin/bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 RUN INPUT_ROOT OUTPUT_DIR REPO_DIR MAX_EVENTS" >&2
    exit 2
fi

run=$1
input_root=$2
output_dir=$3
repo_dir=$4
max_events=$5

for path in "$input_root" "$output_dir" "$repo_dir"; do
    if [[ "$path" != /eos/* ]]; then
        echo "Input, output, and repository must be on /eos: $path" >&2
        exit 2
    fi
done
if [[ ! -f "$input_root" ]]; then
    echo "Missing input ROOT file: $input_root" >&2
    exit 1
fi
if [[ -f "$output_dir/summary.json" || -f "$output_dir/.beamline_prompt.done" ]]; then
    echo "Already complete: run $run"
    exit 0
fi
mkdir -p "$output_dir"
if [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 ! -name '.beamline_prompt_in_progress' -print -quit)" ]]; then
    echo "Output directory has existing incomplete files; refusing to overwrite: $output_dir" >&2
    exit 1
fi
if ! mkdir "$output_dir/.beamline_prompt_in_progress" 2>/dev/null; then
    echo "Study already running or previously interrupted: $output_dir" >&2
    exit 1
fi

set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_swan/x86_64-el9-gcc13-opt/setup.sh
set -u
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"
python3 -c 'import awkward, matplotlib, numpy, uproot'
python3 "$repo_dir/scripts/study_wcte_beamline_prompt.py" \
    "$input_root" \
    --output-dir "$output_dir" \
    --max-events "$max_events"

touch "$output_dir/.beamline_prompt.done"
rmdir "$output_dir/.beamline_prompt_in_progress"
echo "Finished beamline prompt study for run $run: $output_dir"
