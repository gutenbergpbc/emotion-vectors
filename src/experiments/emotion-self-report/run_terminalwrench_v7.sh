#!/usr/bin/env bash
set -u

project=/work/code/model-forensics
python_bin="$project/experiments/canonical/.venv/bin/python"
runner="$project/experiments/canonical/run.py"
config_dir="$project/experiments/canonical/configs"
log_dir=/scratch/terminalwrench-self-report-v7-20261026
mkdir -p "$log_dir"

run_one() {
  local stem="$1"
  local config="$config_dir/$stem.json"
  local run_id
  run_id="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$config")"
  local receipt="$project/experiment-results/emotion-self-report/$run_id/verification.json"
  local attempt
  for attempt in 1 2 3; do
    if [ -f "$receipt" ]; then
      return 0
    fi
    echo "$(date -u +%FT%TZ) $stem attempt $attempt" >> "$log_dir/$stem.log"
    "$python_bin" "$runner" --config "$config" --resume >> "$log_dir/$stem.log" 2>&1
    if [ -f "$receipt" ]; then
      return 0
    fi
    sleep 30
  done
  echo "$(date -u +%FT%TZ) $stem stopped after three attempts" >> "$log_dir/$stem.log"
  return 1
}

run_one "$1"
