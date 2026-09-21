#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo "Usage: $0 IMAGE [IMAGE ...]" >&2
  exit 2
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs_sweep_${RUN_ID}}"
MAX_JOBS="${MAX_JOBS:-1}"
SEEDS="${SEEDS:-0 1 2}"
THRESHOLDS="${THRESHOLDS:-0.000 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.010 0.011 0.012}"
mkdir -p "$LOG_DIR"

for image in "$@"; do
  if [[ ! -f "$image" ]]; then
    echo "Image not found: $image" >&2
    exit 2
  fi
  name="$(basename "${image%.*}")"
  for threshold in $THRESHOLDS; do
    for seed in $SEEDS; do
      log="$LOG_DIR/${name}_std_${threshold}_seed_${seed}.log"
      echo "launch image=$image threshold=$threshold seed=$seed log=$log"
      python run.py "$image" \
        --std-threshold "$threshold" \
        --seed "$seed" >"$log" 2>&1 &
      while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        wait -n
      done
    done
  done
done

wait
echo "Sweep complete: $LOG_DIR"

