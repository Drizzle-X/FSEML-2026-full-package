#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
: "${CORE50_DATA_ROOT:?Set CORE50_DATA_ROOT to the extracted core50_128x128 image directory}"

runs=(0 1 2)
seeds=(9 19 29)

for index in 0 1 2; do
  run="${runs[$index]}"
  seed="${seeds[$index]}"
  protocol="core50_protocols/nicv2_391_run${run}.json"
  "$PYTHON_BIN" prepare_core50_protocol.py \
    --scenario nicv2_391 --run "$run" --output "$protocol"

  MPLCONFIGDIR=/tmp/matplotlib-core50 "$PYTHON_BIN" run_core50_experiment.py \
    --data-root "$CORE50_DATA_ROOT" --scenario nicv2_391 --run "$run" --seed "$seed" \
    --initial-epochs 5 --replay --replay-content decoder --replay-partitions 50 --output-dir core50_results

  MPLCONFIGDIR=/tmp/matplotlib-core50 "$PYTHON_BIN" run_core50_experiment.py \
    --data-root "$CORE50_DATA_ROOT" --scenario nicv2_391 --run "$run" --seed "$seed" \
    --initial-epochs 5 --no-replay --output-dir core50_results
done
