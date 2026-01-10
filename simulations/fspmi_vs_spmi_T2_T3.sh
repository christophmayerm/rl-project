#!/usr/bin/env bash

set -e  # Exit on error

CONFIG="config/config_report.yaml"
MAX_ITER=1000
MAX_ROUNDS=1000
N_SEEDS=5

TRACKS=("T2" "T3")
MODEL_CHOOSERS=("greedy" "gp")

echo "Starting Experiment Set: F-SPMI vs SPMI"

for TRACK in "${TRACKS[@]}"; do
  for CHOOSER in "${MODEL_CHOOSERS[@]}"; do

    echo "Running TRACK=${TRACK}, MODEL_CHOOSER=${CHOOSER}"

    CMD=(
      python simulations/run_experiments.py
      --config "$CONFIG"
      --experiment standard_spmi federated_spmi
      --track "$TRACK"
      --n_seeds "$N_SEEDS"
      --max_iter "$MAX_ITER"
      --max_rounds "$MAX_ROUNDS"
      --model_chooser "$CHOOSER"
    )

    "${CMD[@]}"

  done
done

echo "All experiments completed successfully."
