#!/bin/bash

# Quick test of a single environment with minimal iterations
# Usage: ./test_single_quick.sh <env_name>

ENV=${1:-frozen_lake_4x4}
SEED=0
RUNNER="python examples/experiment_runner.py"

echo "========================================"
echo "QUICK TEST: $ENV"
echo "========================================"

echo ""
echo "Testing Value Iteration (quick)..."
$RUNNER single --env $ENV --algo value_iteration --seed $SEED --max_iter 100 --no-metrics

echo ""
echo "Testing SPMI (quick)..."
$RUNNER single --env $ENV --algo spmi --seed $SEED --max_iter 100 --no-metrics

echo ""
echo "Testing SA-PMI Linear (quick)..."
$RUNNER single --env $ENV --algo sa_pmi_linear --seed $SEED --max_iter 100 --no-metrics

echo ""
echo "✅ Quick test complete for $ENV"