#!/bin/bash

set -e

ENV="frozen_lake_4x4_det"
SEED=0
RUNNER="python examples/experiment_runner.py"

echo "========================================"
echo "TEST: FrozenLake 4x4 (Deterministic)"
echo "========================================"

echo ""
echo "Testing Value Iteration..."
$RUNNER single --env $ENV --algo value_iteration --seed $SEED --max_iter 500 --no-metrics

echo ""
echo "Testing Policy Iteration..."
$RUNNER single --env $ENV --algo policy_iteration --seed $SEED --max_iter 500 --no-metrics

echo ""
echo "Testing Q-Learning..."
$RUNNER single --env $ENV --algo q_learning --seed $SEED --episodes 5000 --no-metrics

echo ""
echo "Testing SPMI (baseline)..."
$RUNNER single --env $ENV --algo spmi --seed $SEED --max_iter 500 --no-metrics

echo ""
echo "Testing SA-PMI Linear..."
$RUNNER single --env $ENV --algo sa_pmi_linear --seed $SEED --max_iter 500 --no-metrics

echo ""
echo "Testing SA-PMI Cosine..."
$RUNNER single --env $ENV --algo sa_pmi_cosine --seed $SEED --max_iter 500 --no-metrics

echo ""
echo "✅ FrozenLake 4x4 (Deterministic) - ALL TESTS PASSED"