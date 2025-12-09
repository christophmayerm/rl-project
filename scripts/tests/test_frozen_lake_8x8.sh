#!/bin/bash

set -e

ENV="frozen_lake_8x8"
SEED=0
RUNNER="python examples/experiment_runner.py"

echo "========================================"
echo "TEST: FrozenLake 8x8 (Stochastic)"
echo "========================================"

echo ""
echo "Testing Value Iteration..."
$RUNNER single --env $ENV --algo value_iteration --seed $SEED --max_iter 2000 --no-metrics

echo ""
echo "Testing Policy Iteration..."
$RUNNER single --env $ENV --algo policy_iteration --seed $SEED --max_iter 2000 --no-metrics

echo ""
echo "Testing Q-Learning..."
$RUNNER single --env $ENV --algo q_learning --seed $SEED --episodes 20000 --no-metrics

echo ""
echo "Testing SPMI (baseline)..."
$RUNNER single --env $ENV --algo spmi --seed $SEED --max_iter 2000 --no-metrics

echo ""
echo "Testing SA-PMI Linear..."
$RUNNER single --env $ENV --algo sa_pmi_linear --seed $SEED --max_iter 2000 --no-metrics

echo ""
echo "Testing SA-PMI Adaptive..."
$RUNNER single --env $ENV --algo sa_pmi_adaptive --seed $SEED --max_iter 2000 --no-metrics

echo ""
echo "✅ FrozenLake 8x8 (Stochastic) - ALL TESTS PASSED"