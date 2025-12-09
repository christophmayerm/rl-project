#!/bin/bash

set -e

ENV="taxi"
SEED=0
RUNNER="python examples/experiment_runner.py"

echo "========================================"
echo "TEST: Taxi"
echo "========================================"

echo ""
echo "⚠️  WARNING: Taxi has 500 states - tests may be slower"
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
echo "✅ Taxi - ALL TESTS PASSED"