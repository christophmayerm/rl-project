#!/bin/bash

set -e

ENV="cartpole"
SEED=0
RUNNER="python examples/experiment_runner.py"

echo "========================================"
echo "TEST: CartPole (Discretized)"
echo "========================================"

echo ""
echo "⚠️  WARNING: CartPole is discretized from continuous space"
echo "⚠️  Tests may take longer due to state space size"
echo ""

echo "Testing Value Iteration..."
$RUNNER single --env $ENV --algo value_iteration --seed $SEED --max_iter 3000 --no-metrics

echo ""
echo "Testing Policy Iteration..."
$RUNNER single --env $ENV --algo policy_iteration --seed $SEED --max_iter 3000 --no-metrics

echo ""
echo "Testing Q-Learning..."
$RUNNER single --env $ENV --algo q_learning --seed $SEED --episodes 25000 --no-metrics

echo ""
echo "Testing SPMI (baseline)..."
$RUNNER single --env $ENV --algo spmi --seed $SEED --max_iter 3000 --no-metrics

echo ""
echo "Testing SA-PMI Exponential..."
$RUNNER single --env $ENV --algo sa_pmi_exponential --seed $SEED --max_iter 3000 --no-metrics

echo ""
echo "✅ CartPole - ALL TESTS PASSED"