#!/bin/bash

set -e

ENV="teacher_student_small"
SEED=0
RUNNER="python examples/experiment_runner.py"

echo "========================================"
echo "TEST: Teacher-Student Small"
echo "========================================"

echo ""
echo "Testing Value Iteration..."
$RUNNER single --env $ENV --algo value_iteration --seed $SEED --max_iter 2000 --no-metrics

echo ""
echo "Testing Policy Iteration..."
$RUNNER single --env $ENV --algo policy_iteration --seed $SEED --max_iter 2000 --no-metrics

echo ""
echo "Testing Q-Learning..."
$RUNNER single --env $ENV --algo q_learning --seed $SEED --episodes 15000 --no-metrics

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
echo "✅ Teacher-Student Small - ALL TESTS PASSED"