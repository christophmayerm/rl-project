#!/bin/bash

# Quick benchmark for testing (uses fewer seeds and smaller environments)

set -e

RUNNER="python examples/experiment_runner.py"
SEEDS="0 1"

echo "=================================="
echo "QUICK BENCHMARK (2 seeds only)"
echo "=================================="

# Test on small stochastic environment
echo ""
echo "FrozenLake 4x4 (Stochastic)"
$RUNNER benchmark \
    --env frozen_lake_4x4 \
    --algos value_iteration policy_iteration q_learning spmi sa_pmi_linear \
    --seeds $SEEDS \
    --no-metrics

# Test on small deterministic environment
echo ""
echo "FrozenLake 4x4 (Deterministic)"
$RUNNER benchmark \
    --env frozen_lake_4x4_det \
    --algos value_iteration policy_iteration q_learning spmi sa_pmi_linear \
    --seeds $SEEDS \
    --no-metrics

echo ""
echo "Quick benchmark complete!"