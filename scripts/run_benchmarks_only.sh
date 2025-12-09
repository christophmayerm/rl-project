#!/bin/bash

# Simple benchmark: Classical vs SPMI variants
# Goal: Show classical algorithms outperform adversarial training

set -e

RUNNER="python examples/experiment_runner.py"

echo "=================================="
echo "BENCHMARK: CLASSICAL vs ADVERSARIAL"
echo "=================================="
echo ""

# All algorithms to test
CLASSICAL="value_iteration policy_iteration q_learning"
SPMI_VARIANTS="spmi sa_pmi_linear sa_pmi_exponential"
ALL_ALGOS="$CLASSICAL $SPMI_VARIANTS"

# Use 3 seeds for statistical significance
SEEDS="0 1 2"

echo "Testing 6 algorithms × 4 environments × 3 seeds = 72 runs"
echo ""

# ============================================
# RUN ALL BENCHMARKS
# ============================================

echo "1/4: FrozenLake 4x4 (Stochastic)"
$RUNNER benchmark \
    --env frozen_lake_4x4 \
    --algos $ALL_ALGOS \
    --seeds $SEEDS \
    --max_iter 1000 \
    --eps 0.001 \
    --project "benchmark-fl4x4-stoch" \
    --no-metrics

echo ""
echo "2/4: FrozenLake 4x4 (Deterministic)"
$RUNNER benchmark \
    --env frozen_lake_4x4_det \
    --algos $ALL_ALGOS \
    --seeds $SEEDS \
    --max_iter 500 \
    --eps 0.001 \
    --project "benchmark-fl4x4-det" \
    --no-metrics

echo ""
echo "3/4: FrozenLake 8x8 (Stochastic)"
$RUNNER benchmark \
    --env frozen_lake_8x8 \
    --algos $ALL_ALGOS \
    --seeds $SEEDS \
    --max_iter 2000 \
    --eps 0.001 \
    --project "benchmark-fl8x8-stoch" \
    --no-metrics

echo ""
echo "4/4: FrozenLake 8x8 (Deterministic)"
$RUNNER benchmark \
    --env frozen_lake_8x8_det \
    --algos $ALL_ALGOS \
    --seeds $SEEDS \
    --max_iter 1500 \
    --eps 0.001 \
    --project "benchmark-fl8x8-det" \
    --no-metrics

echo ""
echo "=================================="
echo "BENCHMARKS COMPLETE!"
echo "=================================="
echo ""
echo "Total runs: 72"
echo "Results saved to: results/benchmarks/"
echo ""
echo "Next: Run analysis script"
echo "  python scripts/analyze_benchmarks.py"