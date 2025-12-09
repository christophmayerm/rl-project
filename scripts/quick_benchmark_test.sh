#!/bin/bash

# Medium-sized benchmark test for validating analysis pipeline
# Tests 2 environments × 5 algorithms × 2 seeds = 20 runs (~30-45 min)

set -e

RUNNER="python examples/experiment_runner.py"

echo "=================================="
echo "QUICK BENCHMARK TEST"
echo "=================================="
echo "Purpose: Test full analysis pipeline"
echo "Runs: 2 envs × 5 algos × 2 seeds = 20 runs"
echo "Time: ~30-45 minutes"
echo ""

# Select representative algorithms
CLASSICAL="value_iteration policy_iteration q_learning"
SPMI_VARIANTS="spmi sa_pmi_exponential"
ALL_ALGOS="$CLASSICAL $SPMI_VARIANTS"

# Use 2 seeds for variability
SEEDS="0 1"

# Test on small and medium environment
echo "1/2: FrozenLake 4x4 (Stochastic) - Fast convergence"
$RUNNER benchmark \
    --env frozen_lake_4x4 \
    --algos $ALL_ALGOS \
    --seeds $SEEDS \
    --max_iter 1000 \
    --project "test-benchmark-fl4x4" \
    --no-metrics

echo ""
echo "2/2: FrozenLake 8x8 (Deterministic) - Shows clear differences"
$RUNNER benchmark \
    --env frozen_lake_8x8_det \
    --algos $ALL_ALGOS \
    --seeds $SEEDS \
    --max_iter 3000 \
    --project "test-benchmark-fl8x8-det" \
    --no-metrics

echo "2/2: FrozenLake 8x8 (Stochastic) - Shows clear differences"
$RUNNER benchmark \
    --env frozen_lake_8x8 \
    --algos $ALL_ALGOS \
    --seeds $SEEDS \
    --max_iter 3000 \
    --project "test-benchmark-fl8x8-det" \
    --no-metrics
echo ""
echo "=================================="
echo "BENCHMARK TEST COMPLETE!"
echo "=================================="
echo ""
echo "✅ Generated 20 benchmark runs"
echo ""
echo "Next: Test analysis pipeline"
echo "  python scripts/analyze_benchmarks.py"
echo ""
echo "If plots look good, run full benchmark:"
echo "  ./scripts/run_benchmarks_only.sh"