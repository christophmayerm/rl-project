#!/bin/bash

# Full benchmark with optimal parameters from sweeps
# Run this AFTER sweeps are complete

set -e

echo "=================================="
echo "FULL BENCHMARK SUITE"
echo "=================================="
echo "5 environments × 7 algorithms × 3 seeds = 105 runs"
echo ""
echo "⚠️  Make sure you've run sweeps first to find optimal parameters!"
echo ""

RUNNER="python examples/experiment_runner.py"
CLASSICAL="value_iteration policy_iteration q_learning"
SPMI_VARIANTS="spmi sa_pmi_linear sa_pmi_exponential sa_pmi_adaptive"
ALL_ALGOS="$CLASSICAL $SPMI_VARIANTS"
SEEDS="0 1 2"

# 1. FrozenLake 4x4 - Stochastic
echo "1/5: FrozenLake 4x4 (Stochastic)"
$RUNNER benchmark \
    --env frozen_lake_4x4 \
    --algos $ALL_ALGOS \
    --seeds $SEEDS \
    --project "benchmark-fl4x4-stochastic" \
    --no-metrics

# 2. FrozenLake 4x4 - Deterministic
echo ""
echo "2/5: FrozenLake 4x4 (Deterministic)"
$RUNNER benchmark \
    --env frozen_lake_4x4_det \
    --algos $ALL_ALGOS \
    --seeds $SEEDS \
    --project "benchmark-fl4x4-deterministic" \
    --no-metrics

# 3. FrozenLake 8x8 - Stochastic
echo ""
echo "3/5: FrozenLake 8x8 (Stochastic)"
$RUNNER benchmark \
    --env frozen_lake_8x8 \
    --algos $ALL_ALGOS \
    --seeds $SEEDS \
    --project "benchmark-fl8x8-stochastic" \
    --no-metrics

# 4. FrozenLake 8x8 - Deterministic
echo ""
echo "4/5: FrozenLake 8x8 (Deterministic)"
$RUNNER benchmark \
    --env frozen_lake_8x8_det \
    --algos $ALL_ALGOS \
    --seeds $SEEDS \
    --project "benchmark-fl8x8-deterministic" \
    --no-metrics

# 5. Taxi
echo ""
echo "5/5: Taxi"
$RUNNER benchmark \
    --env taxi \
    --algos $ALL_ALGOS \
    --seeds $SEEDS \
    --project "benchmark-taxi" \
    --no-metrics

echo ""
echo "=================================="
echo "BENCHMARK COMPLETE!"
echo "=================================="
echo ""
echo "Results saved to: results/benchmarks/"
echo ""
echo "W&B projects:"
echo "  • benchmark-fl4x4-stochastic"
echo "  • benchmark-fl4x4-deterministic"
echo "  • benchmark-fl8x8-stochastic"
echo "  • benchmark-fl8x8-deterministic"
echo "  • benchmark-taxi"
echo ""
echo "Analyze results with:"
echo "  python scripts/analyze_benchmarks.py"