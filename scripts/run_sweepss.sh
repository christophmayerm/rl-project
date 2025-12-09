#!/bin/bash

# Complete experiment pipeline: baselines + sweeps
# Tests ALL algorithms (classical + adversarial)

set -e

RUNNER="python examples/experiment_runner.py"

echo "=================================="
echo "COMPLETE EXPERIMENT PIPELINE"
echo "=================================="
echo ""
echo "Phase 1: Baseline classical algorithms"
echo "Phase 2: SA-PMI hyperparameter sweeps"
echo "Phase 3: Full comparison"
echo ""

# ============================================
# PHASE 1: CLASSICAL BASELINES
# ============================================
echo "============================================"
echo "PHASE 1: Classical Baselines"
echo "============================================"
echo "Running VI, PI, Q-Learning on all environments..."
echo ""

CLASSICAL="value_iteration policy_iteration q_learning"

echo "1a: FrozenLake 4x4 Stochastic - Classical Baselines"
$RUNNER benchmark \
    --env frozen_lake_4x4 \
    --algos $CLASSICAL \
    --seeds 0 1 2 \
    --project "baselines-fl4x4-stochastic" \
    --no-metrics

echo ""
echo "1b: FrozenLake 4x4 Deterministic - Classical Baselines"
$RUNNER benchmark \
    --env frozen_lake_4x4_det \
    --algos $CLASSICAL \
    --seeds 0 1 2 \
    --project "baselines-fl4x4-deterministic" \
    --no-metrics

echo ""
echo "1c: FrozenLake 8x8 Stochastic - Classical Baselines"
$RUNNER benchmark \
    --env frozen_lake_8x8 \
    --algos $CLASSICAL \
    --seeds 0 1 2 \
    --project "baselines-fl8x8-stochastic" \
    --no-metrics

echo ""
echo "1d: FrozenLake 8x8 Deterministic - Classical Baselines"
$RUNNER benchmark \
    --env frozen_lake_8x8_det \
    --algos $CLASSICAL \
    --seeds 0 1 2 \
    --project "baselines-fl8x8-deterministic" \
    --no-metrics

echo ""
echo "1e: Taxi - Classical Baselines"
$RUNNER benchmark \
    --env taxi \
    --algos $CLASSICAL \
    --seeds 0 1 2 \
    --project "baselines-taxi" \
    --no-metrics

echo ""
echo "✅ Phase 1 complete! Classical baselines established."
echo ""

# ============================================
# PHASE 2: SA-PMI HYPERPARAMETER SWEEPS
# ============================================
echo "============================================"
echo "PHASE 2: SA-PMI Hyperparameter Sweeps"
echo "============================================"
echo ""

echo "2a: FrozenLake 4x4 Stochastic - SA-PMI Sweep"
$RUNNER sweep \
    --sweep fl4x4_stoch_sweep \
    --seeds 0 1 2 \
    --project "sweep-fl4x4-stochastic" \
    --no-metrics

echo ""
echo "2b: FrozenLake 4x4 Deterministic - SA-PMI Sweep"
$RUNNER sweep \
    --sweep fl4x4_det_sweep \
    --seeds 0 1 2 \
    --project "sweep-fl4x4-deterministic" \
    --no-metrics

echo ""
echo "2c: FrozenLake 8x8 Stochastic - SA-PMI Sweep"
$RUNNER sweep \
    --sweep fl8x8_stoch_sweep \
    --seeds 0 1 2 \
    --project "sweep-fl8x8-stochastic" \
    --no-metrics

echo ""
echo "2d: FrozenLake 8x8 Deterministic - SA-PMI Sweep"
$RUNNER sweep \
    --sweep fl8x8_det_sweep \
    --seeds 0 1 2 \
    --project "sweep-fl8x8-deterministic" \
    --no-metrics

echo ""
echo "2e: Taxi - SA-PMI Sweep"
$RUNNER sweep \
    --sweep taxi_sweep \
    --seeds 0 1 2 \
    --project "sweep-taxi" \
    --no-metrics

echo ""
echo "✅ Phase 2 complete! SA-PMI parameters optimized."
echo ""

# ============================================
# PHASE 3: FULL ALGORITHM COMPARISON
# ============================================
echo "============================================"
echo "PHASE 3: Full Algorithm Comparison"
echo "============================================"
echo "Comparing ALL algorithms on all environments..."
echo ""

ALL_ALGOS="value_iteration policy_iteration q_learning spmi sa_pmi_linear sa_pmi_exponential sa_pmi_adaptive"

echo "3a: FrozenLake 4x4 Stochastic - Full Comparison"
$RUNNER benchmark \
    --env frozen_lake_4x4 \
    --algos $ALL_ALGOS \
    --seeds 0 1 2 \
    --project "comparison-fl4x4-stochastic" \
    --no-metrics

echo ""
echo "3b: FrozenLake 4x4 Deterministic - Full Comparison"
$RUNNER benchmark \
    --env frozen_lake_4x4_det \
    --algos $ALL_ALGOS \
    --seeds 0 1 2 \
    --project "comparison-fl4x4-deterministic" \
    --no-metrics

echo ""
echo "3c: FrozenLake 8x8 Stochastic - Full Comparison"
$RUNNER benchmark \
    --env frozen_lake_8x8 \
    --algos $ALL_ALGOS \
    --seeds 0 1 2 \
    --project "comparison-fl8x8-stochastic" \
    --no-metrics

echo ""
echo "3d: FrozenLake 8x8 Deterministic - Full Comparison"
$RUNNER benchmark \
    --env frozen_lake_8x8_det \
    --algos $ALL_ALGOS \
    --seeds 0 1 2 \
    --project "comparison-fl8x8-deterministic" \
    --no-metrics

echo ""
echo "3e: Taxi - Full Comparison"
$RUNNER benchmark \
    --env taxi \
    --algos $ALL_ALGOS \
    --seeds 0 1 2 \
    --project "comparison-taxi" \
    --no-metrics

echo ""
echo "✅ Phase 3 complete!"
echo ""

echo "=================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=================================="
echo ""
echo "W&B Projects Created:"
echo "Phase 1 (Baselines):"
echo "  • baselines-fl4x4-stochastic"
echo "  • baselines-fl4x4-deterministic"
echo "  • baselines-fl8x8-stochastic"
echo "  • baselines-fl8x8-deterministic"
echo "  • baselines-taxi"
echo ""
echo "Phase 2 (SA-PMI Sweeps):"
echo "  • sweep-fl4x4-stochastic"
echo "  • sweep-fl4x4-deterministic"
echo "  • sweep-fl8x8-stochastic"
echo "  • sweep-fl8x8-deterministic"
echo "  • sweep-taxi"
echo ""
echo "Phase 3 (Full Comparison):"
echo "  • comparison-fl4x4-stochastic"
echo "  • comparison-fl4x4-deterministic"
echo "  • comparison-fl8x8-stochastic"
echo "  • comparison-fl8x8-deterministic"
echo "  • comparison-taxi"