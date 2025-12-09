#!/bin/bash

# Hyperparameter sweep pipeline
# Run this BEFORE benchmarks to find optimal parameters

set -e

RUNNER="python examples/experiment_runner.py"

echo "=================================="
echo "HYPERPARAMETER SWEEP PIPELINE"
echo "=================================="
echo ""
echo "Phase 1: Quick validation"
echo "Phase 2: Environment-specific sweeps"
echo "Phase 3: Schedule comparison"
echo ""

# ============================================
# PHASE 1: QUICK VALIDATION
# ============================================
echo "============================================"
echo "PHASE 1: Quick Validation"
echo "============================================"
echo "Testing all algorithms work on small env..."
echo ""

$RUNNER sweep \
    --sweep quick_validation \
    --seeds 0 \
    --project "sweeps-validation" \
    --no-metrics

echo ""
echo "✅ Phase 1 complete!"
echo ""

# ============================================
# PHASE 2: ENVIRONMENT-SPECIFIC SWEEPS
# ============================================
echo "============================================"
echo "PHASE 2: Environment-Specific Sweeps"
echo "============================================"
echo ""

echo "1/5: FrozenLake 4x4 Stochastic..."
$RUNNER sweep \
    --sweep fl4x4_stoch_sweep \
    --seeds 0 1 2 \
    --project "sweeps-fl4x4-stochastic" \
    --no-metrics

echo ""
echo "2/5: FrozenLake 4x4 Deterministic..."
$RUNNER sweep \
    --sweep fl4x4_det_sweep \
    --seeds 0 1 2 \
    --project "sweeps-fl4x4-deterministic" \
    --no-metrics

echo ""
echo "3/5: FrozenLake 8x8 Stochastic..."
$RUNNER sweep \
    --sweep fl8x8_stoch_sweep \
    --seeds 0 1 2 \
    --project "sweeps-fl8x8-stochastic" \
    --no-metrics

echo ""
echo "4/5: FrozenLake 8x8 Deterministic..."
$RUNNER sweep \
    --sweep fl8x8_det_sweep \
    --seeds 0 1 2 \
    --project "sweeps-fl8x8-deterministic" \
    --no-metrics

echo ""
echo "5/5: Taxi..."
$RUNNER sweep \
    --sweep taxi_sweep \
    --seeds 0 1 2 \
    --project "sweeps-taxi" \
    --no-metrics

echo ""
echo "✅ Phase 2 complete!"
echo ""

# ============================================
# PHASE 3: SCHEDULE COMPARISON
# ============================================
echo "============================================"
echo "PHASE 3: Schedule Comparison"
echo "============================================"
echo "Comparing curriculum schedules across all environments..."
echo ""

$RUNNER sweep \
    --sweep schedule_comparison \
    --seeds 0 1 2 \
    --project "sweeps-schedule-comparison" \
    --no-metrics

echo ""
echo "✅ Phase 3 complete!"
echo ""

echo "=================================="
echo "ALL SWEEPS COMPLETE!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Analyze sweep results in W&B dashboard"
echo "2. Update best_params_grid sweep with optimal parameters"
echo "3. Run final benchmarks with ./scripts/run_full_benchmark.sh"
echo ""
echo "W&B projects:"
echo "  • sweeps-validation"
echo "  • sweeps-fl4x4-stochastic"
echo "  • sweeps-fl4x4-deterministic"
echo "  • sweeps-fl8x8-stochastic"
echo "  • sweeps-fl8x8-deterministic"
echo "  • sweeps-taxi"
echo "  • sweeps-schedule-comparison"