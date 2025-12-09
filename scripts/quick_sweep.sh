#!/bin/bash

# Quick sweep for testing - runs fast

set -e

RUNNER="python examples/experiment_runner.py"

echo "=================================="
echo "QUICK SWEEP (Testing Only)"
echo "=================================="
echo ""

# Just test on small environment with 1 seed
$RUNNER sweep \
    --sweep quick_validation \
    --seeds 0 \
    --project "quick-sweep-test" \
    --no-metrics

echo ""
echo "✅ Quick sweep complete!"
echo ""
echo "Check results in W&B project: quick-sweep-test"