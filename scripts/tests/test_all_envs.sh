#!/bin/bash

# Master test script - runs all environment tests
# Tests each environment with all algorithms (1 seed only)

set -e  # Exit on error

echo "=================================="
echo "ENVIRONMENT COMPATIBILITY TESTS"
echo "=================================="
echo "Testing all environments with all algorithms"
echo "Using seed=0 only for quick validation"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run each environment test
echo "1. Testing FrozenLake 4x4 (Stochastic)..."
bash "$SCRIPT_DIR/test_frozen_lake_4x4.sh"

echo ""
echo "2. Testing FrozenLake 4x4 (Deterministic)..."
bash "$SCRIPT_DIR/test_frozen_lake_4x4_det.sh"

echo ""
echo "3. Testing FrozenLake 8x8 (Stochastic)..."
bash "$SCRIPT_DIR/test_frozen_lake_8x8.sh"

echo ""
echo "4. Testing FrozenLake 8x8 (Deterministic)..."
bash "$SCRIPT_DIR/test_frozen_lake_8x8_det.sh"

echo ""
echo "5. Testing Taxi..."
bash "$SCRIPT_DIR/test_taxi.sh"

echo ""
echo "6. Testing CartPole..."
bash "$SCRIPT_DIR/test_cartpole.sh"

echo ""
echo "7. Testing Teacher-Student..."
bash "$SCRIPT_DIR/test_teacher_student.sh"

echo ""
echo "=================================="
echo "ALL TESTS COMPLETE!"
echo "=================================="
echo ""
echo "If all tests passed, you can proceed with:"
echo "  1. Hyperparameter sweeps: ./scripts/run_sweeps.sh"
echo "  2. Full benchmarks: ./scripts/run_full_benchmark.sh"