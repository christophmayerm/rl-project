#!/bin/bash
# test_improved_schedules.sh
# Test script for improved SA-PMI curriculum schedules

set -e  # Exit on error

echo "=========================================="
echo "SA-PMI IMPROVED SCHEDULES TEST SUITE"
echo "=========================================="
echo ""

# Configuration
PROJECT_NAME="sa-pmi-schedule-test"
ENVIRONMENT="teacher_student_small"
SEED=42

# Create results directory
mkdir -p results/schedule_tests
mkdir -p logs

# Timestamp for this test run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/schedule_test_${TIMESTAMP}.log"

echo "Test Configuration:"
echo "  Project: $PROJECT_NAME"
echo "  Environment: $ENVIRONMENT"
echo "  Seed: $SEED"
echo "  Log file: $LOG_FILE"
echo ""

# Function to run a single test
run_test() {
    local algo=$1
    local algo_name=$2
    
    echo "=========================================="
    echo "Testing: $algo_name"
    echo "=========================================="
    
    python examples/experiment_runner.py \
        --mode single \
        --algorithm "$algo" \
        --environment "$ENVIRONMENT" \
        --seed $SEED \
        --project "$PROJECT_NAME" \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✓ $algo_name completed successfully"
    else
        echo "✗ $algo_name failed"
        exit 1
    fi
    echo ""
}

# Test 1: Baseline SPMI (no adversarial)
echo "TEST 1/7: Baseline SPMI"
run_test "spmi" "SPMI-Baseline"

# Test 2: Linear schedule (reference)
echo "TEST 2/7: Linear Schedule"
run_test "sa_pmi_linear" "SA-PMI-Linear"

# Test 3: Improved Exponential (quadratic)
echo "TEST 3/7: Improved Exponential Schedule"
run_test "sa_pmi_exponential_improved" "SA-PMI-Exponential-v2"

# Test 4: Square Root schedule
echo "TEST 4/7: Square Root Schedule"
run_test "sa_pmi_sqrt" "SA-PMI-Sqrt"

# Test 5: Cosine schedule
echo "TEST 5/7: Cosine Schedule"
run_test "sa_pmi_cosine" "SA-PMI-Cosine"

# Test 6: Smooth Exponential (sigmoid)
echo "TEST 6/7: Smooth Exponential Schedule"
run_test "sa_pmi_smooth" "SA-PMI-Smooth"

echo ""
echo "=========================================="
echo "ALL TESTS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Results logged to: $LOG_FILE"
echo "View results in W&B project: $PROJECT_NAME"
echo ""

# Optional: Run visualization script
echo "Generating curriculum visualization..."
python -c "
import sys
sys.path.append('.')
from algorithm.adversarial_model_chooser import AdversarialModelChooser
import numpy as np
import matplotlib.pyplot as plt

schedules = {
    'linear': 'Linear',
    'exponential': 'Quadratic (Improved Exp)',
    'sqrt': 'Square Root',
    'cosine': 'Cosine',
    'smooth_exponential': 'Smooth Sigmoid'
}

iterations = np.arange(0, 8000)
plt.figure(figsize=(14, 8))

for schedule_type, label in schedules.items():
    chooser = AdversarialModelChooser(
        nS=10, nA=4,
        budget_schedule=schedule_type,
        B_min=0.0, B_max=0.05, K_warmup=6000
    )
    budgets = [chooser.get_budget(k) for k in iterations]
    plt.plot(iterations, budgets, label=label, linewidth=2.5, alpha=0.8)

plt.axhline(y=0.05, color='red', linestyle='--', label='B_max = 0.05', alpha=0.6, linewidth=1.5)
plt.axvline(x=6000, color='gray', linestyle='--', label='K_warmup = 6000', alpha=0.6, linewidth=1.5)
plt.axvline(x=8000, color='black', linestyle=':', label='max_iter = 8000', alpha=0.4, linewidth=1.5)

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Adversarial Budget B(k)', fontsize=12)
plt.title('Curriculum Schedule Comparison - Improved Schedules', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, linestyle=':')
plt.tight_layout()
plt.savefig('results/schedule_tests/curriculum_comparison_${TIMESTAMP}.png', dpi=300, bbox_inches='tight')
print('✓ Visualization saved to: results/schedule_tests/curriculum_comparison_${TIMESTAMP}.png')
"

echo ""
echo "Test suite finished at: $(date)"