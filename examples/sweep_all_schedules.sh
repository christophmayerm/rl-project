#!/bin/bash
# sweep_all_schedules.sh
# Comprehensive hyperparameter sweeps for all curriculum schedules

set -e  # Exit on error

echo "=========================================="
echo "SA-PMI COMPREHENSIVE SWEEP SUITE"
echo "=========================================="
echo ""

# Configuration
PROJECT_BASE="sa-pmi-sweeps"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/sweeps_${TIMESTAMP}"
RESULTS_DIR="results/sweeps_${TIMESTAMP}"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

echo "Sweep Configuration:"
echo "  Base Project: $PROJECT_BASE"
echo "  Timestamp: $TIMESTAMP"
echo "  Log Directory: $LOG_DIR"
echo "  Results Directory: $RESULTS_DIR"
echo ""

# Environments to test
ENVIRONMENTS=(
    "teacher_student_small"
    "frozen_lake_4x4"
    "frozen_lake_8x8"
    "taxi"
)

# Seeds for sweeps (fewer for speed)
SEEDS="0 1 2"

# ==========================================
# SWEEP 1: Curriculum Schedule Comparison
# ==========================================
echo "=========================================="
echo "SWEEP 1: Curriculum Schedule Comparison"
echo "=========================================="
echo "Testing: linear, exponential, sqrt, cosine, smooth_exponential"
echo "Environments: ${ENVIRONMENTS[@]}"
echo ""

for env in "${ENVIRONMENTS[@]}"; do
    echo "Running curriculum sweep on: $env"
    
    python examples/experiment_runner.py \
        --mode sweep \
        --sweep sa_pmi_curriculum_sweep_20000 \
        --environment "$env" \
        --seeds $SEEDS \
        --project "${PROJECT_BASE}-curriculum" \
        2>&1 | tee "$LOG_DIR/curriculum_sweep_${env}.log"
    
    echo "✓ Curriculum sweep completed for $env"
    echo ""
done

echo "✓ SWEEP 1 COMPLETED"
echo ""
sleep 2

# ==========================================
# SWEEP 2: Budget (B_max) Sweep - Cosine Schedule
# ==========================================
echo "=========================================="
echo "SWEEP 2: Adversarial Budget Sweep"
echo "=========================================="
echo "Testing B_max: 0.01, 0.03, 0.05, 0.07, 0.1"
echo "Using cosine schedule (best from sweep 1)"
echo ""

for env in "${ENVIRONMENTS[@]}"; do
    echo "Running budget sweep on: $env"
    
    python examples/experiment_runner.py \
        --mode sweep \
        --sweep sa_pmi_budget_sweep_20000 \
        --algorithm sa_pmi_cosine \
        --environment "$env" \
        --seeds $SEEDS \
        --project "${PROJECT_BASE}-budget" \
        2>&1 | tee "$LOG_DIR/budget_sweep_${env}.log"
    
    echo "✓ Budget sweep completed for $env"
    echo ""
done

echo "✓ SWEEP 2 COMPLETED"
echo ""
sleep 2

# ==========================================
# SWEEP 3: Warmup (K_warmup) Sweep - Cosine Schedule
# ==========================================
echo "=========================================="
echo "SWEEP 3: Warmup Duration Sweep"
echo "=========================================="
echo "Testing K_warmup: 4000, 5000, 6000, 7000"
echo "Using cosine schedule and B_max=0.05"
echo ""

for env in "${ENVIRONMENTS[@]}"; do
    echo "Running warmup sweep on: $env"
    
    python examples/experiment_runner.py \
        --mode sweep \
        --sweep sa_pmi_warmup_sweep_20000 \
        --algorithm sa_pmi_cosine \
        --environment "$env" \
        --seeds $SEEDS \
        --project "${PROJECT_BASE}-warmup" \
        2>&1 | tee "$LOG_DIR/warmup_sweep_${env}.log"
    
    echo "✓ Warmup sweep completed for $env"
    echo ""
done

echo "✓ SWEEP 3 COMPLETED"
echo ""
sleep 2

# ==========================================
# SWEEP 4: Minimum Budget (B_min) Sweep
# ==========================================
echo "=========================================="
echo "SWEEP 4: Minimum Budget Sweep"
echo "=========================================="
echo "Testing B_min: 0.0, 0.001, 0.005, 0.01"
echo "Using cosine schedule and B_max=0.05"
echo ""

for env in "${ENVIRONMENTS[@]}"; do
    echo "Running B_min sweep on: $env"
    
    python examples/experiment_runner.py \
        --mode sweep \
        --sweep sa_pmi_bmin_sweep_20000 \
        --algorithm sa_pmi_cosine \
        --environment "$env" \
        --seeds $SEEDS \
        --project "${PROJECT_BASE}-bmin" \
        2>&1 | tee "$LOG_DIR/bmin_sweep_${env}.log"
    
    echo "✓ B_min sweep completed for $env"
    echo ""
done

echo "✓ SWEEP 4 COMPLETED"
echo ""

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "ALL SWEEPS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  Logs: $LOG_DIR"
echo "  Data: $RESULTS_DIR"
echo ""
echo "W&B Projects:"
echo "  - ${PROJECT_BASE}-curriculum"
echo "  - ${PROJECT_BASE}-budget"
echo "  - ${PROJECT_BASE}-warmup"
echo "  - ${PROJECT_BASE}-bmin"
echo ""
echo "Sweep completed at: $(date)"