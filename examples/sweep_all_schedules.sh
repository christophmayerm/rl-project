#!/bin/bash
# sweep_all_schedules.sh
# Comprehensive hyperparameter sweeps for ALL curriculum schedules
# Uses environment-specific recommended settings automatically

set -e

echo "=========================================="
echo "SA-PMI COMPREHENSIVE SWEEP SUITE"
echo "=========================================="
echo ""

PROJECT_BASE="sa-pmi-sweeps"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/sweeps_${TIMESTAMP}"
RESULTS_DIR="results/sweeps_${TIMESTAMP}"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

echo "Sweep Configuration:"
echo "  Base Project: $PROJECT_BASE"
echo "  Using environment-specific settings:"
echo "    teacher_student_small: max_iter=10000, eps=1e-7, K_warmup=7500"
echo "    frozen_lake_4x4:       max_iter=15000, eps=1e-7, K_warmup=11250"
echo "    frozen_lake_8x8:       max_iter=20000, eps=1e-8, K_warmup=15000"
echo "    taxi:                  max_iter=20000, eps=1e-8, K_warmup=15000"
echo "  Timestamp: $TIMESTAMP"
echo "  Log Directory: $LOG_DIR"
echo ""

# Environments (focused set for paper)
ENVIRONMENTS=(
    "frozen_lake_8x8"
    "taxi"
)

SEEDS="0 1 2"

# ==========================================
# SWEEP 1: Curriculum Schedule Comparison
# Tests ALL 5 schedules
# ==========================================
echo "=========================================="
echo "SWEEP 1: Curriculum Schedule Comparison"
echo "=========================================="
echo "Testing: linear, exponential, sqrt, cosine, smooth_exponential"
echo "5 schedules × 4 envs × 3 seeds = 60 runs"
echo ""

for env in "${ENVIRONMENTS[@]}"; do
    echo "Running curriculum sweep on: $env"
    echo "  (Using environment-specific max_iter and K_warmup)"
    
    python examples/experiment_runner.py \
        --mode sweep \
        --sweep sa_pmi_curriculum_sweep \
        --environment "$env" \
        --seeds $SEEDS \
        --project "${PROJECT_BASE}-curriculum" \
        2>&1 | tee "$LOG_DIR/curriculum_sweep_${env}.log"
    
    echo "✓ Curriculum sweep completed for $env"
    echo ""
done

echo "✓ SWEEP 1 COMPLETED (60 runs)"
echo ""
sleep 2

# ==========================================
# SWEEP 2: Budget (B_max) Sweep
# Tests ALL 5 schedules with different B_max
# ==========================================
echo "=========================================="
echo "SWEEP 2: Adversarial Budget Sweep"
echo "=========================================="
echo "Testing B_max: 0.01, 0.03, 0.05, 0.07, 0.1"
echo "Testing ALL 5 schedules"
echo "5 B_max × 5 schedules × 4 envs × 3 seeds = 300 runs"
echo ""

for env in "${ENVIRONMENTS[@]}"; do
    echo "Running budget sweep on: $env"
    echo "  (Automatically tests all curriculum schedules)"
    
    python examples/experiment_runner.py \
        --mode sweep \
        --sweep sa_pmi_budget_sweep \
        --environment "$env" \
        --seeds $SEEDS \
        --project "${PROJECT_BASE}-budget" \
        2>&1 | tee "$LOG_DIR/budget_sweep_${env}.log"
    
    echo "✓ Budget sweep completed for $env"
    echo ""
done

echo "✓ SWEEP 2 COMPLETED (300 runs)"
echo ""
sleep 2

# ==========================================
# SWEEP 3: Warmup (K_warmup) Sweep
# Tests ALL 5 schedules with different warmup ratios
# ==========================================
echo "=========================================="
echo "SWEEP 3: Warmup Duration Sweep"
echo "=========================================="
echo "Testing K_warmup_ratio: 0.50, 0.625, 0.75, 0.875"
echo "  (Adapts to each environment's max_iter automatically)"
echo "Testing ALL 5 schedules"
echo "4 K_warmup_ratio × 5 schedules × 4 envs × 3 seeds = 240 runs"
echo ""

for env in "${ENVIRONMENTS[@]}"; do
    echo "Running warmup sweep on: $env"
    echo "  (K_warmup computed as ratio × environment's max_iter)"
    
    python examples/experiment_runner.py \
        --mode sweep \
        --sweep sa_pmi_warmup_sweep \
        --environment "$env" \
        --seeds $SEEDS \
        --project "${PROJECT_BASE}-warmup" \
        2>&1 | tee "$LOG_DIR/warmup_sweep_${env}.log"
    
    echo "✓ Warmup sweep completed for $env"
    echo ""
done

echo "✓ SWEEP 3 COMPLETED (240 runs)"
echo ""
sleep 2

# ==========================================
# SWEEP 4: Minimum Budget (B_min) Sweep
# Tests ALL 5 schedules with different B_min
# ==========================================
echo "=========================================="
echo "SWEEP 4: Minimum Budget Sweep"
echo "=========================================="
echo "Testing B_min: 0.0, 0.001, 0.005, 0.01"
echo "Testing ALL 5 schedules"
echo "4 B_min × 5 schedules × 4 envs × 3 seeds = 240 runs"
echo ""

for env in "${ENVIRONMENTS[@]}"; do
    echo "Running B_min sweep on: $env"
    echo "  (Automatically tests all curriculum schedules)"
    
    python examples/experiment_runner.py \
        --mode sweep \
        --sweep sa_pmi_bmin_sweep \
        --environment "$env" \
        --seeds $SEEDS \
        --project "${PROJECT_BASE}-bmin" \
        2>&1 | tee "$LOG_DIR/bmin_sweep_${env}.log"
    
    echo "✓ B_min sweep completed for $env"
    echo ""
done

echo "✓ SWEEP 4 COMPLETED (240 runs)"
echo ""

# ==========================================
# Summary and Analysis
# ==========================================
echo ""
echo "=========================================="
echo "ALL SWEEPS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Total Runs: 840"
echo "  - Sweep 1 (Curriculum):     60 runs"
echo "  - Sweep 2 (Budget):        300 runs"
echo "  - Sweep 3 (Warmup):        240 runs"
echo "  - Sweep 4 (B_min):         240 runs"
echo ""
echo "Results saved to:"
echo "  Logs:   $LOG_DIR"
echo "  Data:   $RESULTS_DIR"
echo "  JSON:   results/sweeps/"
echo ""
echo "W&B Projects (view online):"
echo "  - ${PROJECT_BASE}-curriculum"
echo "  - ${PROJECT_BASE}-budget"
echo "  - ${PROJECT_BASE}-warmup"
echo "  - ${PROJECT_BASE}-bmin"
echo ""
echo "Environment-Specific Iterations:"
echo "  teacher_student_small: 10,000 iterations per run"
echo "  frozen_lake_4x4:       15,000 iterations per run"
echo "  frozen_lake_8x8:       20,000 iterations per run"
echo "  taxi:                  20,000 iterations per run"
echo ""

# Calculate approximate runtime
SMALL_RUNS=$((1 * 60))  # teacher_student_small: 60 runs
MEDIUM_RUNS=$((1 * 300 + 1 * 240 + 1 * 240))  # frozen_lake_4x4: 780 runs
LARGE_RUNS=$((2 * 300 + 2 * 240 + 2 * 240))  # frozen_lake_8x8, taxi: 1560 runs total

echo "Estimated Runtime (assuming ~1 min per 10k iterations):"
echo "  Fast envs:    ~$((SMALL_RUNS * 1)) minutes"
echo "  Medium envs:  ~$((MEDIUM_RUNS * 2 / 60)) hours"
echo "  Large envs:   ~$((LARGE_RUNS * 2 / 60)) hours"
echo "  Total:        ~$((($SMALL_RUNS + $MEDIUM_RUNS * 2 + $LARGE_RUNS * 2) / 60)) hours"
echo ""
echo "Sweep completed at: $(date)"
echo ""

# Optional: Generate comparison plots
read -p "Generate curriculum comparison plots? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Generating plots..."
    python scripts/analyze_sweeps.py \
        --sweep_dirs "$RESULTS_DIR" \
        --output_dir "figures/sweeps_${TIMESTAMP}"
    echo "✓ Plots saved to: figures/sweeps_${TIMESTAMP}"
fi

echo ""
echo "================================"
echo "Next Steps for Analysis:"
echo "================================"
echo "1. View live results on W&B:"
echo "   https://wandb.ai/your-username/${PROJECT_BASE}-curriculum"
echo ""
echo "2. Analyze sweep results:"
echo "   python scripts/analyze_sweeps.py --sweep_dirs results/sweeps/"
echo ""
echo "3. Generate paper figures:"
echo "   python scripts/generate_paper_figures.py --results_dir results/"
echo ""
echo "4. Compare best schedules:"
echo "   python scripts/compare_schedules.py \\"
echo "     --curriculum results/sweeps/sweep_sa_pmi_curriculum_sweep_*.json \\"
echo "     --budget results/sweeps/sweep_sa_pmi_budget_sweep_*.json"
echo ""