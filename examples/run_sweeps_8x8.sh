#!/bin/bash

# ============================================================================
# Complete SA-PMI Hyperparameter Sweep Suite
# Tests all hyperparameters for both linear and exponential curricula
# ============================================================================

set -e  # Exit on any error

echo "=========================================================================="
echo "COMPLETE SA-PMI HYPERPARAMETER SWEEP SUITE"
echo "=========================================================================="
echo "This script runs ALL sweeps for BOTH SA-PMI variants:"
echo "  - Budget sweep (B_max values)"
echo "  - Warmup sweep (K_warmup values)"
echo "  - Curriculum sweep (linear vs exponential, B_min values)"
echo ""
echo "For algorithms: SA-PMI Linear, SA-PMI Exponential"
echo "Environment: FrozenLake-8x8 (fast standard benchmark)"
echo "Seeds: 0, 1, 2"
echo ""
echo "Total sweeps: 6 (3 sweeps × 2 algorithms)"
echo "Total runs: ~90 experiments"
echo "Estimated time: 2-4 hours"
echo "=========================================================================="
echo ""

read -p "Do you want to proceed? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Create directories
mkdir -p results/sweeps
mkdir -p logs

# Timestamp for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/all_sweeps_${TIMESTAMP}.log"

echo "Logging to: ${LOG_FILE}"
echo ""

# ============================================================================
# SA-PMI LINEAR SWEEPS
# ============================================================================

echo "=========================================================================="
echo "PART 1: SA-PMI LINEAR SWEEPS"
echo "=========================================================================="
echo ""

# ============================================================================
# SWEEP 1: Linear - Budget Sweep
# ============================================================================
echo "----------------------------------------------------------------------"
echo "SWEEP 1/6: SA-PMI Linear - Budget Sweep (B_max)"
echo "----------------------------------------------------------------------"
echo "Testing B_max: [0.01, 0.03, 0.05, 0.07, 0.1]"
echo "Seeds: [0, 1, 2]"
echo "Total runs: 15"
echo "Started at: $(date)"
echo ""

python examples/experiment_runner.py \
    --mode sweep \
    --sweep sa_pmi_budget_sweep \
    --algorithm sa_pmi_linear \
    --environment frozen_lake_8x8 \
    --seeds 0 1 2 \
    --project sa-pmi-sweep-linear-budget_8x8 \
    2>&1 | tee -a ${LOG_FILE}

if [ $? -eq 0 ]; then
    echo "✓ Linear budget sweep completed!"
else
    echo "✗ Linear budget sweep failed! Check ${LOG_FILE}"
    exit 1
fi

echo ""
sleep 2

# ============================================================================
# SWEEP 2: Linear - Warmup Sweep
# ============================================================================
echo "----------------------------------------------------------------------"
echo "SWEEP 2/6: SA-PMI Linear - Warmup Sweep (K_warmup)"
echo "----------------------------------------------------------------------"
echo "Testing K_warmup: [7000, 7250, 7500, 8000]"
echo "Seeds: [0, 1, 2]"
echo "Total runs: 12"
echo "Started at: $(date)"
echo ""

python examples/experiment_runner.py \
    --mode sweep \
    --sweep sa_pmi_warmup_sweep \
    --algorithm sa_pmi_linear \
    --environment frozen_lake_8x8 \
    --seeds 0 1 2 \
    --project sa-pmi-sweep-linear-warmup_8x8 \
    2>&1 | tee -a ${LOG_FILE}

if [ $? -eq 0 ]; then
    echo "✓ Linear warmup sweep completed!"
else
    echo "✗ Linear warmup sweep failed! Check ${LOG_FILE}"
    exit 1
fi

echo ""
sleep 2

# ============================================================================
# SWEEP 3: Linear - Curriculum Sweep
# ============================================================================
echo "----------------------------------------------------------------------"
echo "SWEEP 3/6: SA-PMI Linear - Curriculum Sweep (B_min)"
echo "----------------------------------------------------------------------"
echo "Testing curriculum: [linear, exponential]"
echo "Testing B_min: [0.0, 0.001, 0.005]"
echo "Seeds: [0, 1, 2]"
echo "Total runs: 18"
echo "Started at: $(date)"
echo ""

python examples/experiment_runner.py \
    --mode sweep \
    --sweep sa_pmi_curriculum_sweep \
    --algorithm sa_pmi_linear \
    --environment frozen_lake_8x8 \
    --seeds 0 1 2 \
    --project sa-pmi-sweep-linear-curriculum_8x8 \
    2>&1 | tee -a ${LOG_FILE}

if [ $? -eq 0 ]; then
    echo "✓ Linear curriculum sweep completed!"
else
    echo "✗ Linear curriculum sweep failed! Check ${LOG_FILE}"
    exit 1
fi

echo ""
sleep 2

# ============================================================================
# SA-PMI EXPONENTIAL SWEEPS
# ============================================================================

echo "=========================================================================="
echo "PART 2: SA-PMI EXPONENTIAL SWEEPS"
echo "=========================================================================="
echo ""

# ============================================================================
# SWEEP 4: Exponential - Budget Sweep
# ============================================================================
echo "----------------------------------------------------------------------"
echo "SWEEP 4/6: SA-PMI Exponential - Budget Sweep (B_max)"
echo "----------------------------------------------------------------------"
echo "Testing B_max: [0.01, 0.03, 0.05, 0.07, 0.1]"
echo "Seeds: [0, 1, 2]"
echo "Total runs: 15"
echo "Started at: $(date)"
echo ""

python examples/experiment_runner.py \
    --mode sweep \
    --sweep sa_pmi_budget_sweep \
    --algorithm sa_pmi_exponential \
    --environment frozen_lake_8x8 \
    --seeds 0 1 2 \
    --project sa-pmi-sweep-exp-budget_8x8 \
    2>&1 | tee -a ${LOG_FILE}

if [ $? -eq 0 ]; then
    echo "✓ Exponential budget sweep completed!"
else
    echo "✗ Exponential budget sweep failed! Check ${LOG_FILE}"
    exit 1
fi

echo ""
sleep 2

# ============================================================================
# SWEEP 5: Exponential - Warmup Sweep
# ============================================================================
echo "----------------------------------------------------------------------"
echo "SWEEP 5/6: SA-PMI Exponential - Warmup Sweep (K_warmup)"
echo "----------------------------------------------------------------------"
echo "Testing K_warmup: [7000, 7250, 7500, 8000]"
echo "Seeds: [0, 1, 2]"
echo "Total runs: 12"
echo "Started at: $(date)"
echo ""

python examples/experiment_runner.py \
    --mode sweep \
    --sweep sa_pmi_warmup_sweep \
    --algorithm sa_pmi_exponential \
    --environment frozen_lake_8x8 \
    --seeds 0 1 2 \
    --project sa-pmi-sweep-exp-warmup_8x8 \
    2>&1 | tee -a ${LOG_FILE}

if [ $? -eq 0 ]; then
    echo "✓ Exponential warmup sweep completed!"
else
    echo "✗ Exponential warmup sweep failed! Check ${LOG_FILE}"
    exit 1
fi

echo ""
sleep 2

# ============================================================================
# SWEEP 6: Exponential - Curriculum Sweep
# ============================================================================
echo "----------------------------------------------------------------------"
echo "SWEEP 6/6: SA-PMI Exponential - Curriculum Sweep (B_min)"
echo "----------------------------------------------------------------------"
echo "Testing curriculum: [linear, exponential]"
echo "Testing B_min: [0.0, 0.001, 0.005]"
echo "Seeds: [0, 1, 2]"
echo "Total runs: 18"
echo "Started at: $(date)"
echo ""

python examples/experiment_runner.py \
    --mode sweep \
    --sweep sa_pmi_curriculum_sweep \
    --algorithm sa_pmi_exponential \
    --environment frozen_lake_8x8 \
    --seeds 0 1 2 \
    --project sa-pmi-sweep-exp-curriculum_8x8 \
    2>&1 | tee -a ${LOG_FILE}

if [ $? -eq 0 ]; then
    echo "✓ Exponential curriculum sweep completed!"
else
    echo "✗ Exponential curriculum sweep failed! Check ${LOG_FILE}"
    exit 1
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "=========================================================================="
echo "ALL SWEEPS COMPLETED!"
echo "=========================================================================="
echo ""
echo "Total sweeps completed: 6"
echo "Total runs completed: ~90"
echo "Log file: ${LOG_FILE}"
echo ""
echo "=========================================================================="
echo "NEXT STEPS"
echo "=========================================================================="
echo ""
echo "1. Analyze results in W&B:"
echo "   Linear sweeps:"
echo "     - https://wandb.ai/yourname/sa-pmi-sweep-linear-budget"
echo "     - https://wandb.ai/yourname/sa-pmi-sweep-linear-warmup"
echo "     - https://wandb.ai/yourname/sa-pmi-sweep-linear-curriculum"
echo ""
echo "   Exponential sweeps:"
echo "     - https://wandb.ai/yourname/sa-pmi-sweep-exp-budget"
echo "     - https://wandb.ai/yourname/sa-pmi-sweep-exp-warmup"
echo "     - https://wandb.ai/yourname/sa-pmi-sweep-exp-curriculum"
echo ""
echo "2. For each algorithm (linear and exponential), identify:"
echo "   - Best B_max (from budget sweep)"
echo "   - Best K_warmup (from warmup sweep)"
echo "   - Best B_min (from curriculum sweep)"
echo ""
echo "3. Update experiment_runner.py with best hyperparameters:"
echo "   Edit the ALGORITHMS dictionary in ExperimentConfig class"
echo ""
echo "4. Run final benchmark:"
echo "   ./run_benchmark.sh"
echo ""
echo "=========================================================================="
echo ""

# Create summary template
cat > sweep_results_template.txt << EOF
========================================================================
SWEEP RESULTS SUMMARY
Completed: $(date)
========================================================================

LINEAR ALGORITHM:
-----------------
Budget Sweep (sa-pmi-sweep-linear-budget):
  Best B_max: _____ (performance: _____)
  W&B: https://wandb.ai/yourname/sa-pmi-sweep-linear-budget

Warmup Sweep (sa-pmi-sweep-linear-warmup):
  Best K_warmup: _____ (performance: _____)
  W&B: https://wandb.ai/yourname/sa-pmi-sweep-linear-warmup

Curriculum Sweep (sa-pmi-sweep-linear-curriculum):
  Best B_min: _____ (performance: _____)
  W&B: https://wandb.ai/yourname/sa-pmi-sweep-linear-curriculum

EXPONENTIAL ALGORITHM:
---------------------
Budget Sweep (sa-pmi-sweep-exp-budget):
  Best B_max: _____ (performance: _____)
  W&B: https://wandb.ai/yourname/sa-pmi-sweep-exp-budget

Warmup Sweep (sa-pmi-sweep-exp-warmup):
  Best K_warmup: _____ (performance: _____)
  W&B: https://wandb.ai/yourname/sa-pmi-sweep-exp-warmup

Curriculum Sweep (sa-pmi-sweep-exp-curriculum):
  Best B_min: _____ (performance: _____)
  W&B: https://wandb.ai/yourname/sa-pmi-sweep-exp-curriculum

========================================================================
RECOMMENDED CONFIGURATION FOR BENCHMARK
========================================================================

'sa_pmi_linear': {
    'B_min': _____,
    'B_max': _____,
    'K_warmup': _____,
}

'sa_pmi_exponential': {
    'B_min': _____,
    'B_max': _____,
    'K_warmup': _____,
}

========================================================================
EOF

echo "✓ Created sweep_results_template.txt"
echo "  Fill this out with your sweep results!"
echo ""