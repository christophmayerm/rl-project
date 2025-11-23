#!/bin/bash

# ============================================================================
# SA-PMI Hyperparameter Sweeps
# Run this script to find the best hyperparameters for your algorithms
# ============================================================================

set -e  # Exit on any error

echo "=========================================================================="
echo "SA-PMI HYPERPARAMETER SWEEPS"
echo "=========================================================================="
echo "This script will run 3 sweeps to find optimal hyperparameters:"
echo "  1. Budget Sweep (B_max values)"
echo "  2. Warmup Sweep (K_warmup values)"
echo "  3. Curriculum Sweep (linear vs exponential, B_min values)"
echo ""
echo "Total estimated time: 2-4 hours (depending on your machine)"
echo "=========================================================================="
echo ""

# Ask for confirmation
read -p "Do you want to proceed? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Create results directory
mkdir -p results/sweeps
mkdir -p logs

# Timestamp for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/sweeps_${TIMESTAMP}.log"

echo "Logging to: ${LOG_FILE}"
echo ""

# ============================================================================
# SWEEP 1: Adversarial Budget (B_max)
# ============================================================================
echo "=========================================================================="
echo "SWEEP 1/3: Adversarial Budget (B_max)"
echo "=========================================================================="
echo "Testing B_max values: [0.01, 0.03, 0.05, 0.07, 0.1]"
echo "Seeds: [0, 1, 2]"
echo "Total runs: 15"
echo "Started at: $(date)"
echo ""

python examples/experiment_runner.py \
    --mode sweep \
    --sweep sa_pmi_budget_sweep \
    --algorithm sa_pmi_linear \
    --environment teacher_student_small \
    --seeds 0 1 2 \
    --project sa-pmi-sweep-budget \
    2>&1 | tee -a ${LOG_FILE}

if [ $? -eq 0 ]; then
    echo "✓ Budget sweep completed successfully!"
else
    echo "✗ Budget sweep failed! Check ${LOG_FILE}"
    exit 1
fi

echo ""
echo "Completed at: $(date)"
echo ""
sleep 2

# ============================================================================
# SWEEP 2: Warmup Schedule (K_warmup)
# ============================================================================
echo "=========================================================================="
echo "SWEEP 2/3: Warmup Schedule (K_warmup)"
echo "=========================================================================="
echo "Testing K_warmup values: [5000, 10000, 15000, 20000]"
echo "Seeds: [0, 1, 2]"
echo "Total runs: 12"
echo "Started at: $(date)"
echo ""

python examples/experiment_runner.py \
    --mode sweep \
    --sweep sa_pmi_warmup_sweep \
    --algorithm sa_pmi_linear \
    --environment teacher_student_small \
    --seeds 0 1 2 \
    --project sa-pmi-sweep-warmup \
    2>&1 | tee -a ${LOG_FILE}

if [ $? -eq 0 ]; then
    echo "✓ Warmup sweep completed successfully!"
else
    echo "✗ Warmup sweep failed! Check ${LOG_FILE}"
    exit 1
fi

echo ""
echo "Completed at: $(date)"
echo ""
sleep 2

# ============================================================================
# SWEEP 3: Curriculum Type and B_min
# ============================================================================
echo "=========================================================================="
echo "SWEEP 3/3: Curriculum Type and B_min"
echo "=========================================================================="
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
    --environment teacher_student_small \
    --seeds 0 1 2 \
    --project sa-pmi-sweep-curriculum \
    2>&1 | tee -a ${LOG_FILE}

if [ $? -eq 0 ]; then
    echo "✓ Curriculum sweep completed successfully!"
else
    echo "✗ Curriculum sweep failed! Check ${LOG_FILE}"
    exit 1
fi

echo ""
echo "Completed at: $(date)"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "=========================================================================="
echo "ALL SWEEPS COMPLETED!"
echo "=========================================================================="
echo ""
echo "Total runs completed: 45"
echo "Log file: ${LOG_FILE}"
echo ""
echo "Next steps:"
echo "  1. Go to your W&B dashboard:"
echo "     - https://wandb.ai/yourname/sa-pmi-sweep-budget"
echo "     - https://wandb.ai/yourname/sa-pmi-sweep-warmup"
echo "     - https://wandb.ai/yourname/sa-pmi-sweep-curriculum"
echo ""
echo "  2. Identify the best hyperparameters:"
echo "     - Which B_max gives highest performance?"
echo "     - Which K_warmup works best?"
echo "     - Linear or exponential curriculum?"
echo "     - Which B_min is optimal?"
echo ""
echo "  3. Update experiment_runner.py with best configs:"
echo "     - Edit the ALGORITHMS dictionary in ExperimentConfig class"
echo ""
echo "  4. Run the benchmark:"
echo "     chmod +x run_benchmark.sh"
echo "     ./run_benchmark.sh"
echo ""
echo "=========================================================================="
echo ""

# Create a template file for recording best configs
cat > best_configs_template.txt << EOF
========================================================================
BEST HYPERPARAMETERS FROM SWEEPS
Run date: $(date)
========================================================================

SWEEP RESULTS:

1. Budget Sweep (sa-pmi-sweep-budget):
   Best B_max: _____ (performance: _____)
   W&B link: https://wandb.ai/yourname/sa-pmi-sweep-budget

2. Warmup Sweep (sa-pmi-sweep-warmup):
   Best K_warmup: _____ (performance: _____)
   W&B link: https://wandb.ai/yourname/sa-pmi-sweep-warmup

3. Curriculum Sweep (sa-pmi-sweep-curriculum):
   Best curriculum: _____ (linear/exponential)
   Best B_min: _____ (performance: _____)
   W&B link: https://wandb.ai/yourname/sa-pmi-sweep-curriculum

========================================================================
FINAL CONFIGURATION FOR BENCHMARK:
========================================================================

sa_pmi_linear:
  curriculum_schedule: 'linear'
  B_min: _____
  B_max: _____
  K_warmup: _____

sa_pmi_exponential:
  curriculum_schedule: 'exponential'
  B_min: _____
  B_max: _____
  K_warmup: _____

========================================================================
NOTES:
(Add any observations or decisions here)


========================================================================
EOF

echo "✓ Created best_configs_template.txt"
echo "  Fill this out with your sweep results before running benchmark!"
echo ""