#!/bin/bash

# ============================================================================
# SA-PMI Full Benchmark
# Run this AFTER sweeps and updating configs with best hyperparameters
# ============================================================================

set -e  # Exit on any error

echo "=========================================================================="
echo "SA-PMI FULL BENCHMARK"
echo "=========================================================================="
echo "This script will run the final benchmark with optimized hyperparameters."
echo ""
echo "⚠️  BEFORE RUNNING THIS:"
echo "  1. Make sure you've run ./run_sweeps.sh"
echo "  2. Analyzed results in W&B"
echo "  3. Updated experiment_runner.py with best hyperparameters"
echo ""
echo "Benchmark configuration:"
echo "  - Algorithms: SPMI, SA-PMI Linear, SA-PMI Exponential"
echo "  - Environments: Small, Medium, Large (3 total)"
echo "  - Seeds: 0, 1, 2, 3, 4 (5 total)"
echo "  - Total runs: 3 × 3 × 5 = 45"
echo ""
echo "Estimated time: 4-8 hours (depending on your machine)"
echo "=========================================================================="
echo ""

# Check if best_configs_template.txt exists
if [ ! -f "best_configs_template.txt" ]; then
    echo "⚠️  Warning: best_configs_template.txt not found."
    echo "   Did you run ./run_sweeps.sh first?"
    echo ""
fi

# Ask for confirmation
read -p "Have you updated experiment_runner.py with best hyperparameters? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Please update experiment_runner.py first, then run this script again."
    exit 1
fi

# Create directories
mkdir -p results/benchmarks
mkdir -p logs
mkdir -p figures

# Timestamp for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/benchmark_${TIMESTAMP}.log"

echo "Logging to: ${LOG_FILE}"
echo ""

# ============================================================================
# OPTION: Small Test Benchmark First (Optional)
# ============================================================================
echo "=========================================================================="
echo "OPTIONAL: Test Benchmark (Recommended)"
echo "=========================================================================="
echo "Would you like to run a small test benchmark first?"
echo "  - Only 2 seeds instead of 5"
echo "  - Only small environment"
echo "  - Total: 6 runs (takes ~30 min)"
echo ""
read -p "Run test benchmark first? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "Running test benchmark..."
    echo "Started at: $(date)"
    echo ""
    
    python examples/experiment_runner.py \
        --mode benchmark \
        --algorithms spmi sa_pmi_linear sa_pmi_exponential \
        --environments teacher_student_small \
        --seeds 0 1 \
        --project sa-pmi-benchmark-test \
        2>&1 | tee -a ${LOG_FILE}
    
    if [ $? -eq 0 ]; then
        echo "✓ Test benchmark completed successfully!"
        echo ""
        echo "Results look good? Check W&B: https://wandb.ai/yourname/sa-pmi-benchmark-test"
        echo ""
        read -p "Proceed with full benchmark? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]
        then
            echo "Stopping here. Review test results and run again when ready."
            exit 0
        fi
    else
        echo "✗ Test benchmark failed! Check ${LOG_FILE}"
        echo "Fix issues before running full benchmark."
        exit 1
    fi
fi

# ============================================================================
# FULL BENCHMARK
# ============================================================================
echo ""
echo "=========================================================================="
echo "FULL BENCHMARK - STARTING"
echo "=========================================================================="
echo "Configuration:"
echo "  - Algorithms: spmi, sa_pmi_linear, sa_pmi_exponential"
echo "  - Environments: teacher_student_small, teacher_student_medium, teacher_student_large"
echo "  - Seeds: 0, 1, 2, 3, 4"
echo "  - Total runs: 45"
echo ""
echo "Started at: $(date)"
echo ""
echo "⏰ This will take several hours. You can:"
echo "  - Monitor progress in W&B: https://wandb.ai/yourname/sa-pmi-final-benchmark"
echo "  - Check log file: ${LOG_FILE}"
echo "  - Let it run overnight"
echo ""

# Final confirmation
read -p "Start full benchmark now? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

python examples/experiment_runner.py \
    --mode benchmark \
    --algorithms spmi sa_pmi_linear sa_pmi_exponential \
    --environments teacher_student_small teacher_student_medium teacher_student_large \
    --seeds 0 1 2 3 4 \
    --project sa-pmi-final-benchmark \
    2>&1 | tee -a ${LOG_FILE}

if [ $? -ne 0 ]; then
    echo "✗ Benchmark failed! Check ${LOG_FILE}"
    exit 1
fi

echo ""
echo "Completed at: $(date)"
echo ""

# ============================================================================
# GENERATE FIGURES
# ============================================================================
echo "=========================================================================="
echo "GENERATING FIGURES"
echo "=========================================================================="
echo ""

if [ -f "analyze_results.py" ]; then
    python analyze_results.py 2>&1 | tee -a ${LOG_FILE}
    
    if [ $? -eq 0 ]; then
        echo "✓ Figures generated successfully!"
    else
        echo "⚠️  Figure generation had issues. Check ${LOG_FILE}"
    fi
else
    echo "⚠️  analyze_results.py not found. Skipping figure generation."
    echo "   Run manually: python analyze_results.py"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=========================================================================="
echo "BENCHMARK COMPLETED!"
echo "=========================================================================="
echo ""
echo "Results saved to:"
echo "  - JSON summary: results/benchmarks/"
echo "  - Figures: figures/"
echo "  - Log file: ${LOG_FILE}"
echo ""
echo "Next steps:"
echo "  1. Review results in W&B:"
echo "     https://wandb.ai/yourname/sa-pmi-final-benchmark"
echo ""
echo "  2. Check generated figures in figures/ directory:"
echo "     - algorithm_comparison.pdf"
echo "     - learning_curves.pdf"
echo "     - adversarial_budget.pdf"
echo "     - results_table.tex (for paper)"
echo "     - summary_report.md"
echo ""
echo "  3. Use these results for your thesis/paper!"
echo ""
echo "=========================================================================="
echo ""

# Create completion marker
echo "Benchmark completed at: $(date)" > benchmark_complete_${TIMESTAMP}.txt
echo "W&B project: sa-pmi-final-benchmark" >> benchmark_complete_${TIMESTAMP}.txt
echo "Log file: ${LOG_FILE}" >> benchmark_complete_${TIMESTAMP}.txt

echo "✓ Created benchmark_complete_${TIMESTAMP}.txt"
echo ""