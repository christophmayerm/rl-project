
# Experiment Set

List of commands to run the experiments from the report with.

## Automatic Bash Runs

### Experiments of Track 1 and 4 (MAIN EXPERIMENTS) 

```bash
bash simulations/fspmi_vs_spmi_T1_T4.sh
```

### Experiments of Track 2 and 3 
```bash
bash simulations/fspmi_vs_spmi_T2_T3.sh
```

## Manual Runs

### F-SPMI vs SPMI with Greedy as Model chooser with Racetrack T1 (5 seeds)

```bash
# F-SPMI vs SPMI with Greedy as Model Chooser (5 seeds)
python simulations/run_experiments2.py --config config/config_report.yaml --experiment standard_spmi federated_spmi --track T1 --n_seeds 5 --max_iter 1000 --max_rounds 1000 --model_chooser greedy
```

### F-SPMI vs SPMI with Gaussian Processes as Model chooser with Racetrack T1 (5 seeds)

```bash
# F-SPMI vs SPMI with GP as Model Chooser (5 seeds)
python simulations/run_experiments2.py --config config/config_report.yaml --experiment standard_spmi federated_spmi --track T1 --n_seeds 5 --max_iter 1000 --max_rounds 1000 --model_chooser gp
```

### F-SPMI vs SPMI with Greedy as Model chooser with Racetrack T4 (5 seeds)

```bash
# F-SPMI vs SPMI with Greedy as Model Chooser (5 seeds)
python simulations/run_experiments2.py --config config/config_report.yaml --experiment standard_spmi federated_spmi --track T4 --n_seeds 5 --max_iter 1000 --max_rounds 1000 --model_chooser greedy
```

### F-SPMI vs SPMI with Gaussian Processes as Model chooser with Racetrack T4 (5 seeds)

```bash
# F-SPMI vs SPMI with GP as Model Chooser (5 seeds)
python simulations/run_experiments2.py --config config/config_report.yaml --experiment standard_spmi federated_spmi --track T4 --n_seeds 5 --max_iter 1000 --max_rounds 1000 --model_chooser gp

python simulations/plot_comparison.py --greedy_dir data/experiments/racetrack_T1/greedy/20260110-205927 --gp_dir data/experiments/racetrack_T1/gp/20260111-045142 --faceted
```

## Visualization Plots


### Plots for F-SPMI vs SPMI (Model Chooser Greedy)

N_Agents : 2, 4, 8

Eps/Round: 100, 200, 400, 800

Seeds: 5

```bash
python simulations/plot_results.py --results_dir data/experiments/racetrack_T1/greedy/20260110-205927
```

### Summary of FSPMI vs SPMI (iters=1000)

![Summary_Exp1](../data/experiments/racetrack_T1/greedy/20260110-205927/figures/fig_summary.png)

### Exact vs MC 

![ExactVSMC](../data/experiments/racetrack_T1/greedy/20260110-205927/figures/fig2_mc_vs_exact.png)


