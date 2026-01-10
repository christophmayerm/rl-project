
# Experiment Set


## F-SPMI vs SPMI with Greedy as Model chooser with Racetrack T1 (5 seeds)

```bash
# F-SPMI vs SPMI with Greedy as Model Chooser (5 seeds)
python simulations/run_experiments.py --config config/config_report.yaml --experiment standard_spmi federated_spmi --track T1 --n_seeds 5 --max_iter 1000 --max_rounds 1000 --model_chooser greedy
```

## F-SPMI vs SPMI with Gaussian Processes as Model chooser with Racetrack T1 (5 seeds)

```bash
# F-SPMI vs SPMI with GP as Model Chooser (5 seeds)
python simulations/run_experiments.py --config config/config_report.yaml --experiment standard_spmi federated_spmi --track T1 --n_seeds 5 --max_iter 1000 --max_rounds 1000
--model_chooser gp
```

## F-SPMI vs SPMI with Greedy as Model chooser with Racetrack T4 (5 seeds)

```bash
# F-SPMI vs SPMI with Greedy as Model Chooser (5 seeds)
python simulations/run_experiments.py --config config/config_report.yaml --experiment standard_spmi federated_spmi --track T4 --n_seeds 5 --max_iter 1000 --max_rounds 1000 --model_chooser greedy
```

## F-SPMI vs SPMI with Gaussian Processes as Model chooser with Racetrack T4 (5 seeds)

```bash
# F-SPMI vs SPMI with GP as Model Chooser (5 seeds)
python simulations/run_experiments.py --config config/config_report.yaml --experiment standard_spmi federated_spmi --track T4 --n_seeds 5 --max_iter 1000 --max_rounds 1000
--model_chooser gp
```


