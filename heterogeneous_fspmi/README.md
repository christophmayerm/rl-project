# Experiments Heterogeneous FSPMI

## H-FSMI Experiment for Dynamics, Obstacles, Robustness (5 Seeds)
```python
python simulations/run_heterogeneous_experiments.py --experiment all --n_seeds 5 --max_obstacles 3 --step_boost 1 --config config/hetero_config.yaml

# If someone doesnt care about safety constraints, you can increase step_boost to 50 for example then you have quicker learning but without theoretical safety bounds.

# If you want to isolate experiment settings there is the option to choose from the experiment flag:

# --experiment dynamics
# --experiment robustness
# --experiment obstacles
# for obstacles you also need the flag --max_obstacles [int]
```