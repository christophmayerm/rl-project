# Configurable Markov Decision Processes

This repository contains the code to run the experiments of the paper "Configurable Markov Decision Processes".

## Setup

This project uses `uv` for virtual environment and package management.

1.  **Create a virtual environment:**
    ```bash
    uv venv
    ```
    This will create a virtual environment in a `.venv` directory in the project root.

2.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

3.  **Sync the dependencies:**
    ```bash
    uv pip sync
    ```
    This will install the dependencies specified in `pyproject.toml` and `uv.lock`, ensuring a reproducible environment.

## Running the simulations

To run a simulation, execute the desired script from the `simulations` directory.
For example:

```bash
python simulations/racetrack2.py
```

This will run the `racetrack2` simulation with the default configuration. The output will be saved in the `data/racetrack2_T1` directory.

You can also run the other simulations:
```bash
python simulations/ractrack4.py
python simulations/student_teacher.py
```
