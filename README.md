# Configurable Markov Decision Processes

This repository contains the code to run the experiments of the paper "Configurable Markov Decision Processes".

## Setup

This project uses `uv` for virtual environment and package management.
Make sure you have installd the `uv` package
On iOS, simply run `brew install uv`. 

1.  **Set up the virtual environment:**

    Execute the following command in the root directory of the project.
    ```bash
    uv sync
    ```
    This will create a virtual environment in a `.venv` directory in the project root.

2.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

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
