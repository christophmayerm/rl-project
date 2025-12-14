"""
Federated SPMI Simulation: Student-Teacher Domain (GP Model Selection)

This script runs both standard SPMI and Federated SPMI on the
Student-Teacher environment using the Gaussian Process model chooser
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
import time

# Import existing SPMI components
from algorithm.spmi import SPMI
from algorithm.policy_chooser import GreedyPolicyChooser
from algorithm.model_chooser import GPModelChooser
from utils.tabular_factory import model_from_matrix
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy
from envs.student_teacher import TeacherStudentEnv

# Import F-SPMI components
from federated import FSPMI

def build_gp_model_set(original_model, nS, nA, noise_levels=(0.0, 0.05, 0.1, 0.2)):
    """Create a small set of smoothed models for GP exploration.

    The base model is progressively blended with a uniform transition matrix
    to generate different vertices for the GP simplex. The first element is
    always the unmodified original model.
    """
    base_matrix = TabularModel(original_model, nS, nA).get_matrix()
    uniform_matrix = np.full_like(base_matrix, 1.0 / nS)

    model_set = []
    for noise in noise_levels:
        blended = (1.0 - noise) * base_matrix + noise * uniform_matrix
        model_set.append(model_from_matrix(blended, original_model, nS=nS, nA=nA))

    init_vector = np.zeros(len(model_set))
    init_vector[0] = 1.0
    return model_set, init_vector.tolist()

def run_standard_spmi(
    mdp,
    initial_policy,
    initial_model,
    original_model,
    model_set,
    init_model_vector,
    gp_beta=1.0,
    gp_gp_update_frequency=10,
    max_iter=1000
):
    """Run standard SPMI with GP model chooser and metric logging."""
    print("\n" + "=" * 60)
    print("Running Standard SPMI (GP model chooser)")
    print("=" * 60)

    mdp.set_model(copy.deepcopy(original_model))
    mdp.model_vector = np.array(init_model_vector)

    policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
    model_chooser = GPModelChooser(model_set, mdp.nS, mdp.nA, init_model_vector, gp_beta, original_model)
    model_chooser.gp_update_frequency = gp_gp_update_frequency

    spmi = SPMI(
        mdp,
        eps=0.0,
        policy_chooser=policy_chooser,
        model_chooser=model_chooser,
        max_iter=max_iter,
        persistent=True,
    )

    start_time = time.time()
    final_policy, final_model = spmi.spmi(
        copy.deepcopy(initial_policy),
        copy.deepcopy(initial_model)
    )
    elapsed = time.time() - start_time

    print(f"Standard SPMI completed in {elapsed:.2f} seconds")
    print(f"Iterations: {spmi.logger.iteration}")

    return spmi, final_policy, final_model

def run_federated_spmi(
    mdp,
    initial_policy,
    initial_model,
    original_model,
    model_set,
    init_model_vector,
    n_agents=4,
    episodes_per_round=100,
    max_rounds=200,
    gp_beta=1.0,
    gp_gp_update_frequency=10
):
    """Run Federated SPMI with GP model chooser and metric logging."""
    print("\n" + "=" * 60)
    print(f"Running Federated SPMI (GP) (N={n_agents} agents, {episodes_per_round} episodes/round)")
    print("=" * 60)

    mdp.set_model(copy.deepcopy(original_model))
    mdp.model_vector = np.array(init_model_vector)

    policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
    model_chooser = GPModelChooser(model_set, mdp.nS, mdp.nA, init_model_vector, gp_beta, original_model)
    model_chooser.gp_update_frequency = gp_gp_update_frequency

    fspmi = FSPMI(
        conf_mdp=mdp,
        n_agents=n_agents,
        episodes_per_round=episodes_per_round,
        eps=0.0,
        max_rounds=max_rounds,
        policy_chooser=policy_chooser,
        model_chooser=model_chooser,
        aggregation_method='weighted',
        persistent=True,
        verbose=True
    )

    start_time = time.time()
    final_policy, final_model = fspmi.run(
        copy.deepcopy(initial_policy),
        copy.deepcopy(initial_model)
    )
    elapsed = time.time() - start_time

    print(f"Federated SPMI completed in {elapsed:.2f} seconds")
    print(f"Final average return: {fspmi.logger.avg_returns[-1] if fspmi.logger.avg_returns else 'N/A'}")
    print(f"Rounds: {len(fspmi.logger.iterations)}")
    print(f"Total samples collected: {sum(fspmi.logger.total_samples)}")

    return fspmi, final_policy, final_model

def compare_results(spmi_logger, fspmi_logger):
    """Compare results from standard SPMI and F-SPMI"""
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    print("\nStandard SPMI:")
    if spmi_logger.evaluations:
        print(f"  Final Performance: {spmi_logger.evaluations[-1]:.4f}")
        print(f"  Iterations: {spmi_logger.iteration}")

    print("\nFederated SPMI:")
    if fspmi_logger.true_performances:
        print(f"  Final Performance (true): {fspmi_logger.true_performances[-1]:.4f}")
    if fspmi_logger.performances:
        print(f"  Final Performance (MC est.): {fspmi_logger.performances[-1]:.4f}")
    print(f"  Rounds: {len(fspmi_logger.iterations)}")
    print(f"  Total Samples: {sum(fspmi_logger.total_samples)}")

def main():
    """Main entry point"""
    print("=" * 60)
    print("Federated SPMI vs Standard SPMI (GP): Student-Teacher Domain")
    print("=" * 60)

    # Create environment
    print("\nInitializing Student-Teacher environment...")
    mdp = TeacherStudentEnv(
        n_literals=2,
        max_value=1,
        max_update=1,
        max_literals_in_examples=2,
        horizon=10
    )

    print(f"State space size: {mdp.nS}")
    print(f"Action space size: {mdp.nA}")
    print(f"Discount factor: {mdp.gamma}")
    print(f"Horizon: {mdp.horizon}")

    # Initialize policy and model
    uniform_policy = UniformPolicy(mdp)
    original_model = copy.deepcopy(mdp.P)

    initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
    initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)

    # GP model set and parameters
    gp_noise_levels = (0.0, 0.05, 0.1, 0.2)
    model_set, init_model_vector = build_gp_model_set(original_model, mdp.nS, mdp.nA, gp_noise_levels)
    gp_beta = 1.0
    gp_gp_update_frequency = 10

    # Create output directory
    dir_path = "./data/federated_student_teacher_gp"
    os.makedirs(dir_path, exist_ok=True)

    # Run standard SPMI
    spmi, _, _ = run_standard_spmi(
        mdp,
        initial_policy,
        initial_model,
        original_model,
        model_set,
        init_model_vector,
        gp_beta=gp_beta,
        gp_gp_update_frequency=gp_gp_update_frequency,
        max_iter=500
    )
    spmi.logger.save(dir_path, 'standard_spmi.csv')
    spmi.model_chooser.save_gp_times(f"{dir_path}/standard_spmi")

    # Run Federated SPMI with different configurations
    configurations = [
        {'n_agents': 2, 'episodes_per_round': 100, 'max_rounds': 200},
        {'n_agents': 4, 'episodes_per_round': 100, 'max_rounds': 200},
        {'n_agents': 8, 'episodes_per_round': 100, 'max_rounds': 200},
    ]

    fspmi_results = []
    for config in configurations:
        fspmi, _, _ = run_federated_spmi(
            mdp,
            initial_policy,
            initial_model,
            original_model,
            model_set,
            init_model_vector,
            gp_beta=gp_beta,
            gp_gp_update_frequency=gp_gp_update_frequency,
            **config
        )
        log_path = f"{dir_path}/fspmi_n{config['n_agents']}.csv"
        fspmi.logger.save(log_path)
        fspmi.model_chooser.save_gp_times(log_path.replace('.csv', ''))
        fspmi_results.append((config, fspmi))

    # Summary comparison
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)

    print(f"\nStandard SPMI:")
    print(f"  Iterations: {spmi.logger.iteration}")
    if spmi.logger:
        print(f"  Final Performance: {spmi.logger.evaluations[-1]:.4f}")

    for config, fspmi in fspmi_results:
        print(f"\nF-SPMI (N={config['n_agents']}):")
        print(f"  Rounds: {len(fspmi.logger.iterations)}")
        print(f"  Total Samples: {sum(fspmi.logger.total_samples)}")
        if fspmi.logger.true_performances:
            print(f"  Final Performance (true): {fspmi.logger.true_performances[-1]:.4f}")
        if fspmi.logger.performances:
            print(f"  Final Performance (MC est.): {fspmi.logger.performances[-1]:.4f}")

    print("\nResults saved to:", dir_path)

if __name__ == '__main__':
    main()
