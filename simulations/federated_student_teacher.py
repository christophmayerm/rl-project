"""
Federated SPMI Simulation: Student-Teacher Domain

This script demonstrates the Federated Safe Policy-Model Iteration (F-SPMI)
algorithm on the Student-Teacher domain, comparing it with the standard SPMI.

The Student-Teacher domain is a simple model of concept learning where:
- A student (agent) learns to perform consistent assignments to literals
- An automatic teacher (environment) provides examples
- The goal is to jointly optimize the student's policy and the teacher's model
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
from algorithm.model_chooser import DoNotCreateTransitionsGreedyModelChooser
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy
from envs.student_teacher import TeacherStudentEnv

# Import F-SPMI components
from federated import FSPMI, FederatedAgent, FederatedServer


def run_standard_spmi(mdp, initial_policy, initial_model, original_model, max_iter=1000):
    """Run standard SPMI for comparison"""
    print("\n" + "="*60)
    print("Running Standard SPMI")
    print("="*60)

    mdp.set_model(copy.deepcopy(original_model))

    policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
    model_chooser = DoNotCreateTransitionsGreedyModelChooser(original_model, mdp.nS, mdp.nA)

    spmi = SPMI(mdp, eps=0.0, policy_chooser=policy_chooser,
                model_chooser=model_chooser, max_iter=max_iter, persistent=True)

    start_time = time.time()
    final_policy, final_model = spmi.spmi(
        copy.deepcopy(initial_policy),
        copy.deepcopy(initial_model)
    )
    elapsed = time.time() - start_time

    print(f"Standard SPMI completed in {elapsed:.2f} seconds")
    print(f"Final performance: {spmi.logger.J_history[-1] if spmi.logger.J_history else 'N/A'}")
    print(f"Iterations: {spmi.logger.iteration}")

    return spmi, final_policy, final_model


def run_federated_spmi(mdp, initial_policy, initial_model, original_model,
                       n_agents=4, episodes_per_round=50, max_rounds=100):
    """Run Federated SPMI"""
    print("\n" + "="*60)
    print(f"Running Federated SPMI (N={n_agents} agents, {episodes_per_round} episodes/round)")
    print("="*60)

    mdp.set_model(copy.deepcopy(original_model))

    policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
    model_chooser = DoNotCreateTransitionsGreedyModelChooser(original_model, mdp.nS, mdp.nA)

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
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)

    print("\nStandard SPMI:")
    if spmi_logger.J_history:
        print(f"  Final Performance: {spmi_logger.J_history[-1]:.4f}")
        print(f"  Iterations: {spmi_logger.iteration}")

    print("\nFederated SPMI:")
    if fspmi_logger.performances:
        print(f"  Final Performance (est.): {fspmi_logger.performances[-1]:.4f}")
        print(f"  Rounds: {len(fspmi_logger.iterations)}")
        print(f"  Total Samples: {sum(fspmi_logger.total_samples)}")


def main():
    """Main entry point"""
    print("="*60)
    print("Federated SPMI vs Standard SPMI: Student-Teacher Domain")
    print("="*60)

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

    # Create output directory
    dir_path = "./data/federated_student_teacher"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Run standard SPMI
    spmi, spmi_policy, spmi_model = run_standard_spmi(
        mdp, initial_policy, initial_model, original_model, max_iter=500
    )
    spmi.logger.save(dir_path, 'standard_spmi.csv')

    # Run Federated SPMI with different configurations
    configurations = [
        {'n_agents': 2, 'episodes_per_round': 25, 'max_rounds': 50},
        {'n_agents': 4, 'episodes_per_round': 25, 'max_rounds': 50},
        {'n_agents': 8, 'episodes_per_round': 25, 'max_rounds': 50},
    ]

    fspmi_results = []
    for config in configurations:
        fspmi, fspmi_policy, fspmi_model = run_federated_spmi(
            mdp, initial_policy, initial_model, original_model, **config
        )
        fspmi.logger.save(f"{dir_path}/fspmi_n{config['n_agents']}.csv")
        fspmi_results.append((config, fspmi))

    # Summary comparison
    print("\n" + "="*60)
    print("Final Summary")
    print("="*60)

    print(f"\nStandard SPMI:")
    print(f"  Iterations: {spmi.logger.iteration}")
    if spmi.logger.J_history:
        print(f"  Final Performance: {spmi.logger.J_history[-1]:.4f}")

    for config, fspmi in fspmi_results:
        print(f"\nF-SPMI (N={config['n_agents']}):")
        print(f"  Rounds: {len(fspmi.logger.iterations)}")
        print(f"  Total Samples: {sum(fspmi.logger.total_samples)}")
        if fspmi.logger.performances:
            print(f"  Final Performance: {fspmi.logger.performances[-1]:.4f}")

    print("\nResults saved to:", dir_path)


if __name__ == '__main__':
    main()
