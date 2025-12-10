"""
Federated SPMI Simulation: Student-Teacher Domain
with SA-PMI Curriculum Learning for Policy Robustness

This script demonstrates:
1. Standard SPMI (baseline)
2. SA-PMI with curriculum learning for policy robustness
3. Federated SPMI (baseline)
4. Federated SA-PMI with curriculum learning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
import time

# Import existing SPMI components
from algorithm.spmi import SPMI
from algorithm.policy_chooser import GreedyPolicyChooser, SAPMIPolicyChooser
from algorithm.model_chooser import DoNotCreateTransitionsGreedyModelChooser
from algorithm.curriculum_scheduler import (
    LinearCurriculumScheduler,
    ExponentialCurriculumScheduler,
    CosineCurriculumScheduler,
    ConstantCurriculumScheduler
)
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy
from envs.student_teacher import TeacherStudentEnv

# Import F-SPMI components
from federated import FSPMI


def run_standard_spmi(mdp, initial_policy, initial_model, original_model, max_iter=500):
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
    print(f"Final performance: {spmi.logger.evaluations[-1]:.4f}")
    print(f"Iterations: {spmi.logger.iteration}")

    return spmi, final_policy, final_model


def run_sapmi_policy_robust(
    mdp,
    initial_policy,
    initial_model,
    original_model,
    curriculum_name,
    curriculum_scheduler,
    robustness_temperature=0.3,
    entropy_bonus=0.1,
    max_iter=500
):
    """Run SA-PMI with curriculum learning for POLICY robustness."""
    print("\n" + "=" * 60)
    print(f"Running SA-PMI Policy-Robust ({curriculum_name} curriculum)")
    print("=" * 60)

    mdp.set_model(copy.deepcopy(original_model))

    # Create base greedy policy chooser
    base_policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
    
    # Wrap with SA-PMI policy chooser
    policy_chooser = SAPMIPolicyChooser(
        nS=mdp.nS,
        nA=mdp.nA,
        base_chooser=base_policy_chooser,
        curriculum_scheduler=curriculum_scheduler,
        robustness_temperature=robustness_temperature,
        entropy_bonus=entropy_bonus
    )
    
    # Use standard greedy model chooser
    model_chooser = DoNotCreateTransitionsGreedyModelChooser(original_model, mdp.nS, mdp.nA)

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

    print(f"SA-PMI Policy-Robust ({curriculum_name}) completed in {elapsed:.2f} seconds")
    print(f"Iterations: {spmi.logger.iteration}")
    if spmi.logger.evaluations:
        print(f"Final Performance: {spmi.logger.evaluations[-1]:.4f}")

    return spmi, final_policy, final_model


def run_federated_spmi(mdp, initial_policy, initial_model, original_model,
                       n_agents=4, episodes_per_round=100, max_rounds=200):
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


def run_federated_sapmi(
    mdp,
    initial_policy,
    initial_model,
    original_model,
    curriculum_name,
    curriculum_scheduler,
    robustness_temperature=0.3,
    entropy_bonus=0.1,
    n_agents=4,
    episodes_per_round=100,
    max_rounds=200
):
    """Run Federated SA-PMI with policy robustness"""
    print("\n" + "="*60)
    print(f"Running Federated SA-PMI ({curriculum_name}) (N={n_agents} agents, {episodes_per_round} episodes/round)")
    print("="*60)

    mdp.set_model(copy.deepcopy(original_model))

    # Create base greedy policy chooser
    base_policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
    
    # Wrap with SA-PMI policy chooser
    policy_chooser = SAPMIPolicyChooser(
        nS=mdp.nS,
        nA=mdp.nA,
        base_chooser=base_policy_chooser,
        curriculum_scheduler=curriculum_scheduler,
        robustness_temperature=robustness_temperature,
        entropy_bonus=entropy_bonus
    )
    
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

    print(f"Federated SA-PMI ({curriculum_name}) completed in {elapsed:.2f} seconds")
    print(f"Final average return: {fspmi.logger.avg_returns[-1] if fspmi.logger.avg_returns else 'N/A'}")
    print(f"Rounds: {len(fspmi.logger.iterations)}")
    print(f"Total samples collected: {sum(fspmi.logger.total_samples)}")

    return fspmi, final_policy, final_model


def main():
    """Main entry point"""
    print("="*60)
    print("SPMI Variants with SA-PMI: Student-Teacher Domain")
    print("Standard SPMI | SA-PMI | Federated SPMI | Federated SA-PMI")
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
    dir_path = "./data/sapmi_simple"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # ========================================
    # 1. Run Standard SPMI (Baseline)
    # ========================================
    print("\n" + "=" * 80)
    print("PART 1: Standard SPMI Baseline")
    print("=" * 80)
    
    spmi, _, _ = run_standard_spmi(
        mdp, initial_policy, initial_model, original_model, max_iter=500
    )
    spmi.logger.save(dir_path, 'standard_spmi.csv')

    # ========================================
    # 2. Run SA-PMI with Policy Robustness
    # ========================================
    print("\n" + "=" * 80)
    print("PART 2: SA-PMI with Policy Robustness (Curriculum Learning)")
    print("=" * 80)
    
    # Define curriculum configurations (selected best ones)
    curricula_configs = [
        {
            'name': 'constant_medium',
            'scheduler': ConstantCurriculumScheduler(weight=0.4),
            'description': 'Constant 40% robustness',
            'temp': 0.3,
            'entropy': 0.1
        },
        {
            'name': 'linear',
            'scheduler': LinearCurriculumScheduler(
                start_iter=50,
                end_iter=300,
                start_weight=0.0,
                end_weight=0.5
            ),
            'description': 'Linear ramp: 0→50% robustness',
            'temp': 0.3,
            'entropy': 0.1
        },
        {
            'name': 'exponential',
            'scheduler': ExponentialCurriculumScheduler(
                growth_rate=0.01,
                max_weight=0.5
            ),
            'description': 'Exponential growth to 50% robustness',
            'temp': 0.3,
            'entropy': 0.1
        },
    ]
    
    sapmi_results = []
    for config in curricula_configs:
        print(f"\n{config['description']}")
        print("-" * 60)
        
        sapmi, _, _ = run_sapmi_policy_robust(
            mdp,
            initial_policy,
            initial_model,
            original_model,
            curriculum_name=config['name'],
            curriculum_scheduler=config['scheduler'],
            robustness_temperature=config['temp'],
            entropy_bonus=config['entropy'],
            max_iter=500
        )
        
        sapmi.logger.save(dir_path, f"sapmi_policy_{config['name']}.csv")
        sapmi.policy_chooser.save_sapmi_policy_metrics(f"{dir_path}/sapmi_policy_{config['name']}")
        
        sapmi_results.append((config, sapmi))

    # ========================================
    # 3. Run Federated SPMI (Baseline)
    # ========================================
    print("\n" + "=" * 80)
    print("PART 3: Federated SPMI (Baseline)")
    print("=" * 80)
    
    fspmi_configurations = [
        {'n_agents': 2, 'episodes_per_round': 100, 'max_rounds': 500},
        {'n_agents': 4, 'episodes_per_round': 100, 'max_rounds': 500},
    ]

    fspmi_results = []
    for config in fspmi_configurations:
        fspmi, _, _ = run_federated_spmi(
            mdp, initial_policy, initial_model, original_model, **config
        )
        fspmi.logger.save(f"{dir_path}/fspmi_n{config['n_agents']}.csv")
        fspmi_results.append((config, fspmi))

    # ========================================
    # 4. Run Federated SA-PMI
    # ========================================
    print("\n" + "=" * 80)
    print("PART 4: Federated SA-PMI (Policy Robustness)")
    print("=" * 80)
    
    # Use best performing curricula for federated version
    federated_sapmi_configs = [
        {
            'name': 'constant_medium',
            'scheduler': ConstantCurriculumScheduler(weight=0.4),
            'description': 'Federated with Constant 40% robustness',
            'temp': 0.3,
            'entropy': 0.1
        },
        {
            'name': 'exponential',
            'scheduler': ExponentialCurriculumScheduler(
                growth_rate=0.01,
                max_weight=0.5
            ),
            'description': 'Federated with Exponential curriculum',
            'temp': 0.3,
            'entropy': 0.1
        },
    ]
    
    fsapmi_results = []
    for config in federated_sapmi_configs:
        print(f"\n{config['description']}")
        print("-" * 60)
        
        fsapmi, _, _ = run_federated_sapmi(
            mdp,
            initial_policy,
            initial_model,
            original_model,
            curriculum_name=config['name'],
            curriculum_scheduler=config['scheduler'],
            robustness_temperature=config['temp'],
            entropy_bonus=config['entropy'],
            n_agents=4,
            episodes_per_round=100,
            max_rounds=500
        )
        
        fsapmi.logger.save(f"{dir_path}/fsapmi_{config['name']}_n4.csv")
        fsapmi_results.append((config, fsapmi))

    # ========================================
    # 5. Summary
    # ========================================
    print("\n" + "=" * 80)
    print("Final Summary")
    print("=" * 80)

    print(f"\nStandard SPMI:")
    print(f"  Iterations: {spmi.logger.iteration}")
    print(f"  Final Performance: {spmi.logger.evaluations[-1]:.4f}")

    print(f"\nSA-PMI Variants:")
    for config, sapmi in sapmi_results:
        perf = sapmi.logger.evaluations[-1] if sapmi.logger.evaluations else float('nan')
        print(f"  {config['name']:20s}: {perf:.4f}")

    print(f"\nFederated SPMI:")
    for config, fspmi in fspmi_results:
        perf = fspmi.logger.true_performances[-1] if hasattr(fspmi.logger, 'true_performances') and fspmi.logger.true_performances else fspmi.logger.avg_returns[-1]
        print(f"  N={config['n_agents']:2d}: {perf:.4f}")

    print(f"\nFederated SA-PMI:")
    for config, fsapmi in fsapmi_results:
        perf = fsapmi.logger.true_performances[-1] if hasattr(fsapmi.logger, 'true_performances') and fsapmi.logger.true_performances else fsapmi.logger.avg_returns[-1]
        print(f"  {config['name']:20s}: {perf:.4f}")

    print("\nResults saved to:", dir_path)


if __name__ == '__main__':
    main()