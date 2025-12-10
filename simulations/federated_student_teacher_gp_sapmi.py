"""
Federated SPMI Simulation: Student-Teacher Domain (GP Model Selection)
with SA-PMI Curriculum Learning for Policy Robustness

This script runs:
1. Standard SPMI with GP model chooser
2. SA-PMI with curriculum learning for POLICY robustness (various schedules)
3. Federated SPMI with GP model chooser
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
from algorithm.model_chooser import GPModelChooser
from algorithm.curriculum_scheduler import (
    LinearCurriculumScheduler,
    ExponentialCurriculumScheduler,
    CosineCurriculumScheduler,
    ConstantCurriculumScheduler
)
from utils.tabular_factory import model_from_matrix
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy
from envs.student_teacher import TeacherStudentEnv

# Import F-SPMI components
from federated import FSPMI

def build_gp_model_set(original_model, nS, nA, noise_levels=(0.0, 0.05, 0.1, 0.2)):
    """Create a small set of smoothed models for GP exploration."""
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
    gp_fit_frequency=10,
    max_iter=1000
):
    """Run standard SPMI with GP model chooser."""
    print("\n" + "=" * 60)
    print("Running Standard SPMI (GP model chooser)")
    print("=" * 60)

    mdp.set_model(copy.deepcopy(original_model))
    mdp.model_vector = np.array(init_model_vector)

    policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
    model_chooser = GPModelChooser(
        model_set, mdp.nS, mdp.nA, init_model_vector, gp_beta, original_model
    )
    model_chooser.nr_iterations_pause = gp_fit_frequency

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
    if spmi.logger.evaluations:
        print(f"Final Performance: {spmi.logger.evaluations[-1]:.4f}")

    return spmi, final_policy, final_model

def run_sapmi_policy_robust(
    mdp,
    initial_policy,
    initial_model,
    original_model,
    model_set,
    init_model_vector,
    curriculum_name,
    curriculum_scheduler,
    robustness_temperature=0.3,
    entropy_bonus=0.1,
    gp_beta=1.0,
    gp_fit_frequency=10,
    max_iter=1000
):
    """Run SA-PMI with curriculum learning for POLICY robustness."""
    print("\n" + "=" * 60)
    print(f"Running SA-PMI Policy-Robust ({curriculum_name} curriculum)")
    print("=" * 60)

    mdp.set_model(copy.deepcopy(original_model))
    mdp.model_vector = np.array(init_model_vector)

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
    
    # Use standard GP model chooser
    model_chooser = GPModelChooser(
        model_set, mdp.nS, mdp.nA, init_model_vector, gp_beta, original_model
    )
    model_chooser.nr_iterations_pause = gp_fit_frequency

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

def run_federated_spmi(
    mdp,
    initial_policy,
    initial_model,
    original_model,
    model_set,
    init_model_vector,
    n_agents=4,
    episodes_per_round=100,
    max_rounds=500,
    gp_beta=1.0,
    gp_fit_frequency=10
):
    """Run Federated SPMI with GP model chooser."""
    print("\n" + "=" * 60)
    print(f"Running Federated SPMI (GP) (N={n_agents} agents, {episodes_per_round} episodes/round)")
    print("=" * 60)

    mdp.set_model(copy.deepcopy(original_model))
    mdp.model_vector = np.array(init_model_vector)

    policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
    model_chooser = GPModelChooser(
        model_set, mdp.nS, mdp.nA, init_model_vector, gp_beta, original_model
    )
    model_chooser.nr_iterations_pause = gp_fit_frequency

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
        verbose=True  # Reduce output
    )

    start_time = time.time()
    final_policy, final_model = fspmi.run(
        copy.deepcopy(initial_policy),
        copy.deepcopy(initial_model)
    )
    elapsed = time.time() - start_time

    print(f"Federated SPMI completed in {elapsed:.2f} seconds")
    if fspmi.logger.true_performances:
        print(f"Final Performance (true): {fspmi.logger.true_performances[-1]:.4f}")

    return fspmi, final_policy, final_model


def run_federated_sapmi(
    mdp,
    initial_policy,
    initial_model,
    original_model,
    model_set,
    init_model_vector,
    curriculum_name,
    curriculum_scheduler,
    robustness_temperature=0.3,
    entropy_bonus=0.1,
    n_agents=4,
    episodes_per_round=100,
    max_rounds=500,
    gp_beta=1.0,
    gp_fit_frequency=10
):
    """Run Federated SPMI with GP model chooser."""
    print("\n" + "=" * 60)
    print(f"Running Federated SPMI (GP) (N={n_agents} agents, {episodes_per_round} episodes/round)")
    print("=" * 60)

    mdp.set_model(copy.deepcopy(original_model))
    mdp.model_vector = np.array(init_model_vector)

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

    model_chooser = GPModelChooser(
        model_set, mdp.nS, mdp.nA, init_model_vector, gp_beta, original_model
    )
    model_chooser.nr_iterations_pause = gp_fit_frequency

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

    print(f"Federated SPMI ({curriculum_name}) completed in {elapsed:.2f} seconds")
    if fspmi.logger.true_performances:
        print(f"Final Performance (true): {fspmi.logger.true_performances[-1]:.4f}")

    return fspmi, final_policy, final_model


def main():
    """Main entry point"""
    print("=" * 60)
    print("SPMI Variants Comparison: Policy Robustness Focus")
    print("Standard SPMI | SA-PMI | Federated SPMI | Federated SA-PMI")
    print("=" * 60)

    # Create environment (use original smaller environment)
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
    model_set, init_model_vector = build_gp_model_set(
        original_model, mdp.nS, mdp.nA, gp_noise_levels
    )
    gp_beta = 1.0
    gp_fit_frequency = 10
    max_iter = 500

    # Create output directory
    dir_path = "./data/sapmi_policy_robust"
    os.makedirs(dir_path, exist_ok=True)

    # ========================================
    # 1. Run Standard SPMI (Baseline)
    # ========================================
    print("\n" + "=" * 80)
    print("PART 1: Standard SPMI Baseline")
    print("=" * 80)
    
    spmi, _, _ = run_standard_spmi(
        mdp,
        initial_policy,
        initial_model,
        original_model,
        model_set,
        init_model_vector,
        gp_beta=gp_beta,
        gp_fit_frequency=gp_fit_frequency,
        max_iter=max_iter
    )
    spmi.logger.save(dir_path, 'standard_spmi.csv')
    spmi.model_chooser.save_gp_times(f"{dir_path}/standard_spmi")

    # ========================================
    # 2. Run SA-PMI with Policy Robustness
    # ========================================
    print("\n" + "=" * 80)
    print("PART 2: SA-PMI with Policy Robustness (Curriculum Learning)")
    print("=" * 80)
    
    # Define curriculum configurations
    curricula_configs = [
        {
            'name': 'constant_zero',
            'scheduler': ConstantCurriculumScheduler(weight=0.0),
            'description': 'No robustness (pure greedy - sanity check)',
            'temp': 0.3,
            'entropy': 0.1
        },
        {
            'name': 'constant_low',
            'scheduler': ConstantCurriculumScheduler(weight=0.2),
            'description': 'Constant 20% robustness',
            'temp': 0.3,
            'entropy': 0.1
        },
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
            'description': 'Linear ramp: 0→50% robustness over iterations 50-300',
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
        {
            'name': 'cosine',
            'scheduler': CosineCurriculumScheduler(
                start_iter=50,
                end_iter=300,
                max_weight=0.5
            ),
            'description': 'Cosine annealing: smooth S-curve to 50% robustness',
            'temp': 0.3,
            'entropy': 0.1
        },
        {
            'name': 'high_entropy',
            'scheduler': LinearCurriculumScheduler(
                start_iter=50,
                end_iter=300,
                start_weight=0.0,
                end_weight=0.5
            ),
            'description': 'Linear curriculum with higher entropy (more exploration)',
            'temp': 0.5,
            'entropy': 0.2
        }
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
            model_set,
            init_model_vector,
            curriculum_name=config['name'],
            curriculum_scheduler=config['scheduler'],
            robustness_temperature=config['temp'],
            entropy_bonus=config['entropy'],
            gp_beta=gp_beta,
            gp_fit_frequency=gp_fit_frequency,
            max_iter=max_iter
        )
        
        sapmi.logger.save(dir_path, f"sapmi_policy_{config['name']}.csv")
        sapmi.policy_chooser.save_sapmi_policy_metrics(f"{dir_path}/sapmi_policy_{config['name']}")
        sapmi.model_chooser.save_gp_times(f"{dir_path}/sapmi_policy_{config['name']}")
        
        sapmi_results.append((config, sapmi))

    # ========================================
    # 3. Run Federated SPMI
    # ========================================
    print("\n" + "=" * 80)
    print("PART 3: Federated SPMI (Baseline)")
    print("=" * 80)
    
    fspmi_configurations = [
        {'n_agents': 4, 'episodes_per_round': 100, 'max_rounds': 500},
    ]

    fspmi_results = []
    for config in fspmi_configurations:
        fspmi, _, _ = run_federated_spmi(
            mdp,
            initial_policy,
            initial_model,
            original_model,
            model_set,
            init_model_vector,
            gp_beta=gp_beta,
            gp_fit_frequency=gp_fit_frequency,
            **config
        )
        log_path = f"{dir_path}/fspmi_n{config['n_agents']}.csv"
        fspmi.logger.save(log_path)
        fspmi.model_chooser.save_gp_times(log_path.replace('.csv', ''))
        fspmi_results.append((config, fspmi))

    # ========================================
    # 4. Run Federated SA-PMI
    # ========================================
    print("\n" + "=" * 80)
    print("PART 4: Federated SA-PMI (Policy Robustness)")
    print("=" * 80)
    
    # Select best performing curricula from Part 2 for federated version
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
            model_set,
            init_model_vector,
            curriculum_name=config['name'],
            curriculum_scheduler=config['scheduler'],
            robustness_temperature=config['temp'],
            entropy_bonus=config['entropy'],
            n_agents=4,
            episodes_per_round=100,
            max_rounds=500,
            gp_beta=gp_beta,
            gp_fit_frequency=gp_fit_frequency
        )
        
        log_path = f"{dir_path}/fsapmi_{config['name']}_n4.csv"
        fsapmi.logger.save(log_path)
        fsapmi.model_chooser.save_gp_times(log_path.replace('.csv', ''))
        # Note: Federated version uses model_chooser, not policy_chooser for saving
        # The policy chooser metrics are tracked internally but not exposed in F-SPMI logger
        
        fsapmi_results.append((config, fsapmi))

    # ========================================
    # 5. Summary Comparison
    # ========================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: Complete Comparison")
    print("=" * 80)

    print(f"\n{'Method':<50} {'Iterations/Rounds':<15} {'Final Performance':<20}")
    print("-" * 85)
    
    # Standard SPMI
    perf = spmi.logger.evaluations[-1] if spmi.logger.evaluations else float('nan')
    print(f"{'Standard SPMI (greedy)':<50} {spmi.logger.iteration:<15} {perf:.4f}")
    
    # SA-PMI variants
    for config, sapmi in sapmi_results:
        method_name = f"SA-PMI ({config['name']})"
        perf = sapmi.logger.evaluations[-1] if sapmi.logger.evaluations else float('nan')
        print(f"{method_name:<50} {sapmi.logger.iteration:<15} {perf:.4f}")
    
    # Federated SPMI
    for config, fspmi in fspmi_results:
        method_name = f"F-SPMI (N={config['n_agents']}, greedy)"
        perf_true = fspmi.logger.true_performances[-1] if fspmi.logger.true_performances else float('nan')
        print(f"{method_name:<50} {len(fspmi.logger.iterations):<15} {perf_true:.4f}")
    
    # Federated SA-PMI
    for config, fsapmi in fsapmi_results:
        method_name = f"F-SA-PMI (N=4, {config['name']})"
        perf_true = fsapmi.logger.true_performances[-1] if fsapmi.logger.true_performances else float('nan')
        print(f"{method_name:<50} {len(fsapmi.logger.iterations):<15} {perf_true:.4f}")

    print("\n" + "=" * 80)
    print(f"Results saved to: {dir_path}")
    print("=" * 80)
    
    print("\nGenerated files:")
    print("  Standard & SA-PMI:")
    print("    - standard_spmi.csv")
    print("    - sapmi_policy_*.csv (performance logs)")
    print("    - sapmi_policy_*_robustness_weights.csv (curriculum progression)")
    print("    - sapmi_policy_*_*_advantages.csv (advantage decomposition)")
    print("  Federated:")
    print("    - fspmi_n*.csv (federated baseline)")
    print("    - fsapmi_*_n4.csv (federated with robustness)")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    # Calculate improvements
    baseline_perf = spmi.logger.evaluations[-1] if spmi.logger.evaluations else 0.4587
    
    print(f"\nBaseline (Standard SPMI): {baseline_perf:.4f}")
    
    # Best SA-PMI
    best_sapmi = max(sapmi_results, key=lambda x: x[1].logger.evaluations[-1] if x[1].logger.evaluations else 0)
    best_sapmi_perf = best_sapmi[1].logger.evaluations[-1]
    best_sapmi_improvement = ((best_sapmi_perf - baseline_perf) / baseline_perf) * 100
    print(f"Best SA-PMI ({best_sapmi[0]['name']}): {best_sapmi_perf:.4f} ({best_sapmi_improvement:+.2f}%)")
    
    # Federated SPMI
    if fspmi_results:
        fspmi_perf = fspmi_results[0][1].logger.true_performances[-1]
        fspmi_improvement = ((fspmi_perf - baseline_perf) / baseline_perf) * 100
        print(f"Federated SPMI: {fspmi_perf:.4f} ({fspmi_improvement:+.2f}%)")
    
    # Best Federated SA-PMI
    if fsapmi_results:
        best_fsapmi = max(fsapmi_results, key=lambda x: x[1].logger.true_performances[-1] if x[1].logger.true_performances else 0)
        best_fsapmi_perf = best_fsapmi[1].logger.true_performances[-1]
        best_fsapmi_improvement = ((best_fsapmi_perf - baseline_perf) / baseline_perf) * 100
        print(f"Best F-SA-PMI ({best_fsapmi[0]['name']}): {best_fsapmi_perf:.4f} ({best_fsapmi_improvement:+.2f}%)")
    
    print("\n✓ All experiments completed successfully!")

if __name__ == '__main__':
    main()