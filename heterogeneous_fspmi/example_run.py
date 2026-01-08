"""
Example: Running Heterogeneous F-SPMI

This script demonstrates three heterogeneity modes:
1. Different dynamics (k parameter)
2. Different failure probabilities (robustness training)
3. Mixed heterogeneity (both)

Run from project root:
    python -m heterogeneous_fspmi.example_run
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heterogeneous_fspmi.heterogeneous_fspmi import (
    HeterogeneousFSPMI,
    HeterogeneousConfig,
    EnvironmentVariant,
    create_dynamics_heterogeneous_fspmi,
    create_robustness_heterogeneous_fspmi,
)
from utils.tabular import TabularPolicy, TabularModel


def create_initial_policy(nS: int, nA: int) -> TabularPolicy:
    """Create uniform random initial policy."""
    policy_rep = {s: np.ones(nA) / nA for s in range(nS)}
    return TabularPolicy(policy_rep, nS, nA)


def create_initial_model(env) -> TabularModel:
    """Create initial model from environment."""
    return TabularModel(env.P, env.nS, env.nA)


def run_dynamics_experiment():
    """
    Experiment 1: Different dynamics parameters (k).
    
    Agents see environments with different speed-dependent transition probabilities:
    - Agent 0: k=0.3 (low-speed dynamics favored)
    - Agent 1: k=0.5 (balanced)
    - Agent 2: k=0.7 (high-speed dynamics favored)
    
    The learned policy should be robust across all dynamics regimes.
    Uses shaped rewards and alternating updates for faster learning.
    """
    print("=" * 60)
    print("EXPERIMENT 1: Dynamics Heterogeneity")
    print("=" * 60)
    
    # Use shaped rewards: [goal, offroad, zero_speed, low_speed, high_speed]
    shaped_reward = [1.0, -0.1, -0.05, 0.01, 0.02]
    
    # Create heterogeneous F-SPMI with faster learning settings
    config = HeterogeneousConfig(
        variants=[
            EnvironmentVariant(0, "T1", k=0.3, reward_weight=shaped_reward,
                             description="k=0.3 (low-speed favored)"),
            EnvironmentVariant(1, "T1", k=0.5, reward_weight=shaped_reward,
                             description="k=0.5 (balanced)"),
            EnvironmentVariant(2, "T1", k=0.7, reward_weight=shaped_reward,
                             description="k=0.7 (high-speed favored)"),
        ],
        episodes_per_agent=200,
        n_iterations=100,
        use_parallel=True,
        update_mode='alternating',  # Force both policy and model to update
        target_policy_type='softmax',  # Smoother target for larger step sizes
        softmax_temperature=0.5,
        min_step_size=0.01,  # Ensure minimum progress
    )
    hfspmi = HeterogeneousFSPMI(config)
    
    # Initialize policy and model
    initial_policy = create_initial_policy(hfspmi.nS, hfspmi.nA)
    initial_model = create_initial_model(hfspmi.ref_env)
    
    # Run
    final_policy, final_model = hfspmi.run(
        initial_policy, initial_model, verbose=True
    )
    
    # Save results
    hfspmi.logger.save("results_dynamics_heterogeneous.csv")
    print(f"\nFinal true performance: {hfspmi.logger.true_performances[-1]:.4f}")
    
    return hfspmi, final_policy, final_model


def run_robustness_experiment():
    """
    Experiment 2: Different failure probabilities (pfail).
    
    Agents see environments with different failure rates:
    - Agent 0: pfail=0.0 (no failures)
    - Agent 1: pfail=0.05 (5% failure rate)
    - Agent 2: pfail=0.1 (10% failure rate)
    
    The learned policy should be conservative enough to handle failures.
    Uses shaped rewards and alternating updates for faster learning.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Robustness Heterogeneity")
    print("=" * 60)
    
    # Use shaped rewards
    shaped_reward = [1.0, -0.1, -0.05, 0.01, 0.02]
    
    config = HeterogeneousConfig(
        variants=[
            EnvironmentVariant(0, "T1", k=0.5, pfail=0.0, reward_weight=shaped_reward,
                             description="pfail=0.0 (safe)"),
            EnvironmentVariant(1, "T1", k=0.5, pfail=0.05, reward_weight=shaped_reward,
                             description="pfail=0.05 (moderate)"),
            EnvironmentVariant(2, "T1", k=0.5, pfail=0.1, reward_weight=shaped_reward,
                             description="pfail=0.1 (risky)"),
        ],
        episodes_per_agent=200,
        n_iterations=100,
        use_parallel=True,
        update_mode='alternating',
        target_policy_type='softmax',
        softmax_temperature=0.5,
        min_step_size=0.01,
    )
    hfspmi = HeterogeneousFSPMI(config)
    
    initial_policy = create_initial_policy(hfspmi.nS, hfspmi.nA)
    initial_model = create_initial_model(hfspmi.ref_env)
    
    final_policy, final_model = hfspmi.run(
        initial_policy, initial_model, verbose=True
    )
    
    hfspmi.logger.save("results_robustness_heterogeneous.csv")
    print(f"\nFinal true performance: {hfspmi.logger.true_performances[-1]:.4f}")
    
    return hfspmi, final_policy, final_model


def run_mixed_experiment():
    """
    Experiment 3: Mixed heterogeneity (dynamics + failure).
    
    Most realistic scenario with both types of variation.
    Uses shaped rewards and alternating updates for faster learning.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Mixed Heterogeneity")
    print("=" * 60)
    
    # Use shaped rewards
    shaped_reward = [1.0, -0.1, -0.05, 0.01, 0.02]
    
    variants = [
        EnvironmentVariant(0, "T1", k=0.3, pfail=0.0, reward_weight=shaped_reward,
                         description="low-speed, safe"),
        EnvironmentVariant(1, "T1", k=0.5, pfail=0.0, reward_weight=shaped_reward,
                         description="balanced, safe"),
        EnvironmentVariant(2, "T1", k=0.7, pfail=0.0, reward_weight=shaped_reward,
                         description="high-speed, safe"),
        EnvironmentVariant(3, "T1", k=0.5, pfail=0.05, reward_weight=shaped_reward,
                         description="balanced, moderate risk"),
        EnvironmentVariant(4, "T1", k=0.5, pfail=0.1, reward_weight=shaped_reward,
                         description="balanced, high risk"),
    ]
    
    config = HeterogeneousConfig(
        variants=variants,
        episodes_per_agent=200,
        n_iterations=100,
        use_parallel=True,
        update_mode='alternating',
        target_policy_type='softmax',
        softmax_temperature=0.5,
        min_step_size=0.01,
    )
    
    hfspmi = HeterogeneousFSPMI(config)
    
    initial_policy = create_initial_policy(hfspmi.nS, hfspmi.nA)
    initial_model = create_initial_model(hfspmi.ref_env)
    
    final_policy, final_model = hfspmi.run(
        initial_policy, initial_model, verbose=True
    )
    
    hfspmi.logger.save("results_mixed_heterogeneous.csv")
    print(f"\nFinal true performance: {hfspmi.logger.true_performances[-1]:.4f}")
    
    return hfspmi, final_policy, final_model


def run_custom_experiment():
    """
    Experiment 4: Custom variant configuration with hazard zones.
    
    Demonstrates full flexibility of the configuration system.
    Uses shaped rewards and alternating updates for faster learning.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Custom Variants with Hazard Zones")
    print("=" * 60)
    
    # Use shaped rewards
    shaped_reward = [1.0, -0.1, -0.05, 0.01, 0.02]
    
    # Define custom variants
    variants = [
        EnvironmentVariant(
            variant_id=0,
            track_file="T1",
            k=0.5,
            pfail=0.0,
            reward_weight=shaped_reward,
            description="baseline"
        ),
        EnvironmentVariant(
            variant_id=1,
            track_file="T1",
            k=0.5,
            pfail=0.0,
            reward_weight=shaped_reward,
            hazard_positions=[(3, 3), (3, 4), (4, 3)],  # Add hazard zone
            hazard_penalty=-0.5,
            description="hazard_zone_A"
        ),
        EnvironmentVariant(
            variant_id=2,
            track_file="T1",
            k=0.5,
            pfail=0.0,
            reward_weight=shaped_reward,
            hazard_positions=[(5, 5), (5, 6), (6, 5)],  # Different hazard zone
            hazard_penalty=-0.5,
            description="hazard_zone_B"
        ),
    ]
    
    config = HeterogeneousConfig(
        variants=variants,
        episodes_per_agent=200,
        n_iterations=100,
        use_parallel=True,
        update_mode='alternating',
        target_policy_type='softmax',
        softmax_temperature=0.5,
        min_step_size=0.01,
    )
    
    hfspmi = HeterogeneousFSPMI(config)
    
    initial_policy = create_initial_policy(hfspmi.nS, hfspmi.nA)
    initial_model = create_initial_model(hfspmi.ref_env)
    
    final_policy, final_model = hfspmi.run(
        initial_policy, initial_model, verbose=True
    )
    
    hfspmi.logger.save("results_custom_heterogeneous.csv")
    print(f"\nFinal true performance: {hfspmi.logger.true_performances[-1]:.4f}")
    
    return hfspmi, final_policy, final_model


def compare_homogeneous_vs_heterogeneous():
    """
    Comparison: Standard (homogeneous) vs Heterogeneous F-SPMI.
    
    Shows that heterogeneous training produces more robust policies.
    Uses alternating updates for faster learning.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Homogeneous vs Heterogeneous")
    print("=" * 60)
    
    # Use shaped rewards: [goal, offroad, zero_speed, low_speed, high_speed]
    # This gives the agent more learning signal than sparse goal-only rewards
    shaped_reward = [1.0, -0.1, -0.05, 0.01, 0.02]
    
    # Homogeneous: All agents same environment
    print("\n--- Homogeneous (all k=0.5) ---")
    homo_config = HeterogeneousConfig(
        variants=[
            EnvironmentVariant(i, "T1", k=0.5, reward_weight=shaped_reward,
                             description=f"agent_{i}")
            for i in range(3)
        ],
        episodes_per_agent=200,
        n_iterations=100,
        use_parallel=True,
        update_mode='alternating',
        target_policy_type='softmax',
        softmax_temperature=0.5,
        min_step_size=0.01,
    )
    homo_fspmi = HeterogeneousFSPMI(homo_config)
    
    initial_policy = create_initial_policy(homo_fspmi.nS, homo_fspmi.nA)
    initial_model = create_initial_model(homo_fspmi.ref_env)
    
    homo_policy, homo_model = homo_fspmi.run(
        initial_policy, initial_model, verbose=False
    )
    
    # Heterogeneous: Different environments
    print("\n--- Heterogeneous (k=0.3, 0.5, 0.7) ---")
    hetero_config = HeterogeneousConfig(
        variants=[
            EnvironmentVariant(0, "T1", k=0.3, reward_weight=shaped_reward,
                             description="k=0.3"),
            EnvironmentVariant(1, "T1", k=0.5, reward_weight=shaped_reward,
                             description="k=0.5"),
            EnvironmentVariant(2, "T1", k=0.7, reward_weight=shaped_reward,
                             description="k=0.7"),
        ],
        episodes_per_agent=200,
        n_iterations=100,
        use_parallel=True,
        update_mode='alternating',
        target_policy_type='softmax',
        softmax_temperature=0.5,
        min_step_size=0.01,
    )
    hetero_fspmi = HeterogeneousFSPMI(hetero_config)
    
    hetero_policy, hetero_model = hetero_fspmi.run(
        initial_policy, initial_model, verbose=False
    )
    
    # Compare final performances
    print("\n" + "-" * 40)
    print("RESULTS:")
    print(f"  Homogeneous final perf: {homo_fspmi.logger.true_performances[-1]:.4f}")
    print(f"  Heterogeneous final perf: {hetero_fspmi.logger.true_performances[-1]:.4f}")
    
    # Test both policies across different k values
    print("\n  Robustness test (performance across k values):")
    from envs.racetrack_simulator import RaceTrackConfigurableEnv
    from utils.tabular import TabularReward
    import utils.evaluator as evaluator
    
    for test_k in [0.2, 0.5, 0.8]:
        test_env = RaceTrackConfigurableEnv(
            "T1", initial_configuration=[test_k, 1-test_k, 0, 0],
            reward_weight=shaped_reward
        )
        reward = TabularReward(test_env.P, test_env.nS, test_env.nA)
        
        homo_perf = evaluator.compute_performance(
            test_env.mu, reward, homo_policy, homo_model,
            0.9, 20, test_env.nS, test_env.nA
        )
        hetero_perf = evaluator.compute_performance(
            test_env.mu, reward, hetero_policy, hetero_model,
            0.9, 20, test_env.nS, test_env.nA
        )
        
        print(f"    k={test_k}: Homo={homo_perf:.4f}, Hetero={hetero_perf:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Heterogeneous F-SPMI experiments")
    parser.add_argument("--experiment", type=str, default="dynamics",
                        choices=["dynamics", "robustness", "mixed", "custom", "compare", "all"],
                        help="Which experiment to run")
    
    args = parser.parse_args()
    
    if args.experiment == "dynamics" or args.experiment == "all":
        run_dynamics_experiment()
    
    if args.experiment == "robustness" or args.experiment == "all":
        run_robustness_experiment()
    
    if args.experiment == "mixed" or args.experiment == "all":
        run_mixed_experiment()
    
    if args.experiment == "custom" or args.experiment == "all":
        run_custom_experiment()
    
    if args.experiment == "compare" or args.experiment == "all":
        compare_homogeneous_vs_heterogeneous()
    
    print("\n" + "=" * 60)
    print("All experiments complete!")
    print("=" * 60)