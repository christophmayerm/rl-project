"""
Quick debug script to analyze step size calculation.
Run with: python heterogeneous_fspmi/debug_step_sizes.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heterogeneous_fspmi.heterogeneous_fspmi import (
    HeterogeneousFSPMI,
    HeterogeneousConfig,
    EnvironmentVariant,
)
from utils.tabular import TabularPolicy, TabularModel


def create_initial_policy(nS: int, nA: int) -> TabularPolicy:
    """Create uniform random initial policy."""
    policy_rep = {s: np.ones(nA) / nA for s in range(nS)}
    return TabularPolicy(policy_rep, nS, nA)


def create_initial_model(env) -> TabularModel:
    """Create initial model from environment."""
    return TabularModel(env.P, env.nS, env.nA)


def run_standard_mode():
    """Test standard mode (original issue: α*=0 always)."""
    print("=" * 70)
    print("TEST 1: Standard Mode (original)")
    print("=" * 70)
    
    shaped_reward = [1.0, -0.1, -0.05, 0.01, 0.02]
    
    config = HeterogeneousConfig(
        variants=[
            EnvironmentVariant(0, "T1", k=0.3, reward_weight=shaped_reward, description="k=0.3"),
            EnvironmentVariant(1, "T1", k=0.5, reward_weight=shaped_reward, description="k=0.5"),
            EnvironmentVariant(2, "T1", k=0.7, reward_weight=shaped_reward, description="k=0.7"),
        ],
        episodes_per_agent=200,
        n_iterations=20,
        use_parallel=True,
        update_mode='standard',
        target_policy_type='greedy',
    )
    
    hfspmi = HeterogeneousFSPMI(config)
    initial_policy = create_initial_policy(hfspmi.nS, hfspmi.nA)
    initial_model = create_initial_model(hfspmi.ref_env)
    
    final_policy, final_model = hfspmi.run(initial_policy, initial_model, verbose=True)
    print(f"\nFinal performance: {hfspmi.logger.true_performances[-1]:.4f}")
    return hfspmi


def run_alternating_mode():
    """Test alternating mode (forces both policy and model updates)."""
    print("\n" + "=" * 70)
    print("TEST 2: Alternating Mode (forces progress on both)")
    print("=" * 70)
    
    shaped_reward = [1.0, -0.1, -0.05, 0.01, 0.02]
    
    config = HeterogeneousConfig(
        variants=[
            # EnvironmentVariant(0, "T4", k=0.3, reward_weight=shaped_reward, description="k=0.3"),
            # EnvironmentVariant(1, "T4", k=0.5, reward_weight=shaped_reward, description="k=0.5"),
            # EnvironmentVariant(2, "T4", k=0.7, reward_weight=shaped_reward, description="k=0.7"),
            EnvironmentVariant(3, "T4", pfail=0.01, reward_weight=shaped_reward, description="pfail=0.01"),
            EnvironmentVariant(4, "T4", pfail=0.05, reward_weight=shaped_reward, description="pfail=0.05"),
            EnvironmentVariant(5, "T4", pfail=0.1, reward_weight=shaped_reward, description="pfail=0.1"),
        ],
        episodes_per_agent=400,
        n_iterations=500,
        use_parallel=True,
        update_mode='alternating',  # Force alternating updates
        target_policy_type='greedy',
        min_step_size=0.01,  # Ensure minimum progress
    )
    
    hfspmi = HeterogeneousFSPMI(config)
    initial_policy = create_initial_policy(hfspmi.nS, hfspmi.nA)
    initial_model = create_initial_model(hfspmi.ref_env)
    
    final_policy, final_model = hfspmi.run(initial_policy, initial_model, verbose=True)
    print(f"\nFinal performance: {hfspmi.logger.true_performances[-1]:.4f}")
    return hfspmi


def run_softmax_mode():
    """Test softmax target policy (smaller distances → larger step sizes)."""
    print("\n" + "=" * 70)
    print("TEST 3: Softmax Target Policy (smoother updates)")
    print("=" * 70)
    
    shaped_reward = [1.0, -0.1, -0.05, 0.01, 0.02]
    
    config = HeterogeneousConfig(
        variants=[
            EnvironmentVariant(0, "T1", k=0.3, reward_weight=shaped_reward, description="k=0.3"),
            EnvironmentVariant(1, "T1", k=0.5, reward_weight=shaped_reward, description="k=0.5"),
            EnvironmentVariant(2, "T1", k=0.7, reward_weight=shaped_reward, description="k=0.7"),
        ],
        episodes_per_agent=200,
        n_iterations=20,
        use_parallel=True,
        update_mode='standard',
        target_policy_type='softmax',  # Softmax instead of greedy
        softmax_temperature=0.5,  # Lower = more greedy, higher = more uniform
    )
    
    hfspmi = HeterogeneousFSPMI(config)
    initial_policy = create_initial_policy(hfspmi.nS, hfspmi.nA)
    initial_model = create_initial_model(hfspmi.ref_env)
    
    final_policy, final_model = hfspmi.run(initial_policy, initial_model, verbose=True)
    print(f"\nFinal performance: {hfspmi.logger.true_performances[-1]:.4f}")
    return hfspmi


def run_aggressive_mode():
    """Test aggressive mode (alternating + softmax + min step)."""
    print("\n" + "=" * 70)
    print("TEST 4: Aggressive Mode (alternating + softmax + min_step)")
    print("=" * 70)
    
    shaped_reward = [1.0, -0.1, -0.05, 0.01, 0.02]
    
    config = HeterogeneousConfig(
        variants=[
            EnvironmentVariant(0, "T1", k=0.3, reward_weight=shaped_reward, description="k=0.3"),
            EnvironmentVariant(1, "T1", k=0.5, reward_weight=shaped_reward, description="k=0.5"),
            EnvironmentVariant(2, "T1", k=0.7, reward_weight=shaped_reward, description="k=0.7"),
        ],
        episodes_per_agent=200,
        n_iterations=50,
        use_parallel=True,
        update_mode='alternating',
        target_policy_type='softmax',
        softmax_temperature=0.5,
        min_step_size=0.02,
    )
    
    hfspmi = HeterogeneousFSPMI(config)
    initial_policy = create_initial_policy(hfspmi.nS, hfspmi.nA)
    initial_model = create_initial_model(hfspmi.ref_env)
    
    final_policy, final_model = hfspmi.run(initial_policy, initial_model, verbose=True)
    print(f"\nFinal performance: {hfspmi.logger.true_performances[-1]:.4f}")
    return hfspmi


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all",
                        choices=["standard", "alternating", "softmax", "aggressive", "all"])
    args = parser.parse_args()
    
    results = {}
    
    if args.mode in ["standard", "all"]:
        results['standard'] = run_standard_mode()
    
    if args.mode in ["alternating", "all"]:
        results['alternating'] = run_alternating_mode()
    
    if args.mode in ["softmax", "all"]:
        results['softmax'] = run_softmax_mode()
    
    if args.mode in ["aggressive", "all"]:
        results['aggressive'] = run_aggressive_mode()
    
    if args.mode == "all":
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for name, hfspmi in results.items():
            final_perf = hfspmi.logger.true_performances[-1]
            n_policy_updates = sum(1 for a in hfspmi.logger.alphas if a > 0)
            n_model_updates = sum(1 for b in hfspmi.logger.betas if b > 0)
            print(f"  {name:15s}: perf={final_perf:.4f}, policy_updates={n_policy_updates}, model_updates={n_model_updates}")