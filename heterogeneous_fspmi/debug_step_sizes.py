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


def run_debug():
    """Run a short experiment with detailed debug output."""
    print("=" * 70)
    print("DEBUG: Step Size Calculation Analysis")
    print("=" * 70)
    
    # Use shaped rewards
    shaped_reward = [1.0, -0.1, -0.05, 0.01, 0.02]
    
    # Create config with just 2 iterations for debugging
    config = HeterogeneousConfig(
        variants=[
            EnvironmentVariant(0, "T1", k=0.3, reward_weight=shaped_reward,
                             description="k=0.3"),
            EnvironmentVariant(1, "T1", k=0.5, reward_weight=shaped_reward,
                             description="k=0.5"),
            EnvironmentVariant(2, "T1", k=0.7, reward_weight=shaped_reward,
                             description="k=0.7"),
        ],
        episodes_per_agent=200,
        n_iterations=3,  # Just 3 iterations for debugging
        use_parallel=True
    )
    
    hfspmi = HeterogeneousFSPMI(config)
    
    print(f"\nEnvironment info:")
    print(f"  nS = {hfspmi.nS}")
    print(f"  nA = {hfspmi.nA}")
    print(f"  gamma = {hfspmi.gamma}")
    print(f"  horizon = {hfspmi.horizon}")
    print(f"  delta_q = {hfspmi.delta_q:.4f}")
    print()
    
    initial_policy = create_initial_policy(hfspmi.nS, hfspmi.nA)
    initial_model = create_initial_model(hfspmi.ref_env)
    
    # Run with verbose
    final_policy, final_model = hfspmi.run(
        initial_policy, initial_model, verbose=True
    )
    
    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)
    print("""
The key issue with α* = 0:

alpha0 = ((1 - gamma) * p_adv) / (delta_q * gamma * p_dist_sup * p_dist_mean)

If p_dist_mean is VERY SMALL (because d_mu is sparse), then alpha0 becomes VERY LARGE
and gets clipped to 1.0. But then the bound at (1.0, 0) might be NEGATIVE because
the penalty term dominates.

Alternatively, if p_adv is small relative to the distances, alpha0 is small.

Check the ratio: p_adv / (p_dist_sup * p_dist_mean)
- If small: advantage doesn't justify large step
- If large: step size should be large

The fundamental issue is that MC estimates give:
1. Noisy Q → noisy greedy policy → potentially wrong direction
2. Sparse d_mu → p_dist_mean is small but not zero
3. Overconfident greedy policy → large p_dist_sup
""")


if __name__ == "__main__":
    run_debug()