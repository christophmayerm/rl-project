"""
Quick test to verify stochasticity is working properly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
from algorithm.spmi import SPMI
from algorithm.policy_chooser import GreedyPolicyChooser, SAPMIPolicyChooser
from algorithm.model_chooser import DoNotCreateTransitionsGreedyModelChooser
from algorithm.curriculum_scheduler import LinearCurriculumScheduler
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy
from envs.student_teacher import TeacherStudentEnv


def add_exploration_noise(policy, epsilon=0.05, seed=None):
    """Add small exploration noise to policy"""
    if seed is not None:
        np.random.seed(seed)
    
    # Get policy representation
    policy_dict = policy.get_rep()
    
    # Check if it's already a dict of arrays or dict of dicts
    states = sorted(policy_dict.keys())
    first_value = policy_dict[states[0]]
    
    # Case 1: dict maps state -> array of action probabilities
    if isinstance(first_value, np.ndarray):
        nS = len(states)
        nA = len(first_value)
        
        # Build policy matrix
        policy_matrix = np.zeros((nS, nA))
        for s_idx, state in enumerate(states):
            policy_matrix[s_idx, :] = policy_dict[state]
        
        # Add noise
        noise = np.random.dirichlet(np.ones(nA) * 10, size=nS)
        noisy_policy = (1 - epsilon) * policy_matrix + epsilon * noise
        noisy_policy = noisy_policy / noisy_policy.sum(axis=1, keepdims=True)
        
        # Convert back to dict format
        noisy_dict = {}
        for s_idx, state in enumerate(states):
            noisy_dict[state] = noisy_policy[s_idx, :]
        
        return TabularPolicy(noisy_dict, nS, nA)
    
    # Case 2: dict maps state -> dict of action -> probability
    else:
        actions = sorted(first_value.keys())
        nS = len(states)
        nA = len(actions)
        
        # Build policy matrix
        policy_matrix = np.zeros((nS, nA))
        for s_idx, state in enumerate(states):
            for a_idx, action in enumerate(actions):
                policy_matrix[s_idx, a_idx] = policy_dict[state].get(action, 0.0)
        
        # Add noise
        noise = np.random.dirichlet(np.ones(nA) * 10, size=nS)
        noisy_policy = (1 - epsilon) * policy_matrix + epsilon * noise
        noisy_policy = noisy_policy / noisy_policy.sum(axis=1, keepdims=True)
        
        # Convert back to dict format
        noisy_dict = {}
        for s_idx, state in enumerate(states):
            noisy_dict[state] = {}
            for a_idx, action in enumerate(actions):
                noisy_dict[state][action] = noisy_policy[s_idx, a_idx]
        
        return TabularPolicy(noisy_dict, nS, nA)


def test_without_noise():
    """Test without noise - should be deterministic"""
    print("="*60)
    print("TEST 1: WITHOUT NOISE (should be deterministic)")
    print("="*60)
    
    mdp = TeacherStudentEnv(
        n_literals=2, max_value=1, max_update=1,
        max_literals_in_examples=2, horizon=10
    )
    
    uniform_policy = UniformPolicy(mdp)
    original_model = copy.deepcopy(mdp.P)
    initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
    initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)
    
    curriculum = LinearCurriculumScheduler(50, 300, 0.0, 0.5)
    
    results = []
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        mdp.set_model(copy.deepcopy(original_model))
        
        base_policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
        policy_chooser = SAPMIPolicyChooser(
            nS=mdp.nS, nA=mdp.nA,
            base_chooser=base_policy_chooser,
            curriculum_scheduler=copy.deepcopy(curriculum),
            robustness_temperature=0.3,
            entropy_bonus=0.1
        )
        
        model_chooser = DoNotCreateTransitionsGreedyModelChooser(original_model, mdp.nS, mdp.nA)
        
        spmi = SPMI(mdp, eps=0.0, policy_chooser=policy_chooser,
                   model_chooser=model_chooser, max_iter=300, persistent=True)
        
        _, _ = spmi.spmi(
            copy.deepcopy(initial_policy),
            copy.deepcopy(initial_model)
        )
        
        perf = spmi.logger.evaluations[-1]
        results.append(perf)
        print(f"Seed {seed}: {perf:.6f}")
    
    print(f"\nMean: {np.mean(results):.6f}")
    print(f"Std:  {np.std(results):.6f}")
    print(f"Range: [{np.min(results):.6f}, {np.max(results):.6f}]")
    
    if np.std(results) < 1e-10:
        print("✓ Deterministic as expected (std ≈ 0)")
    else:
        print("⚠ Unexpected variance without noise!")
    
    return results


def test_with_noise():
    """Test with noise - should have variance"""
    print("\n" + "="*60)
    print("TEST 2: WITH NOISE (should have variance)")
    print("="*60)
    
    mdp = TeacherStudentEnv(
        n_literals=2, max_value=1, max_update=1,
        max_literals_in_examples=2, horizon=10
    )
    
    uniform_policy = UniformPolicy(mdp)
    original_model = copy.deepcopy(mdp.P)
    initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
    initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)
    
    curriculum = LinearCurriculumScheduler(50, 300, 0.0, 0.5)
    
    results = []
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        mdp.set_model(copy.deepcopy(original_model))
        
        # Add noise to initial policy
        noisy_policy = add_exploration_noise(initial_policy, epsilon=0.05, seed=seed)
        
        base_policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
        policy_chooser = SAPMIPolicyChooser(
            nS=mdp.nS, nA=mdp.nA,
            base_chooser=base_policy_chooser,
            curriculum_scheduler=copy.deepcopy(curriculum),
            robustness_temperature=0.3,
            entropy_bonus=0.1
        )
        
        model_chooser = DoNotCreateTransitionsGreedyModelChooser(original_model, mdp.nS, mdp.nA)
        
        spmi = SPMI(mdp, eps=0.0, policy_chooser=policy_chooser,
                   model_chooser=model_chooser, max_iter=300, persistent=True)
        
        _, _ = spmi.spmi(
            copy.deepcopy(noisy_policy),
            copy.deepcopy(initial_model)
        )
        
        perf = spmi.logger.evaluations[-1]
        results.append(perf)
        print(f"Seed {seed}: {perf:.6f}")
    
    print(f"\nMean: {np.mean(results):.6f}")
    print(f"Std:  {np.std(results):.6f}")
    print(f"Range: [{np.min(results):.6f}, {np.max(results):.6f}]")
    
    if np.std(results) > 1e-6:
        print("✓ Variance introduced successfully (std > 0)")
    else:
        print("⚠ No variance even with noise!")
    
    return results


def test_multiple_runs():
    """Test with more seeds to get better statistics"""
    print("\n" + "="*60)
    print("TEST 3: MULTIPLE RUNS (10 seeds)")
    print("="*60)
    
    mdp = TeacherStudentEnv(
        n_literals=2, max_value=1, max_update=1,
        max_literals_in_examples=2, horizon=10
    )
    
    uniform_policy = UniformPolicy(mdp)
    original_model = copy.deepcopy(mdp.P)
    initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
    initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)
    
    curriculum = LinearCurriculumScheduler(50, 300, 0.0, 0.5)
    
    results = []
    seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
    
    for i, seed in enumerate(seeds, 1):
        np.random.seed(seed)
        mdp.set_model(copy.deepcopy(original_model))
        
        noisy_policy = add_exploration_noise(initial_policy, epsilon=0.05, seed=seed)
        
        base_policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
        policy_chooser = SAPMIPolicyChooser(
            nS=mdp.nS, nA=mdp.nA,
            base_chooser=base_policy_chooser,
            curriculum_scheduler=copy.deepcopy(curriculum),
            robustness_temperature=0.3,
            entropy_bonus=0.1
        )
        
        model_chooser = DoNotCreateTransitionsGreedyModelChooser(original_model, mdp.nS, mdp.nA)
        
        spmi = SPMI(mdp, eps=0.0, policy_chooser=policy_chooser,
                   model_chooser=model_chooser, max_iter=300, persistent=True)
        
        _, _ = spmi.spmi(
            copy.deepcopy(noisy_policy),
            copy.deepcopy(initial_model)
        )
        
        perf = spmi.logger.evaluations[-1]
        results.append(perf)
        print(f"Run {i:2d} (seed {seed}): {perf:.6f}")
    
    print(f"\n{'='*40}")
    print(f"Mean:     {np.mean(results):.6f}")
    print(f"Std:      {np.std(results):.6f}")
    print(f"Median:   {np.median(results):.6f}")
    print(f"Range:    [{np.min(results):.6f}, {np.max(results):.6f}]")
    print(f"Variance: {np.var(results):.9f}")
    print(f"{'='*40}")
    
    # Check coefficient of variation
    cv = np.std(results) / np.mean(results) * 100
    print(f"Coefficient of Variation: {cv:.3f}%")
    
    if cv > 0.1:
        print("✓ Good variance for statistical analysis")
    else:
        print("⚠ Low variance - consider increasing noise epsilon")
    
    return results


def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# STOCHASTICITY VERIFICATION TESTS")
    print("#"*60)
    
    # Test 1: Without noise (deterministic)
    results_no_noise = test_without_noise()
    
    # Test 2: With noise (stochastic)
    results_with_noise = test_with_noise()
    
    # Test 3: Multiple runs
    results_multiple = test_multiple_runs()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Without noise - Std: {np.std(results_no_noise):.6f} (should be ~0)")
    print(f"With noise    - Std: {np.std(results_with_noise):.6f} (should be >0)")
    print(f"Multiple runs - Std: {np.std(results_multiple):.6f}")
    
    if np.std(results_no_noise) < 1e-10 and np.std(results_with_noise) > 1e-6:
        print("\n✓ All tests passed! Stochasticity is working correctly.")
    else:
        print("\n⚠ Some tests failed - check implementation.")


if __name__ == '__main__':
    main()