"""
Debug federated SPMI to understand the sample efficiency issue
and ensure fair comparison with centralized version
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
import pandas as pd
from pathlib import Path

from algorithm.spmi import SPMI
from algorithm.policy_chooser import GreedyPolicyChooser, SAPMIPolicyChooser
from algorithm.model_chooser import DoNotCreateTransitionsGreedyModelChooser
from algorithm.curriculum_scheduler import ConstantCurriculumScheduler
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy
from envs.student_teacher import TeacherStudentEnv
from federated import FSPMI


def detailed_centralized_run(mdp, initial_policy, initial_model, original_model, 
                             weight=0.67, temp=0.5, entropy=0.0, max_iter=500):
    """Run centralized with detailed logging"""
    print("\n" + "="*60)
    print("CENTRALIZED SPMI (Detailed)")
    print("="*60)
    
    mdp.set_model(copy.deepcopy(original_model))
    
    curriculum = ConstantCurriculumScheduler(weight=weight)
    base_policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
    policy_chooser = SAPMIPolicyChooser(
        nS=mdp.nS, nA=mdp.nA,
        base_chooser=base_policy_chooser,
        curriculum_scheduler=curriculum,
        robustness_temperature=temp,
        entropy_bonus=entropy
    )
    
    model_chooser = DoNotCreateTransitionsGreedyModelChooser(original_model, mdp.nS, mdp.nA)
    
    spmi = SPMI(mdp, eps=0.0, policy_chooser=policy_chooser,
               model_chooser=model_chooser, max_iter=max_iter, persistent=True)
    
    _, _ = spmi.spmi(copy.deepcopy(initial_policy), copy.deepcopy(initial_model))
    
    print(f"Iterations completed: {spmi.logger.iteration}")
    print(f"Final performance: {spmi.logger.evaluations[-1]:.4f}")
    print(f"Performance history length: {len(spmi.logger.evaluations)}")
    
    return spmi


def detailed_federated_run(mdp, initial_policy, initial_model, original_model,
                          weight=0.67, temp=0.5, entropy=0.0,
                          n_agents=4, episodes_per_round=100, max_rounds=100):
    """Run federated with detailed logging"""
    print("\n" + "="*60)
    print(f"FEDERATED SPMI (N={n_agents}, episodes_per_round={episodes_per_round})")
    print("="*60)
    
    mdp.set_model(copy.deepcopy(original_model))
    
    curriculum = ConstantCurriculumScheduler(weight=weight)
    base_policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
    policy_chooser = SAPMIPolicyChooser(
        nS=mdp.nS, nA=mdp.nA,
        base_chooser=base_policy_chooser,
        curriculum_scheduler=curriculum,
        robustness_temperature=temp,
        entropy_bonus=entropy
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
        verbose=True  # Enable verbose to see what's happening
    )
    
    _, _ = fspmi.run(copy.deepcopy(initial_policy), copy.deepcopy(initial_model))
    
    print(f"\nRounds completed: {len(fspmi.logger.iterations)}")
    print(f"Final performance: {fspmi.logger.true_performances[-1]:.4f}")
    print(f"Total samples: {sum(fspmi.logger.total_samples)}")
    print(f"Samples per round: {fspmi.logger.total_samples[:5] if len(fspmi.logger.total_samples) >= 5 else fspmi.logger.total_samples}")
    
    # Check logger structure
    print(f"\nLogger attributes:")
    print(f"  iterations: {len(fspmi.logger.iterations)}")
    print(f"  true_performances: {len(fspmi.logger.true_performances)}")
    print(f"  total_samples: {len(fspmi.logger.total_samples)}")
    print(f"  avg_returns: {len(fspmi.logger.avg_returns) if hasattr(fspmi.logger, 'avg_returns') else 'N/A'}")
    
    return fspmi


def compare_convergence():
    """Compare convergence with same computational budget"""
    print("\n" + "#"*80)
    print("# FAIR COMPARISON: SAME COMPUTATIONAL BUDGET")
    print("#"*80)
    
    # Setup
    mdp = TeacherStudentEnv(
        n_literals=2, max_value=1, max_update=1,
        max_literals_in_examples=2, horizon=10
    )
    
    uniform_policy = UniformPolicy(mdp)
    original_model = copy.deepcopy(mdp.P)
    initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
    initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)
    
    WEIGHT = 0.67
    TEMP = 0.5
    ENTROPY = 0.0
    
    # Centralized: 500 iterations
    cent_spmi = detailed_centralized_run(
        mdp, initial_policy, initial_model, original_model,
        weight=WEIGHT, temp=TEMP, entropy=ENTROPY, max_iter=500
    )
    
    # Federated: Match computational budget
    # If centralized does 500 iterations, and we have N agents doing E episodes each:
    # Total episodes = rounds × N × E
    # To match: rounds × N × E ≈ 500
    
    configs = [
        {'n_agents': 2, 'episodes_per_round': 25, 'max_rounds': 10},   # 2×25×10 = 500
        {'n_agents': 4, 'episodes_per_round': 12, 'max_rounds': 10},   # 4×12×10 = 480
        {'n_agents': 4, 'episodes_per_round': 25, 'max_rounds': 5},    # 4×25×5 = 500
    ]
    
    results = []
    
    for config in configs:
        fed_spmi = detailed_federated_run(
            mdp, initial_policy, initial_model, original_model,
            weight=WEIGHT, temp=TEMP, entropy=ENTROPY, **config
        )
        
        results.append({
            'setting': f"Federated (N={config['n_agents']}, E={config['episodes_per_round']})",
            'n_agents': config['n_agents'],
            'episodes_per_round': config['episodes_per_round'],
            'rounds': len(fed_spmi.logger.iterations),
            'performance': fed_spmi.logger.true_performances[-1],
            'total_samples': sum(fed_spmi.logger.total_samples)
        })
    
    # Add centralized
    results.append({
        'setting': 'Centralized',
        'n_agents': 1,
        'episodes_per_round': 1,
        'rounds': cent_spmi.logger.iteration,
        'performance': cent_spmi.logger.evaluations[-1],
        'total_samples': cent_spmi.logger.iteration
    })
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON (MATCHED BUDGET)")
    print("="*80)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save
    output_dir = Path("./data/federated_debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "fair_comparison.csv", index=False)
    
    return df


def investigate_sample_counting():
    """Investigate why sample counts are so high"""
    print("\n" + "#"*80)
    print("# INVESTIGATING SAMPLE COUNTING")
    print("#"*80)
    
    mdp = TeacherStudentEnv(
        n_literals=2, max_value=1, max_update=1,
        max_literals_in_examples=2, horizon=10
    )
    
    uniform_policy = UniformPolicy(mdp)
    original_model = copy.deepcopy(mdp.P)
    initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
    initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)
    
    # Run very short federated experiment
    mdp.set_model(copy.deepcopy(original_model))
    
    curriculum = ConstantCurriculumScheduler(weight=0.67)
    base_policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
    policy_chooser = SAPMIPolicyChooser(
        nS=mdp.nS, nA=mdp.nA,
        base_chooser=base_policy_chooser,
        curriculum_scheduler=curriculum,
        robustness_temperature=0.5,
        entropy_bonus=0.0
    )
    
    model_chooser = DoNotCreateTransitionsGreedyModelChooser(original_model, mdp.nS, mdp.nA)
    
    print("\nRunning 3 rounds with N=2, episodes_per_round=5")
    print("Expected samples: 3 × 2 × 5 = 30")
    
    fspmi = FSPMI(
        conf_mdp=mdp,
        n_agents=2,
        episodes_per_round=5,
        eps=0.0,
        max_rounds=3,
        policy_chooser=policy_chooser,
        model_chooser=model_chooser,
        aggregation_method='weighted',
        persistent=True,
        verbose=True
    )
    
    _, _ = fspmi.run(copy.deepcopy(initial_policy), copy.deepcopy(initial_model))
    
    print(f"\nActual results:")
    print(f"  Rounds: {len(fspmi.logger.iterations)}")
    print(f"  Total samples reported: {sum(fspmi.logger.total_samples)}")
    print(f"  Samples per round: {fspmi.logger.total_samples}")
    print(f"  Expected vs Actual: 30 vs {sum(fspmi.logger.total_samples)}")
    
    if sum(fspmi.logger.total_samples) > 1000:
        print("\n⚠️  WARNING: Sample count is way too high!")
        print("     This suggests the logger is counting something else")
        print("     (possibly transitions, states visited, or cumulative count)")


def main():
    """Run all debugging"""
    
    # First, investigate the sample counting issue
    investigate_sample_counting()
    
    # Then do fair comparison
    df = compare_convergence()
    
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("\n1. Check the sample counting logic in FSPMI.logger")
    print("2. The high performance (0.51 vs 0.47) might be due to:")
    print("   - Actually running more iterations than intended")
    print("   - Different evaluation methodology")
    print("   - Bug in federated aggregation")
    print("\n3. For fair comparison, we need to ensure:")
    print("   - Same total number of environment interactions")
    print("   - Same evaluation protocol")
    print("   - Proper logging of actual samples used")
    
    return df


if __name__ == '__main__':
    df = main()