"""
Test best curricula in Federated SPMI setting
Compare centralized vs federated performance
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
from algorithm.curriculum_scheduler import (
    ConstantCurriculumScheduler,
    WarmRestartCurriculumScheduler,
    TwoStageCurriculumScheduler,
)
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy
from envs.student_teacher import TeacherStudentEnv
from federated import FSPMI


def run_centralized(mdp, initial_policy, initial_model, original_model,
                    curriculum, temp, entropy, max_iter=500):
    """Run centralized SPMI"""
    mdp.set_model(copy.deepcopy(original_model))
    
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
    
    return {
        'performance': spmi.logger.evaluations[-1] if spmi.logger.evaluations else 0.0,
        'iterations': spmi.logger.iteration,
        'spmi': spmi
    }


def run_federated(mdp, initial_policy, initial_model, original_model,
                  curriculum, temp, entropy, n_agents=4, 
                  episodes_per_round=100, max_rounds=500):
    """Run federated SPMI"""
    mdp.set_model(copy.deepcopy(original_model))
    
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
        verbose=False
    )
    
    _, _ = fspmi.run(copy.deepcopy(initial_policy), copy.deepcopy(initial_model))
    
    return {
        'performance': fspmi.logger.true_performances[-1] if fspmi.logger.true_performances else 0.0,
        'rounds': len(fspmi.logger.iterations),
        'total_samples': sum(fspmi.logger.total_samples),
        'fspmi': fspmi
    }


def main():
    """Compare centralized vs federated"""
    print("="*80)
    print("FEDERATED TESTING: BEST CURRICULA")
    print("="*80)
    
    # Setup
    mdp = TeacherStudentEnv(
        n_literals=2, max_value=1, max_update=1,
        max_literals_in_examples=2, horizon=10
    )
    
    uniform_policy = UniformPolicy(mdp)
    original_model = copy.deepcopy(mdp.P)
    initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
    initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)
    
    # Best hyperparameters
    TEMP = 0.5
    ENTROPY = 0.0
    
    # Test configurations
    configs = [
        ('baseline', ConstantCurriculumScheduler(weight=0.0)),
        ('constant_high', ConstantCurriculumScheduler(weight=0.6)),
        ('warm_restart', WarmRestartCurriculumScheduler(100, 0.0, 0.5, 1.5)),
        ('two_stage', TwoStageCurriculumScheduler(150, 0.1, 0.6, 50)),
    ]
    
    # Federated settings to test
    n_agents_list = [2, 4, 8]
    
    output_dir = Path("./data/federated_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Test each configuration
    for name, curriculum in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        # Centralized
        print("  Running centralized...")
        cent_result = run_centralized(
            mdp, initial_policy, initial_model, original_model,
            copy.deepcopy(curriculum), TEMP, ENTROPY, max_iter=500
        )
        cent_result['spmi'].logger.save(str(output_dir), f"{name}_centralized.csv")
        
        print(f"    Performance: {cent_result['performance']:.4f}")
        print(f"    Iterations: {cent_result['iterations']}")
        
        # Federated with different agent counts
        for n_agents in n_agents_list:
            print(f"  Running federated (N={n_agents})...")
            fed_result = run_federated(
                mdp, initial_policy, initial_model, original_model,
                copy.deepcopy(curriculum), TEMP, ENTROPY,
                n_agents=n_agents, episodes_per_round=100, max_rounds=500
            )
            fed_result['fspmi'].logger.save(f"{output_dir}/{name}_federated_n{n_agents}.csv")
            
            print(f"    Performance: {fed_result['performance']:.4f}")
            print(f"    Rounds: {fed_result['rounds']}")
            print(f"    Total samples: {fed_result['total_samples']}")
            
            # Store result
            results.append({
                'curriculum': name,
                'setting': f'Federated (N={n_agents})',
                'n_agents': n_agents,
                'performance': fed_result['performance'],
                'rounds': fed_result['rounds'],
                'total_samples': fed_result['total_samples']
            })
        
        # Store centralized result
        results.append({
            'curriculum': name,
            'setting': 'Centralized',
            'n_agents': 1,
            'performance': cent_result['performance'],
            'rounds': cent_result['iterations'],
            'total_samples': cent_result['iterations']  # Approximate
        })
    
    # Analysis
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    # Pivot table
    pivot = df.pivot_table(
        values='performance',
        index='curriculum',
        columns='setting',
        aggfunc='first'
    )
    
    print("\nPerformance by Setting:")
    print(pivot.to_string())
    
    # Sample efficiency
    print("\n" + "-"*80)
    print("Sample Efficiency:")
    print("-"*80)
    for curriculum in df['curriculum'].unique():
        cent_perf = df[(df['curriculum']==curriculum) & (df['setting']=='Centralized')]['performance'].values[0]
        cent_samples = df[(df['curriculum']==curriculum) & (df['setting']=='Centralized')]['total_samples'].values[0]
        
        print(f"\n{curriculum}:")
        print(f"  Centralized: {cent_perf:.4f} ({cent_samples} samples)")
        
        for n_agents in n_agents_list:
            fed_row = df[(df['curriculum']==curriculum) & (df['n_agents']==n_agents)]
            if not fed_row.empty:
                fed_perf = fed_row['performance'].values[0]
                fed_samples = fed_row['total_samples'].values[0]
                efficiency = (fed_perf / cent_perf) / (fed_samples / cent_samples)
                print(f"  Federated (N={n_agents}): {fed_perf:.4f} ({fed_samples} samples, "
                      f"efficiency={efficiency:.3f})")
    
    # Save results
    df.to_csv(output_dir / "federated_comparison.csv", index=False)
    print(f"\n✓ Results saved to: {output_dir}")
    
    return df


if __name__ == '__main__':
    df = main()