"""
Hyperparameter Grid Search for SA-PMI
Tests different combinations of robustness_temperature and entropy_bonus
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
import pandas as pd
import time
import itertools
from pathlib import Path

from algorithm.spmi import SPMI
from algorithm.policy_chooser import GreedyPolicyChooser, SAPMIPolicyChooser
from algorithm.model_chooser import DoNotCreateTransitionsGreedyModelChooser
from algorithm.curriculum_scheduler import (
    LinearCurriculumScheduler,
    ExponentialCurriculumScheduler,
    ConstantCurriculumScheduler
)
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy
from envs.student_teacher import TeacherStudentEnv


def run_sapmi_single(mdp, initial_policy, initial_model, original_model,
                     curriculum_scheduler, robustness_temp, entropy_bonus,
                     max_iter=500, seed=None):
    """Run single SA-PMI experiment"""
    if seed is not None:
        np.random.seed(seed)
    
    mdp.set_model(copy.deepcopy(original_model))
    
    base_policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
    policy_chooser = SAPMIPolicyChooser(
        nS=mdp.nS,
        nA=mdp.nA,
        base_chooser=base_policy_chooser,
        curriculum_scheduler=curriculum_scheduler,
        robustness_temperature=robustness_temp,
        entropy_bonus=entropy_bonus
    )
    
    model_chooser = DoNotCreateTransitionsGreedyModelChooser(original_model, mdp.nS, mdp.nA)
    
    spmi = SPMI(
        mdp, eps=0.0,
        policy_chooser=policy_chooser,
        model_chooser=model_chooser,
        max_iter=max_iter,
        persistent=True
    )
    
    start_time = time.time()
    final_policy, final_model = spmi.spmi(
        copy.deepcopy(initial_policy),
        copy.deepcopy(initial_model)
    )
    elapsed = time.time() - start_time
    
    final_perf = spmi.logger.evaluations[-1] if spmi.logger.evaluations else 0.0
    
    return {
        'final_performance': final_perf,
        'iterations': spmi.logger.iteration,
        'time': elapsed,
        'logger': spmi.logger
    }


def grid_search():
    """Run grid search over hyperparameters"""
    print("="*80)
    print("SA-PMI Hyperparameter Grid Search")
    print("="*80)
    
    # Setup environment
    mdp = TeacherStudentEnv(
        n_literals=2, max_value=1, max_update=1,
        max_literals_in_examples=2, horizon=10
    )
    
    uniform_policy = UniformPolicy(mdp)
    original_model = copy.deepcopy(mdp.P)
    initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
    initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)
    
    # Hyperparameter grid
    temperature_grid = [0.1, 0.3, 0.5, 0.7, 1.0]
    entropy_grid = [0.0, 0.05, 0.1, 0.2, 0.3]
    curriculum_grid = [
        ('constant_medium', ConstantCurriculumScheduler(weight=0.4)),
        ('linear', LinearCurriculumScheduler(50, 300, 0.0, 0.5)),
        ('exponential', ExponentialCurriculumScheduler(0.01, 0.5))
    ]
    
    n_seeds = 3
    seeds = [42, 123, 456]
    
    results = []
    total_runs = len(temperature_grid) * len(entropy_grid) * len(curriculum_grid) * n_seeds
    run_count = 0
    
    for curriculum_name, curriculum_scheduler in curriculum_grid:
        for temp in temperature_grid:
            for entropy in entropy_grid:
                print(f"\n{'='*60}")
                print(f"Curriculum: {curriculum_name}, Temp: {temp}, Entropy: {entropy}")
                print(f"{'='*60}")
                
                seed_results = []
                for seed in seeds:
                    run_count += 1
                    print(f"Run {run_count}/{total_runs} (seed={seed})")
                    
                    result = run_sapmi_single(
                        mdp, initial_policy, initial_model, original_model,
                        copy.deepcopy(curriculum_scheduler),
                        temp, entropy,
                        max_iter=500,
                        seed=seed
                    )
                    seed_results.append(result['final_performance'])
                
                mean_perf = np.mean(seed_results)
                std_perf = np.std(seed_results)
                
                results.append({
                    'curriculum': curriculum_name,
                    'temperature': temp,
                    'entropy_bonus': entropy,
                    'mean_performance': mean_perf,
                    'std_performance': std_perf,
                    'seed_performances': seed_results
                })
                
                print(f"Mean Performance: {mean_perf:.4f} ± {std_perf:.4f}")
    
    # Save results
    output_dir = Path("./data/hyperparam_grid_search")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "grid_search_results.csv", index=False)
    
    # Find best configurations
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS")
    print("="*80)
    
    df_sorted = df.sort_values('mean_performance', ascending=False)
    print(df_sorted[['curriculum', 'temperature', 'entropy_bonus', 'mean_performance', 'std_performance']].head(10))
    
    # Best per curriculum
    print("\n" + "="*80)
    print("BEST CONFIGURATION PER CURRICULUM")
    print("="*80)
    for curriculum in df['curriculum'].unique():
        best = df[df['curriculum'] == curriculum].nlargest(1, 'mean_performance').iloc[0]
        print(f"\n{curriculum}:")
        print(f"  Temperature: {best['temperature']}")
        print(f"  Entropy: {best['entropy_bonus']}")
        print(f"  Performance: {best['mean_performance']:.4f} ± {best['std_performance']:.4f}")
    
    return df


if __name__ == '__main__':
    results_df = grid_search()