"""
Comprehensive Hyperparameter Grid Search for SA-PMI
Tests ALL curriculum schedules with added stochasticity
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
    ConstantCurriculumScheduler,
    LinearCurriculumScheduler,
    ExponentialCurriculumScheduler,
    CosineCurriculumScheduler,
    StepCurriculumScheduler,
    SigmoidCurriculumScheduler,
    PolynomialCurriculumScheduler,
    WarmRestartCurriculumScheduler,
    EarlyPeakCurriculumScheduler,
    TwoStageCurriculumScheduler
)
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy
from envs.student_teacher import TeacherStudentEnv


def add_exploration_noise(policy, epsilon=0.05, seed=None):
    """Add small exploration noise to policy to introduce stochasticity"""
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
        
        # Renormalize
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
        
        # Renormalize
        noisy_policy = noisy_policy / noisy_policy.sum(axis=1, keepdims=True)
        
        # Convert back to dict format
        noisy_dict = {}
        for s_idx, state in enumerate(states):
            noisy_dict[state] = {}
            for a_idx, action in enumerate(actions):
                noisy_dict[state][action] = noisy_policy[s_idx, a_idx]
        
        return TabularPolicy(noisy_dict, nS, nA)


def run_sapmi_single(mdp, initial_policy, initial_model, original_model,
                     curriculum_scheduler, robustness_temp, entropy_bonus,
                     max_iter=500, seed=None, add_noise=True):
    """Run single SA-PMI experiment with optional noise"""
    if seed is not None:
        np.random.seed(seed)
    
    mdp.set_model(copy.deepcopy(original_model))
    
    # Add noise to initial policy for stochasticity
    policy_to_use = initial_policy
    if add_noise:
        policy_to_use = add_exploration_noise(initial_policy, epsilon=0.05, seed=seed)
    
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
        copy.deepcopy(policy_to_use),
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


def test_all_schedulers():
    """Quick test to verify all schedulers work correctly"""
    print("="*80)
    print("TESTING ALL CURRICULUM SCHEDULERS")
    print("="*80)
    
    schedulers = [
        ('constant_zero', ConstantCurriculumScheduler(weight=0.0)),
        ('constant_low', ConstantCurriculumScheduler(weight=0.2)),
        ('constant_med', ConstantCurriculumScheduler(weight=0.4)),
        ('constant_high', ConstantCurriculumScheduler(weight=0.6)),
        ('linear_fast', LinearCurriculumScheduler(20, 150, 0.0, 0.5)),
        ('linear_med', LinearCurriculumScheduler(50, 300, 0.0, 0.5)),
        ('linear_slow', LinearCurriculumScheduler(100, 400, 0.0, 0.5)),
        ('exponential_fast', ExponentialCurriculumScheduler(0.02, 0.5)),
        ('exponential_med', ExponentialCurriculumScheduler(0.01, 0.5)),
        ('exponential_slow', ExponentialCurriculumScheduler(0.005, 0.5)),
        ('cosine_fast', CosineCurriculumScheduler(20, 150, 0.5)),
        ('cosine_med', CosineCurriculumScheduler(50, 300, 0.5)),
        ('step', StepCurriculumScheduler([100, 300], [0.0, 0.3, 0.6])),
        ('sigmoid_early', SigmoidCurriculumScheduler(midpoint=100, steepness=0.03, max_weight=0.5)),
        ('sigmoid_late', SigmoidCurriculumScheduler(midpoint=200, steepness=0.02, max_weight=0.5)),
        ('poly_sqrt', PolynomialCurriculumScheduler(0, 300, 0.5, power=0.5)),
        ('poly_quad', PolynomialCurriculumScheduler(0, 300, 0.5, power=2.0)),
        ('poly_cubic', PolynomialCurriculumScheduler(0, 300, 0.5, power=3.0)),
        ('warm_restart', WarmRestartCurriculumScheduler(100, 0.0, 0.5, 1.5)),
        ('early_peak', EarlyPeakCurriculumScheduler(100, 0.5, 0.05)),
        ('two_stage_early', TwoStageCurriculumScheduler(150, 0.1, 0.6, 50)),
        ('two_stage_late', TwoStageCurriculumScheduler(250, 0.1, 0.6, 50)),
    ]
    
    print(f"\nTesting {len(schedulers)} schedulers...")
    print(f"{'Scheduler':<25} {'Weight@0':<12} {'Weight@100':<12} {'Weight@300':<12} {'Weight@500':<12}")
    print("-"*75)
    
    for name, scheduler in schedulers:
        try:
            w0 = scheduler.get_adversarial_weight(0)
            w100 = scheduler.get_adversarial_weight(100)
            w300 = scheduler.get_adversarial_weight(300)
            w500 = scheduler.get_adversarial_weight(500)
            print(f"{name:<25} {w0:<12.4f} {w100:<12.4f} {w300:<12.4f} {w500:<12.4f}")
        except Exception as e:
            print(f"{name:<25} ERROR: {e}")
            return False
    
    print("\n✓ All schedulers working correctly!")
    return True


def quick_grid_search():
    """Quick grid search on reduced parameter space"""
    print("\n" + "="*80)
    print("QUICK GRID SEARCH (Subset of Parameters)")
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
    
    # Reduced parameter grid for quick testing
    temperature_grid = [0.3, 0.5, 0.7]  # Reduced
    entropy_grid = [0.0, 0.1, 0.2]      # Reduced
    
    # Select representative curricula from each category
    curriculum_grid = [
        ('baseline', ConstantCurriculumScheduler(weight=0.0)),
        ('constant_med', ConstantCurriculumScheduler(weight=0.4)),
        ('linear_med', LinearCurriculumScheduler(50, 300, 0.0, 0.5)),
        ('exponential_med', ExponentialCurriculumScheduler(0.01, 0.5)),
        ('cosine_med', CosineCurriculumScheduler(50, 300, 0.5)),
        ('sigmoid_early', SigmoidCurriculumScheduler(100, 0.03, 0.5)),
        ('poly_quad', PolynomialCurriculumScheduler(0, 300, 0.5, 2.0)),
        ('two_stage_early', TwoStageCurriculumScheduler(150, 0.1, 0.6, 50)),
    ]
    
    n_seeds = 3
    seeds = [42, 123, 456]
    
    results = []
    total_runs = len(temperature_grid) * len(entropy_grid) * len(curriculum_grid) * n_seeds
    run_count = 0
    
    print(f"\nTotal configurations: {len(temperature_grid)} temps × {len(entropy_grid)} entropy × {len(curriculum_grid)} curricula = {len(temperature_grid) * len(entropy_grid) * len(curriculum_grid)}")
    print(f"Total runs (with {n_seeds} seeds): {total_runs}\n")
    
    for curriculum_name, curriculum_scheduler in curriculum_grid:
        for temp in temperature_grid:
            for entropy in entropy_grid:
                seed_results = []
                for seed_idx, seed in enumerate(seeds):
                    run_count += 1
                    
                    if run_count % 10 == 1:
                        print(f"Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.1f}%)")
                    
                    result = run_sapmi_single(
                        mdp, initial_policy, initial_model, original_model,
                        copy.deepcopy(curriculum_scheduler),
                        temp, entropy,
                        max_iter=500,
                        seed=seed,
                        add_noise=True  # Enable noise for variance
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
                    'min_performance': np.min(seed_results),
                    'max_performance': np.max(seed_results),
                    'seed_performances': seed_results
                })
    
    return results


def full_grid_search():
    """Full grid search with all schedulers"""
    print("\n" + "="*80)
    print("FULL COMPREHENSIVE GRID SEARCH")
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
    
    # Full parameter grid
    temperature_grid = [0.1, 0.3, 0.5, 0.7, 1.0]
    entropy_grid = [0.0, 0.05, 0.1, 0.2, 0.3]
    
    # ALL curricula
    curriculum_grid = [
        ('baseline', ConstantCurriculumScheduler(weight=0.0)),
        ('constant_low', ConstantCurriculumScheduler(weight=0.2)),
        ('constant_med', ConstantCurriculumScheduler(weight=0.4)),
        ('constant_high', ConstantCurriculumScheduler(weight=0.6)),
        ('linear_fast', LinearCurriculumScheduler(20, 150, 0.0, 0.5)),
        ('linear_med', LinearCurriculumScheduler(50, 300, 0.0, 0.5)),
        ('linear_slow', LinearCurriculumScheduler(100, 400, 0.0, 0.5)),
        ('exponential_fast', ExponentialCurriculumScheduler(0.02, 0.5)),
        ('exponential_med', ExponentialCurriculumScheduler(0.01, 0.5)),
        ('exponential_slow', ExponentialCurriculumScheduler(0.005, 0.5)),
        ('cosine_fast', CosineCurriculumScheduler(20, 150, 0.5)),
        ('cosine_med', CosineCurriculumScheduler(50, 300, 0.5)),
        ('step', StepCurriculumScheduler([100, 300], [0.0, 0.3, 0.6])),
        ('sigmoid_early', SigmoidCurriculumScheduler(100, 0.03, 0.5)),
        ('sigmoid_late', SigmoidCurriculumScheduler(200, 0.02, 0.5)),
        ('poly_sqrt', PolynomialCurriculumScheduler(0, 300, 0.5, 0.5)),
        ('poly_quad', PolynomialCurriculumScheduler(0, 300, 0.5, 2.0)),
        ('poly_cubic', PolynomialCurriculumScheduler(0, 300, 0.5, 3.0)),
        ('warm_restart', WarmRestartCurriculumScheduler(100, 0.0, 0.5, 1.5)),
        ('early_peak', EarlyPeakCurriculumScheduler(100, 0.5, 0.05)),
        ('two_stage_early', TwoStageCurriculumScheduler(150, 0.1, 0.6, 50)),
        ('two_stage_late', TwoStageCurriculumScheduler(250, 0.1, 0.6, 50)),
    ]
    
    n_seeds = 3
    seeds = [42, 123, 456]
    
    total_configs = len(temperature_grid) * len(entropy_grid) * len(curriculum_grid)
    total_runs = total_configs * n_seeds
    
    print(f"\nTotal configurations: {len(temperature_grid)} temps × {len(entropy_grid)} entropy × {len(curriculum_grid)} curricula = {total_configs}")
    print(f"Total runs (with {n_seeds} seeds): {total_runs}")
    print(f"Estimated time: ~{total_runs * 2 / 60:.1f} minutes\n")
    
    results = []
    run_count = 0
    start_time = time.time()
    
    for curriculum_name, curriculum_scheduler in curriculum_grid:
        for temp in temperature_grid:
            for entropy in entropy_grid:
                seed_results = []
                for seed in seeds:
                    run_count += 1
                    
                    if run_count % 20 == 1:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / run_count if run_count > 0 else 0
                        remaining = (total_runs - run_count) * avg_time
                        print(f"[{run_count}/{total_runs}] {100*run_count/total_runs:.1f}% | "
                              f"ETA: {remaining/60:.1f}min | {curriculum_name}")
                    
                    result = run_sapmi_single(
                        mdp, initial_policy, initial_model, original_model,
                        copy.deepcopy(curriculum_scheduler),
                        temp, entropy,
                        max_iter=500,
                        seed=seed,
                        add_noise=True
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
                    'min_performance': np.min(seed_results),
                    'max_performance': np.max(seed_results),
                    'seed_performances': seed_results
                })
    
    return results


def analyze_results(results, output_dir):
    """Comprehensive analysis of results"""
    df = pd.DataFrame(results)
    
    # Save raw results
    df.to_csv(output_dir / "grid_search_results.csv", index=False)
    
    print("\n" + "="*80)
    print("TOP 15 CONFIGURATIONS")
    print("="*80)
    df_sorted = df.sort_values('mean_performance', ascending=False)
    print(df_sorted[['curriculum', 'temperature', 'entropy_bonus', 
                     'mean_performance', 'std_performance']].head(15).to_string(index=False))
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION PER CURRICULUM")
    print("="*80)
    for curriculum in sorted(df['curriculum'].unique()):
        best = df[df['curriculum'] == curriculum].nlargest(1, 'mean_performance').iloc[0]
        print(f"\n{curriculum}:")
        print(f"  Temperature: {best['temperature']}")
        print(f"  Entropy: {best['entropy_bonus']}")
        print(f"  Performance: {best['mean_performance']:.4f} ± {best['std_performance']:.4f}")
        print(f"  Range: [{best['min_performance']:.4f}, {best['max_performance']:.4f}]")
    
    # Category analysis
    print("\n" + "="*80)
    print("CATEGORY ANALYSIS")
    print("="*80)
    
    categories = {
        'Constant': ['baseline', 'constant_low', 'constant_med', 'constant_high'],
        'Linear': ['linear_fast', 'linear_med', 'linear_slow'],
        'Exponential': ['exponential_fast', 'exponential_med', 'exponential_slow'],
        'Cosine': ['cosine_fast', 'cosine_med'],
        'Sigmoid': ['sigmoid_early', 'sigmoid_late'],
        'Polynomial': ['poly_sqrt', 'poly_quad', 'poly_cubic'],
        'Step': ['step'],
        'Two-Stage': ['two_stage_early', 'two_stage_late'],
        'Other': ['warm_restart', 'early_peak']
    }
    
    category_results = []
    for cat_name, members in categories.items():
        cat_df = df[df['curriculum'].isin(members)]
        if not cat_df.empty:
            best_config = cat_df.nlargest(1, 'mean_performance').iloc[0]
            avg_perf = cat_df['mean_performance'].mean()
            category_results.append({
                'Category': cat_name,
                'Best': best_config['curriculum'],
                'Best_Perf': best_config['mean_performance'],
                'Avg_Perf': avg_perf,
                'Count': len(cat_df)
            })
    
    cat_df_summary = pd.DataFrame(category_results).sort_values('Best_Perf', ascending=False)
    print(cat_df_summary.to_string(index=False))
    
    # Statistical significance test
    baseline_perfs = df[df['curriculum'] == 'baseline']['seed_performances'].iloc[0]
    baseline_mean = np.mean(baseline_perfs)
    
    print("\n" + "="*80)
    print("IMPROVEMENTS OVER BASELINE")
    print("="*80)
    print(f"Baseline performance: {baseline_mean:.4f}\n")
    
    improvements = []
    for _, row in df.iterrows():
        if row['curriculum'] != 'baseline':
            improvement = ((row['mean_performance'] - baseline_mean) / baseline_mean) * 100
            improvements.append({
                'curriculum': row['curriculum'],
                'temperature': row['temperature'],
                'entropy': row['entropy_bonus'],
                'improvement_pct': improvement,
                'mean_perf': row['mean_performance']
            })
    
    imp_df = pd.DataFrame(improvements).sort_values('improvement_pct', ascending=False)
    print("Top 10 improvements:")
    print(imp_df[['curriculum', 'temperature', 'entropy', 'improvement_pct', 'mean_perf']].head(10).to_string(index=False))
    
    # Check if std is actually non-zero now
    print("\n" + "="*80)
    print("STOCHASTICITY CHECK")
    print("="*80)
    print(f"Configs with std > 0: {(df['std_performance'] > 0).sum()} / {len(df)}")
    print(f"Configs with std = 0: {(df['std_performance'] == 0).sum()} / {len(df)}")
    print(f"Mean std across all configs: {df['std_performance'].mean():.6f}")
    
    return df


def main():
    """Main entry point"""
    output_dir = Path("./data/hyperparam_grid_search")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Test all schedulers
    if not test_all_schedulers():
        print("\n❌ Scheduler test failed! Fix errors before proceeding.")
        return
    
    # Step 2: Choose search mode
    print("\n" + "="*80)
    print("SELECT GRID SEARCH MODE")
    print("="*80)
    print("1. Quick search (8 curricula, 3×3 params, ~200 runs, ~7 min)")
    print("2. Full search (22 curricula, 5×5 params, ~1650 runs, ~55 min)")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip() or "1"
    
    if choice == "1":
        print("\n🚀 Running QUICK grid search...")
        results = quick_grid_search()
    else:
        print("\n🚀 Running FULL grid search...")
        results = full_grid_search()
    
    # Step 3: Analyze
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    df = analyze_results(results, output_dir)
    
    print(f"\n✓ Grid search completed!")
    print(f"Results saved to: {output_dir}")
    
    return df


if __name__ == '__main__':
    results_df = main()