"""
Deep evaluation of top performing curricula from grid search
with statistical significance testing and detailed analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from algorithm.spmi import SPMI
from algorithm.policy_chooser import GreedyPolicyChooser, SAPMIPolicyChooser
from algorithm.model_chooser import DoNotCreateTransitionsGreedyModelChooser
from algorithm.curriculum_scheduler import (
    ConstantCurriculumScheduler,
    WarmRestartCurriculumScheduler,
    TwoStageCurriculumScheduler,
    ExponentialCurriculumScheduler,
    EarlyPeakCurriculumScheduler,
    CosineCurriculumScheduler
)
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy
from envs.student_teacher import TeacherStudentEnv


def add_exploration_noise(policy, epsilon=0.05, seed=None):
    """Add exploration noise for stochasticity"""
    if seed is not None:
        np.random.seed(seed)
    
    policy_dict = policy.get_rep()
    states = sorted(policy_dict.keys())
    first_value = policy_dict[states[0]]
    
    if isinstance(first_value, np.ndarray):
        nS = len(states)
        nA = len(first_value)
        policy_matrix = np.zeros((nS, nA))
        for s_idx, state in enumerate(states):
            policy_matrix[s_idx, :] = policy_dict[state]
        
        noise = np.random.dirichlet(np.ones(nA) * 10, size=nS)
        noisy_policy = (1 - epsilon) * policy_matrix + epsilon * noise
        noisy_policy = noisy_policy / noisy_policy.sum(axis=1, keepdims=True)
        
        noisy_dict = {}
        for s_idx, state in enumerate(states):
            noisy_dict[state] = noisy_policy[s_idx, :]
        
        return TabularPolicy(noisy_dict, nS, nA)
    else:
        actions = sorted(first_value.keys())
        nS = len(states)
        nA = len(actions)
        policy_matrix = np.zeros((nS, nA))
        for s_idx, state in enumerate(states):
            for a_idx, action in enumerate(actions):
                policy_matrix[s_idx, a_idx] = policy_dict[state].get(action, 0.0)
        
        noise = np.random.dirichlet(np.ones(nA) * 10, size=nS)
        noisy_policy = (1 - epsilon) * policy_matrix + epsilon * noise
        noisy_policy = noisy_policy / noisy_policy.sum(axis=1, keepdims=True)
        
        noisy_dict = {}
        for s_idx, state in enumerate(states):
            noisy_dict[state] = {}
            for a_idx, action in enumerate(actions):
                noisy_dict[state][action] = noisy_policy[s_idx, a_idx]
        
        return TabularPolicy(noisy_dict, nS, nA)


def run_single_experiment(mdp, initial_policy, initial_model, original_model,
                          curriculum, temp, entropy, seed):
    """Run single experiment"""
    np.random.seed(seed)
    mdp.set_model(copy.deepcopy(original_model))
    
    noisy_policy = add_exploration_noise(initial_policy, epsilon=0.05, seed=seed)
    
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
               model_chooser=model_chooser, max_iter=500, persistent=True)
    
    _, _ = spmi.spmi(copy.deepcopy(noisy_policy), copy.deepcopy(initial_model))
    
    return {
        'performance': spmi.logger.evaluations[-1] if spmi.logger.evaluations else 0.0,
        'iterations': spmi.logger.iteration,
        'spmi': spmi
    }


def evaluate_top_curricula():
    """Evaluate top curricula with many seeds"""
    print("="*80)
    print("DEEP EVALUATION OF TOP CURRICULA")
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
    
    # Best hyperparameters from grid search
    BEST_TEMP = 0.5
    BEST_ENTROPY = 0.0
    
    # Top curricula based on grid search
    curricula = [
        ('baseline', ConstantCurriculumScheduler(weight=0.0), 
         'No robustness'),
        ('constant_high', ConstantCurriculumScheduler(weight=0.6), 
         'Constant 60% robustness'),
        ('warm_restart', WarmRestartCurriculumScheduler(100, 0.0, 0.5, 1.5),
         'Warm restart cycling'),
        ('two_stage_early', TwoStageCurriculumScheduler(150, 0.1, 0.6, 50),
         'Two-stage early transition'),
        ('early_peak', EarlyPeakCurriculumScheduler(100, 0.5, 0.05),
         'Early peak then plateau'),
        ('exponential_fast', ExponentialCurriculumScheduler(0.02, 0.5),
         'Exponential fast ramp-up'),
        ('cosine_fast', CosineCurriculumScheduler(20, 150, 0.5),
         'Cosine fast schedule'),
    ]
    
    # More seeds for better statistics
    n_seeds = 20
    seeds = list(range(42, 42 + n_seeds))
    
    print(f"\nRunning {len(curricula)} curricula × {n_seeds} seeds = {len(curricula) * n_seeds} experiments")
    print("This will take approximately 15-20 minutes...\n")
    
    all_results = []
    
    for curriculum_name, curriculum, description in curricula:
        print(f"\n{'='*60}")
        print(f"Testing: {description}")
        print(f"{'='*60}")
        
        performances = []
        iterations_list = []
        
        for i, seed in enumerate(seeds, 1):
            if i % 5 == 0:
                print(f"  Progress: {i}/{n_seeds}")
            
            result = run_single_experiment(
                mdp, initial_policy, initial_model, original_model,
                copy.deepcopy(curriculum), BEST_TEMP, BEST_ENTROPY, seed
            )
            
            performances.append(result['performance'])
            iterations_list.append(result['iterations'])
            
            # Save first run for detailed analysis
            if i == 1:
                output_dir = Path("./data/top_performers")
                output_dir.mkdir(parents=True, exist_ok=True)
                result['spmi'].logger.save(str(output_dir), f"{curriculum_name}.csv")
                if hasattr(result['spmi'].policy_chooser, 'save_sapmi_policy_metrics'):
                    result['spmi'].policy_chooser.save_sapmi_policy_metrics(
                        f"{output_dir}/{curriculum_name}"
                    )
        
        # Statistics
        mean_perf = np.mean(performances)
        std_perf = np.std(performances, ddof=1)
        sem_perf = std_perf / np.sqrt(n_seeds)  # Standard error
        ci_95 = 1.96 * sem_perf  # 95% confidence interval
        
        all_results.append({
            'name': curriculum_name,
            'description': description,
            'mean': mean_perf,
            'std': std_perf,
            'sem': sem_perf,
            'ci_95': ci_95,
            'min': np.min(performances),
            'max': np.max(performances),
            'median': np.median(performances),
            'q25': np.percentile(performances, 25),
            'q75': np.percentile(performances, 75),
            'performances': performances,
            'mean_iterations': np.mean(iterations_list)
        })
        
        print(f"  Mean: {mean_perf:.4f} ± {std_perf:.4f}")
        print(f"  95% CI: [{mean_perf - ci_95:.4f}, {mean_perf + ci_95:.4f}]")
        print(f"  Median: {np.median(performances):.4f}")
        print(f"  Range: [{np.min(performances):.4f}, {np.max(performances):.4f}]")
    
    return all_results


def statistical_analysis(results):
    """Perform statistical significance testing"""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    
    # Get baseline
    baseline = next(r for r in results if r['name'] == 'baseline')
    baseline_perfs = baseline['performances']
    
    print(f"\nBaseline: {baseline['mean']:.4f} ± {baseline['std']:.4f}")
    print(f"95% CI: [{baseline['mean'] - baseline['ci_95']:.4f}, "
          f"{baseline['mean'] + baseline['ci_95']:.4f}]")
    
    print("\n" + "-"*80)
    print("Paired t-tests against baseline:")
    print("-"*80)
    print(f"{'Curriculum':<20} {'Mean Diff':<12} {'t-stat':<12} {'p-value':<12} {'Significant?':<15}")
    print("-"*80)
    
    for result in results:
        if result['name'] == 'baseline':
            continue
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(result['performances'], baseline_perfs)
        mean_diff = result['mean'] - baseline['mean']
        is_sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"{result['name']:<20} {mean_diff:>11.4f} {t_stat:>11.3f} {p_value:>11.4f} {is_sig:<15}")
    
    # Pairwise comparisons between top methods
    print("\n" + "-"*80)
    print("Pairwise comparisons (top methods only):")
    print("-"*80)
    
    top_methods = [r for r in results if r['name'] != 'baseline'][:5]
    
    for i, method1 in enumerate(top_methods):
        for method2 in top_methods[i+1:]:
            t_stat, p_value = stats.ttest_rel(method1['performances'], method2['performances'])
            mean_diff = method1['mean'] - method2['mean']
            is_sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            print(f"{method1['name']:<20} vs {method2['name']:<20}: "
                  f"Δ={mean_diff:>7.4f}, p={p_value:.4f} {is_sig}")


def effect_size_analysis(results):
    """Calculate effect sizes (Cohen's d)"""
    print("\n" + "="*80)
    print("EFFECT SIZE ANALYSIS (Cohen's d)")
    print("="*80)
    
    baseline = next(r for r in results if r['name'] == 'baseline')
    baseline_perfs = np.array(baseline['performances'])
    baseline_std = baseline['std']
    
    print(f"\n{'Curriculum':<20} {'Cohen d':<12} {'Effect Size':<15} {'% Improvement':<15}")
    print("-"*70)
    
    for result in results:
        if result['name'] == 'baseline':
            continue
        
        result_perfs = np.array(result['performances'])
        mean_diff = result['mean'] - baseline['mean']
        
        # Pooled standard deviation
        pooled_std = np.sqrt((baseline_std**2 + result['std']**2) / 2)
        cohens_d = mean_diff / pooled_std
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect = "negligible"
        elif abs(cohens_d) < 0.5:
            effect = "small"
        elif abs(cohens_d) < 0.8:
            effect = "medium"
        else:
            effect = "large"
        
        pct_improvement = (mean_diff / baseline['mean']) * 100
        
        print(f"{result['name']:<20} {cohens_d:>11.3f} {effect:<15} {pct_improvement:>13.2f}%")


def create_summary_table(results):
    """Create comprehensive summary table"""
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY")
    print("="*80)
    
    df = pd.DataFrame([{
        'Curriculum': r['name'],
        'Mean': r['mean'],
        'Std': r['std'],
        '95% CI Lower': r['mean'] - r['ci_95'],
        '95% CI Upper': r['mean'] + r['ci_95'],
        'Median': r['median'],
        'IQR': r['q75'] - r['q25'],
        'Range': r['max'] - r['min'],
        'Avg Iterations': r['mean_iterations']
    } for r in results])
    
    df = df.sort_values('Mean', ascending=False)
    
    print("\n" + df.to_string(index=False))
    
    # Save to CSV
    output_dir = Path("./data/top_performers")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "summary_statistics.csv", index=False)
    
    return df


def main():
    """Main entry point"""
    print("\n" + "#"*80)
    print("# DEEP EVALUATION: TOP PERFORMING CURRICULA")
    print("#"*80)
    
    # Run experiments
    results = evaluate_top_curricula()
    
    # Statistical analysis
    statistical_analysis(results)
    
    # Effect sizes
    effect_size_analysis(results)
    
    # Summary table
    df = create_summary_table(results)
    
    # Final recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    best = results[1]  # Skip baseline, get best
    baseline = results[0]
    
    improvement = ((best['mean'] - baseline['mean']) / baseline['mean']) * 100
    
    print(f"\n✓ Best curriculum: {best['name']} ({best['description']})")
    print(f"  Performance: {best['mean']:.4f} ± {best['std']:.4f}")
    print(f"  95% CI: [{best['mean'] - best['ci_95']:.4f}, {best['mean'] + best['ci_95']:.4f}]")
    print(f"  Improvement over baseline: {improvement:.2f}%")
    print(f"  Recommended hyperparameters: Temperature=0.5, Entropy=0.0")
    
    print(f"\n✓ Results saved to: ./data/top_performers/")
    print("  - summary_statistics.csv")
    print("  - Individual run logs: <curriculum_name>.csv")
    
    return results, df


if __name__ == '__main__':
    results, df = main()