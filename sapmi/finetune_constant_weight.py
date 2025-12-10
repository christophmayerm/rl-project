"""
Fine-tune the constant curriculum weight parameter
Grid search around weight=0.6 to find optimal value
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from algorithm.spmi import SPMI
from algorithm.policy_chooser import GreedyPolicyChooser, SAPMIPolicyChooser
from algorithm.model_chooser import DoNotCreateTransitionsGreedyModelChooser
from algorithm.curriculum_scheduler import ConstantCurriculumScheduler
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy
from envs.student_teacher import TeacherStudentEnv


def add_exploration_noise(policy, epsilon=0.05, seed=None):
    """Add exploration noise"""
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


def evaluate_weight(mdp, initial_policy, initial_model, original_model,
                    weight, temp, entropy, seeds):
    """Evaluate a single weight value"""
    performances = []
    
    for seed in seeds:
        np.random.seed(seed)
        mdp.set_model(copy.deepcopy(original_model))
        
        noisy_policy = add_exploration_noise(initial_policy, epsilon=0.05, seed=seed)
        
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
                   model_chooser=model_chooser, max_iter=500, persistent=True)
        
        _, _ = spmi.spmi(copy.deepcopy(noisy_policy), copy.deepcopy(initial_model))
        performances.append(spmi.logger.evaluations[-1] if spmi.logger.evaluations else 0.0)
    
    return {
        'weight': weight,
        'mean': np.mean(performances),
        'std': np.std(performances, ddof=1),
        'sem': np.std(performances, ddof=1) / np.sqrt(len(seeds)),
        'performances': performances
    }


def fine_grain_search():
    """Fine-grained search around weight=0.6"""
    print("="*80)
    print("FINE-TUNING CONSTANT CURRICULUM WEIGHT")
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
    
    # Fine-grained weight grid around 0.6
    weights = np.linspace(0.45, 0.75, 31)  # 0.45 to 0.75 in steps of 0.01
    
    # Use 10 seeds for each weight
    n_seeds = 10
    seeds = list(range(42, 42 + n_seeds))
    
    print(f"\nTesting {len(weights)} weight values × {n_seeds} seeds = {len(weights) * n_seeds} runs")
    print(f"Weight range: [{weights[0]:.2f}, {weights[-1]:.2f}]")
    print(f"Estimated time: ~{len(weights) * n_seeds * 2 / 60:.1f} minutes\n")
    
    results = []
    for i, weight in enumerate(weights, 1):
        if i % 5 == 1:
            print(f"Progress: {i}/{len(weights)} (weight={weight:.3f})")
        
        result = evaluate_weight(
            mdp, initial_policy, initial_model, original_model,
            weight, TEMP, ENTROPY, seeds
        )
        results.append(result)
    
    return results, weights


def plot_results(results, output_dir):
    """Create visualization of weight vs performance"""
    weights = [r['weight'] for r in results]
    means = [r['mean'] for r in results]
    stds = [r['std'] for r in results]
    sems = [r['sem'] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    # Main plot with error bars
    plt.subplot(1, 2, 1)
    plt.errorbar(weights, means, yerr=[1.96*sem for sem in sems], 
                 fmt='o-', capsize=3, capthick=1, markersize=4)
    plt.xlabel('Adversarial Weight', fontsize=12)
    plt.ylabel('Mean Performance', fontsize=12)
    plt.title('Constant Curriculum: Weight vs Performance', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Mark best
    best_idx = np.argmax(means)
    best_weight = weights[best_idx]
    best_mean = means[best_idx]
    plt.axvline(best_weight, color='r', linestyle='--', alpha=0.5, 
                label=f'Best: w={best_weight:.3f}')
    plt.plot(best_weight, best_mean, 'r*', markersize=15)
    plt.legend()
    
    # Zoom in around best
    plt.subplot(1, 2, 2)
    # Find indices within ±0.1 of best
    zoom_mask = np.abs(np.array(weights) - best_weight) <= 0.1
    zoom_weights = np.array(weights)[zoom_mask]
    zoom_means = np.array(means)[zoom_mask]
    zoom_sems = np.array(sems)[zoom_mask]
    
    plt.errorbar(zoom_weights, zoom_means, yerr=[1.96*sem for sem in zoom_sems],
                 fmt='o-', capsize=3, capthick=1, markersize=6)
    plt.xlabel('Adversarial Weight', fontsize=12)
    plt.ylabel('Mean Performance', fontsize=12)
    plt.title(f'Zoomed: w ∈ [{best_weight-0.1:.2f}, {best_weight+0.1:.2f}]', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(best_weight, color='r', linestyle='--', alpha=0.5)
    plt.plot(best_weight, best_mean, 'r*', markersize=15)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_optimization.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_dir / 'weight_optimization.png'}")
    
    return best_weight, best_mean


def analyze_results(results, output_dir):
    """Analyze and save results"""
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Find best
    means = [r['mean'] for r in results]
    best_idx = np.argmax(means)
    best = results[best_idx]
    
    print(f"\n✓ BEST WEIGHT: {best['weight']:.3f}")
    print(f"  Performance: {best['mean']:.4f} ± {best['std']:.4f}")
    print(f"  95% CI: [{best['mean'] - 1.96*best['sem']:.4f}, "
          f"{best['mean'] + 1.96*best['sem']:.4f}]")
    
    # Top 5
    sorted_results = sorted(results, key=lambda x: x['mean'], reverse=True)
    print("\nTop 5 weights:")
    print(f"{'Rank':<6} {'Weight':<10} {'Mean':<12} {'Std':<12} {'95% CI Width':<15}")
    print("-"*55)
    for i, r in enumerate(sorted_results[:5], 1):
        ci_width = 1.96 * r['sem'] * 2
        print(f"{i:<6} {r['weight']:<10.3f} {r['mean']:<12.4f} {r['std']:<12.4f} {ci_width:<15.4f}")
    
    # Check if there's a clear peak or plateau
    print("\n" + "-"*80)
    print("SENSITIVITY ANALYSIS")
    print("-"*80)
    
    # How much does performance change around best?
    nearby_mask = np.abs(np.array([r['weight'] for r in results]) - best['weight']) <= 0.05
    nearby_results = [r for r, mask in zip(results, nearby_mask) if mask]
    nearby_means = [r['mean'] for r in nearby_results]
    
    variation = (np.max(nearby_means) - np.min(nearby_means)) / best['mean'] * 100
    
    print(f"Performance variation within ±0.05 of best: {variation:.2f}%")
    if variation < 1.0:
        print("→ Relatively flat around optimum (robust to small changes)")
    else:
        print("→ Sensitive to weight changes (precise tuning important)")
    
    # Save detailed results
    df = pd.DataFrame(results)
    df = df.sort_values('mean', ascending=False)
    df.to_csv(output_dir / 'weight_fine_tuning.csv', index=False)
    print(f"\n✓ Detailed results saved: {output_dir / 'weight_fine_tuning.csv'}")
    
    return best


def main():
    """Main entry point"""
    print("\n" + "#"*80)
    print("# FINE-TUNING: CONSTANT CURRICULUM WEIGHT")
    print("#"*80)
    
    output_dir = Path("./data/weight_fine_tuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run fine-grained search
    results, weights = fine_grain_search()
    
    # Analyze
    best = analyze_results(results, output_dir)
    
    # Plot
    best_weight, best_mean = plot_results(results, output_dir)
    
    # Final recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    print(f"\n✓ Optimal constant curriculum weight: {best_weight:.3f}")
    print(f"  Expected performance: {best_mean:.4f}")
    print(f"  Recommended configuration:")
    print(f"    - ConstantCurriculumScheduler(weight={best_weight:.3f})")
    print(f"    - robustness_temperature=0.5")
    print(f"    - entropy_bonus=0.0")
    
    return results, best


if __name__ == '__main__':
    results, best = main()