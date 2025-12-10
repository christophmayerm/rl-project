"""
Create publication-quality visualizations for the report
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def plot_curriculum_progressions():
    """Plot how different curricula evolve over iterations"""
    from algorithm.curriculum_scheduler import (
        ConstantCurriculumScheduler,
        LinearCurriculumScheduler,
        ExponentialCurriculumScheduler,
        CosineCurriculumScheduler,
        SigmoidCurriculumScheduler,
        PolynomialCurriculumScheduler,
        TwoStageCurriculumScheduler,
        WarmRestartCurriculumScheduler,
        EarlyPeakCurriculumScheduler
    )
    
    iterations = np.arange(0, 501)
    
    curricula = [
        ('Baseline', ConstantCurriculumScheduler(0.0), 'gray', '--'),
        ('Constant (0.67)', ConstantCurriculumScheduler(0.67), 'red', '-'),
        ('Linear', LinearCurriculumScheduler(50, 300, 0.0, 0.5), 'blue', '-'),
        ('Exponential', ExponentialCurriculumScheduler(0.02, 0.5), 'green', '-'),
        ('Cosine', CosineCurriculumScheduler(20, 150, 0.5), 'orange', '-'),
        ('Two-Stage', TwoStageCurriculumScheduler(150, 0.1, 0.6, 50), 'purple', '-'),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full range
    for name, scheduler, color, style in curricula:
        weights = [scheduler.get_adversarial_weight(i) for i in iterations]
        ax1.plot(iterations, weights, label=name, color=color, linestyle=style, linewidth=2)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Adversarial Weight λ(t)')
    ax1.set_title('Curriculum Learning Schedules')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 0.75)
    
    # Zoomed first 200 iterations
    zoom_iterations = np.arange(0, 201)
    for name, scheduler, color, style in curricula:
        weights = [scheduler.get_adversarial_weight(i) for i in zoom_iterations]
        ax2.plot(zoom_iterations, weights, label=name, color=color, linestyle=style, linewidth=2)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Adversarial Weight λ(t)')
    ax2.set_title('Early Training Phase (Zoomed)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 0.75)
    
    plt.tight_layout()
    return fig


def plot_performance_comparison():
    """Bar plot comparing all methods"""
    # Data from your results
    methods = ['Baseline', 'Cosine\nFast', 'Exponential\nFast', 'Early\nPeak', 
               'Two-Stage\nEarly', 'Warm\nRestart', 'Constant\n(0.67)']
    performances = [0.4568, 0.4630, 0.4634, 0.4640, 0.4642, 0.4643, 0.4675]
    stds = [0.0032, 0.0032, 0.0032, 0.0032, 0.0032, 0.0032, 0.0032]
    
    colors = ['gray'] + ['lightblue']*5 + ['darkred']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(methods, performances, yerr=[1.96*s for s in stds], 
                   capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, perf in zip(bars, performances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{perf:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Mean Performance', fontsize=12)
    ax.set_title('SA-PMI Curriculum Comparison (20 seeds, 95% CI)', fontsize=14)
    ax.axhline(y=0.4568, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_ylim(0.450, 0.475)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_statistical_significance():
    """Heatmap of pairwise statistical significance"""
    methods = ['Baseline', 'Cosine', 'Exponential', 'Early Peak', 'Two-Stage', 'Warm Restart', 'Constant']
    
    # P-values from your results (all are < 0.0001, so use Cohen's d as proxy)
    # This is a simplified version - you'd fill in actual p-values
    cohens_d = np.array([
        [0.000, 1.932, 2.042, 2.235, 2.305, 2.356, 3.358],  # Baseline vs others
        [1.932, 0.000, 0.110, 0.303, 0.373, 0.424, 1.426],  # Cosine
        [2.042, 0.110, 0.000, 0.193, 0.263, 0.314, 1.316],  # Exponential
        [2.235, 0.303, 0.193, 0.000, 0.070, 0.121, 1.123],  # Early Peak
        [2.305, 0.373, 0.263, 0.070, 0.000, 0.051, 1.053],  # Two-Stage
        [2.356, 0.424, 0.314, 0.121, 0.051, 0.000, 1.002],  # Warm Restart
        [3.358, 1.426, 1.316, 1.123, 1.053, 1.002, 0.000],  # Constant
    ])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create significance levels: 0=ns, 1=small, 2=medium, 3=large
    significance = np.zeros_like(cohens_d)
    significance[cohens_d > 0.2] = 1
    significance[cohens_d > 0.5] = 2
    significance[cohens_d > 0.8] = 3
    
    sns.heatmap(significance, annot=cohens_d, fmt='.2f', 
                cmap='RdYlGn', cbar_kws={'label': 'Effect Size Category'},
                xticklabels=methods, yticklabels=methods,
                vmin=0, vmax=3, ax=ax, linewidths=0.5)
    
    ax.set_title('Effect Sizes (Cohen\'s d) - Pairwise Comparisons', fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_weight_optimization():
    """Plot the fine-tuning results"""
    # Load the fine-tuning data
    data_path = Path("./data/weight_fine_tuning/weight_fine_tuning.csv")
    
    if not data_path.exists():
        print(f"Warning: {data_path} not found, skipping weight optimization plot")
        return None
    
    df = pd.read_csv(data_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full range
    ax1.errorbar(df['weight'], df['mean'], yerr=1.96*df['sem'],
                 fmt='o-', capsize=3, capthick=1, markersize=4, color='blue')
    
    best_idx = df['mean'].idxmax()
    best_weight = df.loc[best_idx, 'weight']
    best_mean = df.loc[best_idx, 'mean']
    
    ax1.axvline(best_weight, color='red', linestyle='--', alpha=0.5, 
                label=f'Optimal: w={best_weight:.3f}')
    ax1.plot(best_weight, best_mean, 'r*', markersize=15)
    ax1.axvline(0.6, color='orange', linestyle=':', alpha=0.5, label='Initial guess (0.6)')
    
    ax1.set_xlabel('Adversarial Weight', fontsize=12)
    ax1.set_ylabel('Mean Performance', fontsize=12)
    ax1.set_title('Fine-tuning Constant Curriculum Weight', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Zoomed around optimum
    zoom_range = 0.1
    zoom_df = df[(df['weight'] >= best_weight - zoom_range) & 
                 (df['weight'] <= best_weight + zoom_range)]
    
    ax2.errorbar(zoom_df['weight'], zoom_df['mean'], yerr=1.96*zoom_df['sem'],
                 fmt='o-', capsize=3, capthick=1, markersize=6, color='blue')
    ax2.axvline(best_weight, color='red', linestyle='--', alpha=0.5)
    ax2.plot(best_weight, best_mean, 'r*', markersize=15)
    
    ax2.set_xlabel('Adversarial Weight', fontsize=12)
    ax2.set_ylabel('Mean Performance', fontsize=12)
    ax2.set_title(f'Zoomed: w ∈ [{best_weight-zoom_range:.2f}, {best_weight+zoom_range:.2f}]', 
                  fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add annotation
    ax2.annotate(f'Optimal: {best_weight:.3f}\nPerf: {best_mean:.4f}',
                xy=(best_weight, best_mean), xytext=(best_weight+0.03, best_mean-0.0005),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    return fig


def plot_convergence_curves():
    """Plot learning curves from log files"""
    data_dir = Path("./data/top_performers")
    
    if not data_dir.exists():
        print(f"Warning: {data_dir} not found, skipping convergence plot")
        return None
    
    methods = [
        ('baseline', 'Baseline', 'gray', '--'),
        ('constant_high', 'Constant (0.67)', 'red', '-'),
        ('warm_restart', 'Warm Restart', 'purple', '-'),
        ('two_stage_early', 'Two-Stage', 'blue', '-'),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for filename, label, color, style in methods:
        csv_path = data_dir / f"{filename}.csv"
        if csv_path.exists():
            try:
                # Read CSV with semicolon separator
                df = pd.read_csv(csv_path, sep=';')
                
                # Strip whitespace from column names
                df.columns = df.columns.str.strip()
                
                # The column is called '# iterations' (with # and space)
                iterations_col = '# iterations' if '# iterations' in df.columns else 'iterations'
                evaluations_col = 'evaluations'
                
                if iterations_col in df.columns and evaluations_col in df.columns:
                    # Create iteration index (the file doesn't have actual iteration numbers)
                    iterations = np.arange(len(df))
                    evaluations = df[evaluations_col].values
                    
                    ax.plot(iterations, evaluations, 
                           label=label, color=color, linestyle=style, linewidth=2, alpha=0.8)
                else:
                    print(f"Warning: Missing required columns in {filename}.csv")
            except Exception as e:
                print(f"Error reading {filename}.csv: {e}")
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Performance', fontsize=12)
    ax.set_title('Learning Curves: SA-PMI with Different Curricula', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_summary_table():
    """Create LaTeX table for paper"""
    data = {
        'Method': ['Baseline (greedy)', 'Cosine Fast', 'Exponential Fast', 
                   'Early Peak', 'Two-Stage Early', 'Warm Restart', 'Constant (0.67)'],
        'Performance': [0.4568, 0.4630, 0.4634, 0.4640, 0.4642, 0.4643, 0.4675],
        'Std': [0.0032, 0.0032, 0.0032, 0.0032, 0.0032, 0.0032, 0.0032],
        'Improvement (%)': [0.0, 1.36, 1.43, 1.57, 1.62, 1.65, 2.34],
        'Cohen\'s d': [0.0, 1.932, 2.042, 2.235, 2.305, 2.356, 3.358],
        'p-value': ['—', '<0.001', '<0.001', '<0.001', '<0.001', '<0.001', '<0.001']
    }
    
    df = pd.DataFrame(data)
    
    # Create LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.4f", 
                              caption="SA-PMI Curriculum Learning Results",
                              label="tab:sapmi_results")
    
    output_dir = Path("./data/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results_table.tex", 'w') as f:
        f.write(latex_table)
    
    print(f"✓ LaTeX table saved: {output_dir / 'results_table.tex'}")
    
    return df


def main():
    """Generate all visualizations"""
    print("="*80)
    print("GENERATING FINAL VISUALIZATIONS")
    print("="*80)
    
    output_dir = Path("./data/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Curriculum progressions
    print("\n1. Creating curriculum progression plot...")
    fig = plot_curriculum_progressions()
    fig.savefig(output_dir / "curriculum_progressions.png", dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: curriculum_progressions.png")
    plt.close(fig)
    
    # 2. Performance comparison
    print("\n2. Creating performance comparison...")
    fig = plot_performance_comparison()
    fig.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: performance_comparison.png")
    plt.close(fig)
    
    # 3. Statistical significance
    print("\n3. Creating effect size heatmap...")
    fig = plot_statistical_significance()
    fig.savefig(output_dir / "effect_sizes.png", dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: effect_sizes.png")
    plt.close(fig)
    
    # 4. Weight optimization
    print("\n4. Creating weight optimization plot...")
    fig = plot_weight_optimization()
    if fig:
        fig.savefig(output_dir / "weight_optimization.png", dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: weight_optimization.png")
        plt.close(fig)
    
    # 5. Convergence curves
    print("\n5. Creating convergence curves...")
    fig = plot_convergence_curves()
    if fig:
        fig.savefig(output_dir / "convergence_curves.png", dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: convergence_curves.png")
        plt.close(fig)
    
    # 6. Summary table
    print("\n6. Creating results table...")
    df = create_summary_table()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nFiles created:")
    print("  1. curriculum_progressions.png - Shows how λ(t) evolves")
    print("  2. performance_comparison.png - Bar chart of all methods")
    print("  3. effect_sizes.png - Heatmap of pairwise comparisons")
    print("  4. weight_optimization.png - Fine-tuning results")
    print("  5. convergence_curves.png - Learning curves")
    print("  6. results_table.tex - LaTeX table for paper")
    
    print("\n✓ Visualization generation complete!")
    
    return output_dir


if __name__ == '__main__':
    output_dir = main()