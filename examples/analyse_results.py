"""
Result Analysis Tool for SA-PMI Experiments
Generates publication-quality figures and statistical comparisons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os
from scipy import stats
from pathlib import Path

# Set publication-quality plot defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})


def load_benchmark_results(results_dir='results/sweeps'):
    """Load all benchmark JSON files"""
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    
    if not json_files:
        print(f"No results found in {results_dir}")
        return None
    
    print(f"Found {len(json_files)} result files")
    
    all_results = []
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            all_results.extend(data)
    
    df = pd.DataFrame(all_results)
    print(f"Loaded {len(df)} experiment runs")
    return df


def compute_statistics(df):
    """Compute mean, std, and confidence intervals"""
    stats_df = df.groupby(['algorithm', 'environment']).agg({
        'final_performance': ['mean', 'std', 'count'],
        'improvement': ['mean', 'std'],
        'iterations': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns]
    
    # Compute 95% confidence intervals
    stats_df['final_performance_ci'] = 1.96 * stats_df['final_performance_std'] / np.sqrt(stats_df['final_performance_count'])
    stats_df['improvement_ci'] = 1.96 * stats_df['improvement_std'] / np.sqrt(stats_df['final_performance_count'])
    
    return stats_df


def plot_algorithm_comparison(df, save_path='figures'):
    """Create bar plot comparing algorithms"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.barplot(
        data=df, 
        x='algorithm', 
        y='final_performance',
        hue='environment',
        ci='sd',  # Show standard deviation
        ax=ax,
        capsize=0.1
    )
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Final Performance')
    ax.set_title('Algorithm Performance Comparison Across Environments')
    ax.legend(title='Environment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filepath = os.path.join(save_path, 'algorithm_comparison.pdf')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def plot_learning_curves(csv_dir='results', save_path='figures'):
    """Plot learning curves from CSV files"""
    os.makedirs(save_path, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for csv_file in csv_files:
        try:
            # Load CSV (handle the semicolon delimiter from your logger)
            df = pd.read_csv(csv_file, sep=';', comment='#')
            
            # Extract algorithm name from filename
            algo_name = os.path.basename(csv_file).replace('.csv', '').replace('teacher_student_', '')
            
            # Plot performance over iterations
            ax.plot(df['iterations'], df['evaluations'], label=algo_name, linewidth=2)
            
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Performance')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(save_path, 'learning_curves.pdf')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def plot_adversarial_budget_evolution(csv_dir='results', save_path='figures'):
    """Plot adversarial budget evolution for SA-PMI variants"""
    os.makedirs(save_path, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(csv_dir, '*sa_pmi*.csv'))
    
    if not csv_files:
        print(f"No SA-PMI CSV files found in {csv_dir}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, sep=';', comment='#')
            
            # Check if adversarial budget column exists
            if 'adv_budget' not in df.columns:
                continue
            
            algo_name = os.path.basename(csv_file).replace('.csv', '').replace('teacher_student_', '')
            
            ax.plot(df['iterations'], df['adv_budget'], label=algo_name, linewidth=2)
            
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Adversarial Budget')
    ax.set_title('Adversarial Budget Evolution (Curriculum Learning)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(save_path, 'adversarial_budget.pdf')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def statistical_comparison(df):
    """Perform statistical tests comparing algorithms"""
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON (t-tests)")
    print("="*70)
    
    algorithms = df['algorithm'].unique()
    environments = df['environment'].unique()
    
    for env in environments:
        print(f"\nEnvironment: {env}")
        print("-" * 70)
        
        env_data = df[df['environment'] == env]
        
        # Compare each pair of algorithms
        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i+1:]:
                data1 = env_data[env_data['algorithm'] == algo1]['final_performance']
                data2 = env_data[env_data['algorithm'] == algo2]['final_performance']
                
                if len(data1) < 2 or len(data2) < 2:
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(data1, data2)
                
                mean1, std1 = data1.mean(), data1.std()
                mean2, std2 = data2.mean(), data2.std()
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                print(f"\n{algo1} vs {algo2}:")
                print(f"  {algo1}: {mean1:.4f} ± {std1:.4f}")
                print(f"  {algo2}: {mean2:.4f} ± {std2:.4f}")
                print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f} {significance}")


def create_latex_table(stats_df, save_path='figures'):
    """Create LaTeX table for paper"""
    os.makedirs(save_path, exist_ok=True)
    
    # Pivot table for better layout
    pivot = stats_df.pivot(
        index='algorithm',
        columns='environment',
        values=['final_performance_mean', 'final_performance_std']
    )
    
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Algorithm Performance Comparison}",
        "\\label{tab:results}",
        "\\begin{tabular}{l" + "c" * len(stats_df['environment'].unique()) + "}",
        "\\toprule",
    ]
    
    # Header
    header = "Algorithm & " + " & ".join(stats_df['environment'].unique()) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Data rows
    for algo in stats_df['algorithm'].unique():
        row_data = []
        for env in stats_df['environment'].unique():
            mask = (stats_df['algorithm'] == algo) & (stats_df['environment'] == env)
            if mask.any():
                mean = stats_df[mask]['final_performance_mean'].values[0]
                std = stats_df[mask]['final_performance_std'].values[0]
                row_data.append(f"{mean:.3f} $\\pm$ {std:.3f}")
            else:
                row_data.append("-")
        
        row = algo.replace('_', '\\_') + " & " + " & ".join(row_data) + " \\\\"
        latex_lines.append(row)
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    filepath = os.path.join(save_path, 'results_table.tex')
    with open(filepath, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"✓ Saved LaTeX table: {filepath}")


def generate_summary_report(df, stats_df, save_path='figures'):
    """Generate a markdown summary report"""
    os.makedirs(save_path, exist_ok=True)
    
    report = ["# Experiment Results Summary\n"]
    report.append(f"**Total runs**: {len(df)}\n")
    report.append(f"**Algorithms**: {', '.join(df['algorithm'].unique())}\n")
    report.append(f"**Environments**: {', '.join(df['environment'].unique())}\n")
    report.append(f"**Seeds per config**: {df.groupby(['algorithm', 'environment']).size().max()}\n")
    
    report.append("\n## Performance Statistics\n")
    report.append("\n| Algorithm | Environment | Mean ± Std | 95% CI |\n")
    report.append("|-----------|-------------|------------|--------|\n")
    
    for _, row in stats_df.iterrows():
        report.append(
            f"| {row['algorithm']} | {row['environment']} | "
            f"{row['final_performance_mean']:.4f} ± {row['final_performance_std']:.4f} | "
            f"±{row['final_performance_ci']:.4f} |\n"
        )
    
    report.append("\n## Best Performing Configurations\n")
    best_per_env = stats_df.loc[stats_df.groupby('environment')['final_performance_mean'].idxmax()]
    for _, row in best_per_env.iterrows():
        report.append(
            f"- **{row['environment']}**: {row['algorithm']} "
            f"({row['final_performance_mean']:.4f} ± {row['final_performance_std']:.4f})\n"
        )
    
    filepath = os.path.join(save_path, 'summary_report.md')
    with open(filepath, 'w') as f:
        f.writelines(report)
    
    print(f"✓ Saved summary report: {filepath}")


def main():
    """Main analysis pipeline"""
    print("\n" + "="*70)
    print("SA-PMI RESULT ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_benchmark_results()
    if df is None or len(df) == 0:
        print("\n⚠️  No data to analyze. Run experiments first!")
        return
    
    # Compute statistics
    print("\nComputing statistics...")
    stats_df = compute_statistics(df)
    
    # Print statistics
    print("\n" + "="*70)
    print("PERFORMANCE STATISTICS")
    print("="*70)
    print(stats_df.to_string(index=False))
    
    # Statistical tests
    statistical_comparison(df)
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)
    
    plot_algorithm_comparison(df)
    plot_learning_curves()
    plot_adversarial_budget_evolution()
    
    # Generate tables and reports
    create_latex_table(stats_df)
    generate_summary_report(df, stats_df)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - figures/algorithm_comparison.pdf")
    print("  - figures/learning_curves.pdf")
    print("  - figures/adversarial_budget.pdf")
    print("  - figures/results_table.tex")
    print("  - figures/summary_report.md")
    print("\nUse these for your thesis/paper!")


if __name__ == '__main__':
    main()