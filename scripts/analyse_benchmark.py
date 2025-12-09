"""
Analyze benchmark results and create comparison tables
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_benchmarks(results_dir="results/benchmarks"):
    """Load all benchmark JSON files"""
    results_path = Path(results_dir)
    all_results = []
    
    for json_file in results_path.glob("benchmark_frozen_lake_4x4_det_2*.json"):
        with open(json_file) as f:
            data = json.load(f)
            for result in data:
                result['benchmark_file'] = json_file.name
                all_results.append(result)
    
    return pd.DataFrame(all_results)

def create_summary_table(df):
    """Create summary statistics table"""
    # Group by environment and algorithm
    summary = df.groupby(['environment', 'algorithm']).agg({
        'final_performance': ['mean', 'std', 'min', 'max'],
        'iterations': ['mean', 'std'],
        'improvement': ['mean', 'std']
    }).round(4)
    
    return summary

def create_comparison_plot(df, output_path="results/benchmark_comparison.png"):
    """Create visual comparison of algorithms"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    environments = df['environment'].unique()
    
    for idx, env in enumerate(environments[:4]):  # Plot first 4 envs
        ax = axes[idx // 2, idx % 2]
        
        env_data = df[df['environment'] == env]
        
        # Plot performance by algorithm
        env_summary = env_data.groupby('algorithm')['final_performance'].agg(['mean', 'std'])
        
        env_summary.plot(y='mean', kind='bar', ax=ax, yerr=env_summary['std'], 
                         capsize=4, alpha=0.7)
        ax.set_title(f"{env}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Final Performance")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")

def main():
    print("Loading benchmark results...")
    df = load_all_benchmarks()
    
    if df.empty:
        print("No benchmark results found in results/benchmarks/")
        return
    
    print(f"\nLoaded {len(df)} benchmark runs")
    print(f"Environments: {df['environment'].unique()}")
    print(f"Algorithms: {df['algorithm'].unique()}")
    
    # Create summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    summary = create_summary_table(df)
    print(summary)
    
    # Save summary to CSV
    summary.to_csv("results/benchmark_summary.csv")
    print("\nSummary saved to: results/benchmark_summary.csv")
    
    # Create comparison plot
    create_comparison_plot(df)
    
    # Find best algorithm per environment
    print("\n" + "="*80)
    print("BEST ALGORITHM PER ENVIRONMENT")
    print("="*80)
    best_per_env = df.groupby(['environment', 'algorithm'])['final_performance'].mean().reset_index()
    best_per_env = best_per_env.loc[best_per_env.groupby('environment')['final_performance'].idxmax()]
    print(best_per_env[['environment', 'algorithm', 'final_performance']])
    
    # Statistical significance tests (if scipy available)
    try:
        from scipy import stats
        print("\n" + "="*80)
        print("STATISTICAL TESTS")
        print("="*80)
        
        for env in df['environment'].unique():
            print(f"\nEnvironment: {env}")
            env_data = df[df['environment'] == env]
            
            # Compare classical vs adversarial
            classical = env_data[env_data['algorithm'].isin(['value_iteration', 'policy_iteration', 'q_learning'])]
            adversarial = env_data[env_data['algorithm'].str.contains('sa_pmi')]
            
            if len(classical) > 0 and len(adversarial) > 0:
                t_stat, p_value = stats.ttest_ind(
                    classical['final_performance'], 
                    adversarial['final_performance']
                )
                print(f"  Classical vs Adversarial: t={t_stat:.4f}, p={p_value:.4f}")
                
                if p_value < 0.05:
                    classical_mean = classical['final_performance'].mean()
                    adversarial_mean = adversarial['final_performance'].mean()
                    
                    if classical_mean > adversarial_mean:
                        print(f"  ✓ Classical significantly better (p<0.05)")
                    else:
                        print(f"  ✗ Adversarial significantly better (p<0.05)")
                else:
                    print(f"  ~ No significant difference (p>=0.05)")
    
    except ImportError:
        print("\nInstall scipy for statistical significance tests")

if __name__ == '__main__':
    main()