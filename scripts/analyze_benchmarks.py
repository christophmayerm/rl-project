"""
Publication-Grade Benchmark Analysis
Comprehensive analysis of Classical vs Adversarial RL algorithms
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.3)
sns.set_palette("Set2")

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ============================================================================
# LOAD DATA
# ============================================================================
def load_all_benchmarks(results_dir="results/benchmarks"):
    results_path = Path(results_dir)
    all_results = []
    
    if not results_path.exists():
        print(f"❌ Directory not found: {results_dir}")
        return pd.DataFrame()
    
    json_files = sorted(results_path.glob("benchmark_*.json"))
    
    if not json_files:
        print(f"❌ No benchmark files found in {results_dir}")
        return pd.DataFrame()
    
    for json_file in json_files:
        print(f"📂 Loading: {json_file.name}")
        with open(json_file) as f:
            data = json.load(f)
            for result in data:
                result['benchmark_file'] = json_file.name
                all_results.append(result)
    
    return pd.DataFrame(all_results)

print("="*80)
print("PUBLICATION-GRADE BENCHMARK ANALYSIS")
print("="*80)
print()

df = load_all_benchmarks()

if df.empty:
    print("❌ No benchmark results found!")
    exit(1)

print(f"✅ Loaded {len(df)} runs")
print(f"   Environments: {df['environment'].nunique()}")
print(f"   Algorithms: {df['algorithm'].nunique()}")
print(f"   Seeds: {df['seed'].nunique()}")

# ============================================================================
# DATA PREPARATION
# ============================================================================
classical_algos = ['value_iteration', 'policy_iteration', 'q_learning']
spmi_algos = ['spmi', 'sa_pmi_linear', 'sa_pmi_exponential']

df['algorithm_category'] = df['algorithm'].apply(
    lambda x: 'Classical' if x in classical_algos else 'Adversarial'
)

algo_names = {
    'value_iteration': 'VI',
    'policy_iteration': 'PI',
    'q_learning': 'Q-Learning',
    'spmi': 'SPMI',
    'sa_pmi_linear': 'SA-PMI (Linear)',
    'sa_pmi_exponential': 'SA-PMI (Exp)',
}

env_names = {
    'frozen_lake_4x4': 'FL-4×4 (S)',
    'frozen_lake_4x4_det': 'FL-4×4 (D)',
    'frozen_lake_8x8': 'FL-8×8 (S)',
    'frozen_lake_8x8_det': 'FL-8×8 (D)',
}

df['algo_short'] = df['algorithm'].map(algo_names).fillna(df['algorithm'])
df['env_short'] = df['environment'].map(env_names).fillna(df['environment'])
df['efficiency'] = df['final_performance'] / df['iterations'] * 1000

Path('results/plots').mkdir(parents=True, exist_ok=True)

# ============================================================================
# FIGURE 1: OVERVIEW - 2×2 GRID OF KEY METRICS
# ============================================================================
print("\n" + "="*80)
print("FIGURE 1: Overview - Key Performance Metrics")
print("="*80)

fig1 = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, figure=fig1, hspace=0.3, wspace=0.3)

# Subplot A: Mean Performance by Algorithm
ax1 = fig1.add_subplot(gs[0, 0])
summary_perf = df.groupby(['algorithm_category', 'algorithm']).agg({
    'final_performance': ['mean', 'std']
}).reset_index()

classical_data = summary_perf[summary_perf['algorithm_category'] == 'Classical']
adversarial_data = summary_perf[summary_perf['algorithm_category'] == 'Adversarial']

x_pos = np.arange(len(classical_data))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, 
                classical_data[('final_performance', 'mean')],
                width, 
                yerr=classical_data[('final_performance', 'std')],
                label='Classical', color='#2E7D32', alpha=0.8, capsize=3)

x_pos2 = np.arange(len(adversarial_data))
bars2 = ax1.bar(x_pos2 + len(classical_data) + 0.5 + width/2,
                adversarial_data[('final_performance', 'mean')],
                width,
                yerr=adversarial_data[('final_performance', 'std')],
                label='Adversarial', color='#C62828', alpha=0.8, capsize=3)

all_algos = list(classical_data['algorithm']) + list(adversarial_data['algorithm'])
all_x = list(x_pos) + list(x_pos2 + len(classical_data) + 0.5)
ax1.set_xticks(all_x)
ax1.set_xticklabels([algo_names.get(a, a) for a in all_algos], rotation=45, ha='right')
ax1.set_ylabel('Mean Return')
ax1.set_title('(A) Mean Performance by Algorithm', fontweight='bold', loc='left')
ax1.legend(frameon=True, fancybox=True)
ax1.axvline(len(classical_data) - 0.25, color='gray', linestyle='--', alpha=0.5)

# Subplot B: Performance Distribution
ax2 = fig1.add_subplot(gs[0, 1])
classical_perf = df[df['algorithm_category'] == 'Classical']['final_performance']
adversarial_perf = df[df['algorithm_category'] == 'Adversarial']['final_performance']

parts = ax2.violinplot([classical_perf, adversarial_perf], 
                        positions=[0, 1], 
                        showmeans=True, showmedians=True,
                        widths=0.6)
for pc, color in zip(parts['bodies'], ['#2E7D32', '#C62828']):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)

ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Classical', 'Adversarial'])
ax2.set_ylabel('Return Distribution')
ax2.set_title('(B) Overall Distribution', fontweight='bold', loc='left')

# Add statistical annotation
t_stat, p_val = stats.ttest_ind(classical_perf, adversarial_perf)
sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
ax2.text(0.5, 0.95, f'p = {p_val:.4f} {sig}', 
         transform=ax2.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Subplot C: Convergence Speed
ax3 = fig1.add_subplot(gs[1, 0])
summary_iters = df.groupby('algorithm').agg({
    'iterations': ['mean', 'std']
}).reset_index()

colors_iters = ['#2E7D32' if a in classical_algos else '#C62828' 
                for a in summary_iters['algorithm']]

x_pos = np.arange(len(summary_iters))
bars = ax3.bar(x_pos, summary_iters[('iterations', 'mean')],
               yerr=summary_iters[('iterations', 'std')],
               color=colors_iters, alpha=0.8, capsize=3)

ax3.set_xticks(x_pos)
ax3.set_xticklabels([algo_names.get(a, a) for a in summary_iters['algorithm']], 
                    rotation=45, ha='right')
ax3.set_ylabel('Iterations')
ax3.set_title('(C) Convergence Speed', fontweight='bold', loc='left')

# Subplot D: Efficiency
ax4 = fig1.add_subplot(gs[1, 1])
summary_eff = df.groupby('algorithm').agg({
    'efficiency': ['mean', 'std']
}).reset_index()

bars = ax4.bar(x_pos, summary_eff[('efficiency', 'mean')],
               yerr=summary_eff[('efficiency', 'std')],
               color=colors_iters, alpha=0.8, capsize=3)

ax4.set_xticks(x_pos)
ax4.set_xticklabels([algo_names.get(a, a) for a in summary_eff['algorithm']], 
                    rotation=45, ha='right')
ax4.set_ylabel('Efficiency (Return/Iter × 1000)')
ax4.set_title('(D) Learning Efficiency', fontweight='bold', loc='left')

plt.suptitle('Figure 1: Performance Overview', fontsize=14, fontweight='bold', y=0.995)
output1 = 'results/plots/fig1_overview.png'
plt.savefig(output1, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output1}")
plt.close()

# ============================================================================
# FIGURE 2: ENVIRONMENT-SPECIFIC PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("FIGURE 2: Environment-Specific Analysis")
print("="*80)

fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

envs = sorted(df['environment'].unique())

for idx, env in enumerate(envs):
    ax = axes[idx]
    env_data = df[df['environment'] == env]
    
    summary = env_data.groupby('algorithm').agg({
        'final_performance': ['mean', 'std']
    })
    
    algo_order = [a for a in classical_algos + spmi_algos if a in summary.index]
    summary = summary.loc[algo_order]
    
    x_pos = np.arange(len(summary))
    colors = ['#2E7D32' if a in classical_algos else '#C62828' 
              for a in algo_order]
    
    bars = ax.bar(x_pos, summary[('final_performance', 'mean')],
                  yerr=summary[('final_performance', 'std')],
                  color=colors, alpha=0.8, capsize=3, edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([algo_names.get(a, a) for a in algo_order], 
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Return')
    ax.set_title(f'{env_names.get(env, env)}', fontweight='bold')
    
    # Add baseline reference
    if len(summary) > 0:
        best_perf = summary[('final_performance', 'mean')].max()
        ax.axhline(best_perf, color='gold', linestyle=':', alpha=0.5, linewidth=1.5)

plt.suptitle('Figure 2: Performance Across Environments', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
output2 = 'results/plots/fig2_environments.png'
plt.savefig(output2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output2}")
plt.close()

# ============================================================================
# FIGURE 3: ALGORITHM COMPARISON HEATMAP
# ============================================================================
print("\n" + "="*80)
print("FIGURE 3: Comprehensive Heatmap")
print("="*80)

fig3, ax = plt.subplots(figsize=(10, 6))

pivot = df.groupby(['env_short', 'algo_short'])['final_performance'].mean().unstack()

# Sort columns
classical_cols = [algo_names[a] for a in classical_algos if algo_names[a] in pivot.columns]
adversarial_cols = [c for c in pivot.columns if c not in classical_cols]
pivot = pivot[classical_cols + adversarial_cols]

# Create heatmap
im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0)

# Annotations
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        value = pivot.values[i, j]
        if not np.isnan(value):
            # Best in row?
            is_best = value == pivot.iloc[i].max()
            
            text_color = 'white' if value < pivot.values.mean() else 'black'
            fontweight = 'bold' if is_best else 'normal'
            fontsize = 9 if is_best else 7
            
            ax.text(j, i, f'{value:.3f}',
                   ha='center', va='center',
                   color=text_color, fontsize=fontsize,
                   fontweight=fontweight)
            
            if is_best:
                ax.add_patch(plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8,
                            fill=False, edgecolor='gold', linewidth=2.5))

ax.set_xticks(np.arange(len(pivot.columns)))
ax.set_yticks(np.arange(len(pivot.index)))
ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_xlabel('Algorithm', fontweight='bold')
ax.set_ylabel('Environment', fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Return', rotation=270, labelpad=15)

# Separator
if len(classical_cols) > 0:
    sep_pos = len(classical_cols) - 0.5
    ax.axvline(sep_pos, color='black', linewidth=2, linestyle='--', alpha=0.6)

plt.title('Figure 3: Algorithm × Environment Performance Matrix\n' +
          '(Gold box = best per environment)',
          fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
output3 = 'results/plots/fig3_heatmap.png'
plt.savefig(output3, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output3}")
plt.close()

# ============================================================================
# FIGURE 4: STATISTICAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("FIGURE 4: Statistical Significance")
print("="*80)

fig4, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot A: Box plots by algorithm
ax1 = axes[0, 0]
algo_order = [a for a in classical_algos + spmi_algos if a in df['algorithm'].unique()]
data_boxes = [df[df['algorithm'] == a]['final_performance'].values for a in algo_order]
labels_boxes = [algo_names.get(a, a) for a in algo_order]

bp = ax1.boxplot(data_boxes, labels=labels_boxes, patch_artist=True,
                 showmeans=True, meanline=True,
                 medianprops=dict(color='red', linewidth=1.5),
                 meanprops=dict(color='blue', linewidth=1.5, linestyle='--'))

for i, box in enumerate(bp['boxes']):
    if algo_order[i] in classical_algos:
        box.set_facecolor('#2E7D32')
    else:
        box.set_facecolor('#C62828')
    box.set_alpha(0.6)

ax1.set_xticklabels(labels_boxes, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Return')
ax1.set_title('(A) Distribution by Algorithm', fontweight='bold', loc='left')
ax1.grid(axis='y', alpha=0.3)

# Subplot B: Pairwise t-tests
ax2 = axes[0, 1]
n_algos = len(algo_order)
p_matrix = np.ones((n_algos, n_algos))

for i in range(n_algos):
    for j in range(i+1, n_algos):
        data_i = df[df['algorithm'] == algo_order[i]]['final_performance']
        data_j = df[df['algorithm'] == algo_order[j]]['final_performance']
        if len(data_i) > 1 and len(data_j) > 1:
            _, p = stats.ttest_ind(data_i, data_j)
            p_matrix[i, j] = p
            p_matrix[j, i] = p

im2 = ax2.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.05, aspect='auto')
ax2.set_xticks(np.arange(n_algos))
ax2.set_yticks(np.arange(n_algos))
ax2.set_xticklabels([algo_names.get(a, a) for a in algo_order], 
                     rotation=45, ha='right', fontsize=7)
ax2.set_yticklabels([algo_names.get(a, a) for a in algo_order], fontsize=7)
ax2.set_title('(B) Pairwise p-values', fontweight='bold', loc='left')

for i in range(n_algos):
    for j in range(n_algos):
        if i != j:
            p = p_matrix[i, j]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            ax2.text(j, i, sig, ha='center', va='center', 
                    color='white' if p < 0.025 else 'black', fontweight='bold')

plt.colorbar(im2, ax=ax2, label='p-value')

# Subplot C: Effect sizes (Cohen's d)
ax3 = axes[1, 0]
classical_perf = df[df['algorithm_category'] == 'Classical']['final_performance']
adversarial_perf = df[df['algorithm_category'] == 'Adversarial']['final_performance']

mean_diff = classical_perf.mean() - adversarial_perf.mean()
pooled_std = np.sqrt((classical_perf.std()**2 + adversarial_perf.std()**2) / 2)
cohens_d = mean_diff / pooled_std

categories = ['Classical\nvs\nAdversarial']
effect_sizes = [cohens_d]
colors_effect = ['#2E7D32' if d > 0 else '#C62828' for d in effect_sizes]

bars = ax3.barh(categories, effect_sizes, color=colors_effect, alpha=0.8)
ax3.axvline(0, color='black', linewidth=1)
ax3.axvline(0.2, color='gray', linestyle=':', alpha=0.5, label='Small effect')
ax3.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
ax3.axvline(0.8, color='gray', linestyle='-', alpha=0.5, label='Large effect')
ax3.set_xlabel("Cohen's d")
ax3.set_title("(C) Effect Size", fontweight='bold', loc='left')
ax3.legend(fontsize=7, loc='lower right')
ax3.text(cohens_d, 0, f'd = {cohens_d:.2f}', 
         ha='right' if cohens_d > 0 else 'left', va='center', fontweight='bold')

# Subplot D: Sample means with confidence intervals
ax4 = axes[1, 1]
means = []
cis = []
labels_ci = []

for algo in algo_order:
    data = df[df['algorithm'] == algo]['final_performance']
    mean = data.mean()
    ci = 1.96 * data.std() / np.sqrt(len(data))  # 95% CI
    means.append(mean)
    cis.append(ci)
    labels_ci.append(algo_names.get(algo, algo))

x_pos = np.arange(len(means))
colors_ci = ['#2E7D32' if algo in classical_algos else '#C62828' 
             for algo in algo_order]

ax4.errorbar(x_pos, means, yerr=cis, fmt='o', capsize=5, capthick=2,
             color='black', markersize=8, elinewidth=2)
for i, (x, y, c) in enumerate(zip(x_pos, means, colors_ci)):
    ax4.scatter(x, y, s=150, color=c, alpha=0.8, zorder=3)

ax4.set_xticks(x_pos)
ax4.set_xticklabels(labels_ci, rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Return (95% CI)')
ax4.set_title('(D) Confidence Intervals', fontweight='bold', loc='left')
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Figure 4: Statistical Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
output4 = 'results/plots/fig4_statistics.png'
plt.savefig(output4, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output4}")
plt.close()

# ============================================================================
# FIGURE 5: PERFORMANCE VS COMPUTATIONAL COST
# ============================================================================
print("\n" + "="*80)
print("FIGURE 5: Performance-Cost Trade-off")
print("="*80)

fig5, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot A: Performance vs Iterations
ax1 = axes[0]
for env in df['environment'].unique():
    env_data = df[df['environment'] == env]
    for algo in algo_order:
        algo_data = env_data[env_data['algorithm'] == algo]
        if len(algo_data) > 0:
            color = '#2E7D32' if algo in classical_algos else '#C62828'
            marker = 'o' if algo in classical_algos else '^'
            ax1.scatter(algo_data['iterations'], 
                       algo_data['final_performance'],
                       c=color, marker=marker, s=60, alpha=0.6,
                       label=f'{algo_names.get(algo, algo)} ({env_names.get(env, env)})')

ax1.set_xlabel('Iterations')
ax1.set_ylabel('Final Return')
ax1.set_title('(A) Return vs Computational Cost', fontweight='bold', loc='left')
ax1.legend(fontsize=6, ncol=2, loc='best', framealpha=0.9)
ax1.grid(alpha=0.3)

# Subplot B: Efficiency frontier
ax2 = axes[1]
summary_scatter = df.groupby('algorithm').agg({
    'final_performance': 'mean',
    'iterations': 'mean',
    'efficiency': 'mean'
}).reset_index()

colors_scatter = ['#2E7D32' if a in classical_algos else '#C62828' 
                  for a in summary_scatter['algorithm']]
markers_scatter = ['o' if a in classical_algos else '^' 
                   for a in summary_scatter['algorithm']]

for i, row in summary_scatter.iterrows():
    ax2.scatter(row['iterations'], row['final_performance'],
               s=row['efficiency']*50, c=colors_scatter[i],
               marker=markers_scatter[i], alpha=0.7, edgecolors='black')
    ax2.annotate(algo_names.get(row['algorithm'], row['algorithm']),
                (row['iterations'], row['final_performance']),
                xytext=(5, 5), textcoords='offset points', fontsize=7)

ax2.set_xlabel('Mean Iterations')
ax2.set_ylabel('Mean Return')
ax2.set_title('(B) Efficiency Frontier\n(bubble size ∝ efficiency)', 
             fontweight='bold', loc='left')
ax2.grid(alpha=0.3)

plt.suptitle('Figure 5: Performance-Cost Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
output5 = 'results/plots/fig5_cost_analysis.png'
plt.savefig(output5, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output5}")
plt.close()

# ============================================================================
# FIGURE 6: LEARNING CURVES (If iteration data available)
# ============================================================================
print("\n" + "="*80)
print("FIGURE 6: Algorithm Characteristics")
print("="*80)

fig6, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot A: Variance analysis
ax1 = axes[0, 0]
variance_data = df.groupby('algorithm').agg({
    'final_performance': 'std'
}).reset_index()

x_pos = np.arange(len(variance_data))
colors_var = ['#2E7D32' if a in classical_algos else '#C62828' 
              for a in variance_data['algorithm']]

bars = ax1.bar(x_pos, variance_data['final_performance'],
              color=colors_var, alpha=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([algo_names.get(a, a) for a in variance_data['algorithm']], 
                    rotation=45, ha='right')
ax1.set_ylabel('Std Dev of Return')
ax1.set_title('(A) Performance Stability', fontweight='bold', loc='left')
ax1.grid(axis='y', alpha=0.3)

# Subplot B: Range analysis (min-max) - FIXED
ax2 = axes[0, 1]
range_data = df.groupby('algorithm').agg({
    'final_performance': ['min', 'max', 'mean']
}).reset_index()

for i, row in range_data.iterrows():
    algo = row[('algorithm', '')]  # Fixed: correct column access
    color = '#2E7D32' if algo in classical_algos else '#C62828'
    ax2.plot([i, i], 
            [row[('final_performance', 'min')], row[('final_performance', 'max')]],
            'o-', color=color, linewidth=2, markersize=6, alpha=0.7)
    ax2.scatter(i, row[('final_performance', 'mean')], 
               s=100, color=color, marker='s', zorder=3, edgecolors='black')

ax2.set_xticks(range(len(range_data)))
ax2.set_xticklabels([algo_names.get(a, a) for a in range_data[('algorithm', '')].values],  # Fixed
                    rotation=45, ha='right')
ax2.set_ylabel('Return Range')
ax2.set_title('(B) Performance Range\n(square = mean)', fontweight='bold', loc='left')
ax2.grid(axis='y', alpha=0.3)

# Subplot C: Success rate (above threshold)
ax3 = axes[1, 0]
thresholds = {}
for env in df['environment'].unique():
    env_data = df[df['environment'] == env]
    thresholds[env] = env_data['final_performance'].quantile(0.75)

success_rates = []
for algo in algo_order:
    algo_data = df[df['algorithm'] == algo]
    successes = []
    for env in df['environment'].unique():
        env_algo_data = algo_data[algo_data['environment'] == env]
        if len(env_algo_data) > 0:
            success_rate = (env_algo_data['final_performance'] > thresholds[env]).mean()
            successes.append(success_rate)
    success_rates.append(np.mean(successes) if successes else 0)

x_pos = np.arange(len(algo_order))
colors_success = ['#2E7D32' if a in classical_algos else '#C62828' 
                  for a in algo_order]

bars = ax3.bar(x_pos, success_rates, color=colors_success, alpha=0.8)
ax3.axhline(0.75, color='gray', linestyle='--', alpha=0.5, label='Target (75th percentile)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([algo_names.get(a, a) for a in algo_order], 
                    rotation=45, ha='right')
ax3.set_ylabel('Success Rate')
ax3.set_title('(C) Reliability (% above Q3)', fontweight='bold', loc='left')
ax3.legend(fontsize=8)
ax3.grid(axis='y', alpha=0.3)

# Subplot D: Improvement over baseline
ax4 = axes[1, 1]
baseline_perf = df[df['algorithm'] == 'value_iteration']['final_performance'].mean()

improvements = []
for algo in algo_order:
    algo_mean = df[df['algorithm'] == algo]['final_performance'].mean()
    improvement = (algo_mean - baseline_perf) / baseline_perf * 100
    improvements.append(improvement)

colors_imp = ['green' if imp >= 0 else 'red' for imp in improvements]
bars = ax4.barh(range(len(improvements)), improvements, color=colors_imp, alpha=0.8)
ax4.axvline(0, color='black', linewidth=1)
ax4.set_yticks(range(len(algo_order)))
ax4.set_yticklabels([algo_names.get(a, a) for a in algo_order])
ax4.set_xlabel('Improvement over VI (%)')
ax4.set_title('(D) Relative Performance', fontweight='bold', loc='left')
ax4.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvements)):
    ax4.text(val, i, f'{val:+.1f}%', 
            ha='left' if val >= 0 else 'right', 
            va='center', fontsize=8, fontweight='bold')

plt.suptitle('Figure 6: Algorithm Characteristics', fontsize=14, fontweight='bold')
plt.tight_layout()
output6 = 'results/plots/fig6_characteristics.png'
plt.savefig(output6, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output6}")
plt.close()

# ============================================================================
# FIGURE 7: SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("FIGURE 7: Summary Table")
print("="*80)

fig7, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Prepare summary statistics - FIXED
summary_stats = df.groupby('algorithm').agg({
    'final_performance': ['mean', 'std', 'min', 'max'],
    'iterations': ['mean', 'std'],
    'efficiency': ['mean']
}).round(4)

table_data = []
for algo in algo_order:
    if algo in summary_stats.index:
        row = summary_stats.loc[algo]
        table_data.append([
            algo_names.get(algo, algo),
            'Classical' if algo in classical_algos else 'Adversarial',
            f"{row[('final_performance', 'mean')]:.4f}",
            f"{row[('final_performance', 'std')]:.4f}",
            f"{row[('final_performance', 'min')]:.4f}",
            f"{row[('final_performance', 'max')]:.4f}",
            f"{int(row[('iterations', 'mean')])}",
            f"{row[('efficiency', 'mean')]:.2f}"
        ])

columns = ['Algorithm', 'Type', 'Mean', 'Std', 'Min', 'Max', 'Iters', 'Efficiency']

table = ax.table(cellText=table_data, colLabels=columns,
                cellLoc='center', loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(len(columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(table_data) + 1):
    algo_type = table_data[i-1][1]
    color = '#E7F4E7' if algo_type == 'Classical' else '#F4E7E7'
    
    for j in range(len(columns)):
        cell = table[(i, j)]
        cell.set_facecolor(color)
        
        # Bold best values
        if j == 2:  # Mean column
            values = [float(row[2]) for row in table_data]
            if float(table_data[i-1][2]) == max(values):
                cell.set_text_props(weight='bold', color='darkgreen')

plt.title('Figure 7: Comprehensive Performance Summary', 
         fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
output7 = 'results/plots/fig7_summary_table.png'
plt.savefig(output7, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output7}")
plt.close()

# ============================================================================
# FINAL SUMMARY & LATEX TABLE
# ============================================================================
print("\n" + "="*80)
print("GENERATING LATEX TABLE")
print("="*80)

latex_table = summary_stats.to_latex(float_format="%.4f")
with open('results/plots/table1_latex.txt', 'w') as f:
    f.write(latex_table)
print("✅ LaTeX table saved: results/plots/table1_latex.txt")

# Summary statistics
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

classical_perf = df[df['algorithm_category'] == 'Classical']['final_performance']
adversarial_perf = df[df['algorithm_category'] == 'Adversarial']['final_performance']

classical_mean = classical_perf.mean()
adversarial_mean = adversarial_perf.mean()
difference = classical_mean - adversarial_mean
percent_diff = (difference / adversarial_mean) * 100

print(f"\n📊 Performance Summary:")
print(f"  • Classical algorithms:    {classical_mean:.4f} ± {classical_perf.std():.4f}")
print(f"  • Adversarial algorithms:  {adversarial_mean:.4f} ± {adversarial_perf.std():.4f}")
print(f"  • Absolute difference:     {difference:.4f}")
print(f"  • Relative difference:     {percent_diff:.1f}%")

t_stat, p_val = stats.ttest_ind(classical_perf, adversarial_perf)
sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
print(f"  • Statistical significance: t={t_stat:.3f}, p={p_val:.6f} {sig}")

# Win rate analysis
print(f"\n🏆 Win Rate Analysis:")
win_counts = {}
for env in df['environment'].unique():
    env_data = df[df['environment'] == env]
    best_algo = env_data.groupby('algorithm')['final_performance'].mean().idxmax()
    win_counts[best_algo] = win_counts.get(best_algo, 0) + 1

for algo, count in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(df['environment'].unique()) * 100
    category = 'CLASSICAL' if algo in classical_algos else 'ADVERSARIAL'
    print(f"  • [{category:12s}] {algo_names.get(algo, algo):20s}: {count}/{len(df['environment'].unique())} ({pct:.0f}%)")

print(f"\n📁 Generated Figures:")
print(f"  1. Overview (4-panel):        {output1}")
print(f"  2. Environment analysis:      {output2}")
print(f"  3. Performance heatmap:       {output3}")
print(f"  4. Statistical analysis:      {output4}")
print(f"  5. Cost-benefit analysis:     {output5}")
print(f"  6. Algorithm characteristics: {output6}")
print(f"  7. Summary table:             {output7}")

print("\n" + "="*80)
print("✅ PUBLICATION-GRADE ANALYSIS COMPLETE")
print("="*80)