# analyze_bmin.py
import json
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Load B_min sweep results
files = glob.glob('results/sweeps/sweep_ts_small_bmin_sweep_*.json')
if not files:
    print("❌ No B_min sweep results found!")
    exit(1)

latest = sorted(files)[-1]
print(f"📂 Loading: {latest}")

with open(latest) as f:
    results = json.load(f)

df = pd.DataFrame(results)

# Group by B_min
grouped = df.groupby('B_min').agg({
    'final_performance': ['mean', 'std'],
    'improvement': 'mean',
    'iterations': 'mean',
    'final_state_coverage': 'mean',
    'final_state_entropy': 'mean',
    'final_novelty_yield': 'mean',
    'final_model_tv_mean': 'mean',
    'final_model_tv_max': 'mean'
}).round(4)

print("\n" + "="*80)
print("B_MIN SWEEP RESULTS")
print("="*80)
print(grouped.sort_values(('final_performance', 'mean'), ascending=False))

# Best B_min
best_bmin = grouped[('final_performance', 'mean')].idxmax()
best_performance = grouped.loc[best_bmin, ('final_performance', 'mean')]

print("\n" + "="*80)
print("OPTIMAL CONFIGURATION")
print("="*80)
print(f"B_min: {best_bmin}")
print(f"Performance: {best_performance:.6f}")
print(f"State Coverage: {grouped.loc[best_bmin, ('final_state_coverage', 'mean')]:.4f}")
print(f"Model TV Mean: {grouped.loc[best_bmin, ('final_model_tv_mean', 'mean')]:.4f}")
print("="*80)

# ============================================================================
# FIGURE 1: MAIN PERFORMANCE ANALYSIS (4 subplots)
# ============================================================================
fig1 = plt.figure(figsize=(16, 10))
gs = fig1.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ---------------------------------------------------------------------------
# Plot 1.1: Performance vs B_min
# ---------------------------------------------------------------------------
ax1 = fig1.add_subplot(gs[0, 0])

perf_stats = df.groupby('B_min')['final_performance'].agg(['mean', 'std'])
x = np.array(perf_stats.index)
y_mean = perf_stats['mean'].values
y_std = perf_stats['std'].values

# Bar plot with error bars
bars = ax1.bar(x, y_mean, width=x[1]-x[0] if len(x) > 1 else 0.0005, 
               color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.errorbar(x, y_mean, yerr=y_std, fmt='none', ecolor='#A23B72', 
            capsize=8, capthick=2, linewidth=2)

# Highlight best
best_idx = list(x).index(best_bmin)
bars[best_idx].set_color('#90EE90')
bars[best_idx].set_edgecolor('red')
bars[best_idx].set_linewidth(2.5)

ax1.axvline(best_bmin, color='red', linestyle='--', alpha=0.5, linewidth=2,
           label=f'Optimal: B_min={best_bmin}')

ax1.set_xlabel('Minimum Adversarial Budget (B_min)', fontweight='bold')
ax1.set_ylabel('Final Performance', fontweight='bold')
ax1.set_title('Performance vs Initial Adversarial Pressure', fontweight='bold', pad=15)
ax1.legend(framealpha=0.9, loc='best')
ax1.grid(True, alpha=0.3, axis='y')

# ---------------------------------------------------------------------------
# Plot 1.2: Improvement vs B_min
# ---------------------------------------------------------------------------
ax2 = fig1.add_subplot(gs[0, 1])

improve_stats = df.groupby('B_min')['improvement'].agg(['mean', 'std'])
x = np.array(improve_stats.index)
y_mean = improve_stats['mean'].values
y_std = improve_stats['std'].values

ax2.plot(x, y_mean, marker='o', linewidth=2.5, markersize=10, 
         color='#F18F01', label='Improvement', alpha=0.8)
ax2.fill_between(x, y_mean - y_std, y_mean + y_std, 
                 alpha=0.2, color='#F18F01')

ax2.axvline(best_bmin, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax2.scatter([best_bmin], [improve_stats.loc[best_bmin, 'mean']], 
           color='red', s=300, marker='*', zorder=5, 
           edgecolors='black', linewidth=2)

ax2.set_xlabel('Minimum Adversarial Budget (B_min)', fontweight='bold')
ax2.set_ylabel('Performance Improvement', fontweight='bold')
ax2.set_title('Learning Efficiency vs Initial Pressure', fontweight='bold', pad=15)
ax2.legend(framealpha=0.9, loc='best')
ax2.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 1.3: Convergence Speed
# ---------------------------------------------------------------------------
ax3 = fig1.add_subplot(gs[1, 0])

iter_stats = df.groupby('B_min')['iterations'].agg(['mean', 'std'])
x = np.array(iter_stats.index)
y_mean = iter_stats['mean'].values
y_std = iter_stats['std'].values

bars = ax3.bar(x, y_mean, width=x[1]-x[0] if len(x) > 1 else 0.0005,
               color='#A23B72', alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.errorbar(x, y_mean, yerr=y_std, fmt='none', ecolor='black', 
            capsize=8, capthick=2, linewidth=2)

# Highlight best
bars[best_idx].set_color('#FFD700')
bars[best_idx].set_edgecolor('red')
bars[best_idx].set_linewidth(2.5)

ax3.set_xlabel('Minimum Adversarial Budget (B_min)', fontweight='bold')
ax3.set_ylabel('Iterations to Convergence', fontweight='bold')
ax3.set_title('Convergence Speed vs Initial Pressure', fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3, axis='y')

# ---------------------------------------------------------------------------
# Plot 1.4: Performance-Efficiency Scatter
# ---------------------------------------------------------------------------
ax4 = fig1.add_subplot(gs[1, 1])

bmin_values = []
performance_values = []
iteration_values = []

for bmin in sorted(df['B_min'].unique()):
    bmin_df = df[df['B_min'] == bmin]
    bmin_values.append(bmin)
    performance_values.append(bmin_df['final_performance'].mean())
    iteration_values.append(bmin_df['iterations'].mean())

# Scatter with bubble size
scatter = ax4.scatter(iteration_values, performance_values, 
                     s=np.array(bmin_values) * 10000 + 100,
                     c=bmin_values, cmap='viridis', alpha=0.7,
                     edgecolors='black', linewidth=2)

# Highlight optimal
optimal_idx = bmin_values.index(best_bmin)
ax4.scatter([iteration_values[optimal_idx]], [performance_values[optimal_idx]], 
           s=500, marker='*', color='red', 
           edgecolors='black', linewidth=2, zorder=5)

ax4.set_xlabel('Iterations to Convergence', fontweight='bold')
ax4.set_ylabel('Final Performance', fontweight='bold')
ax4.set_title('Performance-Efficiency Trade-off', fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('B_min Value', rotation=270, labelpad=20, fontweight='bold')

# Pareto front line
sorted_indices = np.argsort(iteration_values)
ax4.plot(np.array(iteration_values)[sorted_indices], 
        np.array(performance_values)[sorted_indices],
        '--', color='gray', alpha=0.5, linewidth=1.5, label='Trend')
ax4.legend(loc='lower right', framealpha=0.9)

fig1.suptitle('SA-PMI B_min Sweep: Performance Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
output_path1 = 'results/bmin_performance_analysis.png'
plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Performance analysis saved: {output_path1}")

# ============================================================================
# FIGURE 2: AUXILIARY METRICS ANALYSIS (Exploration & Robustness)
# ============================================================================
fig2 = plt.figure(figsize=(16, 10))
gs = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ---------------------------------------------------------------------------
# Plot 2.1: State Coverage
# ---------------------------------------------------------------------------
ax1 = fig2.add_subplot(gs[0, 0])

coverage_stats = df.groupby('B_min')['final_state_coverage'].agg(['mean', 'std'])
x = np.array(coverage_stats.index)
y_mean = coverage_stats['mean'].values
y_std = coverage_stats['std'].values

ax1.plot(x, y_mean, marker='o', linewidth=2.5, markersize=10, 
         color='#2E86AB', label='State Coverage', alpha=0.8)
ax1.fill_between(x, y_mean - y_std, y_mean + y_std, 
                 alpha=0.2, color='#2E86AB')

ax1.axvline(best_bmin, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax1.axhline(1.0, color='green', linestyle=':', alpha=0.5, linewidth=2, 
           label='Full Coverage')

ax1.set_xlabel('Minimum Adversarial Budget (B_min)', fontweight='bold')
ax1.set_ylabel('State Coverage', fontweight='bold')
ax1.set_title('Exploration: State Coverage vs B_min', fontweight='bold', pad=15)
ax1.legend(framealpha=0.9, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.1])

# ---------------------------------------------------------------------------
# Plot 2.2: State Entropy (Exploration Diversity)
# ---------------------------------------------------------------------------
ax2 = fig2.add_subplot(gs[0, 1])

entropy_stats = df.groupby('B_min')['final_state_entropy'].agg(['mean', 'std'])
x = np.array(entropy_stats.index)
y_mean = entropy_stats['mean'].values
y_std = entropy_stats['std'].values

ax2.plot(x, y_mean, marker='s', linewidth=2.5, markersize=10, 
         color='#F18F01', label='State Entropy', alpha=0.8)
ax2.fill_between(x, y_mean - y_std, y_mean + y_std, 
                 alpha=0.2, color='#F18F01')

ax2.axvline(best_bmin, color='red', linestyle='--', alpha=0.5, linewidth=2)

ax2.set_xlabel('Minimum Adversarial Budget (B_min)', fontweight='bold')
ax2.set_ylabel('Normalized State Entropy', fontweight='bold')
ax2.set_title('Exploration: State Visitation Entropy vs B_min', fontweight='bold', pad=15)
ax2.legend(framealpha=0.9, loc='best')
ax2.grid(True, alpha=0.3)

# Add interpretation text
entropy_interpretation = "Higher = More uniform exploration\nLower = More focused policy"
ax2.text(0.05, 0.95, entropy_interpretation, transform=ax2.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ---------------------------------------------------------------------------
# Plot 2.3: Model Divergence (Robustness)
# ---------------------------------------------------------------------------
ax3 = fig2.add_subplot(gs[1, 0])

tv_mean_stats = df.groupby('B_min')['final_model_tv_mean'].agg(['mean', 'std'])
tv_max_stats = df.groupby('B_min')['final_model_tv_max'].agg(['mean', 'std'])
x = np.array(tv_mean_stats.index)

# Plot both mean and max TV distance
ax3.plot(x, tv_mean_stats['mean'], marker='o', linewidth=2.5, markersize=10, 
         color='#2E86AB', label='Mean TV Distance', alpha=0.8)
ax3.fill_between(x, 
                 tv_mean_stats['mean'] - tv_mean_stats['std'], 
                 tv_mean_stats['mean'] + tv_mean_stats['std'], 
                 alpha=0.2, color='#2E86AB')

ax3.plot(x, tv_max_stats['mean'], marker='s', linewidth=2.5, markersize=10, 
         color='#A23B72', label='Max TV Distance', alpha=0.8)
ax3.fill_between(x, 
                 tv_max_stats['mean'] - tv_max_stats['std'], 
                 tv_max_stats['mean'] + tv_max_stats['std'], 
                 alpha=0.2, color='#A23B72')

ax3.axvline(best_bmin, color='red', linestyle='--', alpha=0.5, linewidth=2,
           label=f'Optimal B_min')

ax3.set_xlabel('Minimum Adversarial Budget (B_min)', fontweight='bold')
ax3.set_ylabel('Total Variation Distance', fontweight='bold')
ax3.set_title('Robustness: Model Divergence vs B_min', fontweight='bold', pad=15)
ax3.legend(framealpha=0.9, loc='best')
ax3.grid(True, alpha=0.3)

# Add interpretation text
tv_interpretation = "Higher = More adversarial perturbation\n= Stronger robustness training"
ax3.text(0.05, 0.95, tv_interpretation, transform=ax3.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ---------------------------------------------------------------------------
# Plot 2.4: Multi-Metric Radar Chart
# ---------------------------------------------------------------------------
ax4 = fig2.add_subplot(gs[1, 1], projection='polar')

# Prepare normalized metrics for radar chart
metrics_to_plot = ['final_performance', 'final_state_coverage', 
                   'final_state_entropy', 'final_model_tv_mean']
metric_labels = ['Performance', 'Coverage', 'Entropy', 'Robustness']

# Normalize each metric to 0-1 range
normalized_data = {}
for bmin in sorted(df['B_min'].unique()):
    bmin_df = df[df['B_min'] == bmin]
    normalized_data[bmin] = []
    for metric in metrics_to_plot:
        value = bmin_df[metric].mean()
        # Normalize
        all_values = df[metric].values
        norm_value = (value - all_values.min()) / (all_values.max() - all_values.min() + 1e-8)
        normalized_data[bmin].append(norm_value)

# Plot radar chart for each B_min
angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

colors = plt.cm.viridis(np.linspace(0, 1, len(normalized_data)))

for idx, (bmin, values) in enumerate(normalized_data.items()):
    values += values[:1]  # Complete the circle
    linewidth = 3 if bmin == best_bmin else 1.5
    alpha = 0.8 if bmin == best_bmin else 0.4
    label = f'B_min={bmin}' + (' ⭐' if bmin == best_bmin else '')
    
    ax4.plot(angles, values, 'o-', linewidth=linewidth, 
            color=colors[idx], label=label, alpha=alpha)
    ax4.fill(angles, values, alpha=0.15 if bmin == best_bmin else 0.05, 
            color=colors[idx])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(metric_labels, fontweight='bold')
ax4.set_ylim(0, 1)
ax4.set_title('Multi-Metric Comparison', fontweight='bold', pad=20, y=1.08)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
ax4.grid(True, alpha=0.3)

fig2.suptitle('SA-PMI B_min Sweep: Exploration & Robustness Metrics', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
output_path2 = 'results/bmin_auxiliary_metrics.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Auxiliary metrics analysis saved: {output_path2}")

# ============================================================================
# FIGURE 3: SUMMARY TABLE
# ============================================================================
fig3, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

# Prepare comprehensive summary
summary_data = []
for bmin in sorted(df['B_min'].unique()):
    bmin_df = df[df['B_min'] == bmin]
    summary_data.append([
        f'{bmin:.4f}',
        f'{bmin_df["final_performance"].mean():.6f}',
        f'±{bmin_df["final_performance"].std():.6f}',
        f'{bmin_df["improvement"].mean():.6f}',
        f'{int(bmin_df["iterations"].mean())}',
        f'{bmin_df["final_state_coverage"].mean():.4f}',
        f'{bmin_df["final_state_entropy"].mean():.4f}',
        f'{bmin_df["final_model_tv_mean"].mean():.4f}'
    ])

summary_df = pd.DataFrame(summary_data, 
                          columns=['B_min', 'Performance', 'Std', 'Improvement', 
                                  'Iterations', 'Coverage', 'Entropy', 'Model TV'])

# Sort by performance
summary_df['sort_perf'] = summary_df['Performance'].astype(float)
summary_df = summary_df.sort_values('sort_perf', ascending=False).drop('sort_perf', axis=1)

# Create table
table = ax.table(cellText=summary_df.values,
                colLabels=summary_df.columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.8)

# Style header
for i in range(len(summary_df.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', size=11)

# Alternate row colors
for i in range(1, len(summary_df) + 1):
    for j in range(len(summary_df.columns)):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('#FFFFFF')
    
    # Highlight best row
    if i == 1:
        for j in range(len(summary_df.columns)):
            cell = table[(i, j)]
            cell.set_facecolor('#90EE90')
            cell.set_text_props(weight='bold')

# Add interpretation annotation
interpretation_text = (
    f"Optimal Configuration: B_min = {best_bmin:.4f}\n\n"
    "Key Insights:\n"
    "• Coverage: Exploration completeness (1.0 = all states visited)\n"
    "• Entropy: Exploration uniformity (higher = more diverse)\n"
    "• Model TV: Adversarial perturbation strength (higher = more robust training)"
)
ax.text(0.5, -0.08, interpretation_text, 
        ha='center', va='top', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        transform=ax.transAxes)

plt.title('B_min Sweep: Comprehensive Summary', 
          fontweight='bold', fontsize=14, pad=20)
plt.tight_layout()

output_path3 = 'results/bmin_summary_table.png'
plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Summary table saved: {output_path3}")

# ============================================================================
# CONSOLE OUTPUT: KEY FINDINGS
# ============================================================================
print("\n" + "="*80)
print("📊 KEY FINDINGS")
print("="*80)

# Compare B_min=0 vs non-zero
if 0.0 in df['B_min'].values:
    baseline = df[df['B_min'] == 0.0]['final_performance'].mean()
    print(f"\nBaseline (B_min=0.0) Performance: {baseline:.6f}")
    
    for bmin in sorted(df['B_min'].unique()):
        if bmin > 0:
            bmin_perf = df[df['B_min'] == bmin]['final_performance'].mean()
            improvement = ((bmin_perf - baseline) / baseline) * 100
            print(f"B_min={bmin:.4f}: {bmin_perf:.6f} ({improvement:+.2f}% vs baseline)")

print("\n" + "="*80)
print("📊 ALL PLOTS GENERATED")
print("="*80)
print(f"1. Performance analysis:    {output_path1}")
print(f"2. Auxiliary metrics:       {output_path2}")
print(f"3. Summary table:           {output_path3}")
print("="*80)

print(f"\n🎯 RECOMMENDATION: Use B_min = {best_bmin:.4f}")

# Print interpretation
print("\n💡 INTERPRETATION:")
if best_bmin == 0.0:
    print("  • Starting with NO adversarial pressure (B_min=0) is optimal")
    print("  • Curriculum should gradually introduce adversarial training")
    print("  • Early adversarial pressure may hurt initial exploration")
else:
    print(f"  • Starting with B_min={best_bmin:.4f} provides better robustness")
    print("  • Early adversarial pressure helps shape exploration")
    print("  • Policy learns robust behavior from the start")