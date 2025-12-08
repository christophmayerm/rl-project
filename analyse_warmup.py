# analyze_warmup.py
import json
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Load warmup sweep results
files = glob.glob('results/sweeps/sweep_ts_small_warmup_sweep_*.json')
if not files:
    print("❌ No warmup sweep results found!")
    exit(1)

latest = sorted(files)[-1]
print(f"📂 Loading: {latest}")

with open(latest) as f:
    results = json.load(f)

df = pd.DataFrame(results)

# Group by K_warmup_ratio
grouped = df.groupby('K_warmup_ratio').agg({
    'final_performance': ['mean', 'std'],
    'improvement': 'mean',
    'iterations': 'mean'
}).round(4)

print("\n" + "="*80)
print("WARMUP SWEEP RESULTS")
print("="*80)
print(grouped.sort_values(('final_performance', 'mean'), ascending=False))

# Best warmup
best_warmup = grouped[('final_performance', 'mean')].idxmax()
best_performance = grouped.loc[best_warmup, ('final_performance', 'mean')]

print("\n" + "="*80)
print("OPTIMAL CONFIGURATION")
print("="*80)
print(f"K_warmup_ratio: {best_warmup}")
print(f"Performance: {best_performance:.6f}")
print("="*80)

# ============================================================================
# PUBLICATION-READY PLOTS
# ============================================================================

# Create figure with 3 subplots
fig = plt.figure(figsize=(16, 5))
gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

# ---------------------------------------------------------------------------
# Plot 1: Performance vs K_warmup_ratio with Smooth Curve
# ---------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])

warmup_stats = df.groupby('K_warmup_ratio')['final_performance'].agg(['mean', 'std'])
x = np.array(warmup_stats.index)
y_mean = warmup_stats['mean'].values
y_std = warmup_stats['std'].values

# Plot with error bars
ax1.errorbar(x, y_mean, yerr=y_std, fmt='o', markersize=10, 
             capsize=6, capthick=2, linewidth=2, 
             color='#2E86AB', ecolor='#A23B72', 
             label='Measured Performance', alpha=0.8)

# Add smooth interpolation curve
if len(x) > 3:
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y_mean, k=3)
    y_smooth = spl(x_smooth)
    ax1.plot(x_smooth, y_smooth, '--', color='#F18F01', 
             linewidth=2, alpha=0.6, label='Trend')

# Mark optimal point
ax1.axvline(best_warmup, color='red', linestyle='--', alpha=0.5, linewidth=2,
            label=f'Optimal: {best_warmup:.2f}')
ax1.scatter([best_warmup], [best_performance], 
            color='red', s=300, marker='*', zorder=5, 
            edgecolors='black', linewidth=2)

# Add shaded region for "good" performance (within 1 std of best)
performance_threshold = best_performance - y_std[list(x).index(best_warmup)]
ax1.axhspan(performance_threshold, best_performance + 0.01, 
            alpha=0.1, color='green', label='High Performance Zone')

ax1.set_xlabel('Warmup Ratio (K_warmup / max_iter)', fontweight='bold')
ax1.set_ylabel('Final Performance', fontweight='bold')
ax1.set_title('Performance vs Curriculum Warmup Duration', fontweight='bold', pad=15)
ax1.legend(framealpha=0.9, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([x.min() - 0.05, x.max() + 0.05])

# ---------------------------------------------------------------------------
# Plot 2: Performance and Iterations (Dual Y-axis)
# ---------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2_twin = ax2.twinx()

iter_stats = df.groupby('K_warmup_ratio')['iterations'].agg(['mean', 'std'])
perf_stats = df.groupby('K_warmup_ratio')['final_performance'].agg(['mean', 'std'])

x = np.array(perf_stats.index)

# Performance line
line1 = ax2.plot(x, perf_stats['mean'], marker='o', linewidth=2.5, 
                 markersize=10, color='#2E86AB', label='Performance', alpha=0.8)
ax2.fill_between(x, perf_stats['mean'] - perf_stats['std'], 
                  perf_stats['mean'] + perf_stats['std'], 
                  alpha=0.2, color='#2E86AB')

# Iterations line
line2 = ax2_twin.plot(x, iter_stats['mean'], marker='s', linewidth=2.5, 
                      markersize=8, color='#F18F01', label='Iterations', alpha=0.8)
ax2_twin.fill_between(x, iter_stats['mean'] - iter_stats['std'], 
                       iter_stats['mean'] + iter_stats['std'], 
                       alpha=0.2, color='#F18F01')

ax2.set_xlabel('Warmup Ratio', fontweight='bold')
ax2.set_ylabel('Final Performance', fontweight='bold', color='#2E86AB')
ax2_twin.set_ylabel('Iterations to Convergence', fontweight='bold', color='#F18F01')
ax2.set_title('Performance vs Convergence Speed', fontweight='bold', pad=15)
ax2.tick_params(axis='y', labelcolor='#2E86AB')
ax2_twin.tick_params(axis='y', labelcolor='#F18F01')
ax2.grid(True, alpha=0.3)

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper left', framealpha=0.9)

# ---------------------------------------------------------------------------
# Plot 3: Distribution Analysis (Violin Plot)
# ---------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])

# Prepare data for violin plot
violin_data = []
violin_labels = []
for warmup in sorted(df['K_warmup_ratio'].unique()):
    violin_data.append(df[df['K_warmup_ratio'] == warmup]['final_performance'].values)
    violin_labels.append(f'{warmup:.2f}')

parts = ax3.violinplot(violin_data, positions=range(len(violin_labels)),
                        showmeans=True, showmedians=True, widths=0.7)

# Customize violin plot colors
for pc in parts['bodies']:
    pc.set_facecolor('#2E86AB')
    pc.set_alpha(0.6)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

parts['cmeans'].set_color('#F18F01')
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color('red')
parts['cmedians'].set_linewidth(2)
parts['cbars'].set_color('black')
parts['cmaxes'].set_color('black')
parts['cmins'].set_color('black')

# Highlight optimal
optimal_idx = violin_labels.index(f'{best_warmup:.2f}')
ax3.axvline(optimal_idx, color='red', linestyle='--', alpha=0.5, linewidth=2)

ax3.set_xticks(range(len(violin_labels)))
ax3.set_xticklabels(violin_labels, rotation=45, ha='right')
ax3.set_xlabel('Warmup Ratio', fontweight='bold')
ax3.set_ylabel('Final Performance', fontweight='bold')
ax3.set_title('Performance Distribution by Warmup', fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3, axis='y')

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=2, label='Median'),
    Line2D([0], [0], color='#F18F01', linewidth=2, label='Mean'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Optimal')
]
ax3.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

# Main title
fig.suptitle('SA-PMI Warmup Sweep Analysis: Teacher-Student Small Environment', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save high-resolution figure
output_path = 'results/warmup_sweep_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ High-resolution plot saved: {output_path}")

# ============================================================================
# ADDITIONAL PLOT: Improvement vs Warmup
# ============================================================================
fig2, ax = plt.subplots(figsize=(10, 6))

improvement_stats = df.groupby('K_warmup_ratio')['improvement'].agg(['mean', 'std'])
x = np.array(improvement_stats.index)
y_mean = improvement_stats['mean'].values
y_std = improvement_stats['std'].values

# Plot bars with error bars
bars = ax.bar(x, y_mean, width=0.04, color='#2E86AB', alpha=0.7, 
              edgecolor='black', linewidth=1.5)
ax.errorbar(x, y_mean, yerr=y_std, fmt='none', ecolor='#A23B72', 
            capsize=8, capthick=2, linewidth=2)

# Highlight best
best_idx = list(x).index(best_warmup)
bars[best_idx].set_color('#90EE90')
bars[best_idx].set_edgecolor('red')
bars[best_idx].set_linewidth(2.5)

ax.axvline(best_warmup, color='red', linestyle='--', alpha=0.5, linewidth=2,
           label=f'Optimal: {best_warmup:.2f}')

ax.set_xlabel('Warmup Ratio', fontweight='bold', fontsize=12)
ax.set_ylabel('Performance Improvement', fontweight='bold', fontsize=12)
ax.set_title('Performance Improvement vs Warmup Ratio', 
             fontweight='bold', fontsize=14, pad=15)
ax.legend(framealpha=0.9, loc='best', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path2 = 'results/warmup_improvement_analysis.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Improvement plot saved: {output_path2}")

# ============================================================================
# SUMMARY TABLE IMAGE
# ============================================================================
fig3, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Prepare summary data
summary_data = []
for warmup in sorted(df['K_warmup_ratio'].unique(), reverse=True):
    warmup_df = df[df['K_warmup_ratio'] == warmup]
    summary_data.append([
        f'{warmup:.2f}',
        f'{warmup_df["final_performance"].mean():.6f}',
        f'±{warmup_df["final_performance"].std():.6f}',
        f'{warmup_df["improvement"].mean():.6f}',
        f'{int(warmup_df["iterations"].mean())}',
        f'±{int(warmup_df["iterations"].std())}'
    ])

summary_df = pd.DataFrame(summary_data, 
                          columns=['K_warmup', 'Performance', 'Std', 
                                  'Improvement', 'Iterations', 'Iter Std'])

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
table.set_fontsize(11)
table.scale(1, 3)

# Style header
for i in range(len(summary_df.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', size=12)

# Alternate row colors
for i in range(1, len(summary_df) + 1):
    for j in range(len(summary_df.columns)):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('#FFFFFF')
    
    # Highlight best row (green)
    if i == 1:
        for j in range(len(summary_df.columns)):
            cell = table[(i, j)]
            cell.set_facecolor('#90EE90')
            cell.set_text_props(weight='bold', size=11)

# Add annotation
annotation_text = (
    f"Optimal Configuration: K_warmup_ratio = {best_warmup:.2f}\n"
    f"Best Performance: {best_performance:.6f}\n"
    f"Note: Warmup ratio determines when full adversarial budget is reached"
)
ax.text(0.5, -0.05, annotation_text, 
        ha='center', va='top', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        transform=ax.transAxes)

plt.title('Warmup Sweep Summary: All Configurations', 
          fontweight='bold', fontsize=14, pad=20)
plt.tight_layout()

output_path3 = 'results/warmup_summary_table.png'
plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Summary table saved: {output_path3}")

# ============================================================================
# CONVERGENCE SPEED ANALYSIS
# ============================================================================
fig4, ax = plt.subplots(figsize=(10, 6))

# Calculate efficiency metric: performance / iterations
efficiency_data = []
for warmup in sorted(df['K_warmup_ratio'].unique()):
    warmup_df = df[df['K_warmup_ratio'] == warmup]
    perf = warmup_df['final_performance'].mean()
    iters = warmup_df['iterations'].mean()
    efficiency = perf / iters * 1000  # Scale for readability
    efficiency_data.append((warmup, efficiency, perf, iters))

efficiency_df = pd.DataFrame(efficiency_data, 
                             columns=['warmup', 'efficiency', 'performance', 'iterations'])

# Create scatter plot with size proportional to performance
scatter = ax.scatter(efficiency_df['warmup'], 
                     efficiency_df['efficiency'],
                     s=efficiency_df['performance'] * 100,
                     c=efficiency_df['iterations'],
                     cmap='viridis_r', alpha=0.7,
                     edgecolors='black', linewidth=2)

# Highlight optimal
optimal_row = efficiency_df[efficiency_df['warmup'] == best_warmup].iloc[0]
ax.scatter([best_warmup], [optimal_row['efficiency']], 
           s=500, marker='*', color='red', 
           edgecolors='black', linewidth=2, zorder=5,
           label=f'Optimal: {best_warmup:.2f}')

ax.set_xlabel('Warmup Ratio', fontweight='bold', fontsize=12)
ax.set_ylabel('Training Efficiency (Performance/Iteration × 1000)', 
              fontweight='bold', fontsize=12)
ax.set_title('Convergence Efficiency Analysis', fontweight='bold', fontsize=14, pad=15)
ax.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Iterations to Convergence', rotation=270, labelpad=20, fontweight='bold')

# Add legend
from matplotlib.patches import Circle
legend_elements = [
    ax.scatter([], [], s=100, c='gray', alpha=0.7, edgecolors='black', label='Size ∝ Performance'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
           markersize=15, label='Optimal', markeredgecolor='black', markeredgewidth=1.5)
]
ax.legend(handles=legend_elements, loc='best', framealpha=0.9, fontsize=11)

plt.tight_layout()
output_path4 = 'results/warmup_efficiency_analysis.png'
plt.savefig(output_path4, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Efficiency plot saved: {output_path4}")

print("\n" + "="*80)
print("📊 ALL PLOTS GENERATED")
print("="*80)
print(f"1. Main analysis:       {output_path}")
print(f"2. Improvement curve:   {output_path2}")
print(f"3. Summary table:       {output_path3}")
print(f"4. Efficiency analysis: {output_path4}")
print("="*80)

print(f"\n🎯 RECOMMENDATION: Use K_warmup_ratio = {best_warmup:.2f}")