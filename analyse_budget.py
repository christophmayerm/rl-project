# analyze_budget.py
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

# Load budget sweep results
files = glob.glob('results/sweeps/sweep_ts_small_budget_sweep_*.json')
if not files:
    print("❌ No budget sweep results found!")
    exit(1)

latest = sorted(files)[-1]
print(f"📂 Loading: {latest}")

with open(latest) as f:
    results = json.load(f)

df = pd.DataFrame(results)

# Group by B_max and schedule
grouped = df.groupby(['B_max', 'curriculum_schedule']).agg({
    'final_performance': ['mean', 'std'],
    'improvement': 'mean',
    'iterations': 'mean'
}).round(4)

print("\n" + "="*80)
print("BUDGET SWEEP RESULTS")
print("="*80)
print(grouped.sort_values(('final_performance', 'mean'), ascending=False))

# Find optimal B_max
best = df.loc[df['final_performance'].idxmax()]
print("\n" + "="*80)
print("OPTIMAL CONFIGURATION")
print("="*80)
print(f"B_max: {best['B_max']}")
print(f"Schedule: {best['curriculum_schedule']}")
print(f"Performance: {best['final_performance']:.6f}")
print(f"Improvement: {best['improvement']:.6f}")
print("="*80)

# Recommend top 3 B_max values
top_b_max = df.groupby('B_max')['final_performance'].mean().nlargest(3)
print("\nTop 3 B_max values:")
for bmax, perf in top_b_max.items():
    print(f"  B_max={bmax}: {perf:.6f}")

# ============================================================================
# PUBLICATION-READY PLOTS
# ============================================================================

# Create figure with 3 subplots
fig = plt.figure(figsize=(16, 5))
gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

# ---------------------------------------------------------------------------
# Plot 1: Performance vs B_max (Main Result)
# ---------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])

for schedule in df['curriculum_schedule'].unique():
    schedule_df = df[df['curriculum_schedule'] == schedule]
    grouped_schedule = schedule_df.groupby('B_max').agg({
        'final_performance': ['mean', 'std']
    })
    
    x = grouped_schedule.index
    y_mean = grouped_schedule[('final_performance', 'mean')]
    y_std = grouped_schedule[('final_performance', 'std')]
    
    ax1.plot(x, y_mean, marker='o', linewidth=2, markersize=8, 
             label=schedule.capitalize(), alpha=0.8)
    ax1.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.15)

# Mark optimal point
ax1.axvline(best['B_max'], color='red', linestyle='--', alpha=0.5, 
            label=f'Optimal: B_max={best["B_max"]}')
ax1.scatter([best['B_max']], [best['final_performance']], 
            color='red', s=200, marker='*', zorder=5, edgecolors='black', linewidth=1.5)

ax1.set_xlabel('Adversarial Budget (B_max)', fontweight='bold')
ax1.set_ylabel('Final Performance', fontweight='bold')
ax1.set_title('Performance vs Adversarial Budget', fontweight='bold', pad=15)
ax1.legend(framealpha=0.9, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# ---------------------------------------------------------------------------
# Plot 2: Box Plot - Performance Distribution by B_max
# ---------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])

# Prepare data for box plot
box_data = []
box_labels = []
for bmax in sorted(df['B_max'].unique()):
    box_data.append(df[df['B_max'] == bmax]['final_performance'].values)
    box_labels.append(f'{bmax}')

bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True,
                  showmeans=True, meanline=True,
                  medianprops=dict(color='red', linewidth=2),
                  meanprops=dict(color='blue', linewidth=2, linestyle='--'),
                  boxprops=dict(facecolor='lightblue', alpha=0.6),
                  whiskerprops=dict(linewidth=1.5),
                  capprops=dict(linewidth=1.5))

ax2.set_xlabel('Adversarial Budget (B_max)', fontweight='bold')
ax2.set_ylabel('Final Performance', fontweight='bold')
ax2.set_title('Performance Distribution by Budget', fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', rotation=45)

# Add legend for median/mean
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=2, label='Median'),
    Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
]
ax2.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

# ---------------------------------------------------------------------------
# Plot 3: Heatmap - Schedule vs Budget
# ---------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])

# Create pivot table for heatmap
pivot = df.pivot_table(
    values='final_performance',
    index='curriculum_schedule',
    columns='B_max',
    aggfunc='mean'
)

# Plot heatmap
im = ax3.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=pivot.values.min(), vmax=pivot.values.max())

# Set ticks and labels
ax3.set_xticks(np.arange(len(pivot.columns)))
ax3.set_yticks(np.arange(len(pivot.index)))
ax3.set_xticklabels([f'{x:.3f}' for x in pivot.columns], rotation=45, ha='right')
ax3.set_yticklabels([s.capitalize() for s in pivot.index])

# Add colorbar
cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label('Final Performance', rotation=270, labelpad=20, fontweight='bold')

# Add text annotations
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        value = pivot.values[i, j]
        if not np.isnan(value):
            text_color = 'white' if value < (pivot.values.min() + 0.7*(pivot.values.max()-pivot.values.min())) else 'black'
            ax3.text(j, i, f'{value:.3f}', ha='center', va='center',
                    color=text_color, fontsize=9, fontweight='bold')

ax3.set_xlabel('Adversarial Budget (B_max)', fontweight='bold')
ax3.set_ylabel('Curriculum Schedule', fontweight='bold')
ax3.set_title('Performance Heatmap', fontweight='bold', pad=15)

# Main title
fig.suptitle('SA-PMI Budget Sweep Analysis: Teacher-Student Small Environment', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save high-resolution figure
output_path = 'results/budget_sweep_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ High-resolution plot saved: {output_path}")

# ============================================================================
# ADDITIONAL PLOT: Improvement vs Budget
# ============================================================================
fig2, ax = plt.subplots(figsize=(10, 6))

for schedule in df['curriculum_schedule'].unique():
    schedule_df = df[df['curriculum_schedule'] == schedule]
    grouped_schedule = schedule_df.groupby('B_max').agg({
        'improvement': ['mean', 'std']
    })
    
    x = grouped_schedule.index
    y_mean = grouped_schedule[('improvement', 'mean')]
    y_std = grouped_schedule[('improvement', 'std')]
    
    ax.plot(x, y_mean, marker='s', linewidth=2, markersize=8, 
            label=schedule.capitalize(), alpha=0.8)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.15)

ax.axvline(best['B_max'], color='red', linestyle='--', alpha=0.5, 
           label=f'Optimal: B_max={best["B_max"]}')

ax.set_xlabel('Adversarial Budget (B_max)', fontweight='bold', fontsize=12)
ax.set_ylabel('Performance Improvement', fontweight='bold', fontsize=12)
ax.set_title('Performance Improvement vs Adversarial Budget', 
             fontweight='bold', fontsize=14, pad=15)
ax.legend(framealpha=0.9, loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

plt.tight_layout()
output_path2 = 'results/budget_improvement_analysis.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Improvement plot saved: {output_path2}")

# ============================================================================
# SUMMARY TABLE IMAGE
# ============================================================================
fig3, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Prepare summary data
summary_data = []
for schedule in df['curriculum_schedule'].unique():
    schedule_df = df[df['curriculum_schedule'] == schedule]
    for bmax in sorted(df['B_max'].unique()):
        bmax_df = schedule_df[schedule_df['B_max'] == bmax]
        if len(bmax_df) > 0:
            summary_data.append([
                schedule.capitalize(),
                f'{bmax:.3f}',
                f'{bmax_df["final_performance"].mean():.4f} ± {bmax_df["final_performance"].std():.4f}',
                f'{bmax_df["improvement"].mean():.4f}',
                f'{int(bmax_df["iterations"].mean())}'
            ])

# Sort by performance
summary_df = pd.DataFrame(summary_data, 
                          columns=['Schedule', 'B_max', 'Performance (mean ± std)', 
                                  'Improvement', 'Iterations'])
summary_df['sort_perf'] = summary_df['Performance (mean ± std)'].str.split(' ').str[0].astype(float)
summary_df = summary_df.sort_values('sort_perf', ascending=False).drop('sort_perf', axis=1)

# Create table
table = ax.table(cellText=summary_df.head(15).values,
                colLabels=summary_df.columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(len(summary_df.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, min(16, len(summary_df) + 1)):
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

plt.title('Budget Sweep Summary: Top 15 Configurations', 
          fontweight='bold', fontsize=14, pad=20)
plt.tight_layout()

output_path3 = 'results/budget_summary_table.png'
plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Summary table saved: {output_path3}")

print("\n" + "="*80)
print("📊 ALL PLOTS GENERATED")
print("="*80)
print(f"1. Main analysis:     {output_path}")
print(f"2. Improvement curve: {output_path2}")
print(f"3. Summary table:     {output_path3}")
print("="*80)