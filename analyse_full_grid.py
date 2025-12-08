# analyze_full_grid.py
import json
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from itertools import product

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Load full grid sweep results
files = glob.glob('results/sweeps/sweep_ts_small_full_grid_*.json')
if not files:
    print("❌ No full grid sweep results found!")
    exit(1)

latest = sorted(files)[-1]
print(f"📂 Loading: {latest}")

with open(latest) as f:
    results = json.load(f)

df = pd.DataFrame(results)

print(f"\n📊 Dataset Summary:")
print(f"  Total runs: {len(df)}")
print(f"  Unique seeds: {df['seed'].nunique()}")
print(f"  Parameters swept:")
print(f"    - B_max: {sorted(df['B_max'].unique())}")
print(f"    - B_min: {sorted(df['B_min'].unique())}")
print(f"    - K_warmup_ratio: {sorted(df['K_warmup_ratio'].unique())}")
print(f"    - curriculum_schedule: {sorted(df['curriculum_schedule'].unique())}")

# Aggregate by hyperparameters
grouped = df.groupby(['B_max', 'B_min', 'K_warmup_ratio', 'curriculum_schedule']).agg({
    'final_performance': ['mean', 'std', 'min', 'max'],
    'improvement': ['mean', 'std'],
    'iterations': ['mean', 'std'],
    'final_state_coverage': 'mean',
    'final_state_entropy': 'mean',
    'final_model_tv_mean': 'mean'
}).round(6)

print("\n" + "="*80)
print("TOP 10 CONFIGURATIONS BY PERFORMANCE")
print("="*80)
top10 = grouped.sort_values(('final_performance', 'mean'), ascending=False).head(10)
print(top10)

# Find absolute best
best_config = grouped[('final_performance', 'mean')].idxmax()
best_perf = grouped.loc[best_config, ('final_performance', 'mean')]
best_std = grouped.loc[best_config, ('final_performance', 'std')]

print("\n" + "="*80)
print("🏆 OPTIMAL CONFIGURATION")
print("="*80)
print(f"B_max: {best_config[0]}")
print(f"B_min: {best_config[1]}")
print(f"K_warmup_ratio: {best_config[2]}")
print(f"Schedule: {best_config[3]}")
print(f"Performance: {best_perf:.6f} ± {best_std:.6f}")
print("="*80)

# ============================================================================
# FIGURE 1: COMPREHENSIVE HEATMAP GRID (Main Result)
# ============================================================================
fig1 = plt.figure(figsize=(20, 12))
gs = fig1.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

schedules = sorted(df['curriculum_schedule'].unique())
b_mins = sorted(df['B_min'].unique())

for schedule_idx, schedule in enumerate(schedules):
    for bmin_idx, b_min in enumerate(b_mins):
        ax = fig1.add_subplot(gs[schedule_idx, bmin_idx])
        
        # Filter data
        subset = df[(df['curriculum_schedule'] == schedule) & (df['B_min'] == b_min)]
        
        # Create pivot table
        pivot = subset.pivot_table(
            values='final_performance',
            index='K_warmup_ratio',
            columns='B_max',
            aggfunc='mean'
        )
        
        # Plot heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                      vmin=df['final_performance'].min(),
                      vmax=df['final_performance'].max())
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f'{x:.3f}' for x in pivot.columns], rotation=45, ha='right')
        ax.set_yticklabels([f'{y:.2f}' for y in pivot.index])
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if not np.isnan(value):
                    # Check if this is the best config
                    is_best = (pivot.columns[j] == best_config[0] and 
                              pivot.index[i] == best_config[2] and
                              schedule == best_config[3] and
                              b_min == best_config[1])
                    
                    text_color = 'white' if value < (pivot.values.min() + 0.7*(pivot.values.max()-pivot.values.min())) else 'black'
                    fontweight = 'bold' if is_best else 'normal'
                    fontsize = 11 if is_best else 9
                    
                    text = ax.text(j, i, f'{value:.4f}', ha='center', va='center',
                                 color=text_color, fontsize=fontsize, fontweight=fontweight)
                    
                    if is_best:
                        # Add star marker for best
                        ax.scatter([j], [i], s=300, marker='*', color='gold',
                                 edgecolors='red', linewidth=2, zorder=10)
        
        ax.set_xlabel('B_max (Adversarial Budget)', fontweight='bold')
        ax.set_ylabel('K_warmup_ratio', fontweight='bold')
        ax.set_title(f'{schedule.capitalize()} | B_min={b_min:.4f}', 
                    fontweight='bold', pad=10)
        
        # Colorbar for last plot in each row
        if bmin_idx == len(b_mins) - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Final Performance', rotation=270, labelpad=20, fontweight='bold')

fig1.suptitle('Full Grid Search: Performance Heatmaps', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()
output_path1 = 'results/full_grid_heatmaps.png'
plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Heatmap grid saved: {output_path1}")

# ============================================================================
# FIGURE 2: MAIN EFFECTS ANALYSIS (4 subplots)
# ============================================================================
fig2 = plt.figure(figsize=(16, 10))
gs = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ---------------------------------------------------------------------------
# Plot 2.1: B_max Effect
# ---------------------------------------------------------------------------
ax1 = fig2.add_subplot(gs[0, 0])

for schedule in schedules:
    schedule_df = df[df['curriculum_schedule'] == schedule]
    grouped_schedule = schedule_df.groupby('B_max').agg({
        'final_performance': ['mean', 'std']
    })
    
    x = grouped_schedule.index
    y_mean = grouped_schedule[('final_performance', 'mean')]
    y_std = grouped_schedule[('final_performance', 'std')]
    
    ax1.plot(x, y_mean, marker='o', linewidth=2.5, markersize=10, 
            label=schedule.capitalize(), alpha=0.8)
    ax1.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.15)

ax1.axvline(best_config[0], color='red', linestyle='--', alpha=0.5, linewidth=2,
           label=f'Optimal: {best_config[0]:.3f}')
ax1.set_xlabel('Adversarial Budget (B_max)', fontweight='bold', fontsize=13)
ax1.set_ylabel('Final Performance', fontweight='bold', fontsize=13)
ax1.set_title('Main Effect: Adversarial Budget', fontweight='bold', pad=15)
ax1.legend(framealpha=0.9, loc='best')
ax1.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 2.2: K_warmup_ratio Effect
# ---------------------------------------------------------------------------
ax2 = fig2.add_subplot(gs[0, 1])

for schedule in schedules:
    schedule_df = df[df['curriculum_schedule'] == schedule]
    grouped_schedule = schedule_df.groupby('K_warmup_ratio').agg({
        'final_performance': ['mean', 'std']
    })
    
    x = grouped_schedule.index
    y_mean = grouped_schedule[('final_performance', 'mean')]
    y_std = grouped_schedule[('final_performance', 'std')]
    
    ax2.plot(x, y_mean, marker='s', linewidth=2.5, markersize=10, 
            label=schedule.capitalize(), alpha=0.8)
    ax2.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.15)

ax2.axvline(best_config[2], color='red', linestyle='--', alpha=0.5, linewidth=2,
           label=f'Optimal: {best_config[2]:.2f}')
ax2.set_xlabel('Warmup Ratio', fontweight='bold', fontsize=13)
ax2.set_ylabel('Final Performance', fontweight='bold', fontsize=13)
ax2.set_title('Main Effect: Warmup Duration', fontweight='bold', pad=15)
ax2.legend(framealpha=0.9, loc='best')
ax2.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 2.3: B_min Effect
# ---------------------------------------------------------------------------
ax3 = fig2.add_subplot(gs[1, 0])

for schedule in schedules:
    schedule_df = df[df['curriculum_schedule'] == schedule]
    grouped_schedule = schedule_df.groupby('B_min').agg({
        'final_performance': ['mean', 'std']
    })
    
    x = grouped_schedule.index
    y_mean = grouped_schedule[('final_performance', 'mean')]
    y_std = grouped_schedule[('final_performance', 'std')]
    
    ax3.plot(x, y_mean, marker='^', linewidth=2.5, markersize=10, 
            label=schedule.capitalize(), alpha=0.8)
    ax3.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.15)

ax3.axvline(best_config[1], color='red', linestyle='--', alpha=0.5, linewidth=2,
           label=f'Optimal: {best_config[1]:.4f}')
ax3.set_xlabel('Minimum Adversarial Budget (B_min)', fontweight='bold', fontsize=13)
ax3.set_ylabel('Final Performance', fontweight='bold', fontsize=13)
ax3.set_title('Main Effect: Initial Adversarial Pressure', fontweight='bold', pad=15)
ax3.legend(framealpha=0.9, loc='best')
ax3.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 2.4: Schedule Comparison (Box Plot)
# ---------------------------------------------------------------------------
ax4 = fig2.add_subplot(gs[1, 1])

box_data = []
box_labels = []
for schedule in schedules:
    box_data.append(df[df['curriculum_schedule'] == schedule]['final_performance'].values)
    box_labels.append(schedule.capitalize())

bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True,
                 showmeans=True, meanline=True,
                 medianprops=dict(color='red', linewidth=2.5),
                 meanprops=dict(color='blue', linewidth=2.5, linestyle='--'),
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 whiskerprops=dict(linewidth=2),
                 capprops=dict(linewidth=2))

# Statistical test
if len(schedules) == 2:
    data1 = df[df['curriculum_schedule'] == schedules[0]]['final_performance']
    data2 = df[df['curriculum_schedule'] == schedules[1]]['final_performance']
    t_stat, p_value = stats.ttest_ind(data1, data2)
    
    sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    ax4.text(0.5, 0.98, f't-test: p={p_value:.4f} {sig_text}',
            transform=ax4.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11, fontweight='bold')

ax4.set_ylabel('Final Performance', fontweight='bold', fontsize=13)
ax4.set_title('Schedule Comparison', fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, axis='y')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=2.5, label='Median'),
    Line2D([0], [0], color='blue', linewidth=2.5, linestyle='--', label='Mean')
]
ax4.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

fig2.suptitle('Main Effects Analysis: Hyperparameter Impact', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()
output_path2 = 'results/full_grid_main_effects.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Main effects analysis saved: {output_path2}")

# ============================================================================
# FIGURE 3: INTERACTION EFFECTS (3D Surfaces)
# ============================================================================
fig3 = plt.figure(figsize=(18, 10))

for schedule_idx, schedule in enumerate(schedules):
    # B_max vs K_warmup interaction
    ax1 = fig3.add_subplot(2, 2, schedule_idx*2 + 1, projection='3d')
    
    subset = df[(df['curriculum_schedule'] == schedule)]
    pivot = subset.pivot_table(
        values='final_performance',
        index='K_warmup_ratio',
        columns='B_max',
        aggfunc='mean'
    )
    
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values
    
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_xlabel('B_max', fontweight='bold', labelpad=10)
    ax1.set_ylabel('K_warmup', fontweight='bold', labelpad=10)
    ax1.set_zlabel('Performance', fontweight='bold', labelpad=10)
    ax1.set_title(f'{schedule.capitalize()}: B_max × K_warmup', 
                 fontweight='bold', pad=15)
    fig3.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # B_max vs B_min interaction
    ax2 = fig3.add_subplot(2, 2, schedule_idx*2 + 2, projection='3d')
    
    pivot2 = subset.pivot_table(
        values='final_performance',
        index='B_min',
        columns='B_max',
        aggfunc='mean'
    )
    
    X2, Y2 = np.meshgrid(pivot2.columns, pivot2.index)
    Z2 = pivot2.values
    
    surf2 = ax2.plot_surface(X2, Y2, Z2, cmap='plasma', alpha=0.8, edgecolor='none')
    ax2.set_xlabel('B_max', fontweight='bold', labelpad=10)
    ax2.set_ylabel('B_min', fontweight='bold', labelpad=10)
    ax2.set_zlabel('Performance', fontweight='bold', labelpad=10)
    ax2.set_title(f'{schedule.capitalize()}: B_max × B_min', 
                 fontweight='bold', pad=15)
    fig3.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

fig3.suptitle('Interaction Effects: Hyperparameter Synergies', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()
output_path3 = 'results/full_grid_interactions.png'
plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Interaction effects saved: {output_path3}")

# ============================================================================
# FIGURE 4: ROBUSTNESS ANALYSIS (Auxiliary Metrics)
# ============================================================================
fig4 = plt.figure(figsize=(16, 10))
gs = fig4.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ---------------------------------------------------------------------------
# Plot 4.1: Performance vs Model Divergence
# ---------------------------------------------------------------------------
ax1 = fig4.add_subplot(gs[0, 0])

for schedule in schedules:
    schedule_df = df[df['curriculum_schedule'] == schedule]
    ax1.scatter(schedule_df['final_model_tv_mean'], 
               schedule_df['final_performance'],
               s=100, alpha=0.6, label=schedule.capitalize(), edgecolors='black')

# Mark optimal
optimal_runs = df[(df['B_max'] == best_config[0]) & 
                 (df['B_min'] == best_config[1]) & 
                 (df['K_warmup_ratio'] == best_config[2]) & 
                 (df['curriculum_schedule'] == best_config[3])]

ax1.scatter(optimal_runs['final_model_tv_mean'],
           optimal_runs['final_performance'],
           s=300, marker='*', color='gold', 
           edgecolors='red', linewidth=2, zorder=10, label='Optimal Config')

ax1.set_xlabel('Model TV Distance (Robustness)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Final Performance', fontweight='bold', fontsize=12)
ax1.set_title('Performance vs Adversarial Robustness', fontweight='bold', pad=15)
ax1.legend(framealpha=0.9, loc='best')
ax1.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 4.2: Performance vs State Coverage
# ---------------------------------------------------------------------------
ax2 = fig4.add_subplot(gs[0, 1])

for schedule in schedules:
    schedule_df = df[df['curriculum_schedule'] == schedule]
    ax2.scatter(schedule_df['final_state_coverage'], 
               schedule_df['final_performance'],
               s=100, alpha=0.6, label=schedule.capitalize(), edgecolors='black')

ax2.scatter(optimal_runs['final_state_coverage'],
           optimal_runs['final_performance'],
           s=300, marker='*', color='gold', 
           edgecolors='red', linewidth=2, zorder=10)

ax2.set_xlabel('State Coverage', fontweight='bold', fontsize=12)
ax2.set_ylabel('Final Performance', fontweight='bold', fontsize=12)
ax2.set_title('Performance vs Exploration', fontweight='bold', pad=15)
ax2.legend(framealpha=0.9, loc='best')
ax2.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 4.3: Convergence Speed Analysis
# ---------------------------------------------------------------------------
ax3 = fig4.add_subplot(gs[1, 0])

for schedule in schedules:
    schedule_df = df[df['curriculum_schedule'] == schedule]
    grouped_schedule = schedule_df.groupby('B_max').agg({
        'iterations': ['mean', 'std']
    })
    
    x = grouped_schedule.index
    y_mean = grouped_schedule[('iterations', 'mean')]
    y_std = grouped_schedule[('iterations', 'std')]
    
    ax3.plot(x, y_mean, marker='o', linewidth=2.5, markersize=10, 
            label=schedule.capitalize(), alpha=0.8)
    ax3.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.15)

ax3.set_xlabel('Adversarial Budget (B_max)', fontweight='bold', fontsize=12)
ax3.set_ylabel('Iterations to Convergence', fontweight='bold', fontsize=12)
ax3.set_title('Convergence Speed vs Budget', fontweight='bold', pad=15)
ax3.legend(framealpha=0.9, loc='best')
ax3.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 4.4: Efficiency Frontier (Pareto Front)
# ---------------------------------------------------------------------------
ax4 = fig4.add_subplot(gs[1, 1])

# Calculate efficiency: performance / iterations
df['efficiency'] = df['final_performance'] / df['iterations'] * 1000

for schedule in schedules:
    schedule_df = df[df['curriculum_schedule'] == schedule]
    ax4.scatter(schedule_df['iterations'], 
               schedule_df['final_performance'],
               s=schedule_df['efficiency']*50,  # Bubble size = efficiency
               alpha=0.6, label=schedule.capitalize(), edgecolors='black')

ax4.scatter(optimal_runs['iterations'],
           optimal_runs['final_performance'],
           s=300, marker='*', color='gold', 
           edgecolors='red', linewidth=2, zorder=10, label='Optimal')

ax4.set_xlabel('Iterations to Convergence', fontweight='bold', fontsize=12)
ax4.set_ylabel('Final Performance', fontweight='bold', fontsize=12)
ax4.set_title('Efficiency Frontier (bubble size ∝ efficiency)', fontweight='bold', pad=15)
ax4.legend(framealpha=0.9, loc='best')
ax4.grid(True, alpha=0.3)

fig4.suptitle('Robustness & Efficiency Analysis', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()
output_path4 = 'results/full_grid_robustness.png'
plt.savefig(output_path4, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Robustness analysis saved: {output_path4}")

# ============================================================================
# FIGURE 5: COMPREHENSIVE SUMMARY TABLE
# ============================================================================
fig5, ax = plt.subplots(figsize=(18, 10))
ax.axis('off')

# Prepare summary data
summary_data = []
for idx, row in grouped.sort_values(('final_performance', 'mean'), ascending=False).head(20).iterrows():
    b_max, b_min, k_warmup, schedule = idx
    summary_data.append([
        f'{schedule.capitalize()}',
        f'{b_max:.3f}',
        f'{b_min:.4f}',
        f'{k_warmup:.2f}',
        f'{row[("final_performance", "mean")]:.6f}',
        f'±{row[("final_performance", "std")]:.6f}',
        f'{row[("improvement", "mean")]:.6f}',
        f'{int(row[("iterations", "mean")])}',
        f'{row[("final_state_coverage", "mean")]:.4f}',
        f'{row[("final_model_tv_mean", "mean")]:.4f}'
    ])

summary_df = pd.DataFrame(summary_data,
                          columns=['Schedule', 'B_max', 'B_min', 'K_warmup',
                                  'Performance', 'Std', 'Improvement', 'Iters',
                                  'Coverage', 'Robustness'])

# Create table
table = ax.table(cellText=summary_df.values,
                colLabels=summary_df.columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Style header
for i in range(len(summary_df.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', size=10)

# Alternate row colors
for i in range(1, len(summary_df) + 1):
    for j in range(len(summary_df.columns)):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('#FFFFFF')
    
    # Highlight top 3
    if i <= 3:
        for j in range(len(summary_df.columns)):
            cell = table[(i, j)]
            if i == 1:
                cell.set_facecolor('#FFD700')  # Gold
            elif i == 2:
                cell.set_facecolor('#C0C0C0')  # Silver
            elif i == 3:
                cell.set_facecolor('#CD7F32')  # Bronze
            cell.set_text_props(weight='bold', size=9)

# Add comprehensive annotation
annotation_text = (
    f"🏆 OPTIMAL HYPERPARAMETERS 🏆\n"
    f"Schedule: {best_config[3].capitalize()} | "
    f"B_max: {best_config[0]:.3f} | "
    f"B_min: {best_config[1]:.4f} | "
    f"K_warmup: {best_config[2]:.2f}\n"
    f"Performance: {best_perf:.6f} ± {best_std:.6f}\n\n"
    f"Key Insights:\n"
    f"• Robustness (Model TV): Higher values indicate stronger adversarial training\n"
    f"• Coverage: 1.0 = Complete state space exploration\n"
    f"• Top 3 configurations shown in Gold/Silver/Bronze"
)

ax.text(0.5, -0.08, annotation_text, 
        ha='center', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        transform=ax.transAxes, family='monospace')

plt.title('Full Grid Search: Top 20 Configurations', 
          fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()

output_path5 = 'results/full_grid_summary_table.png'
plt.savefig(output_path5, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Summary table saved: {output_path5}")

# ============================================================================
# STATISTICAL ANALYSIS OUTPUT
# ============================================================================
print("\n" + "="*80)
print("📊 STATISTICAL ANALYSIS")
print("="*80)

# ANOVA for each hyperparameter
from scipy.stats import f_oneway

print("\nOne-way ANOVA Results:")
print("-" * 80)

# B_max effect
groups_bmax = [df[df['B_max'] == val]['final_performance'].values 
               for val in df['B_max'].unique()]
f_stat, p_val = f_oneway(*groups_bmax)
print(f"B_max:          F={f_stat:.4f}, p={p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

# K_warmup effect
groups_warmup = [df[df['K_warmup_ratio'] == val]['final_performance'].values 
                for val in df['K_warmup_ratio'].unique()]
f_stat, p_val = f_oneway(*groups_warmup)
print(f"K_warmup_ratio: F={f_stat:.4f}, p={p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

# B_min effect
groups_bmin = [df[df['B_min'] == val]['final_performance'].values 
              for val in df['B_min'].unique()]
f_stat, p_val = f_oneway(*groups_bmin)
print(f"B_min:          F={f_stat:.4f}, p={p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

# Schedule effect
groups_schedule = [df[df['curriculum_schedule'] == val]['final_performance'].values 
                  for val in df['curriculum_schedule'].unique()]
f_stat, p_val = f_oneway(*groups_schedule)
print(f"Schedule:       F={f_stat:.4f}, p={p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

print("\n" + "="*80)
print("📊 ALL PLOTS GENERATED")
print("="*80)
print(f"1. Heatmap grid:        {output_path1}")
print(f"2. Main effects:        {output_path2}")
print(f"3. Interactions:        {output_path3}")
print(f"4. Robustness analysis: {output_path4}")
print(f"5. Summary table:       {output_path5}")
print("="*80)

print(f"\n🎯 FINAL RECOMMENDATION:")
print(f"   Use configuration: {best_config}")
print(f"   Expected performance: {best_perf:.6f} ± {best_std:.6f}")