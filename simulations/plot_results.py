"""
Plot F-SPMI Results with Confidence Bands
==========================================
Generates publication-quality figures from aggregated CSV files.

Usage:
    # Plot all results (standard plots)
    python plot_results.py --results_dir ./data/report_experiments/racetrack4_T1/greedy/TIMESTAMP
    
    # Generate faceted plots (one subplot per eps/round: eps100, eps200, eps400, eps800)
    python plot_results.py --results_dir ... --faceted
    
    # Only faceted plots (skip standard)
    python plot_results.py --results_dir ... --faceted --no-standard
    
    # Filter by N agents (compare eps/round for fixed N)
    python plot_results.py --results_dir ... --n_agents 4
    
    # Filter by eps/round (compare N agents for fixed eps)
    python plot_results.py --results_dir ... --eps_per_round 400
    
    # Both filters
    python plot_results.py --results_dir ... --n_agents 4 --eps_per_round 400
    
Faceted Plots Generated (--faceted):
    - facet_convergence.png: Convergence curves, one subplot per eps/round
    - facet_mc_vs_exact.png: MC vs Exact evaluator, one subplot per eps/round
    - facet_safety_bounds.png: Safety bounds, one subplot per eps/round
    - facet_step_sizes.png: Alpha/Beta step sizes, 2 rows x N cols (eps values)
    - facet_sample_efficiency.png: Sample efficiency, one subplot per eps/round
    - facet_final_performance.png: Final performance bars, one subplot per eps/round
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Publication-quality settings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.dpi'] = 150


def parse_config_key(key: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract n_agents and eps_per_round from config key like 'n4_eps100'"""
    n_match = re.search(r'n(\d+)', key)
    eps_match = re.search(r'eps(\d+)', key)
    
    n_agents = int(n_match.group(1)) if n_match else None
    eps = int(eps_match.group(1)) if eps_match else None
    
    return n_agents, eps


def filter_results(results: Dict[str, pd.DataFrame], 
                   n_agents: Optional[int] = None,
                   eps_per_round: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Filter results by n_agents and/or eps_per_round"""
    filtered = {}
    
    for key, df in results.items():
        # Always keep SPMI baseline
        if key == 'spmi':
            filtered[key] = df
            continue
        
        # Parse config
        n, eps = parse_config_key(key)
        
        # Apply filters
        if n_agents is not None and n != n_agents:
            continue
        if eps_per_round is not None and eps != eps_per_round:
            continue
        
        filtered[key] = df
    
    return filtered


def load_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all CSV results from directory"""
    results = {}
    
    # Load aggregated F-SPMI results first (preferred)
    for f in results_dir.glob("fspmi_*_aggregated.csv"):
        key = f.stem.replace("fspmi_", "").replace("_aggregated", "")
        results[key] = pd.read_csv(f)
        results[key]['source'] = 'aggregated'
        print(f"  Loaded: {key} ({len(results[key])} iterations) [aggregated]")
    
    # If no aggregated results, try loading seed0 files
    if not results:
        for f in results_dir.glob("fspmi_*_seed0.csv"):
            key = f.stem.replace("fspmi_", "").replace("_seed0", "")
            df = pd.read_csv(f)
            # Rename columns to match aggregated format
            df = df.rename(columns={
                'performance_true': 'true_perf_mean',
                'performance_mc': 'mc_perf_mean',
                'bound': 'bound_mean',
                'alpha': 'alpha_mean',
                'beta': 'beta_mean',
                'policy_advantage': 'policy_advantage_mean',
                'model_advantage': 'model_advantage_mean',
            })
            # Add dummy std columns
            for col in ['true_perf', 'mc_perf', 'bound', 'alpha', 'beta', 
                       'policy_advantage', 'model_advantage']:
                df[f'{col}_std'] = 0
            # Compute cumulative samples
            if 'total_samples' in df.columns:
                df['cum_samples_mean'] = df['total_samples'].cumsum()
            df['source'] = 'single_seed'
            results[key] = df
            print(f"  Loaded: {key} ({len(df)} iterations) [single seed]")
    
    # Load standard SPMI if exists (semicolon separated!)
    spmi_file = results_dir / "standard_spmi.csv"
    if spmi_file.exists():
        spmi_df = pd.read_csv(spmi_file, sep=';')
        spmi_df = spmi_df.rename(columns={
            '# iterations': 'iteration',
            'evaluations': 'evaluation',
            'alfa': 'alpha',
        })
        # Create iteration column if needed
        if 'iteration' not in spmi_df.columns:
            spmi_df['iteration'] = range(len(spmi_df))
        spmi_df['source'] = 'spmi'
        results['spmi'] = spmi_df
        print(f"  Loaded: SPMI baseline ({len(results['spmi'])} iterations)")
        print(f"    Columns: {list(spmi_df.columns)[:10]}...")
    
    return results


def get_colors(n: int) -> np.ndarray:
    """Get distinct colors for plotting"""
    if n <= 1:
        return plt.cm.viridis(np.array([0.5]))
    return plt.cm.viridis(np.linspace(0.2, 0.8, n))


def load_seed_data(results_dir: Path) -> Dict[str, List[pd.DataFrame]]:
    """Load individual seed files for each config"""
    from collections import defaultdict
    
    seed_data = defaultdict(list)
    
    for f in sorted(results_dir.glob("fspmi_*_seed*.csv")):
        # Extract config key
        parts = f.stem.split("_seed")
        config_key = parts[0].replace("fspmi_", "")
        
        df = pd.read_csv(f)
        # Rename columns to match aggregated format
        df = df.rename(columns={
            'performance_true': 'true_perf_mean',
            'performance_mc': 'mc_perf_mean',
        })
        seed_data[config_key].append(df)
    
    return dict(seed_data)


def format_label(config_key: str) -> str:
    """Format config key for legend labels"""
    return config_key.replace('_', ', ').replace('n', 'N=').replace('eps', 'eps=')


def plot_convergence(results: Dict[str, pd.DataFrame], save_path: Path, title: str = "",
                     seed_data: Optional[Dict[str, List[pd.DataFrame]]] = None):
    """Plot 1: Performance convergence with std bands or individual seeds"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fspmi_keys = sorted([k for k in results.keys() if k != 'spmi'])
    colors = get_colors(len(fspmi_keys))
    
    # Plot SPMI baseline
    if 'spmi' in results:
        spmi_df = results['spmi']
        ax.plot(spmi_df['iteration'], spmi_df['evaluation'], 
                'k-', linewidth=2.5, label='Standard SPMI', zorder=10)
    
    # Plot F-SPMI results
    for idx, config_key in enumerate(fspmi_keys):
        df = results[config_key]
        color = colors[idx]
        label = format_label(config_key)
        
        # If we have individual seed data, plot those as thin lines
        if seed_data and config_key in seed_data:
            for i, seed_df in enumerate(seed_data[config_key]):
                ax.plot(seed_df['iteration'], seed_df['true_perf_mean'], 
                       color=color, linewidth=0.8, alpha=0.3)
            # Plot mean as thicker line
            ax.plot(df['iteration'], df['true_perf_mean'], 
                   color=color, linewidth=2.5, label=f'F-SPMI ({label})')
        else:
            # Plot mean line
            ax.plot(df['iteration'], df['true_perf_mean'], 
                    color=color, linewidth=2, label=f'F-SPMI ({label})')
            
            # Add std band (scale up for visibility if too small)
            if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
                std = df['true_perf_std']
                # Check if std is visible (> 1% of mean range)
                mean_range = df['true_perf_mean'].max() - df['true_perf_mean'].min()
                avg_std = std.mean()
                
                if avg_std < mean_range * 0.01:
                    # Std too small to see - note in legend
                    ax.plot([], [], ' ', label=f'  (σ={avg_std:.2e})')
                else:
                    ax.fill_between(df['iteration'],
                                   df['true_perf_mean'] - std,
                                   df['true_perf_mean'] + std,
                                   color=color, alpha=0.2)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Return')
    ax.set_title(f'Convergence Comparison{title}')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_mc_vs_exact(results: Dict[str, pd.DataFrame], save_path: Path):
    """Plot 2: MC estimate vs Exact evaluator with std bands"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fspmi_keys = sorted([k for k in results.keys() if k != 'spmi'])
    colors = get_colors(len(fspmi_keys))
    
    for idx, config_key in enumerate(fspmi_keys):
        df = results[config_key]
        color = colors[idx]
        label = format_label(config_key)
        
        # Exact evaluator (solid) with std band
        ax.plot(df['iteration'], df['true_perf_mean'], 
                color=color, linestyle='-', linewidth=2,
                label=f'Exact ({label})')
        if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
            ax.fill_between(df['iteration'],
                           df['true_perf_mean'] - df['true_perf_std'],
                           df['true_perf_mean'] + df['true_perf_std'],
                           color=color, alpha=0.15)
        
        # MC estimate (dashed) with std band
        ax.plot(df['iteration'], df['mc_perf_mean'], 
                color=color, linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'MC ({label})')
        if 'mc_perf_std' in df.columns and df['mc_perf_std'].sum() > 0:
            ax.fill_between(df['iteration'],
                           df['mc_perf_mean'] - df['mc_perf_std'],
                           df['mc_perf_mean'] + df['mc_perf_std'],
                           color=color, alpha=0.1)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Return')
    ax.set_title('MC vs Exact Evaluator')
    ax.legend(loc='lower right', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_safety_bounds(results: Dict[str, pd.DataFrame], save_path: Path):
    """Plot 3: Safety bounds over time (log scale) with std bands"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fspmi_keys = sorted([k for k in results.keys() if k != 'spmi'])
    colors = get_colors(len(fspmi_keys))
    
    # SPMI baseline
    if 'spmi' in results:
        spmi_df = results['spmi']
        if 'bound' in spmi_df.columns:
            ax.semilogy(spmi_df['iteration'], 
                       np.maximum(spmi_df['bound'], 1e-10),
                       'k-', linewidth=2, label='Standard SPMI')
    
    # F-SPMI
    for idx, config_key in enumerate(fspmi_keys):
        df = results[config_key]
        color = colors[idx]
        label = format_label(config_key)
        
        ax.semilogy(df['iteration'], 
                   np.maximum(df['bound_mean'], 1e-10),
                   color=color, linewidth=2, label=f'F-SPMI ({label})')
        
        # Std band (careful with log scale)
        if 'bound_std' in df.columns and df['bound_std'].sum() > 0:
            lower = np.maximum(df['bound_mean'] - df['bound_std'], 1e-10)
            upper = df['bound_mean'] + df['bound_std']
            ax.fill_between(df['iteration'], lower, upper, color=color, alpha=0.2)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Bound Value (log scale)')
    ax.set_title('Safety Bounds Over Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_sample_efficiency(results: Dict[str, pd.DataFrame], save_path: Path):
    """Plot 4: Performance vs Total Samples with std bands"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fspmi_keys = sorted([k for k in results.keys() if k != 'spmi'])
    colors = get_colors(len(fspmi_keys))
    
    # SPMI final performance as horizontal line
    if 'spmi' in results:
        spmi_df = results['spmi']
        final_perf = spmi_df['evaluation'].iloc[-1]
        ax.axhline(y=final_perf, color='k', linestyle='--', linewidth=2,
                   label=f'SPMI final ({final_perf:.4f})')
    
    # F-SPMI
    for idx, config_key in enumerate(fspmi_keys):
        df = results[config_key]
        color = colors[idx]
        label = format_label(config_key)
        
        ax.plot(df['cum_samples_mean'], df['true_perf_mean'], 
                color=color, linewidth=2, label=f'F-SPMI ({label})')
        
        if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
            ax.fill_between(df['cum_samples_mean'],
                           df['true_perf_mean'] - df['true_perf_std'],
                           df['true_perf_mean'] + df['true_perf_std'],
                           color=color, alpha=0.2)
    
    ax.set_xlabel('Total Samples Collected')
    ax.set_ylabel('Return')
    ax.set_title('Sample Efficiency')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_returns_and_bounds(results: Dict[str, pd.DataFrame], save_path: Path, 
                            config_to_show: Optional[str] = None):
    """Plot 5: Returns (solid) & Bounds (dotted) - dual axis"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    fspmi_configs = [k for k in results.keys() if k != 'spmi']
    if config_to_show and config_to_show in fspmi_configs:
        config_key = config_to_show
    else:
        config_key = fspmi_configs[0] if fspmi_configs else None
    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Return', color='black')
    
    lines = []
    labels = []
    
    # SPMI return
    if 'spmi' in results:
        spmi_df = results['spmi']
        l1, = ax1.plot(spmi_df['iteration'], spmi_df['evaluation'], 
                       'b-', linewidth=2, label='SPMI return')
        lines.append(l1)
        labels.append('SPMI return')
    
    # F-SPMI return with std
    if config_key:
        df = results[config_key]
        l2, = ax1.plot(df['iteration'], df['true_perf_mean'], 
                       'r-', linewidth=2, label='F-SPMI return')
        if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
            ax1.fill_between(df['iteration'],
                            df['true_perf_mean'] - df['true_perf_std'],
                            df['true_perf_mean'] + df['true_perf_std'],
                            color='r', alpha=0.2)
        lines.append(l2)
        labels.append('F-SPMI return')
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(left=0)
    
    # Right axis: Bounds
    ax2 = ax1.twinx()
    ax2.set_ylabel('Safety Bound', color='gray')
    
    if 'spmi' in results and 'bound' in results['spmi'].columns:
        l3, = ax2.plot(results['spmi']['iteration'], results['spmi']['bound'], 
                       'b:', linewidth=1.5, alpha=0.7, label='SPMI bound')
        lines.append(l3)
        labels.append('SPMI bound')
    
    if config_key:
        l4, = ax2.plot(df['iteration'], df['bound_mean'], 
                       'r:', linewidth=1.5, alpha=0.7, label='F-SPMI bound')
        lines.append(l4)
        labels.append('F-SPMI bound')
    
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    ax1.legend(lines, labels, loc='center right')
    
    title_suffix = f" ({config_key})" if config_key else ""
    ax1.set_title(f'Returns (solid) & Safety Bounds (dotted){title_suffix}')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_step_sizes(results: Dict[str, pd.DataFrame], save_path: Path):
    """Plot 6: Step sizes alpha and beta over iterations with std bands"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    fspmi_keys = sorted([k for k in results.keys() if k != 'spmi'])
    colors = get_colors(len(fspmi_keys))
    
    # Alpha (policy step size)
    ax1 = axes[0]
    if 'spmi' in results:
        spmi_df = results['spmi']
        if 'alpha' in spmi_df.columns:
            ax1.semilogy(spmi_df['iteration'], 
                        np.maximum(spmi_df['alpha'], 1e-10),
                        'k-', linewidth=2, label='SPMI', alpha=0.8)
    
    for idx, config_key in enumerate(fspmi_keys):
        df = results[config_key]
        color = colors[idx]
        label = format_label(config_key)
        
        if 'alpha_mean' in df.columns:
            ax1.semilogy(df['iteration'], 
                        np.maximum(df['alpha_mean'], 1e-10),
                        color=color, linewidth=2, label=f'F-SPMI ({label})', alpha=0.8)
    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('α (Policy Step Size)')
    ax1.set_title('Policy Update Step Sizes')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # Beta (model step size)
    ax2 = axes[1]
    if 'spmi' in results:
        spmi_df = results['spmi']
        if 'beta' in spmi_df.columns:
            ax2.semilogy(spmi_df['iteration'], 
                        np.maximum(spmi_df['beta'], 1e-10),
                        'k-', linewidth=2, label='SPMI', alpha=0.8)
    
    for idx, config_key in enumerate(fspmi_keys):
        df = results[config_key]
        color = colors[idx]
        label = format_label(config_key)
        
        if 'beta_mean' in df.columns:
            ax2.semilogy(df['iteration'], 
                        np.maximum(df['beta_mean'], 1e-10),
                        color=color, linewidth=2, label=f'F-SPMI ({label})', alpha=0.8)
    
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('β (Model Step Size)')
    ax2.set_title('Model Update Step Sizes')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_bar_chart(results: Dict[str, pd.DataFrame], save_path: Path, 
                   iteration: Optional[int] = None):
    """Plot 7: Bar chart comparing final performance with error bars"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect data for bar chart
    labels = []
    means = []
    stds = []
    
    # SPMI baseline
    if 'spmi' in results:
        spmi_df = results['spmi']
        labels.append('SPMI')
        means.append(spmi_df['evaluation'].iloc[-1])
        stds.append(0)  # No std for SPMI (single run)
    
    # F-SPMI configs
    fspmi_keys = sorted([k for k in results.keys() if k != 'spmi'])
    for config_key in fspmi_keys:
        df = results[config_key]
        label = format_label(config_key)
        labels.append(f'F-SPMI\n({label})')
        
        # Use specified iteration or last one
        idx = iteration if iteration is not None else -1
        means.append(df['true_perf_mean'].iloc[idx])
        
        if 'true_perf_std' in df.columns:
            stds.append(df['true_perf_std'].iloc[idx])
        else:
            stds.append(0)
    
    # Create bar chart
    x = np.arange(len(labels))
    colors = ['black'] + list(get_colors(len(fspmi_keys)))
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Final Return')
    ax.set_title('Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 + (std * 100 if std > 0 else 0)),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_bar_chart_grouped(results: Dict[str, pd.DataFrame], save_path: Path):
    """Plot 8: Grouped bar chart - N agents vs eps/round"""
    # Parse all configs
    configs = []
    for key in results.keys():
        if key == 'spmi':
            continue
        n, eps = parse_config_key(key)
        if n is not None and eps is not None:
            df = results[key]
            final_mean = df['true_perf_mean'].iloc[-1]
            final_std = df['true_perf_std'].iloc[-1] if 'true_perf_std' in df.columns else 0
            configs.append({'n': n, 'eps': eps, 'mean': final_mean, 'std': final_std, 'key': key})
    
    if not configs:
        print("  Skipping grouped bar chart (no F-SPMI configs)")
        return
    
    # Get unique values
    n_values = sorted(set(c['n'] for c in configs))
    eps_values = sorted(set(c['eps'] for c in configs))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(eps_values))
    width = 0.8 / len(n_values)
    colors = get_colors(len(n_values))
    
    # SPMI baseline
    if 'spmi' in results:
        spmi_final = results['spmi']['evaluation'].iloc[-1]
        ax.axhline(y=spmi_final, color='k', linestyle='--', linewidth=2,
                   label=f'SPMI ({spmi_final:.4f})')
    
    # Grouped bars
    for i, n in enumerate(n_values):
        means = []
        stds = []
        for eps in eps_values:
            # Find config
            cfg = next((c for c in configs if c['n'] == n and c['eps'] == eps), None)
            if cfg:
                means.append(cfg['mean'])
                stds.append(cfg['std'])
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - len(n_values)/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                     label=f'N={n}', color=colors[i], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Episodes per Round')
    ax.set_ylabel('Final Return')
    ax.set_title('Final Performance: N Agents × Episodes/Round')
    ax.set_xticks(x)
    ax.set_xticklabels(eps_values)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_summary_2x2(results: Dict[str, pd.DataFrame], save_path: Path):
    """Summary 2x2 figure for report"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    fspmi_keys = sorted([k for k in results.keys() if k != 'spmi'])
    colors = get_colors(len(fspmi_keys))
    
    # (a) Convergence
    ax = axes[0, 0]
    if 'spmi' in results:
        ax.plot(results['spmi']['iteration'], results['spmi']['evaluation'], 
                'k-', linewidth=2.5, label='Standard SPMI')
    
    for idx, config_key in enumerate(fspmi_keys):
        df = results[config_key]
        color = colors[idx]
        label = format_label(config_key)
        ax.plot(df['iteration'], df['true_perf_mean'], 
                color=color, linewidth=2, label=f'F-SPMI ({label})')
        if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
            ax.fill_between(df['iteration'],
                           df['true_perf_mean'] - df['true_perf_std'],
                           df['true_perf_mean'] + df['true_perf_std'],
                           color=color, alpha=0.15)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Return')
    ax.set_title('(a) Convergence')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    # (b) Sample Efficiency
    ax = axes[0, 1]
    if 'spmi' in results:
        final_perf = results['spmi']['evaluation'].iloc[-1]
        ax.axhline(y=final_perf, color='k', linestyle='--', linewidth=2, label='SPMI final')
    
    for idx, config_key in enumerate(fspmi_keys):
        df = results[config_key]
        color = colors[idx]
        label = format_label(config_key)
        ax.plot(df['cum_samples_mean'], df['true_perf_mean'], 
                color=color, linewidth=2, label=f'F-SPMI ({label})')
        if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
            ax.fill_between(df['cum_samples_mean'],
                           df['true_perf_mean'] - df['true_perf_std'],
                           df['true_perf_mean'] + df['true_perf_std'],
                           color=color, alpha=0.15)
    
    ax.set_xlabel('Total Samples')
    ax.set_ylabel('Return')
    ax.set_title('(b) Sample Efficiency')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    # (c) Safety Bounds
    ax = axes[1, 0]
    if 'spmi' in results and 'bound' in results['spmi'].columns:
        ax.semilogy(results['spmi']['iteration'], 
                   np.maximum(results['spmi']['bound'], 1e-10),
                   'k-', linewidth=2, label='SPMI')
    
    for idx, config_key in enumerate(fspmi_keys):
        df = results[config_key]
        color = colors[idx]
        label = format_label(config_key)
        ax.semilogy(df['iteration'], np.maximum(df['bound_mean'], 1e-10),
                   color=color, linewidth=2, label=f'F-SPMI ({label})')
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Bound (log)')
    ax.set_title('(c) Safety Bounds')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    # (d) Final Performance Bar Chart
    ax = axes[1, 1]
    
    bar_labels = []
    bar_means = []
    bar_stds = []
    bar_colors = []
    
    if 'spmi' in results:
        bar_labels.append('SPMI')
        bar_means.append(results['spmi']['evaluation'].iloc[-1])
        bar_stds.append(0)
        bar_colors.append('black')
    
    for idx, config_key in enumerate(fspmi_keys):
        df = results[config_key]
        bar_labels.append(format_label(config_key))
        bar_means.append(df['true_perf_mean'].iloc[-1])
        bar_stds.append(df['true_perf_std'].iloc[-1] if 'true_perf_std' in df.columns else 0)
        bar_colors.append(colors[idx])
    
    x = np.arange(len(bar_labels))
    ax.bar(x, bar_means, yerr=bar_stds, capsize=4, color=bar_colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Final Return')
    ax.set_title('(d) Final Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


# ============================================================================
# FACETED PLOTS BY EPS/ROUND
# ============================================================================

def group_by_eps(results: Dict[str, pd.DataFrame]) -> Dict[int, Dict[str, pd.DataFrame]]:
    """Group results by eps/round value"""
    grouped = {}
    
    for key, df in results.items():
        if key == 'spmi':
            continue
        _, eps = parse_config_key(key)
        if eps is not None:
            if eps not in grouped:
                grouped[eps] = {}
            grouped[eps][key] = df
    
    return grouped


def get_n_colors(results_for_eps: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
    """Get consistent colors for N agents across all facets"""
    all_n = set()
    for key in results_for_eps.keys():
        n, _ = parse_config_key(key)
        if n is not None:
            all_n.add(n)
    
    n_values = sorted(all_n)
    color_map = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(n_values)))
    for i, n in enumerate(n_values):
        color_map[n] = colors[i]
    
    return color_map


def plot_convergence_faceted(results: Dict[str, pd.DataFrame], save_path: Path):
    """Faceted convergence plot - one subplot per eps/round"""
    grouped = group_by_eps(results)
    
    if not grouped:
        print("  Skipping faceted convergence (no eps configs found)")
        return
    
    eps_values = sorted(grouped.keys())
    n_cols = min(len(eps_values), 4)
    n_rows = (len(eps_values) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    axes = axes.flatten()
    
    # Get consistent colors for N agents
    all_results = {}
    for eps_results in grouped.values():
        all_results.update(eps_results)
    color_map = get_n_colors(all_results)
    
    for idx, eps in enumerate(eps_values):
        ax = axes[idx]
        eps_results = grouped[eps]
        
        # SPMI baseline
        if 'spmi' in results:
            ax.plot(results['spmi']['iteration'], results['spmi']['evaluation'], 
                    'k-', linewidth=2, label='SPMI')
        
        # F-SPMI for this eps
        for config_key in sorted(eps_results.keys()):
            df = eps_results[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            ax.plot(df['iteration'], df['true_perf_mean'], 
                    color=color, linewidth=2, label=f'N={n}')
            
            if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
                ax.fill_between(df['iteration'],
                               df['true_perf_mean'] - df['true_perf_std'],
                               df['true_perf_mean'] + df['true_perf_std'],
                               color=color, alpha=0.2)
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Return')
        ax.set_title(f'eps/round = {eps}')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
    
    # Hide unused subplots
    for idx in range(len(eps_values), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Convergence by Episodes/Round', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_mc_vs_exact_faceted(results: Dict[str, pd.DataFrame], save_path: Path):
    """Faceted MC vs Exact plot - one subplot per eps/round"""
    grouped = group_by_eps(results)
    
    if not grouped:
        print("  Skipping faceted MC vs Exact (no eps configs found)")
        return
    
    eps_values = sorted(grouped.keys())
    n_cols = min(len(eps_values), 4)
    n_rows = (len(eps_values) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    axes = axes.flatten()
    
    all_results = {}
    for eps_results in grouped.values():
        all_results.update(eps_results)
    color_map = get_n_colors(all_results)
    
    for idx, eps in enumerate(eps_values):
        ax = axes[idx]
        eps_results = grouped[eps]
        
        for config_key in sorted(eps_results.keys()):
            df = eps_results[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            # Exact (solid)
            ax.plot(df['iteration'], df['true_perf_mean'], 
                    color=color, linestyle='-', linewidth=2, label=f'N={n} (Exact)')
            
            # MC (dashed)
            ax.plot(df['iteration'], df['mc_perf_mean'], 
                    color=color, linestyle='--', linewidth=1.5, alpha=0.7, label=f'N={n} (MC)')
            
            if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
                ax.fill_between(df['iteration'],
                               df['true_perf_mean'] - df['true_perf_std'],
                               df['true_perf_mean'] + df['true_perf_std'],
                               color=color, alpha=0.1)
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Return')
        ax.set_title(f'eps/round = {eps}')
        ax.legend(loc='lower right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
    
    for idx in range(len(eps_values), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('MC vs Exact Evaluator by Episodes/Round', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_safety_bounds_faceted(results: Dict[str, pd.DataFrame], save_path: Path):
    """Faceted safety bounds plot - one subplot per eps/round"""
    grouped = group_by_eps(results)
    
    if not grouped:
        print("  Skipping faceted safety bounds (no eps configs found)")
        return
    
    eps_values = sorted(grouped.keys())
    n_cols = min(len(eps_values), 4)
    n_rows = (len(eps_values) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    axes = axes.flatten()
    
    all_results = {}
    for eps_results in grouped.values():
        all_results.update(eps_results)
    color_map = get_n_colors(all_results)
    
    for idx, eps in enumerate(eps_values):
        ax = axes[idx]
        eps_results = grouped[eps]
        
        # SPMI baseline
        if 'spmi' in results and 'bound' in results['spmi'].columns:
            ax.semilogy(results['spmi']['iteration'], 
                       np.maximum(results['spmi']['bound'], 1e-10),
                       'k-', linewidth=2, label='SPMI')
        
        for config_key in sorted(eps_results.keys()):
            df = eps_results[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            ax.semilogy(df['iteration'], np.maximum(df['bound_mean'], 1e-10),
                       color=color, linewidth=2, label=f'N={n}')
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Bound (log)')
        ax.set_title(f'eps/round = {eps}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
    
    for idx in range(len(eps_values), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Safety Bounds by Episodes/Round', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_step_sizes_faceted(results: Dict[str, pd.DataFrame], save_path: Path):
    """Faceted step sizes plot - 2 rows (alpha/beta) x N cols (eps values)"""
    grouped = group_by_eps(results)
    
    if not grouped:
        print("  Skipping faceted step sizes (no eps configs found)")
        return
    
    eps_values = sorted(grouped.keys())
    n_cols = len(eps_values)
    
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8), squeeze=False)
    
    all_results = {}
    for eps_results in grouped.values():
        all_results.update(eps_results)
    color_map = get_n_colors(all_results)
    
    for col_idx, eps in enumerate(eps_values):
        eps_results = grouped[eps]
        
        # Row 0: Alpha
        ax_alpha = axes[0, col_idx]
        
        if 'spmi' in results and 'alpha' in results['spmi'].columns:
            ax_alpha.semilogy(results['spmi']['iteration'], 
                             np.maximum(results['spmi']['alpha'], 1e-10),
                             'k-', linewidth=2, label='SPMI', alpha=0.8)
        
        for config_key in sorted(eps_results.keys()):
            df = eps_results[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            if 'alpha_mean' in df.columns:
                ax_alpha.semilogy(df['iteration'], 
                                 np.maximum(df['alpha_mean'], 1e-10),
                                 color=color, linewidth=2, label=f'N={n}', alpha=0.8)
        
        ax_alpha.set_xlabel('Iterations')
        ax_alpha.set_ylabel('α (Policy)')
        ax_alpha.set_title(f'eps/round = {eps}')
        ax_alpha.legend(fontsize=8)
        ax_alpha.grid(True, alpha=0.3)
        ax_alpha.set_xlim(left=0)
        
        # Row 1: Beta
        ax_beta = axes[1, col_idx]
        
        if 'spmi' in results and 'beta' in results['spmi'].columns:
            ax_beta.semilogy(results['spmi']['iteration'], 
                            np.maximum(results['spmi']['beta'], 1e-10),
                            'k-', linewidth=2, label='SPMI', alpha=0.8)
        
        for config_key in sorted(eps_results.keys()):
            df = eps_results[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            if 'beta_mean' in df.columns:
                ax_beta.semilogy(df['iteration'], 
                                np.maximum(df['beta_mean'], 1e-10),
                                color=color, linewidth=2, label=f'N={n}', alpha=0.8)
        
        ax_beta.set_xlabel('Iterations')
        ax_beta.set_ylabel('β (Model)')
        ax_beta.legend(fontsize=8)
        ax_beta.grid(True, alpha=0.3)
        ax_beta.set_xlim(left=0)
    
    plt.suptitle('Step Sizes by Episodes/Round', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_sample_efficiency_faceted(results: Dict[str, pd.DataFrame], save_path: Path):
    """Faceted sample efficiency plot - one subplot per eps/round"""
    grouped = group_by_eps(results)
    
    if not grouped:
        print("  Skipping faceted sample efficiency (no eps configs found)")
        return
    
    eps_values = sorted(grouped.keys())
    n_cols = min(len(eps_values), 4)
    n_rows = (len(eps_values) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    axes = axes.flatten()
    
    all_results = {}
    for eps_results in grouped.values():
        all_results.update(eps_results)
    color_map = get_n_colors(all_results)
    
    # SPMI final performance
    spmi_final = None
    if 'spmi' in results:
        spmi_final = results['spmi']['evaluation'].iloc[-1]
    
    for idx, eps in enumerate(eps_values):
        ax = axes[idx]
        eps_results = grouped[eps]
        
        if spmi_final is not None:
            ax.axhline(y=spmi_final, color='k', linestyle='--', linewidth=2, label='SPMI final')
        
        for config_key in sorted(eps_results.keys()):
            df = eps_results[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            ax.plot(df['cum_samples_mean'], df['true_perf_mean'], 
                    color=color, linewidth=2, label=f'N={n}')
            
            if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
                ax.fill_between(df['cum_samples_mean'],
                               df['true_perf_mean'] - df['true_perf_std'],
                               df['true_perf_mean'] + df['true_perf_std'],
                               color=color, alpha=0.2)
        
        ax.set_xlabel('Total Samples')
        ax.set_ylabel('Return')
        ax.set_title(f'eps/round = {eps}')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
    
    for idx in range(len(eps_values), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Sample Efficiency by Episodes/Round', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_final_performance_faceted(results: Dict[str, pd.DataFrame], save_path: Path):
    """Faceted bar chart - one subplot per eps/round showing N agents comparison"""
    grouped = group_by_eps(results)
    
    if not grouped:
        print("  Skipping faceted final performance (no eps configs found)")
        return
    
    eps_values = sorted(grouped.keys())
    n_cols = min(len(eps_values), 4)
    n_rows = (len(eps_values) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    axes = axes.flatten()
    
    all_results = {}
    for eps_results in grouped.values():
        all_results.update(eps_results)
    color_map = get_n_colors(all_results)
    
    # SPMI final
    spmi_final = None
    if 'spmi' in results:
        spmi_final = results['spmi']['evaluation'].iloc[-1]
    
    for idx, eps in enumerate(eps_values):
        ax = axes[idx]
        eps_results = grouped[eps]
        
        # Collect data
        labels = []
        means = []
        stds = []
        colors_list = []
        
        if spmi_final is not None:
            labels.append('SPMI')
            means.append(spmi_final)
            stds.append(0)
            colors_list.append('black')
        
        for config_key in sorted(eps_results.keys()):
            df = eps_results[config_key]
            n, _ = parse_config_key(config_key)
            
            labels.append(f'N={n}')
            means.append(df['true_perf_mean'].iloc[-1])
            stds.append(df['true_perf_std'].iloc[-1] if 'true_perf_std' in df.columns else 0)
            colors_list.append(color_map.get(n, 'gray'))
        
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors_list, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Final Return')
        ax.set_title(f'eps/round = {eps}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.annotate(f'{mean:.4f}', xy=(i, mean), xytext=(0, 3),
                       textcoords='offset points', ha='center', fontsize=8)
    
    for idx in range(len(eps_values), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Final Performance by Episodes/Round', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot F-SPMI results")
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to results directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures (default: results_dir/figures)')
    parser.add_argument('--n_agents', type=int, default=None,
                        help='Filter by N agents (e.g., --n_agents 4)')
    parser.add_argument('--eps_per_round', type=int, default=None,
                        help='Filter by episodes per round (e.g., --eps_per_round 400)')
    parser.add_argument('--show_seeds', action='store_true',
                        help='Show individual seed traces instead of just mean')
    parser.add_argument('--faceted', action='store_true',
                        help='Generate faceted plots (one subplot per eps/round value)')
    parser.add_argument('--no-standard', action='store_true',
                        help='Skip standard (non-faceted) plots')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Create output subdirectory based on filters
    filter_suffix = ""
    if args.n_agents is not None:
        filter_suffix += f"_n{args.n_agents}"
    if args.eps_per_round is not None:
        filter_suffix += f"_eps{args.eps_per_round}"
    if args.faceted:
        filter_suffix += "_faceted"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / f"figures{filter_suffix}"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("F-SPMI Results Plotter")
    print("=" * 60)
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    if args.n_agents:
        print(f"Filter: N agents = {args.n_agents}")
    if args.eps_per_round:
        print(f"Filter: eps/round = {args.eps_per_round}")
    if args.show_seeds:
        print("Mode: Showing individual seed traces")
    if args.faceted:
        print("Mode: Generating faceted plots (by eps/round)")
    
    print("\nLoading results...")
    results = load_results(results_dir)
    
    # Load seed data if requested
    seed_data = None
    if args.show_seeds:
        print("Loading individual seed files...")
        seed_data = load_seed_data(results_dir)
        for key, seeds in seed_data.items():
            print(f"  {key}: {len(seeds)} seeds")
    
    if not results:
        print("ERROR: No results found!")
        return
    
    # Apply filters
    if args.n_agents is not None or args.eps_per_round is not None:
        print("\nApplying filters...")
        results = filter_results(results, args.n_agents, args.eps_per_round)
        if seed_data:
            seed_data = {k: v for k, v in seed_data.items() 
                        if k in results}
        fspmi_count = len([k for k in results.keys() if k != 'spmi'])
        print(f"  After filtering: {fspmi_count} F-SPMI configs + SPMI baseline")
    
    if len(results) == 0 or (len(results) == 1 and 'spmi' in results):
        print("ERROR: No F-SPMI results match the filters!")
        return
    
    print("\nGenerating figures...")
    
    # Add filter info to titles
    title_suffix = ""
    if args.n_agents:
        title_suffix += f" (N={args.n_agents})"
    if args.eps_per_round:
        title_suffix += f" (eps={args.eps_per_round})"
    
    # Generate standard plots (unless --no-standard)
    if not args.no_standard:
        print("\n--- Standard Plots ---")
        plot_convergence(results, output_dir / "fig1_convergence.png", title_suffix, seed_data)
        plot_mc_vs_exact(results, output_dir / "fig2_mc_vs_exact.png")
        plot_safety_bounds(results, output_dir / "fig3_safety_bounds.png")
        plot_sample_efficiency(results, output_dir / "fig4_sample_efficiency.png")
        plot_returns_and_bounds(results, output_dir / "fig5_returns_bounds.png")
        plot_step_sizes(results, output_dir / "fig6_step_sizes.png")
        plot_bar_chart(results, output_dir / "fig7_bar_final.png")
        plot_bar_chart_grouped(results, output_dir / "fig8_bar_grouped.png")
        plot_summary_2x2(results, output_dir / "fig_summary.png")
    
    # Generate faceted plots (if --faceted)
    if args.faceted:
        print("\n--- Faceted Plots (by eps/round) ---")
        plot_convergence_faceted(results, output_dir / "facet_convergence.png")
        plot_mc_vs_exact_faceted(results, output_dir / "facet_mc_vs_exact.png")
        plot_safety_bounds_faceted(results, output_dir / "facet_safety_bounds.png")
        plot_step_sizes_faceted(results, output_dir / "facet_step_sizes.png")
        plot_sample_efficiency_faceted(results, output_dir / "facet_sample_efficiency.png")
        plot_final_performance_faceted(results, output_dir / "facet_final_performance.png")
    
    print("\n" + "=" * 60)
    print("✓ All figures generated!")
    print(f"  Output: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()