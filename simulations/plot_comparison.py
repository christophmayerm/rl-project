"""
Plot Greedy vs GP Comparison
=============================
Compares F-SPMI results between greedy and GP model chooser strategies.

Usage:
    python plot_comparison.py \
        --greedy_dir ./data/report_experiments/racetrack4_T1/greedy/TIMESTAMP \
        --gp_dir ./data/report_experiments/racetrack4_T1/gp/TIMESTAMP
    
    # With faceted plots
    python plot_comparison.py --greedy_dir ... --gp_dir ... --faceted
    
    # Filter by N agents or eps
    python plot_comparison.py --greedy_dir ... --gp_dir ... --n_agents 4
    python plot_comparison.py --greedy_dir ... --gp_dir ... --eps_per_round 400

Standard Plots Generated:
    - cmp_summary.pdf: 2x2 summary (convergence, sample efficiency, bounds, final bars)
    - cmp_convergence.pdf: Convergence comparison
    - cmp_safety_bounds.pdf: Safety bounds comparison
    - cmp_returns_bounds.pdf: Returns & bounds on dual axes with std bands
    - cmp_step_sizes.pdf: Alpha & beta step sizes
    - cmp_sample_efficiency.pdf: Performance vs total samples
    - cmp_final_performance.pdf: Grouped bar chart
    - cmp_mc_vs_exact.pdf: MC vs Exact for each strategy

Faceted Plots Generated (--faceted):
    Layout: 2 rows (Greedy on top, GP on bottom) x N columns (one per eps value)
    
    - cmp_facet_convergence.pdf: Convergence curves
    - cmp_facet_safety_bounds.pdf: Safety bounds
    - cmp_facet_returns_bounds.pdf: Returns & bounds (dual axes)
    - cmp_facet_step_sizes.pdf: 4 rows (Greedy α, GP α, Greedy β, GP β) x N cols
    - cmp_facet_sample_efficiency.pdf: Sample efficiency
    - cmp_facet_final_performance.pdf: Bar charts

Color Scheme:
    - Greedy: Blue (solid lines)
    - GP: Orange (dashed lines)
    - SPMI: Black (baseline)
    
All plots include std bands where data is available. No log scale is used.
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

# Color schemes for greedy vs GP
GREEDY_COLOR = '#1f77b4'  # Blue
GP_COLOR = '#ff7f0e'       # Orange
SPMI_COLOR = 'black'


def parse_config_key(key: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract n_agents and eps_per_round from config key like 'n4_eps100'"""
    n_match = re.search(r'n(\d+)', key)
    eps_match = re.search(r'eps(\d+)', key)
    
    n_agents = int(n_match.group(1)) if n_match else None
    eps = int(eps_match.group(1)) if eps_match else None
    
    return n_agents, eps


def load_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all CSV results from directory"""
    results = {}
    
    # Load aggregated F-SPMI results first (preferred)
    for f in results_dir.glob("fspmi_*_aggregated.csv"):
        key = f.stem.replace("fspmi_", "").replace("_aggregated", "")
        results[key] = pd.read_csv(f)
        results[key]['source'] = 'aggregated'
    
    # If no aggregated results, try loading seed0 files
    if not results:
        for f in results_dir.glob("fspmi_*_seed0.csv"):
            key = f.stem.replace("fspmi_", "").replace("_seed0", "")
            df = pd.read_csv(f)
            df = df.rename(columns={
                'performance_true': 'true_perf_mean',
                'performance_mc': 'mc_perf_mean',
                'bound': 'bound_mean',
                'alpha': 'alpha_mean',
                'beta': 'beta_mean',
            })
            for col in ['true_perf', 'mc_perf', 'bound', 'alpha', 'beta']:
                df[f'{col}_std'] = 0
            if 'total_samples' in df.columns:
                df['cum_samples_mean'] = df['total_samples'].cumsum()
            df['source'] = 'single_seed'
            results[key] = df
    
    # Load standard SPMI if exists
    spmi_file = results_dir / "standard_spmi.csv"
    if spmi_file.exists():
        spmi_df = pd.read_csv(spmi_file, sep=';')
        spmi_df = spmi_df.rename(columns={
            '# iterations': 'iteration',
            'evaluations': 'evaluation',
            'alfa': 'alpha',
        })
        if 'iteration' not in spmi_df.columns:
            spmi_df['iteration'] = range(len(spmi_df))
        spmi_df['source'] = 'spmi'
        results['spmi'] = spmi_df
    
    return results


def filter_results(results: Dict[str, pd.DataFrame], 
                   n_agents: Optional[int] = None,
                   eps_per_round: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Filter results by n_agents and/or eps_per_round"""
    filtered = {}
    
    for key, df in results.items():
        if key == 'spmi':
            filtered[key] = df
            continue
        
        n, eps = parse_config_key(key)
        
        if n_agents is not None and n != n_agents:
            continue
        if eps_per_round is not None and eps != eps_per_round:
            continue
        
        filtered[key] = df
    
    return filtered


def format_label(config_key: str) -> str:
    """Format config key for legend labels"""
    return config_key.replace('_', ', ').replace('n', 'N=').replace('eps', 'eps=')


def get_matching_configs(greedy: Dict, gp: Dict) -> List[str]:
    """Get config keys that exist in both greedy and GP results"""
    greedy_keys = set(k for k in greedy.keys() if k != 'spmi')
    gp_keys = set(k for k in gp.keys() if k != 'spmi')
    return sorted(greedy_keys & gp_keys)


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


# =============================================================================
# COMPARISON PLOTS
# =============================================================================

def plot_convergence_comparison(greedy: Dict, gp: Dict, save_path: Path):
    """Compare convergence between greedy and GP for all matching configs"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    matching = get_matching_configs(greedy, gp)
    
    # SPMI baseline (use greedy's if available)
    if 'spmi' in greedy:
        ax.plot(greedy['spmi']['iteration'], greedy['spmi']['evaluation'], 
                color=SPMI_COLOR, linestyle='-', linewidth=2.5, label='SPMI')
    
    # Plot each config for both strategies
    for config_key in matching:
        label = format_label(config_key)
        
        # Greedy (solid)
        df_g = greedy[config_key]
        ax.plot(df_g['iteration'], df_g['true_perf_mean'], 
                color=GREEDY_COLOR, linestyle='-', linewidth=1.5, alpha=0.7,
                label=f'Greedy ({label})')
        if 'true_perf_std' in df_g.columns and df_g['true_perf_std'].sum() > 0:
            ax.fill_between(df_g['iteration'],
                           df_g['true_perf_mean'] - df_g['true_perf_std'],
                           df_g['true_perf_mean'] + df_g['true_perf_std'],
                           color=GREEDY_COLOR, alpha=0.1)
        
        # GP (dashed)
        df_gp = gp[config_key]
        ax.plot(df_gp['iteration'], df_gp['true_perf_mean'], 
                color=GP_COLOR, linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'GP ({label})')
        if 'true_perf_std' in df_gp.columns and df_gp['true_perf_std'].sum() > 0:
            ax.fill_between(df_gp['iteration'],
                           df_gp['true_perf_mean'] - df_gp['true_perf_std'],
                           df_gp['true_perf_mean'] + df_gp['true_perf_std'],
                           color=GP_COLOR, alpha=0.1)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Return')
    ax.set_title('Convergence: Greedy vs GP')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_convergence_comparison_faceted(greedy: Dict, gp: Dict, save_path: Path):
    """Faceted convergence comparison - Row 1: Greedy, Row 2: GP"""
    greedy_grouped = group_by_eps(greedy)
    gp_grouped = group_by_eps(gp)
    
    # Get all eps values from both
    all_eps = sorted(set(greedy_grouped.keys()) | set(gp_grouped.keys()))
    
    if not all_eps:
        print("  Skipping faceted convergence (no eps configs found)")
        return
    
    n_cols = len(all_eps)
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 8), squeeze=False)
    
    # Get consistent colors for N values
    all_n = set()
    for eps_dict in list(greedy_grouped.values()) + list(gp_grouped.values()):
        for key in eps_dict.keys():
            n, _ = parse_config_key(key)
            if n is not None:
                all_n.add(n)
    n_values = sorted(all_n)
    n_colors = plt.cm.tab10(np.linspace(0, 1, len(n_values)))
    color_map = {n: n_colors[i] for i, n in enumerate(n_values)}
    
    for col_idx, eps in enumerate(all_eps):
        # Row 0: Greedy
        ax_greedy = axes[0, col_idx]
        
        if 'spmi' in greedy:
            ax_greedy.plot(greedy['spmi']['iteration'], greedy['spmi']['evaluation'], 
                          color=SPMI_COLOR, linestyle='-', linewidth=2, label='SPMI')
        
        greedy_configs = greedy_grouped.get(eps, {})
        for config_key in sorted(greedy_configs.keys()):
            df = greedy_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            ax_greedy.plot(df['iteration'], df['true_perf_mean'], 
                          color=color, linestyle='-', linewidth=2, label=f'N={n}')
            if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
                ax_greedy.fill_between(df['iteration'],
                                       df['true_perf_mean'] - df['true_perf_std'],
                                       df['true_perf_mean'] + df['true_perf_std'],
                                       color=color, alpha=0.2)
        
        ax_greedy.set_xlabel('Iterations')
        ax_greedy.set_ylabel('Return')
        ax_greedy.set_title(f'Greedy - eps={eps}')
        ax_greedy.legend(loc='lower right', fontsize=8)
        ax_greedy.grid(True, alpha=0.3)
        ax_greedy.set_xlim(left=0)
        
        # Row 1: GP
        ax_gp = axes[1, col_idx]
        
        if 'spmi' in gp:
            ax_gp.plot(gp['spmi']['iteration'], gp['spmi']['evaluation'], 
                      color=SPMI_COLOR, linestyle='-', linewidth=2, label='SPMI')
        elif 'spmi' in greedy:
            ax_gp.plot(greedy['spmi']['iteration'], greedy['spmi']['evaluation'], 
                      color=SPMI_COLOR, linestyle='-', linewidth=2, label='SPMI')
        
        gp_configs = gp_grouped.get(eps, {})
        for config_key in sorted(gp_configs.keys()):
            df = gp_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            ax_gp.plot(df['iteration'], df['true_perf_mean'], 
                      color=color, linestyle='-', linewidth=2, label=f'N={n}')
            if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
                ax_gp.fill_between(df['iteration'],
                                   df['true_perf_mean'] - df['true_perf_std'],
                                   df['true_perf_mean'] + df['true_perf_std'],
                                   color=color, alpha=0.2)
        
        ax_gp.set_xlabel('Iterations')
        ax_gp.set_ylabel('Return')
        ax_gp.set_title(f'GP - eps={eps}')
        ax_gp.legend(loc='lower right', fontsize=8)
        ax_gp.grid(True, alpha=0.3)
        ax_gp.set_xlim(left=0)
    
    plt.suptitle('Convergence Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_safety_bounds_comparison(greedy: Dict, gp: Dict, save_path: Path):
    """Compare safety bounds between greedy and GP (no log scale)"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    matching = get_matching_configs(greedy, gp)
    
    # SPMI baseline
    if 'spmi' in greedy and 'bound' in greedy['spmi'].columns:
        ax.plot(greedy['spmi']['iteration'], greedy['spmi']['bound'],
               color=SPMI_COLOR, linestyle='-', linewidth=2.5, label='SPMI')
    
    for config_key in matching:
        label = format_label(config_key)
        
        # Greedy
        df_g = greedy[config_key]
        ax.plot(df_g['iteration'], df_g['bound_mean'],
               color=GREEDY_COLOR, linestyle='-', linewidth=1.5, alpha=0.7,
               label=f'Greedy ({label})')
        if 'bound_std' in df_g.columns and df_g['bound_std'].sum() > 0:
            ax.fill_between(df_g['iteration'],
                           df_g['bound_mean'] - df_g['bound_std'],
                           df_g['bound_mean'] + df_g['bound_std'],
                           color=GREEDY_COLOR, alpha=0.1)
        
        # GP
        df_gp = gp[config_key]
        ax.plot(df_gp['iteration'], df_gp['bound_mean'],
               color=GP_COLOR, linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'GP ({label})')
        if 'bound_std' in df_gp.columns and df_gp['bound_std'].sum() > 0:
            ax.fill_between(df_gp['iteration'],
                           df_gp['bound_mean'] - df_gp['bound_std'],
                           df_gp['bound_mean'] + df_gp['bound_std'],
                           color=GP_COLOR, alpha=0.1)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Bound')
    ax.set_title('Safety Bounds: Greedy vs GP')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_safety_bounds_comparison_faceted(greedy: Dict, gp: Dict, save_path: Path):
    """Faceted safety bounds comparison - Row 1: Greedy, Row 2: GP (no log scale)"""
    greedy_grouped = group_by_eps(greedy)
    gp_grouped = group_by_eps(gp)
    
    all_eps = sorted(set(greedy_grouped.keys()) | set(gp_grouped.keys()))
    
    if not all_eps:
        print("  Skipping faceted safety bounds (no eps configs found)")
        return
    
    n_cols = len(all_eps)
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 8), squeeze=False)
    
    # Get consistent colors for N values
    all_n = set()
    for eps_dict in list(greedy_grouped.values()) + list(gp_grouped.values()):
        for key in eps_dict.keys():
            n, _ = parse_config_key(key)
            if n is not None:
                all_n.add(n)
    n_values = sorted(all_n)
    n_colors = plt.cm.tab10(np.linspace(0, 1, len(n_values)))
    color_map = {n: n_colors[i] for i, n in enumerate(n_values)}
    
    for col_idx, eps in enumerate(all_eps):
        # Row 0: Greedy
        ax_greedy = axes[0, col_idx]
        
        if 'spmi' in greedy and 'bound' in greedy['spmi'].columns:
            ax_greedy.plot(greedy['spmi']['iteration'], greedy['spmi']['bound'],
                          color=SPMI_COLOR, linestyle='-', linewidth=2, label='SPMI')
        
        greedy_configs = greedy_grouped.get(eps, {})
        for config_key in sorted(greedy_configs.keys()):
            df = greedy_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            ax_greedy.plot(df['iteration'], df['bound_mean'],
                          color=color, linestyle='-', linewidth=2, label=f'N={n}')
            if 'bound_std' in df.columns and df['bound_std'].sum() > 0:
                ax_greedy.fill_between(df['iteration'],
                                       df['bound_mean'] - df['bound_std'],
                                       df['bound_mean'] + df['bound_std'],
                                       color=color, alpha=0.2)
        
        ax_greedy.set_xlabel('Iterations')
        ax_greedy.set_ylabel('Bound')
        ax_greedy.set_title(f'Greedy - eps={eps}')
        ax_greedy.legend(loc='upper right', fontsize=8)
        ax_greedy.grid(True, alpha=0.3)
        ax_greedy.set_xlim(left=0)
        
        # Row 1: GP
        ax_gp = axes[1, col_idx]
        
        if 'spmi' in gp and 'bound' in gp['spmi'].columns:
            ax_gp.plot(gp['spmi']['iteration'], gp['spmi']['bound'],
                      color=SPMI_COLOR, linestyle='-', linewidth=2, label='SPMI')
        elif 'spmi' in greedy and 'bound' in greedy['spmi'].columns:
            ax_gp.plot(greedy['spmi']['iteration'], greedy['spmi']['bound'],
                      color=SPMI_COLOR, linestyle='-', linewidth=2, label='SPMI')
        
        gp_configs = gp_grouped.get(eps, {})
        for config_key in sorted(gp_configs.keys()):
            df = gp_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            ax_gp.plot(df['iteration'], df['bound_mean'],
                      color=color, linestyle='-', linewidth=2, label=f'N={n}')
            if 'bound_std' in df.columns and df['bound_std'].sum() > 0:
                ax_gp.fill_between(df['iteration'],
                                   df['bound_mean'] - df['bound_std'],
                                   df['bound_mean'] + df['bound_std'],
                                   color=color, alpha=0.2)
        
        ax_gp.set_xlabel('Iterations')
        ax_gp.set_ylabel('Bound')
        ax_gp.set_title(f'GP - eps={eps}')
        ax_gp.legend(loc='upper right', fontsize=8)
        ax_gp.grid(True, alpha=0.3)
        ax_gp.set_xlim(left=0)
    
    plt.suptitle('Safety Bounds Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_step_sizes_comparison(greedy: Dict, gp: Dict, save_path: Path):
    """Compare step sizes between greedy and GP (no log scale)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    matching = get_matching_configs(greedy, gp)
    
    # Alpha
    ax1 = axes[0]
    if 'spmi' in greedy and 'alpha' in greedy['spmi'].columns:
        ax1.plot(greedy['spmi']['iteration'], greedy['spmi']['alpha'],
                color=SPMI_COLOR, linewidth=2, label='SPMI')
    
    for config_key in matching:
        label = format_label(config_key)
        
        df_g = greedy[config_key]
        if 'alpha_mean' in df_g.columns:
            ax1.plot(df_g['iteration'], df_g['alpha_mean'],
                    color=GREEDY_COLOR, linestyle='-', linewidth=1.5, alpha=0.7,
                    label=f'Greedy ({label})')
            if 'alpha_std' in df_g.columns and df_g['alpha_std'].sum() > 0:
                ax1.fill_between(df_g['iteration'],
                                df_g['alpha_mean'] - df_g['alpha_std'],
                                df_g['alpha_mean'] + df_g['alpha_std'],
                                color=GREEDY_COLOR, alpha=0.1)
        
        df_gp = gp[config_key]
        if 'alpha_mean' in df_gp.columns:
            ax1.plot(df_gp['iteration'], df_gp['alpha_mean'],
                    color=GP_COLOR, linestyle='--', linewidth=1.5, alpha=0.7,
                    label=f'GP ({label})')
            if 'alpha_std' in df_gp.columns and df_gp['alpha_std'].sum() > 0:
                ax1.fill_between(df_gp['iteration'],
                                df_gp['alpha_mean'] - df_gp['alpha_std'],
                                df_gp['alpha_mean'] + df_gp['alpha_std'],
                                color=GP_COLOR, alpha=0.1)
    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('α (Policy Step Size)')
    ax1.set_title('Policy Step Size: Greedy vs GP')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # Beta
    ax2 = axes[1]
    if 'spmi' in greedy and 'beta' in greedy['spmi'].columns:
        ax2.plot(greedy['spmi']['iteration'], greedy['spmi']['beta'],
                color=SPMI_COLOR, linewidth=2, label='SPMI')
    
    for config_key in matching:
        label = format_label(config_key)
        
        df_g = greedy[config_key]
        if 'beta_mean' in df_g.columns:
            ax2.plot(df_g['iteration'], df_g['beta_mean'],
                    color=GREEDY_COLOR, linestyle='-', linewidth=1.5, alpha=0.7,
                    label=f'Greedy ({label})')
            if 'beta_std' in df_g.columns and df_g['beta_std'].sum() > 0:
                ax2.fill_between(df_g['iteration'],
                                df_g['beta_mean'] - df_g['beta_std'],
                                df_g['beta_mean'] + df_g['beta_std'],
                                color=GREEDY_COLOR, alpha=0.1)
        
        df_gp = gp[config_key]
        if 'beta_mean' in df_gp.columns:
            ax2.plot(df_gp['iteration'], df_gp['beta_mean'],
                    color=GP_COLOR, linestyle='--', linewidth=1.5, alpha=0.7,
                    label=f'GP ({label})')
            if 'beta_std' in df_gp.columns and df_gp['beta_std'].sum() > 0:
                ax2.fill_between(df_gp['iteration'],
                                df_gp['beta_mean'] - df_gp['beta_std'],
                                df_gp['beta_mean'] + df_gp['beta_std'],
                                color=GP_COLOR, alpha=0.1)
    
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('β (Model Step Size)')
    ax2.set_title('Model Step Size: Greedy vs GP')
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_step_sizes_comparison_faceted(greedy: Dict, gp: Dict, save_path: Path):
    """Faceted step sizes - 4 rows (Greedy α, GP α, Greedy β, GP β) x N cols (eps)"""
    greedy_grouped = group_by_eps(greedy)
    gp_grouped = group_by_eps(gp)
    
    all_eps = sorted(set(greedy_grouped.keys()) | set(gp_grouped.keys()))
    
    if not all_eps:
        print("  Skipping faceted step sizes (no eps configs found)")
        return
    
    n_cols = len(all_eps)
    fig, axes = plt.subplots(4, n_cols, figsize=(4*n_cols, 14), squeeze=False)
    
    # Get consistent colors for N values
    all_n = set()
    for eps_dict in list(greedy_grouped.values()) + list(gp_grouped.values()):
        for key in eps_dict.keys():
            n, _ = parse_config_key(key)
            if n is not None:
                all_n.add(n)
    n_values = sorted(all_n)
    n_colors = plt.cm.tab10(np.linspace(0, 1, len(n_values)))
    color_map = {n: n_colors[i] for i, n in enumerate(n_values)}
    
    for col_idx, eps in enumerate(all_eps):
        greedy_configs = greedy_grouped.get(eps, {})
        gp_configs = gp_grouped.get(eps, {})
        
        # Row 0: Greedy Alpha
        ax = axes[0, col_idx]
        if 'spmi' in greedy and 'alpha' in greedy['spmi'].columns:
            ax.plot(greedy['spmi']['iteration'], greedy['spmi']['alpha'],
                   color=SPMI_COLOR, linewidth=2, label='SPMI')
        
        for config_key in sorted(greedy_configs.keys()):
            df = greedy_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            if 'alpha_mean' in df.columns:
                ax.plot(df['iteration'], df['alpha_mean'],
                       color=color, linewidth=2, label=f'N={n}')
                if 'alpha_std' in df.columns and df['alpha_std'].sum() > 0:
                    ax.fill_between(df['iteration'],
                                   df['alpha_mean'] - df['alpha_std'],
                                   df['alpha_mean'] + df['alpha_std'],
                                   color=color, alpha=0.2)
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel('α')
        ax.set_title(f'Greedy α - eps={eps}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        
        # Row 1: GP Alpha
        ax = axes[1, col_idx]
        if 'spmi' in greedy and 'alpha' in greedy['spmi'].columns:
            ax.plot(greedy['spmi']['iteration'], greedy['spmi']['alpha'],
                   color=SPMI_COLOR, linewidth=2, label='SPMI')
        
        for config_key in sorted(gp_configs.keys()):
            df = gp_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            if 'alpha_mean' in df.columns:
                ax.plot(df['iteration'], df['alpha_mean'],
                       color=color, linewidth=2, label=f'N={n}')
                if 'alpha_std' in df.columns and df['alpha_std'].sum() > 0:
                    ax.fill_between(df['iteration'],
                                   df['alpha_mean'] - df['alpha_std'],
                                   df['alpha_mean'] + df['alpha_std'],
                                   color=color, alpha=0.2)
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel('α')
        ax.set_title(f'GP α - eps={eps}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        
        # Row 2: Greedy Beta
        ax = axes[2, col_idx]
        if 'spmi' in greedy and 'beta' in greedy['spmi'].columns:
            ax.plot(greedy['spmi']['iteration'], greedy['spmi']['beta'],
                   color=SPMI_COLOR, linewidth=2, label='SPMI')
        
        for config_key in sorted(greedy_configs.keys()):
            df = greedy_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            if 'beta_mean' in df.columns:
                ax.plot(df['iteration'], df['beta_mean'],
                       color=color, linewidth=2, label=f'N={n}')
                if 'beta_std' in df.columns and df['beta_std'].sum() > 0:
                    ax.fill_between(df['iteration'],
                                   df['beta_mean'] - df['beta_std'],
                                   df['beta_mean'] + df['beta_std'],
                                   color=color, alpha=0.2)
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel('β')
        ax.set_title(f'Greedy β - eps={eps}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        
        # Row 3: GP Beta
        ax = axes[3, col_idx]
        if 'spmi' in greedy and 'beta' in greedy['spmi'].columns:
            ax.plot(greedy['spmi']['iteration'], greedy['spmi']['beta'],
                   color=SPMI_COLOR, linewidth=2, label='SPMI')
        
        for config_key in sorted(gp_configs.keys()):
            df = gp_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            if 'beta_mean' in df.columns:
                ax.plot(df['iteration'], df['beta_mean'],
                       color=color, linewidth=2, label=f'N={n}')
                if 'beta_std' in df.columns and df['beta_std'].sum() > 0:
                    ax.fill_between(df['iteration'],
                                   df['beta_mean'] - df['beta_std'],
                                   df['beta_mean'] + df['beta_std'],
                                   color=color, alpha=0.2)
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel('β')
        ax.set_title(f'GP β - eps={eps}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
    
    plt.suptitle('Step Sizes Comparison', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_sample_efficiency_comparison(greedy: Dict, gp: Dict, save_path: Path):
    """Compare sample efficiency between greedy and GP"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    matching = get_matching_configs(greedy, gp)
    
    # SPMI final
    if 'spmi' in greedy:
        spmi_final = greedy['spmi']['evaluation'].iloc[-1]
        ax.axhline(y=spmi_final, color=SPMI_COLOR, linestyle='--', linewidth=2,
                   label=f'SPMI final ({spmi_final:.4f})')
    
    for config_key in matching:
        label = format_label(config_key)
        
        df_g = greedy[config_key]
        ax.plot(df_g['cum_samples_mean'], df_g['true_perf_mean'], 
                color=GREEDY_COLOR, linestyle='-', linewidth=1.5, alpha=0.7,
                label=f'Greedy ({label})')
        if 'true_perf_std' in df_g.columns and df_g['true_perf_std'].sum() > 0:
            ax.fill_between(df_g['cum_samples_mean'],
                           df_g['true_perf_mean'] - df_g['true_perf_std'],
                           df_g['true_perf_mean'] + df_g['true_perf_std'],
                           color=GREEDY_COLOR, alpha=0.1)
        
        df_gp = gp[config_key]
        ax.plot(df_gp['cum_samples_mean'], df_gp['true_perf_mean'], 
                color=GP_COLOR, linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'GP ({label})')
        if 'true_perf_std' in df_gp.columns and df_gp['true_perf_std'].sum() > 0:
            ax.fill_between(df_gp['cum_samples_mean'],
                           df_gp['true_perf_mean'] - df_gp['true_perf_std'],
                           df_gp['true_perf_mean'] + df_gp['true_perf_std'],
                           color=GP_COLOR, alpha=0.1)
    
    ax.set_xlabel('Total Samples')
    ax.set_ylabel('Return')
    ax.set_title('Sample Efficiency: Greedy vs GP')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_sample_efficiency_comparison_faceted(greedy: Dict, gp: Dict, save_path: Path):
    """Faceted sample efficiency - Row 1: Greedy, Row 2: GP"""
    greedy_grouped = group_by_eps(greedy)
    gp_grouped = group_by_eps(gp)
    
    all_eps = sorted(set(greedy_grouped.keys()) | set(gp_grouped.keys()))
    
    if not all_eps:
        print("  Skipping faceted sample efficiency (no eps configs found)")
        return
    
    n_cols = len(all_eps)
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 8), squeeze=False)
    
    # Get consistent colors for N values
    all_n = set()
    for eps_dict in list(greedy_grouped.values()) + list(gp_grouped.values()):
        for key in eps_dict.keys():
            n, _ = parse_config_key(key)
            if n is not None:
                all_n.add(n)
    n_values = sorted(all_n)
    n_colors = plt.cm.tab10(np.linspace(0, 1, len(n_values)))
    color_map = {n: n_colors[i] for i, n in enumerate(n_values)}
    
    # SPMI final
    spmi_final = None
    if 'spmi' in greedy:
        spmi_final = greedy['spmi']['evaluation'].iloc[-1]
    
    for col_idx, eps in enumerate(all_eps):
        # Row 0: Greedy
        ax_greedy = axes[0, col_idx]
        
        if spmi_final is not None:
            ax_greedy.axhline(y=spmi_final, color=SPMI_COLOR, linestyle='--', 
                             linewidth=2, label='SPMI')
        
        greedy_configs = greedy_grouped.get(eps, {})
        for config_key in sorted(greedy_configs.keys()):
            df = greedy_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            ax_greedy.plot(df['cum_samples_mean'], df['true_perf_mean'], 
                          color=color, linestyle='-', linewidth=2, label=f'N={n}')
            if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
                ax_greedy.fill_between(df['cum_samples_mean'],
                                       df['true_perf_mean'] - df['true_perf_std'],
                                       df['true_perf_mean'] + df['true_perf_std'],
                                       color=color, alpha=0.2)
        
        ax_greedy.set_xlabel('Total Samples')
        ax_greedy.set_ylabel('Return')
        ax_greedy.set_title(f'Greedy - eps={eps}')
        ax_greedy.legend(loc='lower right', fontsize=8)
        ax_greedy.grid(True, alpha=0.3)
        ax_greedy.set_xlim(left=0)
        
        # Row 1: GP
        ax_gp = axes[1, col_idx]
        
        if spmi_final is not None:
            ax_gp.axhline(y=spmi_final, color=SPMI_COLOR, linestyle='--', 
                         linewidth=2, label='SPMI')
        
        gp_configs = gp_grouped.get(eps, {})
        for config_key in sorted(gp_configs.keys()):
            df = gp_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            ax_gp.plot(df['cum_samples_mean'], df['true_perf_mean'], 
                      color=color, linestyle='-', linewidth=2, label=f'N={n}')
            if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
                ax_gp.fill_between(df['cum_samples_mean'],
                                   df['true_perf_mean'] - df['true_perf_std'],
                                   df['true_perf_mean'] + df['true_perf_std'],
                                   color=color, alpha=0.2)
        
        ax_gp.set_xlabel('Total Samples')
        ax_gp.set_ylabel('Return')
        ax_gp.set_title(f'GP - eps={eps}')
        ax_gp.legend(loc='lower right', fontsize=8)
        ax_gp.grid(True, alpha=0.3)
        ax_gp.set_xlim(left=0)
    
    plt.suptitle('Sample Efficiency Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_final_performance_comparison(greedy: Dict, gp: Dict, save_path: Path):
    """Grouped bar chart comparing final performance: Greedy vs GP"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    matching = get_matching_configs(greedy, gp)
    
    if not matching:
        print("  Skipping final performance comparison (no matching configs)")
        return
    
    x = np.arange(len(matching))
    width = 0.35
    
    greedy_means = []
    greedy_stds = []
    gp_means = []
    gp_stds = []
    
    for config_key in matching:
        df_g = greedy[config_key]
        greedy_means.append(df_g['true_perf_mean'].iloc[-1])
        greedy_stds.append(df_g['true_perf_std'].iloc[-1] if 'true_perf_std' in df_g.columns else 0)
        
        df_gp = gp[config_key]
        gp_means.append(df_gp['true_perf_mean'].iloc[-1])
        gp_stds.append(df_gp['true_perf_std'].iloc[-1] if 'true_perf_std' in df_gp.columns else 0)
    
    # SPMI baseline
    if 'spmi' in greedy:
        spmi_final = greedy['spmi']['evaluation'].iloc[-1]
        ax.axhline(y=spmi_final, color=SPMI_COLOR, linestyle='--', linewidth=2,
                   label=f'SPMI ({spmi_final:.4f})')
    
    ax.bar(x - width/2, greedy_means, width, yerr=greedy_stds, capsize=4,
           label='Greedy', color=GREEDY_COLOR, alpha=0.8, edgecolor='black')
    ax.bar(x + width/2, gp_means, width, yerr=gp_stds, capsize=4,
           label='GP', color=GP_COLOR, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Final Return')
    ax.set_title('Final Performance: Greedy vs GP')
    ax.set_xticks(x)
    ax.set_xticklabels([format_label(k) for k in matching], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_final_performance_comparison_faceted(greedy: Dict, gp: Dict, save_path: Path):
    """Faceted bar chart - Row 1: Greedy, Row 2: GP"""
    greedy_grouped = group_by_eps(greedy)
    gp_grouped = group_by_eps(gp)
    
    all_eps = sorted(set(greedy_grouped.keys()) | set(gp_grouped.keys()))
    
    if not all_eps:
        print("  Skipping faceted final performance (no eps configs found)")
        return
    
    n_cols = len(all_eps)
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 8), squeeze=False)
    
    # Get consistent colors for N values
    all_n = set()
    for eps_dict in list(greedy_grouped.values()) + list(gp_grouped.values()):
        for key in eps_dict.keys():
            n, _ = parse_config_key(key)
            if n is not None:
                all_n.add(n)
    n_values = sorted(all_n)
    n_colors = plt.cm.tab10(np.linspace(0, 1, len(n_values)))
    color_map = {n: n_colors[i] for i, n in enumerate(n_values)}
    
    spmi_final = None
    if 'spmi' in greedy:
        spmi_final = greedy['spmi']['evaluation'].iloc[-1]

    spmi_final_gp = None
    if 'spmi' in gp:
        spmi_final_gp = gp['spmi']['evaluation'].iloc[-1]
    
    for col_idx, eps in enumerate(all_eps):
        greedy_configs = greedy_grouped.get(eps, {})
        gp_configs = gp_grouped.get(eps, {})
        
        # Row 0: Greedy
        ax_greedy = axes[0, col_idx]
        
        greedy_n_values = sorted(set(parse_config_key(k)[0] for k in greedy_configs.keys()))
        if greedy_n_values:
            x = np.arange(len(greedy_n_values))
            means = []
            stds = []
            colors_list = []
            
            for n in greedy_n_values:
                config_key = f'n{n}_eps{eps}'
                if config_key in greedy_configs:
                    df = greedy_configs[config_key]
                    means.append(df['true_perf_mean'].iloc[-1])
                    stds.append(df['true_perf_std'].iloc[-1] if 'true_perf_std' in df.columns else 0)
                    colors_list.append(color_map.get(n, 'gray'))
            
            if spmi_final is not None:
                ax_greedy.axhline(y=spmi_final, color=SPMI_COLOR, linestyle='--', 
                                 linewidth=2, label='SPMI')
            
            ax_greedy.bar(x, means, yerr=stds, capsize=4, color=colors_list, 
                         alpha=0.8, edgecolor='black')
            ax_greedy.set_xticks(x)
            ax_greedy.set_xticklabels([f'N={n}' for n in greedy_n_values])
        
        ax_greedy.set_xlabel('N agents')
        ax_greedy.set_ylabel('Final Return')
        ax_greedy.set_title(f'Greedy - eps={eps}')
        ax_greedy.legend(fontsize=8)
        ax_greedy.grid(True, alpha=0.3, axis='y')
        
        # Row 1: GP
        ax_gp = axes[1, col_idx]
        
        gp_n_values = sorted(set(parse_config_key(k)[0] for k in gp_configs.keys()))
        if gp_n_values:
            x = np.arange(len(gp_n_values))
            means = []
            stds = []
            colors_list = []
            
            for n in gp_n_values:
                config_key = f'n{n}_eps{eps}'
                if config_key in gp_configs:
                    df = gp_configs[config_key]
                    means.append(df['true_perf_mean'].iloc[-1])
                    stds.append(df['true_perf_std'].iloc[-1] if 'true_perf_std' in df.columns else 0)
                    colors_list.append(color_map.get(n, 'gray'))
            
            if spmi_final_gp is not None:
                ax_gp.axhline(y=spmi_final_gp, color=SPMI_COLOR, linestyle='--', 
                             linewidth=2, label='SPMI')
            
            ax_gp.bar(x, means, yerr=stds, capsize=4, color=colors_list, 
                     alpha=0.8, edgecolor='black')
            ax_gp.set_xticks(x)
            ax_gp.set_xticklabels([f'N={n}' for n in gp_n_values])
        
        ax_gp.set_xlabel('N agents')
        ax_gp.set_ylabel('Final Return')
        ax_gp.set_title(f'GP - eps={eps}')
        ax_gp.legend(fontsize=8)
        ax_gp.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Final Performance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_mc_vs_exact_comparison(greedy: Dict, gp: Dict, save_path: Path):
    """Compare MC vs Exact evaluation for greedy and GP"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    matching = get_matching_configs(greedy, gp)
    
    # Greedy subplot
    ax1 = axes[0]
    for config_key in matching:
        label = format_label(config_key)
        df = greedy[config_key]
        
        ax1.plot(df['iteration'], df['true_perf_mean'], 
                color=GREEDY_COLOR, linestyle='-', linewidth=1.5, alpha=0.7,
                label=f'Exact ({label})')
        ax1.plot(df['iteration'], df['mc_perf_mean'], 
                color=GREEDY_COLOR, linestyle='--', linewidth=1, alpha=0.5,
                label=f'MC ({label})')
    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Return')
    ax1.set_title('Greedy: MC vs Exact')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # GP subplot
    ax2 = axes[1]
    for config_key in matching:
        label = format_label(config_key)
        df = gp[config_key]
        
        ax2.plot(df['iteration'], df['true_perf_mean'], 
                color=GP_COLOR, linestyle='-', linewidth=1.5, alpha=0.7,
                label=f'Exact ({label})')
        ax2.plot(df['iteration'], df['mc_perf_mean'], 
                color=GP_COLOR, linestyle='--', linewidth=1, alpha=0.5,
                label=f'MC ({label})')
    
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Return')
    ax2.set_title('GP: MC vs Exact')
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_returns_and_bounds_comparison(greedy: Dict, gp: Dict, save_path: Path):
    """Compare returns and bounds between greedy and GP - dual axis plot (no log scale)"""
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    matching = get_matching_configs(greedy, gp)
    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Return', color='black')
    
    # SPMI baseline return
    if 'spmi' in greedy:
        ax1.plot(greedy['spmi']['iteration'], greedy['spmi']['evaluation'], 
                color=SPMI_COLOR, linestyle='-', linewidth=2.5, label='SPMI return')
    
    # Returns for all configs
    for config_key in matching:
        label = format_label(config_key)
        
        # Greedy return (solid)
        df_g = greedy[config_key]
        ax1.plot(df_g['iteration'], df_g['true_perf_mean'], 
                color=GREEDY_COLOR, linestyle='-', linewidth=1.5, alpha=0.8,
                label=f'Greedy ({label})')
        if 'true_perf_std' in df_g.columns and df_g['true_perf_std'].sum() > 0:
            ax1.fill_between(df_g['iteration'],
                            df_g['true_perf_mean'] - df_g['true_perf_std'],
                            df_g['true_perf_mean'] + df_g['true_perf_std'],
                            color=GREEDY_COLOR, alpha=0.1)
        
        # GP return (dashed)
        df_gp = gp[config_key]
        ax1.plot(df_gp['iteration'], df_gp['true_perf_mean'], 
                color=GP_COLOR, linestyle='--', linewidth=1.5, alpha=0.8,
                label=f'GP ({label})')
        if 'true_perf_std' in df_gp.columns and df_gp['true_perf_std'].sum() > 0:
            ax1.fill_between(df_gp['iteration'],
                            df_gp['true_perf_mean'] - df_gp['true_perf_std'],
                            df_gp['true_perf_mean'] + df_gp['true_perf_std'],
                            color=GP_COLOR, alpha=0.1)
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(left=0)
    ax1.grid(True, alpha=0.3)
    
    # Right axis: Bounds (NO log scale)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Safety Bound', color='gray')
    
    # SPMI baseline bound
    if 'spmi' in greedy and 'bound' in greedy['spmi'].columns:
        ax2.plot(greedy['spmi']['iteration'], greedy['spmi']['bound'], 
                color=SPMI_COLOR, linestyle=':', linewidth=1.5, alpha=0.6,
                label='SPMI bound')
    
    # Bounds for all configs
    for config_key in matching:
        label = format_label(config_key)
        
        # Greedy bound
        df_g = greedy[config_key]
        if 'bound_mean' in df_g.columns:
            ax2.plot(df_g['iteration'], df_g['bound_mean'], 
                    color=GREEDY_COLOR, linestyle=':', linewidth=1.5, alpha=0.6)
            if 'bound_std' in df_g.columns and df_g['bound_std'].sum() > 0:
                ax2.fill_between(df_g['iteration'],
                                df_g['bound_mean'] - df_g['bound_std'],
                                df_g['bound_mean'] + df_g['bound_std'],
                                color=GREEDY_COLOR, alpha=0.08)
        
        # GP bound
        df_gp = gp[config_key]
        if 'bound_mean' in df_gp.columns:
            ax2.plot(df_gp['iteration'], df_gp['bound_mean'], 
                    color=GP_COLOR, linestyle=':', linewidth=1.5, alpha=0.6)
            if 'bound_std' in df_gp.columns and df_gp['bound_std'].sum() > 0:
                ax2.fill_between(df_gp['iteration'],
                                df_gp['bound_mean'] - df_gp['bound_std'],
                                df_gp['bound_mean'] + df_gp['bound_std'],
                                color=GP_COLOR, alpha=0.08)
    
    ax2.tick_params(axis='y', labelcolor='gray')
    
    ax1.set_title('Returns (solid/dashed) & Safety Bounds (dotted): Greedy vs GP')
    ax1.legend(loc='center right', fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_returns_and_bounds_comparison_faceted(greedy: Dict, gp: Dict, save_path: Path):
    """Faceted returns & bounds - Row 1: Greedy, Row 2: GP (no log scale)"""
    greedy_grouped = group_by_eps(greedy)
    gp_grouped = group_by_eps(gp)
    
    all_eps = sorted(set(greedy_grouped.keys()) | set(gp_grouped.keys()))
    
    if not all_eps:
        print("  Skipping faceted returns & bounds comparison (no eps configs found)")
        return
    
    n_cols = len(all_eps)
    fig, axes = plt.subplots(2, n_cols, figsize=(6*n_cols, 10), squeeze=False)
    
    # Get consistent colors for N values
    all_n = set()
    for eps_dict in list(greedy_grouped.values()) + list(gp_grouped.values()):
        for key in eps_dict.keys():
            n, _ = parse_config_key(key)
            if n is not None:
                all_n.add(n)
    n_values = sorted(all_n)
    n_colors = plt.cm.tab10(np.linspace(0, 1, len(n_values)))
    color_map = {n: n_colors[i] for i, n in enumerate(n_values)}
    
    for col_idx, eps in enumerate(all_eps):
        # Row 0: Greedy
        ax1 = axes[0, col_idx]
        
        # SPMI baseline return
        if 'spmi' in greedy:
            ax1.plot(greedy['spmi']['iteration'], greedy['spmi']['evaluation'], 
                    color=SPMI_COLOR, linestyle='-', linewidth=2.5, label='SPMI')
        
        greedy_configs = greedy_grouped.get(eps, {})
        for config_key in sorted(greedy_configs.keys()):
            df = greedy_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            ax1.plot(df['iteration'], df['true_perf_mean'], 
                    color=color, linestyle='-', linewidth=2, label=f'N={n}')
            if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
                ax1.fill_between(df['iteration'],
                                df['true_perf_mean'] - df['true_perf_std'],
                                df['true_perf_mean'] + df['true_perf_std'],
                                color=color, alpha=0.2)
        
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Return')
        ax1.set_xlim(left=0)
        ax1.grid(True, alpha=0.3)
        
        # Right axis: Bounds
        ax2 = ax1.twinx()
        ax2.set_ylabel('Bound', color='gray', fontsize=9)
        
        if 'spmi' in greedy and 'bound' in greedy['spmi'].columns:
            ax2.plot(greedy['spmi']['iteration'], greedy['spmi']['bound'], 
                    color=SPMI_COLOR, linestyle=':', linewidth=1.5, alpha=0.6)
        
        for config_key in sorted(greedy_configs.keys()):
            df = greedy_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            if 'bound_mean' in df.columns:
                ax2.plot(df['iteration'], df['bound_mean'], 
                        color=color, linestyle=':', linewidth=1.5, alpha=0.6)
                if 'bound_std' in df.columns and df['bound_std'].sum() > 0:
                    ax2.fill_between(df['iteration'],
                                    df['bound_mean'] - df['bound_std'],
                                    df['bound_mean'] + df['bound_std'],
                                    color=color, alpha=0.1)
        
        ax2.tick_params(axis='y', labelcolor='gray')
        ax1.set_title(f'Greedy - eps={eps}')
        ax1.legend(loc='center right', fontsize=7)
        
        # Row 1: GP
        ax1 = axes[1, col_idx]
        
        # SPMI baseline
        if 'spmi' in greedy:
            ax1.plot(greedy['spmi']['iteration'], greedy['spmi']['evaluation'], 
                    color=SPMI_COLOR, linestyle='-', linewidth=2.5, label='SPMI')
        
        gp_configs = gp_grouped.get(eps, {})
        for config_key in sorted(gp_configs.keys()):
            df = gp_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            ax1.plot(df['iteration'], df['true_perf_mean'], 
                    color=color, linestyle='-', linewidth=2, label=f'N={n}')
            if 'true_perf_std' in df.columns and df['true_perf_std'].sum() > 0:
                ax1.fill_between(df['iteration'],
                                df['true_perf_mean'] - df['true_perf_std'],
                                df['true_perf_mean'] + df['true_perf_std'],
                                color=color, alpha=0.2)
        
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Return')
        ax1.set_xlim(left=0)
        ax1.grid(True, alpha=0.3)
        
        # Right axis: Bounds
        ax2 = ax1.twinx()
        ax2.set_ylabel('Bound', color='gray', fontsize=9)
        
        if 'spmi' in greedy and 'bound' in greedy['spmi'].columns:
            ax2.plot(greedy['spmi']['iteration'], greedy['spmi']['bound'], 
                    color=SPMI_COLOR, linestyle=':', linewidth=1.5, alpha=0.6)
        
        for config_key in sorted(gp_configs.keys()):
            df = gp_configs[config_key]
            n, _ = parse_config_key(config_key)
            color = color_map.get(n, 'gray')
            
            if 'bound_mean' in df.columns:
                ax2.plot(df['iteration'], df['bound_mean'], 
                        color=color, linestyle=':', linewidth=1.5, alpha=0.6)
                if 'bound_std' in df.columns and df['bound_std'].sum() > 0:
                    ax2.fill_between(df['iteration'],
                                    df['bound_mean'] - df['bound_std'],
                                    df['bound_mean'] + df['bound_std'],
                                    color=color, alpha=0.1)
        
        ax2.tick_params(axis='y', labelcolor='gray')
        ax1.set_title(f'GP - eps={eps}')
        ax1.legend(loc='center right', fontsize=7)
    
    plt.suptitle('Returns (solid) & Bounds (dotted) Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_summary_comparison(greedy: Dict, gp: Dict, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    matching = get_matching_configs(greedy, gp)
    
    # (a) Convergence
    ax = axes[0, 0]
    if 'spmi' in greedy:
        ax.plot(greedy['spmi']['iteration'], greedy['spmi']['evaluation'], 
                color=SPMI_COLOR, linestyle='-', linewidth=2.5, label='SPMI')
    
    for config_key in matching:
        label = format_label(config_key)
        
        df_g = greedy[config_key]
        ax.plot(df_g['iteration'], df_g['true_perf_mean'], 
                color=GREEDY_COLOR, linestyle='-', linewidth=1.5, alpha=0.7,
                label=f'Greedy ({label})')
        if 'true_perf_std' in df_g.columns and df_g['true_perf_std'].sum() > 0:
            ax.fill_between(df_g['iteration'],
                           df_g['true_perf_mean'] - df_g['true_perf_std'],
                           df_g['true_perf_mean'] + df_g['true_perf_std'],
                           color=GREEDY_COLOR, alpha=0.1)
        
        df_gp = gp[config_key]
        ax.plot(df_gp['iteration'], df_gp['true_perf_mean'], 
                color=GP_COLOR, linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'GP ({label})')
        if 'true_perf_std' in df_gp.columns and df_gp['true_perf_std'].sum() > 0:
            ax.fill_between(df_gp['iteration'],
                           df_gp['true_perf_mean'] - df_gp['true_perf_std'],
                           df_gp['true_perf_mean'] + df_gp['true_perf_std'],
                           color=GP_COLOR, alpha=0.1)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Return')
    ax.set_title('(a) Convergence')
    ax.legend(loc='lower right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    # (b) Sample Efficiency
    ax = axes[0, 1]
    if 'spmi' in greedy:
        spmi_final = greedy['spmi']['evaluation'].iloc[-1]
        ax.axhline(y=spmi_final, color=SPMI_COLOR, linestyle='--', linewidth=2, label='SPMI final')
    
    for config_key in matching:
        label = format_label(config_key)
        
        df_g = greedy[config_key]
        ax.plot(df_g['cum_samples_mean'], df_g['true_perf_mean'], 
                color=GREEDY_COLOR, linestyle='-', linewidth=1.5, alpha=0.7,
                label=f'Greedy ({label})')
        if 'true_perf_std' in df_g.columns and df_g['true_perf_std'].sum() > 0:
            ax.fill_between(df_g['cum_samples_mean'],
                           df_g['true_perf_mean'] - df_g['true_perf_std'],
                           df_g['true_perf_mean'] + df_g['true_perf_std'],
                           color=GREEDY_COLOR, alpha=0.1)
        
        df_gp = gp[config_key]
        ax.plot(df_gp['cum_samples_mean'], df_gp['true_perf_mean'], 
                color=GP_COLOR, linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'GP ({label})')
        if 'true_perf_std' in df_gp.columns and df_gp['true_perf_std'].sum() > 0:
            ax.fill_between(df_gp['cum_samples_mean'],
                           df_gp['true_perf_mean'] - df_gp['true_perf_std'],
                           df_gp['true_perf_mean'] + df_gp['true_perf_std'],
                           color=GP_COLOR, alpha=0.1)
    
    ax.set_xlabel('Total Samples')
    ax.set_ylabel('Return')
    ax.set_title('(b) Sample Efficiency')
    ax.legend(loc='lower right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    # (c) Safety Bounds
    ax = axes[1, 0]
    if 'spmi' in greedy and 'bound' in greedy['spmi'].columns:
        ax.semilogy(greedy['spmi']['iteration'], 
                   np.maximum(greedy['spmi']['bound'], 1e-10),
                   color=SPMI_COLOR, linestyle='-', linewidth=2, label='SPMI')
    
    for config_key in matching:
        label = format_label(config_key)
        
        df_g = greedy[config_key]
        ax.semilogy(df_g['iteration'], np.maximum(df_g['bound_mean'], 1e-10),
                   color=GREEDY_COLOR, linestyle='-', linewidth=1.5, alpha=0.7,
                   label=f'Greedy ({label})')
        if 'bound_std' in df_g.columns and df_g['bound_std'].sum() > 0:
            lower = np.maximum(df_g['bound_mean'] - df_g['bound_std'], 1e-10)
            upper = df_g['bound_mean'] + df_g['bound_std']
            ax.fill_between(df_g['iteration'], lower, upper, color=GREEDY_COLOR, alpha=0.1)
        
        df_gp = gp[config_key]
        ax.semilogy(df_gp['iteration'], np.maximum(df_gp['bound_mean'], 1e-10),
                   color=GP_COLOR, linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'GP ({label})')
        if 'bound_std' in df_gp.columns and df_gp['bound_std'].sum() > 0:
            lower = np.maximum(df_gp['bound_mean'] - df_gp['bound_std'], 1e-10)
            upper = df_gp['bound_mean'] + df_gp['bound_std']
            ax.fill_between(df_gp['iteration'], lower, upper, color=GP_COLOR, alpha=0.1)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Bound (log)')
    ax.set_title('(c) Safety Bounds')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    # (d) Final Performance Bar Chart
    ax = axes[1, 1]
    
    if matching:
        x = np.arange(len(matching))
        width = 0.35
        
        greedy_means = [greedy[k]['true_perf_mean'].iloc[-1] for k in matching]
        greedy_stds = [greedy[k]['true_perf_std'].iloc[-1] if 'true_perf_std' in greedy[k].columns else 0 for k in matching]
        gp_means = [gp[k]['true_perf_mean'].iloc[-1] for k in matching]
        gp_stds = [gp[k]['true_perf_std'].iloc[-1] if 'true_perf_std' in gp[k].columns else 0 for k in matching]
        
        if 'spmi' in greedy:
            spmi_final = greedy['spmi']['evaluation'].iloc[-1]
            ax.axhline(y=spmi_final, color=SPMI_COLOR, linestyle='--', linewidth=2, label='SPMI')
        
        ax.bar(x - width/2, greedy_means, width, yerr=greedy_stds, capsize=3,
               label='Greedy', color=GREEDY_COLOR, alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, gp_means, width, yerr=gp_stds, capsize=3,
               label='GP', color=GP_COLOR, alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Final Return')
        ax.set_title('(d) Final Performance')
        ax.set_xticks(x)
        ax.set_xticklabels([format_label(k) for k in matching], rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Greedy vs GP F-SPMI results")
    parser.add_argument('--greedy_dir', type=str, required=True,
                        help='Path to greedy results directory')
    parser.add_argument('--gp_dir', type=str, required=True,
                        help='Path to GP results directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures')
    parser.add_argument('--n_agents', type=int, default=None,
                        help='Filter by N agents')
    parser.add_argument('--eps_per_round', type=int, default=None,
                        help='Filter by episodes per round')
    parser.add_argument('--faceted', action='store_true',
                        help='Generate faceted plots')
    parser.add_argument('--no-standard', action='store_true',
                        help='Skip standard plots')
    args = parser.parse_args()
    
    greedy_dir = Path(args.greedy_dir)
    gp_dir = Path(args.gp_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = greedy_dir.parent / "comparison_greedy_vs_gp"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Greedy vs GP Comparison Plotter")
    print("=" * 60)
    print(f"Greedy dir: {greedy_dir}")
    print(f"GP dir: {gp_dir}")
    print(f"Output dir: {output_dir}")
    
    print("\nLoading greedy results...")
    greedy = load_results(greedy_dir)
    print(f"  Loaded {len(greedy)} configs")
    
    print("\nLoading GP results...")
    gp = load_results(gp_dir)
    print(f"  Loaded {len(gp)} configs")
    
    # Apply filters
    if args.n_agents is not None or args.eps_per_round is not None:
        print("\nApplying filters...")
        greedy = filter_results(greedy, args.n_agents, args.eps_per_round)
        gp = filter_results(gp, args.n_agents, args.eps_per_round)
        print(f"  Greedy: {len(greedy)} configs after filter")
        print(f"  GP: {len(gp)} configs after filter")
    
    matching = get_matching_configs(greedy, gp)
    print(f"\nMatching configs: {len(matching)}")
    for k in matching:
        print(f"  - {k}")
    
    if not matching:
        print("ERROR: No matching configurations between greedy and GP!")
        return
    
    print("\nGenerating figures...")
    
    # Standard plots
    if not args.no_standard:
        print("\n--- Standard Comparison Plots ---")
        plot_summary_comparison(greedy, gp, output_dir / "cmp_summary.pdf")
        plot_convergence_comparison(greedy, gp, output_dir / "cmp_convergence.pdf")
        plot_safety_bounds_comparison(greedy, gp, output_dir / "cmp_safety_bounds.pdf")
        plot_returns_and_bounds_comparison(greedy, gp, output_dir / "cmp_returns_bounds.pdf")
        plot_step_sizes_comparison(greedy, gp, output_dir / "cmp_step_sizes.pdf")
        plot_sample_efficiency_comparison(greedy, gp, output_dir / "cmp_sample_efficiency.pdf")
        plot_final_performance_comparison(greedy, gp, output_dir / "cmp_final_performance.pdf")
        plot_mc_vs_exact_comparison(greedy, gp, output_dir / "cmp_mc_vs_exact.pdf")
    
    # Faceted plots
    if args.faceted:
        print("\n--- Faceted Comparison Plots ---")
        plot_convergence_comparison_faceted(greedy, gp, output_dir / "cmp_facet_convergence.pdf")
        plot_safety_bounds_comparison_faceted(greedy, gp, output_dir / "cmp_facet_safety_bounds.pdf")
        plot_returns_and_bounds_comparison_faceted(greedy, gp, output_dir / "cmp_facet_returns_bounds.pdf")
        plot_step_sizes_comparison_faceted(greedy, gp, output_dir / "cmp_facet_step_sizes.pdf")
        plot_sample_efficiency_comparison_faceted(greedy, gp, output_dir / "cmp_facet_sample_efficiency.pdf")
        plot_final_performance_comparison_faceted(greedy, gp, output_dir / "cmp_facet_final_performance.pdf")
    
    print("\n" + "=" * 60)
    print("✓ All comparison figures generated!")
    print(f"  Output: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()