"""
Federated aggregation strategies for combining local statistics from multiple agents.

Implements various aggregation methods:
- Simple federated averaging
- Weighted federated averaging (by sample count)
- Robust federated averaging (with outlier detection)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class AggregationResult:
    """Result of federated aggregation"""
    Q_global: np.ndarray
    delta_mu_global: np.ndarray
    d_mu_global: np.ndarray
    total_samples: int
    n_agents: int
    confidence_radius: float
    Q_counts: np.ndarray = None  # Aggregated per-(s,a) sample counts


def federated_average(
    local_Q_estimates: List[np.ndarray],
    local_Q_counts: List[np.ndarray],
    local_delta_mu_estimates: List[np.ndarray],
    local_d_mu_estimates: List[np.ndarray],
    local_sample_counts: List[int]
) -> AggregationResult:
    """
    Simple federated averaging with uniform weights.

    Q̂_global = (1/N) Σ_j Q̂_j
    δ̂_μ_global = (1/N) Σ_j δ̂_μ,j

    Args:
        local_Q_estimates: List of Q-function estimates from each agent
        local_Q_counts: List of sample counts per (s,a) from each agent
        local_delta_mu_estimates: List of δ_μ estimates from each agent
        local_d_mu_estimates: List of d_μ estimates from each agent
        local_sample_counts: Number of samples from each agent

    Returns:
        AggregationResult with global estimates
    """
    N = len(local_Q_estimates)
    if N == 0:
        raise ValueError("No local estimates provided")

    # Simple averaging
    Q_global = np.mean(local_Q_estimates, axis=0)
    delta_mu_global = np.mean(local_delta_mu_estimates, axis=0)
    d_mu_global = np.mean(local_d_mu_estimates, axis=0)

    total_samples = sum(local_sample_counts)

    # Confidence radius (approximate, assuming bounded estimates)
    confidence_radius = 1.0 / np.sqrt(total_samples + 1)

    # Sum up Q counts from all agents
    Q_count_total = np.sum(np.stack(local_Q_counts, axis=0), axis=0)

    return AggregationResult(
        Q_global=Q_global,
        delta_mu_global=delta_mu_global,
        d_mu_global=d_mu_global,
        total_samples=total_samples,
        n_agents=N,
        confidence_radius=confidence_radius,
        Q_counts=Q_count_total
    )


def weighted_federated_average(
    local_Q_estimates: List[np.ndarray],
    local_Q_counts: List[np.ndarray],
    local_delta_mu_estimates: List[np.ndarray],
    local_d_mu_estimates: List[np.ndarray],
    local_sample_counts: List[int]
) -> AggregationResult:
    """
    Weighted federated averaging based on sample counts.

    This is the preferred method as it accounts for different amounts
    of exploration by different agents.

    Q̂_global(s,a) = Σ_j [n_j(s,a) * Q̂_j(s,a)] / Σ_j n_j(s,a)

    Args:
        local_Q_estimates: List of Q-function estimates from each agent
        local_Q_counts: List of sample counts per (s,a) from each agent
        local_delta_mu_estimates: List of δ_μ estimates from each agent
        local_d_mu_estimates: List of d_μ estimates from each agent
        local_sample_counts: Total samples from each agent

    Returns:
        AggregationResult with global estimates
    """
    N = len(local_Q_estimates)
    if N == 0:
        raise ValueError("No local estimates provided")

    nSA = len(local_Q_estimates[0])

    # Weighted average for Q-function based on per-(s,a) counts
    Q_weighted_sum = np.zeros(nSA)
    Q_count_total = np.zeros(nSA)

    for j in range(N):
        Q_weighted_sum += local_Q_estimates[j] * local_Q_counts[j]
        Q_count_total += local_Q_counts[j]

    Q_global = np.divide(Q_weighted_sum, Q_count_total,
                         out=np.zeros_like(Q_weighted_sum),
                         where=Q_count_total > 0)

    # Weighted average for distributions based on total sample counts
    total_samples = sum(local_sample_counts)
    weights = np.array(local_sample_counts) / (total_samples + 1e-10)

    delta_mu_global = np.zeros_like(local_delta_mu_estimates[0])
    d_mu_global = np.zeros_like(local_d_mu_estimates[0])

    for j in range(N):
        delta_mu_global += weights[j] * local_delta_mu_estimates[j]
        d_mu_global += weights[j] * local_d_mu_estimates[j]

    # Confidence radius based on effective sample size
    confidence_radius = 1.0 / np.sqrt(total_samples + 1)

    return AggregationResult(
        Q_global=Q_global,
        delta_mu_global=delta_mu_global,
        d_mu_global=d_mu_global,
        total_samples=total_samples,
        n_agents=N,
        confidence_radius=confidence_radius,
        Q_counts=Q_count_total
    )


def robust_federated_average(
    local_Q_estimates: List[np.ndarray],
    local_Q_counts: List[np.ndarray],
    local_delta_mu_estimates: List[np.ndarray],
    local_d_mu_estimates: List[np.ndarray],
    local_sample_counts: List[int],
    trim_fraction: float = 0.1
) -> AggregationResult:
    """
    Robust federated averaging with outlier trimming.

    Uses trimmed mean to handle potential Byzantine agents or
    outlier estimates from agents with limited exploration.

    Args:
        local_Q_estimates: List of Q-function estimates from each agent
        local_Q_counts: List of sample counts per (s,a) from each agent
        local_delta_mu_estimates: List of δ_μ estimates from each agent
        local_d_mu_estimates: List of d_μ estimates from each agent
        local_sample_counts: Total samples from each agent
        trim_fraction: Fraction of extreme values to trim (from each tail)

    Returns:
        AggregationResult with global estimates
    """
    N = len(local_Q_estimates)
    if N == 0:
        raise ValueError("No local estimates provided")

    if N < 3:
        # Not enough agents for trimming, fall back to weighted average
        return weighted_federated_average(
            local_Q_estimates, local_Q_counts,
            local_delta_mu_estimates, local_d_mu_estimates,
            local_sample_counts
        )

    nSA = len(local_Q_estimates[0])
    n_trim = max(1, int(N * trim_fraction))

    # Trimmed mean for Q-function (per state-action)
    Q_global = np.zeros(nSA)
    Q_stacked = np.stack(local_Q_estimates, axis=0)  # Shape: (N, nSA)

    for sa in range(nSA):
        values = Q_stacked[:, sa]
        sorted_indices = np.argsort(values)
        trimmed_indices = sorted_indices[n_trim:-n_trim] if n_trim < N // 2 else sorted_indices
        Q_global[sa] = np.mean(values[trimmed_indices])

    # Trimmed mean for distributions
    nS = len(local_d_mu_estimates[0])
    d_mu_global = np.zeros(nS)
    d_mu_stacked = np.stack(local_d_mu_estimates, axis=0)

    for s in range(nS):
        values = d_mu_stacked[:, s]
        sorted_indices = np.argsort(values)
        trimmed_indices = sorted_indices[n_trim:-n_trim] if n_trim < N // 2 else sorted_indices
        d_mu_global[s] = np.mean(values[trimmed_indices])

    delta_mu_global = np.zeros(nSA)
    delta_mu_stacked = np.stack(local_delta_mu_estimates, axis=0)

    for sa in range(nSA):
        values = delta_mu_stacked[:, sa]
        sorted_indices = np.argsort(values)
        trimmed_indices = sorted_indices[n_trim:-n_trim] if n_trim < N // 2 else sorted_indices
        delta_mu_global[sa] = np.mean(values[trimmed_indices])

    # Normalize distributions
    d_mu_global = d_mu_global / (np.sum(d_mu_global) + 1e-10)
    delta_mu_global = delta_mu_global / (np.sum(delta_mu_global) + 1e-10)

    total_samples = sum(local_sample_counts)
    confidence_radius = 1.0 / np.sqrt(total_samples + 1)

    # Sum up Q counts from all agents
    Q_count_total = np.sum(np.stack(local_Q_counts, axis=0), axis=0)

    return AggregationResult(
        Q_global=Q_global,
        delta_mu_global=delta_mu_global,
        d_mu_global=d_mu_global,
        total_samples=total_samples,
        n_agents=N,
        confidence_radius=confidence_radius,
        Q_counts=Q_count_total
    )


def aggregate_advantages(
    local_policy_advantages: List[float],
    local_model_advantages: List[float],
    local_sample_counts: List[int],
    method: str = 'weighted'
) -> Tuple[float, float]:
    """
    Aggregate expected relative advantages from multiple agents.

    Args:
        local_policy_advantages: A^{P,π'}_{P,π,μ} estimates from each agent
        local_model_advantages: A^{P',π}_{P,π,μ} estimates from each agent
        local_sample_counts: Sample counts from each agent
        method: 'uniform', 'weighted', or 'median'

    Returns:
        (global_policy_advantage, global_model_advantage)
    """
    N = len(local_policy_advantages)
    if N == 0:
        return 0.0, 0.0

    if method == 'uniform':
        return np.mean(local_policy_advantages), np.mean(local_model_advantages)

    elif method == 'weighted':
        total = sum(local_sample_counts)
        weights = np.array(local_sample_counts) / (total + 1e-10)
        p_adv = np.sum(weights * np.array(local_policy_advantages))
        m_adv = np.sum(weights * np.array(local_model_advantages))
        return p_adv, m_adv

    elif method == 'median':
        return np.median(local_policy_advantages), np.median(local_model_advantages)

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def aggregate_distances(
    local_sup_distances: List[float],
    local_mean_distances: List[float],
    local_sample_counts: List[int]
) -> Tuple[float, float]:
    """
    Aggregate distance metrics from multiple agents.

    For sup-norm distances, we take the maximum (conservative bound).
    For mean distances, we take weighted average.

    Args:
        local_sup_distances: D_∞ estimates from each agent
        local_mean_distances: D_E estimates from each agent
        local_sample_counts: Sample counts from each agent

    Returns:
        (global_sup_distance, global_mean_distance)
    """
    if not local_sup_distances:
        return 0.0, 0.0

    # Conservative: take max of sup-norm distances
    global_sup = np.max(local_sup_distances)

    # Weighted average for mean distances
    total = sum(local_sample_counts)
    weights = np.array(local_sample_counts) / (total + 1e-10)
    global_mean = np.sum(weights * np.array(local_mean_distances))

    return global_sup, global_mean
