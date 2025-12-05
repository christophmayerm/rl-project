"""
Federated Server for Safe Policy-Model Iteration.

The central server:
1. Aggregates local statistics from all agents
2. Computes global estimates (Q̂, δ̂_μ)
3. Computes the Decoupled Bound from Theorem 3.3
4. Performs the safe update step to find optimal α*, β*
5. Updates the global policy and model
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .agent import LocalStatistics
from .aggregators import (
    weighted_federated_average,
    federated_average,
    robust_federated_average,
    AggregationResult
)


@dataclass
class GlobalStatistics:
    """
    Global statistics computed by the server after aggregation.
    """
    Q_global: np.ndarray                 # Aggregated Q-function
    U_global: np.ndarray                 # Aggregated U-function
    d_mu_global: np.ndarray              # Aggregated γ-discounted state distribution
    delta_mu_global: np.ndarray          # Aggregated γ-discounted state-action distribution
    total_samples: int                   # Total samples across all agents
    n_agents: int                        # Number of contributing agents
    confidence_radius: float             # Confidence bound on estimates
    Q_counts: np.ndarray = None          # Per-(s,a) sample counts for confidence scaling


@dataclass
class UpdateResult:
    """
    Result of the safe update computation.
    """
    alpha_star: float                    # Optimal policy step size
    beta_star: float                     # Optimal model step size
    bound_value: float                   # Value of the decoupled bound
    policy_advantage: float              # Expected policy advantage
    model_advantage: float               # Expected model advantage
    policy_dist_sup: float               # D_∞^{π',π}
    policy_dist_mean: float              # D_E^{π',π}
    model_dist_sup: float                # D_∞^{P',P}
    model_dist_mean: float               # D_E^{P',P}


class FederatedServer:
    """
    Central server for Federated Safe Policy-Model Iteration.

    Implements the server-side operations of F-SPMI:
    1. Aggregation: Q̂ = (1/N) Σ_j Q̂_j, δ̂_μ = (1/N) Σ_j δ̂_μ,j
    2. Bound computation: B̂(P', π') from Theorem 3.3
    3. Safe update: Solve for optimal α*, β*
    """

    def __init__(
        self,
        nS: int,
        nA: int,
        gamma: float,
        horizon: int,
        delta_q: Optional[float] = None,
        aggregation_method: str = 'weighted'
    ):
        """
        Initialize the federated server.

        Args:
            nS: Number of states
            nA: Number of actions
            gamma: Discount factor
            horizon: Episode horizon
            delta_q: ΔQ value for the bound (if None, uses 1/(1-γ))
            aggregation_method: 'uniform', 'weighted', or 'robust'
        """
        self.nS = nS
        self.nA = nA
        self.nSA = nS * nA
        self.gamma = gamma
        self.horizon = horizon
        self.aggregation_method = aggregation_method

        # Default delta_q = (1-gamma^H)/(1-gamma)
        if delta_q is None:
            self.delta_q = (1.0 - gamma ** horizon) / (1 - gamma)
        else:
            self.delta_q = delta_q

        # Storage for latest aggregated statistics
        self.global_stats: Optional[GlobalStatistics] = None

    def aggregate_statistics(
        self,
        local_stats: List[LocalStatistics]
    ) -> GlobalStatistics:
        """
        Aggregate local statistics from all agents using federated averaging.

        Implements the aggregation step of F-SPMI:
        Q̂^{P_k,π_k} = (1/N) Σ_{j=1}^N Q̂_j^{P_k,π_k}
        δ̂_μ^{P_k,π_k} = (1/N) Σ_{j=1}^N δ̂_{μ,j}^{P_k,π_k}

        Args:
            local_stats: List of LocalStatistics from each agent

        Returns:
            GlobalStatistics containing aggregated estimates
        """
        if not local_stats:
            raise ValueError("No local statistics provided")

        N = len(local_stats)

        # Extract arrays for aggregation
        local_Q_estimates = [s.Q_hat for s in local_stats]
        local_Q_counts = [s.Q_counts for s in local_stats]
        local_delta_mu_estimates = [s.delta_mu_hat for s in local_stats]
        local_d_mu_estimates = [s.d_mu_hat for s in local_stats]
        local_sample_counts = [s.n_samples for s in local_stats]

        # Perform aggregation based on method
        if self.aggregation_method == 'weighted':
            agg_result = weighted_federated_average(
                local_Q_estimates, local_Q_counts,
                local_delta_mu_estimates, local_d_mu_estimates,
                local_sample_counts
            )
        elif self.aggregation_method == 'robust':
            agg_result = robust_federated_average(
                local_Q_estimates, local_Q_counts,
                local_delta_mu_estimates, local_d_mu_estimates,
                local_sample_counts
            )
        else:  # uniform
            agg_result = federated_average(
                local_Q_estimates, local_Q_counts,
                local_delta_mu_estimates, local_d_mu_estimates,
                local_sample_counts
            )

        # Aggregate U-function (weighted average)
        total_samples = sum(local_sample_counts)
        weights = np.array(local_sample_counts) / (total_samples + 1e-10)
        U_global = np.zeros_like(local_stats[0].U_hat)
        for j, stats in enumerate(local_stats):
            U_global += weights[j] * stats.U_hat

        self.global_stats = GlobalStatistics(
            Q_global=agg_result.Q_global,
            U_global=U_global,
            d_mu_global=agg_result.d_mu_global,
            delta_mu_global=agg_result.delta_mu_global,
            total_samples=agg_result.total_samples,
            n_agents=N,
            confidence_radius=agg_result.confidence_radius,
            Q_counts=agg_result.Q_counts
        )

        return self.global_stats

    def compute_policy_advantage(
        self,
        target_policy_matrix: np.ndarray,
        current_policy_matrix: np.ndarray,
        Q: np.ndarray,
        d_mu: np.ndarray
    ) -> float:
        """
        Compute expected relative advantage for policy.

        A^{P,π'}_{P,π,μ} = E_{s~d_μ}[Σ_a (π'(a|s) - π(a|s)) Q(s,a)]

        Args:
            target_policy_matrix: Target policy π' as matrix
            current_policy_matrix: Current policy π as matrix
            Q: Q-function
            d_mu: γ-discounted state distribution

        Returns:
            Expected relative advantage
        """
        A = np.dot(target_policy_matrix - current_policy_matrix, Q)
        return np.dot(d_mu, A)

    def compute_model_advantage(
        self,
        target_model_matrix: np.ndarray,
        current_model_matrix: np.ndarray,
        U: np.ndarray,
        delta_mu: np.ndarray
    ) -> float:
        """
        Compute expected relative advantage for model.

        A^{P',π}_{P,π,μ} = E_{(s,a)~δ_μ}[Σ_s' (P'(s'|s,a) - P(s'|s,a)) U(s,a,s')]

        Args:
            target_model_matrix: Target model P' as matrix
            current_model_matrix: Current model P as matrix
            U: U-function
            delta_mu: γ-discounted state-action distribution

        Returns:
            Expected relative advantage
        """
        A = np.sum((target_model_matrix - current_model_matrix) * U, axis=1)
        return np.dot(delta_mu, A)

    def compute_policy_distances(
        self,
        target_policy_matrix: np.ndarray,
        current_policy_matrix: np.ndarray,
        d_mu: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute policy distance metrics.

        D_∞^{π',π} = sup_s ||π'(·|s) - π(·|s)||_1
        D_E^{π',π} = E_{s~d_μ}[||π'(·|s) - π(·|s)||_1]

        Args:
            target_policy_matrix: Target policy matrix
            current_policy_matrix: Current policy matrix
            d_mu: γ-discounted state distribution

        Returns:
            (D_∞, D_E) tuple
        """
        diff = np.abs(target_policy_matrix - current_policy_matrix)
        per_state_dist = np.sum(diff, axis=1)  # ||π'(·|s) - π(·|s)||_1 for each s

        dist_sup = np.max(per_state_dist)
        dist_mean = np.dot(d_mu, per_state_dist)

        return dist_sup, dist_mean

    def compute_model_distances(
        self,
        target_model_matrix: np.ndarray,
        current_model_matrix: np.ndarray,
        delta_mu: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute model distance metrics.

        D_∞^{P',P} = sup_{s,a} ||P'(·|s,a) - P(·|s,a)||_1
        D_E^{P',P} = E_{(s,a)~δ_μ}[||P'(·|s,a) - P(·|s,a)||_1]

        Args:
            target_model_matrix: Target model matrix
            current_model_matrix: Current model matrix
            delta_mu: γ-discounted state-action distribution

        Returns:
            (D_∞, D_E) tuple
        """
        diff = np.abs(target_model_matrix - current_model_matrix)
        per_sa_dist = np.sum(diff, axis=1)  # ||P'(·|s,a) - P(·|s,a)||_1 for each (s,a)

        dist_sup = np.max(per_sa_dist)
        dist_mean = np.dot(delta_mu, per_sa_dist)

        return dist_sup, dist_mean

    def compute_decoupled_bound(
        self,
        p_er_adv: float,
        m_er_adv: float,
        p_dist_sup: float,
        p_dist_mean: float,
        m_dist_sup: float,
        m_dist_mean: float,
        alpha: float,
        beta: float
    ) -> float:
        """
        Compute the Decoupled Bound from Theorem 3.3.

        B(P', π') = [A^{P',π}_{P,π,μ} + A^{P,π'}_{P,π,μ}]/(1-γ)
                    - γΔQ·D / [2(1-γ)²]

        where D is the dissimilarity term.

        Args:
            p_er_adv: Policy expected relative advantage
            m_er_adv: Model expected relative advantage
            p_dist_sup: D_∞^{π',π}
            p_dist_mean: D_E^{π',π}
            m_dist_sup: D_∞^{P',P}
            m_dist_mean: D_E^{P',P}
            alpha: Policy step size
            beta: Model step size

        Returns:
            Value of the decoupled bound
        """
        gamma = self.gamma

        # Advantage term
        advantage = alpha * p_er_adv + beta * m_er_adv

        # Dissimilarity penalization term
        # D = D_E^{π',π}(D_∞^{π',π} + D_∞^{P',P}) + D_E^{P',P}(D_∞^{π',π} + γD_∞^{P',P})
        # For the bound, we use the scaled distances: α·dist for policy, β·dist for model
        penalization = (gamma / (1 - gamma) * self.delta_q / 2) * (
            (alpha ** 2) * p_dist_sup * p_dist_mean +
            gamma * (beta ** 2) * m_dist_sup * m_dist_mean +
            alpha * beta * p_dist_sup * m_dist_mean +
            alpha * beta * p_dist_mean * m_dist_sup
        )

        return advantage - penalization

    def compute_safe_update(
        self,
        current_policy,
        current_model,
        target_policy,
        target_model,
        global_stats: Optional[GlobalStatistics] = None
    ) -> UpdateResult:
        """
        Compute the safe update step by solving for optimal α*, β*.

        This implements Algorithm 1 from the paper, finding the (α,β) pair
        that maximizes the decoupled bound.

        Args:
            current_policy: Current policy (TabularPolicy)
            current_model: Current model (TabularModel)
            target_policy: Target policy (TabularPolicy)
            target_model: Target model (TabularModel)
            global_stats: Aggregated global statistics (uses stored if None)

        Returns:
            UpdateResult containing optimal coefficients and metrics
        """
        if global_stats is None:
            global_stats = self.global_stats
        if global_stats is None:
            raise ValueError("No global statistics available")

        gamma = self.gamma
        Q = global_stats.Q_global
        U = global_stats.U_global
        d_mu = global_stats.d_mu_global
        delta_mu = global_stats.delta_mu_global

        # Get matrices
        current_policy_matrix = current_policy.get_matrix()
        target_policy_matrix = target_policy.get_matrix()
        current_model_matrix = current_model.get_matrix()
        target_model_matrix = target_model.get_matrix()

        # Compute advantages
        p_er_adv = self.compute_policy_advantage(
            target_policy_matrix, current_policy_matrix, Q, d_mu
        )
        m_er_adv = self.compute_model_advantage(
            target_model_matrix, current_model_matrix, U, delta_mu
        )

        # CONFIDENCE-AWARE SCALING: Scale advantages based on sample counts
        # This prevents overconfident updates when estimates have high variance
        # Use mean of VISITED (s,a) pairs, not min (which would be 0 for unvisited pairs)
        if global_stats.Q_counts is not None:
            visited_counts = global_stats.Q_counts[global_stats.Q_counts > 0]
            if len(visited_counts) > 0:
                mean_visits = np.mean(visited_counts)
                # Need at least 20 visits on average for full confidence
                confidence = min(1.0, mean_visits / 20.0)
            else:
                confidence = 0.0
            p_er_adv *= confidence
            m_er_adv *= confidence

        # Compute distances
        p_dist_sup, p_dist_mean = self.compute_policy_distances(
            target_policy_matrix, current_policy_matrix, d_mu
        )
        m_dist_sup, m_dist_mean = self.compute_model_distances(
            target_model_matrix, current_model_matrix, delta_mu
        )

        # Add small epsilon to avoid division by zero
        eps = 1e-24

        # Compute candidate optimal coefficients (from Table 1 in paper)
        alpha0 = ((1 - gamma) * p_er_adv) / (
            self.delta_q * gamma * p_dist_sup * p_dist_mean + eps
        )
        alpha1 = alpha0 - 0.5 * (
            m_dist_mean / (p_dist_mean + eps) + m_dist_sup / (p_dist_sup + eps)
        )
        beta0 = ((1 - gamma) * m_er_adv) / (
            self.delta_q * (gamma ** 2) * m_dist_sup * m_dist_mean + eps
        )
        beta1 = beta0 - 0.5 / gamma * (
            p_dist_mean / (m_dist_mean + eps) + p_dist_sup / (m_dist_sup + eps)
        )

        # Clip to [0, 1]
        alpha0 = np.clip(alpha0, 0.0, 1.0)
        alpha1 = np.clip(alpha1, 0.0, 1.0)
        beta0 = np.clip(beta0, 0.0, 1.0)
        beta1 = np.clip(beta1, 0.0, 1.0)

        # Find the (α, β) pair maximizing the bound
        candidates = [
            (alpha0, 0.0),
            (0.0, beta0),
            (alpha1, 1.0),
            (1.0, beta1)
        ]

        best_bound = float('-inf')
        alpha_star, beta_star = 0.0, 0.0

        for alpha, beta in candidates:
            bound = self.compute_decoupled_bound(
                p_er_adv, m_er_adv,
                p_dist_sup, p_dist_mean,
                m_dist_sup, m_dist_mean,
                alpha, beta
            )
            if bound > best_bound:
                best_bound = bound
                alpha_star = alpha
                beta_star = beta

        # STEP SIZE FLOOR: Prevent collapse to zero when advantages are positive
        # This ensures learning continues even with noisy estimates
        min_step = 1e-4
        if p_er_adv > 0 and alpha_star < min_step:
            alpha_star = min_step
        if m_er_adv > 0 and beta_star < min_step:
            beta_star = min_step

        return UpdateResult(
            alpha_star=alpha_star,
            beta_star=beta_star,
            bound_value=best_bound,
            policy_advantage=p_er_adv,
            model_advantage=m_er_adv,
            policy_dist_sup=p_dist_sup,
            policy_dist_mean=p_dist_mean,
            model_dist_sup=m_dist_sup,
            model_dist_mean=m_dist_mean
        )

    def compute_safe_update_with_multiple_targets(
        self,
        current_policy,
        current_model,
        target_policies: List,
        target_models: List,
        global_stats: Optional[GlobalStatistics] = None
    ) -> Tuple[UpdateResult, Any, Any]:
        """
        Compute safe update considering multiple target policies and models.

        This extends the basic safe update to handle persistent target choice,
        where we compare greedy target with previous target.

        Args:
            current_policy: Current policy
            current_model: Current model
            target_policies: List of candidate target policies
            target_models: List of candidate target models
            global_stats: Aggregated global statistics

        Returns:
            (UpdateResult, best_target_policy, best_target_model)
        """
        best_result = None
        best_target_policy = None
        best_target_model = None

        for target_policy in target_policies:
            for target_model in target_models:
                result = self.compute_safe_update(
                    current_policy, current_model,
                    target_policy, target_model,
                    global_stats
                )
                if best_result is None or result.bound_value > best_result.bound_value:
                    best_result = result
                    best_target_policy = target_policy
                    best_target_model = target_model

        return best_result, best_target_policy, best_target_model
