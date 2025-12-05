"""
Sample-based estimators for Q-function, V-function, U-function,
and discounted state/state-action distributions.

These estimators enable the sample-based version of SPMI as mentioned
in Metelli et al. (2018) Section 2: "when the model space is unknown,
we could resort to a sample-based version of SPMI"
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Trajectory:
    """A trajectory is a sequence of (state, action, reward, next_state) tuples"""
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        for i in range(len(self)):
            yield (self.states[i], self.actions[i],
                   self.rewards[i], self.next_states[i])


def estimate_q_function_mc(
    trajectories: List[Trajectory],
    gamma: float,
    nS: int,
    nA: int,
    method: str = 'first_visit'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo estimation of Q-function from trajectories.

    Args:
        trajectories: List of trajectory objects
        gamma: Discount factor
        nS: Number of states
        nA: Number of actions
        method: 'first_visit' or 'every_visit' MC

    Returns:
        Q_hat: Estimated Q-function of shape (nS * nA,)
        Q_counts: Number of samples for each (s,a) pair
    """
    nSA = nS * nA
    Q_sum = np.zeros(nSA)
    Q_count = np.zeros(nSA)

    for trajectory in trajectories:
        T = len(trajectory)
        if T == 0:
            continue

        # Compute returns backwards
        G = 0.0
        returns_from_t = np.zeros(T)
        for t in reversed(range(T)):
            r = trajectory.rewards[t]
            G = r + gamma * G
            returns_from_t[t] = G

        if method == 'first_visit':
            visited = set()
            for t in range(T):
                s = int(trajectory.states[t])
                a = int(trajectory.actions[t])
                sa = s * nA + a
                if sa not in visited:
                    visited.add(sa)
                    Q_sum[sa] += returns_from_t[t]
                    Q_count[sa] += 1
        else:  # every_visit
            for t in range(T):
                s = int(trajectory.states[t])
                a = int(trajectory.actions[t])
                sa = s * nA + a
                Q_sum[sa] += returns_from_t[t]
                Q_count[sa] += 1

    # Avoid division by zero
    Q_hat = np.divide(Q_sum, Q_count, out=np.zeros_like(Q_sum), where=Q_count > 0)

    return Q_hat, Q_count


def estimate_v_function_mc(
    trajectories: List[Trajectory],
    gamma: float,
    nS: int,
    method: str = 'first_visit'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo estimation of V-function from trajectories.

    Args:
        trajectories: List of trajectory objects
        gamma: Discount factor
        nS: Number of states
        method: 'first_visit' or 'every_visit' MC

    Returns:
        V_hat: Estimated V-function of shape (nS,)
        V_counts: Number of samples for each state
    """
    V_sum = np.zeros(nS)
    V_count = np.zeros(nS)

    for trajectory in trajectories:
        T = len(trajectory)
        if T == 0:
            continue

        # Compute returns backwards
        G = 0.0
        returns_from_t = np.zeros(T)
        for t in reversed(range(T)):
            r = trajectory.rewards[t]
            G = r + gamma * G
            returns_from_t[t] = G

        if method == 'first_visit':
            visited = set()
            for t in range(T):
                s = int(trajectory.states[t])
                if s not in visited:
                    visited.add(s)
                    V_sum[s] += returns_from_t[t]
                    V_count[s] += 1
        else:  # every_visit
            for t in range(T):
                s = int(trajectory.states[t])
                V_sum[s] += returns_from_t[t]
                V_count[s] += 1

    V_hat = np.divide(V_sum, V_count, out=np.zeros_like(V_sum), where=V_count > 0)

    return V_hat, V_count


def estimate_u_function_mc(
    trajectories: List[Trajectory],
    gamma: float,
    nS: int,
    nA: int,
    V_hat: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Estimate U-function (state-action-next-state value function).

    U^{P,π}(s, a, s') = R(s, a, s') + γV^{P,π}(s')

    Since we observe (s, a, r, s') tuples, we can estimate R(s,a,s') directly
    from transitions. This correctly handles transition-dependent rewards.

    Args:
        trajectories: List of trajectory objects
        gamma: Discount factor
        nS: Number of states
        nA: Number of actions
        V_hat: Pre-computed V-function estimate (if None, will be estimated)

    Returns:
        U_hat: Estimated U-function of shape (nS * nA, nS)
    """
    nSA = nS * nA

    # Estimate V if not provided
    if V_hat is None:
        V_hat, _ = estimate_v_function_mc(trajectories, gamma, nS)

    # Estimate R(s,a,s') directly from observed transitions
    # This handles transition-dependent rewards correctly
    R_sas_sum = np.zeros((nSA, nS))
    R_sas_count = np.zeros((nSA, nS))

    for trajectory in trajectories:
        for t in range(len(trajectory)):
            s = int(trajectory.states[t])
            a = int(trajectory.actions[t])
            r = trajectory.rewards[t]
            s_next = int(trajectory.next_states[t])
            sa = s * nA + a
            R_sas_sum[sa, s_next] += r
            R_sas_count[sa, s_next] += 1

    R_sas_hat = np.divide(R_sas_sum, R_sas_count,
                          out=np.zeros_like(R_sas_sum),
                          where=R_sas_count > 0)

    # Build U matrix: U[sa, s'] = R(s,a,s') + gamma * V(s')
    U_hat = R_sas_hat + gamma * V_hat[np.newaxis, :]

    return U_hat


def estimate_discounted_s_distribution(
    trajectories: List[Trajectory],
    gamma: float,
    nS: int,
    horizon: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Estimate the γ-discounted state distribution d_μ^{P,π}.

    d_μ^{P,π}(s) = (1-γ) Σ_{t=0}^{H-1} γ^t P(s_t = s | π, P, μ)

    Args:
        trajectories: List of trajectory objects
        gamma: Discount factor
        nS: Number of states
        horizon: Episode horizon (if None, uses trajectory lengths)

    Returns:
        d_mu_hat: Estimated discounted state distribution
        total_samples: Total number of weighted samples
    """
    d_mu = np.zeros(nS)
    total_weight = 0.0

    for trajectory in trajectories:
        T = len(trajectory)
        H = horizon if horizon is not None else T

        for t in range(min(T, H)):
            s = int(trajectory.states[t])
            weight = gamma ** t
            d_mu[s] += weight
            total_weight += weight

    if total_weight > 0:
        d_mu /= total_weight
        # Normalization factor: (1-γ)/(1-γ^H)
        if horizon is not None:
            d_mu *= (1 - gamma) / (1 - gamma ** horizon + 1e-10)
        else:
            d_mu *= (1 - gamma)

    return d_mu, int(total_weight)


def estimate_discounted_sa_distribution(
    trajectories: List[Trajectory],
    gamma: float,
    nS: int,
    nA: int,
    horizon: Optional[int] = None,
    d_mu: Optional[np.ndarray] = None,
    policy_matrix: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, int]:
    """
    Estimate the γ-discounted state-action distribution δ_μ^{P,π}.

    δ_μ^{P,π}(s,a) = π(a|s) * d_μ^{P,π}(s)

    Can be estimated directly from trajectories or computed from d_μ and π.

    Args:
        trajectories: List of trajectory objects
        gamma: Discount factor
        nS: Number of states
        nA: Number of actions
        horizon: Episode horizon
        d_mu: Pre-computed d_μ (if available)
        policy_matrix: Policy matrix π (if available, for indirect computation)

    Returns:
        delta_mu_hat: Estimated discounted state-action distribution
        total_samples: Total number of weighted samples
    """
    nSA = nS * nA

    # Direct estimation from trajectories
    delta_mu = np.zeros(nSA)
    total_weight = 0.0

    for trajectory in trajectories:
        T = len(trajectory)
        H = horizon if horizon is not None else T

        for t in range(min(T, H)):
            s = int(trajectory.states[t])
            a = int(trajectory.actions[t])
            sa = s * nA + a
            weight = gamma ** t
            delta_mu[sa] += weight
            total_weight += weight

    if total_weight > 0:
        delta_mu /= total_weight
        if horizon is not None:
            delta_mu *= (1 - gamma) / (1 - gamma ** horizon + 1e-10)
        else:
            delta_mu *= (1 - gamma)

    return delta_mu, int(total_weight)


def estimate_transition_probabilities(
    trajectories: List[Trajectory],
    nS: int,
    nA: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate transition probabilities P(s'|s,a) from trajectories.

    Args:
        trajectories: List of trajectory objects
        nS: Number of states
        nA: Number of actions

    Returns:
        P_hat: Estimated transition matrix of shape (nS * nA, nS)
        P_counts: Count of (s,a,s') observations
    """
    nSA = nS * nA
    transition_counts = np.zeros((nSA, nS))
    sa_counts = np.zeros(nSA)

    for trajectory in trajectories:
        for t in range(len(trajectory)):
            s = int(trajectory.states[t])
            a = int(trajectory.actions[t])
            s_next = int(trajectory.next_states[t])
            sa = s * nA + a
            transition_counts[sa, s_next] += 1
            sa_counts[sa] += 1

    # Normalize to get probabilities
    P_hat = np.zeros((nSA, nS))
    for sa in range(nSA):
        if sa_counts[sa] > 0:
            P_hat[sa, :] = transition_counts[sa, :] / sa_counts[sa]
        else:
            # Uniform distribution for unvisited state-actions
            P_hat[sa, :] = 1.0 / nS

    return P_hat, transition_counts


def compute_confidence_bound(
    n_samples: int,
    delta: float = 0.05,
    value_range: float = 1.0
) -> float:
    """
    Compute confidence bound for sample mean estimates using Hoeffding's inequality.

    P(|X̄ - E[X]| > ε) ≤ 2exp(-2nε²/R²)

    Solving for ε: ε = R * sqrt(ln(2/δ) / (2n))

    Args:
        n_samples: Number of samples
        delta: Confidence level (e.g., 0.05 for 95% confidence)
        value_range: Range of the random variable

    Returns:
        epsilon: Confidence bound
    """
    if n_samples == 0:
        return float('inf')
    return value_range * np.sqrt(np.log(2.0 / delta) / (2 * n_samples))


def merge_q_estimates(
    Q_estimates: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge multiple Q-function estimates (from different agents).

    Uses weighted average based on sample counts.

    Args:
        Q_estimates: List of (Q_hat, Q_count) tuples

    Returns:
        Q_merged: Merged Q-function estimate
        Q_count_merged: Merged sample counts
    """
    if not Q_estimates:
        raise ValueError("No estimates to merge")

    nSA = len(Q_estimates[0][0])
    Q_sum = np.zeros(nSA)
    Q_count_total = np.zeros(nSA)

    for Q_hat, Q_count in Q_estimates:
        Q_sum += Q_hat * Q_count
        Q_count_total += Q_count

    Q_merged = np.divide(Q_sum, Q_count_total,
                         out=np.zeros_like(Q_sum),
                         where=Q_count_total > 0)

    return Q_merged, Q_count_total
