"""
Numba-optimized functions for policy evaluation and distribution computations.
These functions provide significant speedup for core RL computations.
"""

import numpy as np
from numba import njit, prange
import numpy.linalg as la


@njit(cache=True, fastmath=True)
def compute_q_function_numba(pi, P, R_sas, gamma, nS, nA, horizon=None):
    """
    Numba-optimized Q-function computation.

    Args:
        pi: Policy matrix (nS, nSA)
        P: Model matrix (nSA, nS)
        R_sas: Reward matrix (nSA, nS)
        gamma: Discount factor
        nS: Number of states
        nA: Number of actions
        horizon: Horizon for iterative computation (None for exact)

    Returns:
        Q: Q-function values (nSA,)
    """
    nSA = nS * nA

    # Compute expected immediate rewards
    R_sa = np.zeros(nSA, dtype=np.float64)
    for sa in range(nSA):
        for s in range(nS):
            R_sa[sa] += R_sas[sa, s] * P[sa, s]

    if horizon is None:
        # Exact computation using linear system solver
        # Q = (I - gamma * P * pi)^{-1} * R_sa
        I = np.eye(nSA, dtype=np.float64)
        P_pi = np.dot(P, pi)
        A = I - gamma * P_pi

        # Use numpy's solve (will be compiled by Numba)
        Q = np.linalg.solve(A, R_sa)
    else:
        # Iterative computation
        Q = np.zeros(nSA, dtype=np.float64)
        P_pi = np.dot(P, pi)

        for h in range(horizon):
            Q = R_sa + gamma * np.dot(P_pi, Q)

    return Q


@njit(cache=True, fastmath=True)
def compute_v_function_numba(pi, P, R_sas, gamma, nS, nA, horizon=None):
    """
    Numba-optimized V-function computation.

    Args:
        pi: Policy matrix (nS, nSA)
        P: Model matrix (nSA, nS)
        R_sas: Reward matrix (nSA, nS)
        gamma: Discount factor
        nS: Number of states
        nA: Number of actions
        horizon: Horizon for iterative computation (None for exact)

    Returns:
        V: V-function values (nS,)
    """
    nSA = nS * nA

    # Compute expected immediate rewards
    R_sa = np.zeros(nSA, dtype=np.float64)
    for sa in range(nSA):
        for s in range(nS):
            R_sa[sa] += R_sas[sa, s] * P[sa, s]

    R_s = np.dot(pi, R_sa)

    if horizon is None:
        # Exact computation using linear system solver
        # V = (I - gamma * pi * P)^{-1} * R_s
        I = np.eye(nS, dtype=np.float64)
        pi_P = np.dot(pi, P)
        A = I - gamma * pi_P

        V = np.linalg.solve(A, R_s)
    else:
        # Iterative computation
        V = np.zeros(nS, dtype=np.float64)
        pi_P = np.dot(pi, P)

        for h in range(horizon):
            V = R_s + gamma * np.dot(pi_P, V)

    return V


@njit(cache=True, fastmath=True)
def compute_u_function_numba(pi, P, R_sas, gamma, nS, nA, horizon=None):
    """
    Numba-optimized U-function computation.

    Args:
        pi: Policy matrix (nS, nSA)
        P: Model matrix (nSA, nS)
        R_sas: Reward matrix (nSA, nS)
        gamma: Discount factor
        nS: Number of states
        nA: Number of actions
        horizon: Horizon for U-function computation

    Returns:
        U: U-function values (nSA, nS)
    """
    # Compute V-function first
    if horizon is not None and horizon > 0:
        V = compute_v_function_numba(pi, P, R_sas, gamma, nS, nA, horizon - 1)
    else:
        V = compute_v_function_numba(pi, P, R_sas, gamma, nS, nA, horizon)

    # U = R_sas + gamma * V (broadcasted)
    nSA = nS * nA
    U = np.zeros((nSA, nS), dtype=np.float64)

    for sa in range(nSA):
        for s in range(nS):
            U[sa, s] = R_sas[sa, s] + gamma * V[s]

    return U


@njit(cache=True, fastmath=True)
def compute_performance_numba(mu, pi, P, R_sas, gamma, nS, nA, horizon=None):
    """
    Numba-optimized performance computation.

    Args:
        mu: Initial state distribution (nS,)
        pi: Policy matrix (nS, nSA)
        P: Model matrix (nSA, nS)
        R_sas: Reward matrix (nSA, nS)
        gamma: Discount factor
        nS: Number of states
        nA: Number of actions
        horizon: Horizon for V-function computation

    Returns:
        J: Expected discounted return
    """
    V = compute_v_function_numba(pi, P, R_sas, gamma, nS, nA, horizon)
    return np.dot(mu, V)


@njit(cache=True, fastmath=True)
def compute_discounted_s_distribution_numba(mu, pi, P, gamma, nS, nA, horizon=None):
    """
    Numba-optimized computation of discounted state distribution.

    Args:
        mu: Initial state distribution (nS,)
        pi: Policy matrix (nS, nSA)
        P: Model matrix (nSA, nS)
        gamma: Discount factor
        nS: Number of states
        nA: Number of actions
        horizon: Horizon for iterative computation (None for exact)

    Returns:
        d_mu: Discounted state distribution (nS,)
    """
    if horizon is None:
        # Exact computation using linear system solver
        # d_mu = (I - gamma * pi * P)^{-T} * mu
        I = np.eye(nS, dtype=np.float64)
        pi_P = np.dot(pi, P)
        A = I - gamma * pi_P

        d_mu = np.linalg.solve(A.T, mu)
        d_mu = d_mu * (1 - gamma)
    else:
        # Iterative computation
        d_mu = np.zeros(nS, dtype=np.float64)
        pi_P = np.dot(pi, P)

        for h in range(horizon):
            d_mu = mu + gamma * np.dot(d_mu, pi_P)

        if horizon > 0:
            d_mu = d_mu * (1 - gamma) / (1 - gamma ** horizon)

    return d_mu


@njit(cache=True, fastmath=True)
def compute_discounted_sa_distribution_numba(mu, pi, P, gamma, nS, nA, horizon=None, d_mu=None):
    """
    Numba-optimized computation of discounted state-action distribution.

    Args:
        mu: Initial state distribution (nS,)
        pi: Policy matrix (nS, nSA)
        P: Model matrix (nSA, nS)
        gamma: Discount factor
        nS: Number of states
        nA: Number of actions
        horizon: Horizon for computation
        d_mu: Pre-computed state distribution (optional)

    Returns:
        delta_mu: Discounted state-action distribution (nSA,)
    """
    # Compute d_mu if not provided
    if d_mu is None:
        d_mu = compute_discounted_s_distribution_numba(mu, pi, P, gamma, nS, nA, horizon)

    # delta_mu = d_mu * pi
    delta_mu = np.dot(d_mu, pi)
    return delta_mu


@njit(cache=True, fastmath=True)
def compute_policy_er_advantage_numba(target_matrix, policy_matrix, Q, d_mu_pi):
    """
    Numba-optimized computation of expected relative advantage for policies.

    Args:
        target_matrix: Target policy matrix (nS, nSA)
        policy_matrix: Current policy matrix (nS, nSA)
        Q: Q-function values (nSA,)
        d_mu_pi: State distribution under current policy (nS,)

    Returns:
        er_advantage: Expected relative advantage
    """
    # A = (target - policy) * Q
    diff = target_matrix - policy_matrix
    A = np.dot(diff, Q)

    # Expected relative advantage
    er_advantage = np.dot(d_mu_pi, A)
    return er_advantage


@njit(cache=True, fastmath=True)
def compute_model_er_advantage_numba(target_matrix, model_matrix, U, delta_mu_pi):
    """
    Numba-optimized computation of expected relative advantage for models.

    Args:
        target_matrix: Target model matrix (nSA, nS)
        model_matrix: Current model matrix (nSA, nS)
        U: U-function values (nSA, nS)
        delta_mu_pi: State-action distribution (nSA,)

    Returns:
        er_advantage: Expected relative advantage
    """
    # Relative advantage as array of state-actions
    diff = target_matrix - model_matrix
    nSA = diff.shape[0]
    nS = diff.shape[1]

    A = np.zeros(nSA, dtype=np.float64)
    for sa in range(nSA):
        for s in range(nS):
            A[sa] += diff[sa, s] * U[sa, s]

    # Expected relative advantage
    er_advantage = np.dot(delta_mu_pi, A)
    return er_advantage