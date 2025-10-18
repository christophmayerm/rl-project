"""
Numba-optimized distance computation and utility functions.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, parallel=True)
def policy_sup_tv_distance_numba(policy1_matrix, policy2_matrix):
    """
    Numba-optimized computation of supremum total variation distance between policies.

    Args:
        policy1_matrix: First policy matrix (nS, nSA)
        policy2_matrix: Second policy matrix (nS, nSA)

    Returns:
        distance: Supremum TV distance
    """
    nS = policy1_matrix.shape[0]
    max_dist = 0.0

    for s in prange(nS):
        row_sum = 0.0
        for sa in range(policy1_matrix.shape[1]):
            row_sum += abs(policy1_matrix[s, sa] - policy2_matrix[s, sa])
        if row_sum > max_dist:
            max_dist = row_sum

    return max_dist


@njit(cache=True, fastmath=True)
def policy_mean_tv_distance_numba(policy1_matrix, policy2_matrix, d_mu_pi):
    """
    Numba-optimized computation of mean total variation distance between policies.

    Args:
        policy1_matrix: First policy matrix (nS, nSA)
        policy2_matrix: Second policy matrix (nS, nSA)
        d_mu_pi: State distribution (nS,)

    Returns:
        distance: Mean TV distance
    """
    nS = policy1_matrix.shape[0]
    nSA = policy1_matrix.shape[1]

    # Compute row-wise sum of absolute differences
    row_sums = np.zeros(nS, dtype=np.float64)
    for s in range(nS):
        for sa in range(nSA):
            row_sums[s] += abs(policy1_matrix[s, sa] - policy2_matrix[s, sa])

    # Weighted average with state distribution
    distance = np.dot(d_mu_pi, row_sums)
    return distance


@njit(cache=True, fastmath=True, parallel=True)
def model_sup_tv_distance_numba(P1_sa_matrix, P2_sa_matrix):
    """
    Numba-optimized computation of supremum total variation distance between models.

    Args:
        P1_sa_matrix: First model matrix (nSA, nS)
        P2_sa_matrix: Second model matrix (nSA, nS)

    Returns:
        distance: Supremum TV distance
    """
    nSA = P1_sa_matrix.shape[0]
    max_dist = 0.0

    for sa in prange(nSA):
        row_sum = 0.0
        for s in range(P1_sa_matrix.shape[1]):
            row_sum += abs(P1_sa_matrix[sa, s] - P2_sa_matrix[sa, s])
        if row_sum > max_dist:
            max_dist = row_sum

    return max_dist


@njit(cache=True, fastmath=True)
def model_mean_tv_distance_numba(P1_sa_matrix, P2_sa_matrix, delta_mu):
    """
    Numba-optimized computation of mean total variation distance between models.

    Args:
        P1_sa_matrix: First model matrix (nSA, nS)
        P2_sa_matrix: Second model matrix (nSA, nS)
        delta_mu: State-action distribution (nSA,)

    Returns:
        distance: Mean TV distance
    """
    nSA = P1_sa_matrix.shape[0]
    nS = P1_sa_matrix.shape[1]

    # Compute row-wise sum of absolute differences
    row_sums = np.zeros(nSA, dtype=np.float64)
    for sa in range(nSA):
        for s in range(nS):
            row_sums[sa] += abs(P1_sa_matrix[sa, s] - P2_sa_matrix[sa, s])

    # Weighted average with state-action distribution
    distance = np.dot(delta_mu, row_sums)
    return distance


@njit(cache=True)
def policy_convex_combination_numba(policy1_matrix, policy2_matrix, coeff):
    """
    Numba-optimized convex combination of two policies.

    Args:
        policy1_matrix: First policy matrix (nS, nSA)
        policy2_matrix: Second policy matrix (nS, nSA)
        coeff: Combination coefficient (0 to 1)

    Returns:
        Combined policy matrix
    """
    if coeff < 0 or coeff > 1:
        raise ValueError("Coefficient must be between 0 and 1")

    return coeff * policy1_matrix + (1.0 - coeff) * policy2_matrix


@njit(cache=True)
def model_convex_combination_numba(model1_matrix, model2_matrix, coeff):
    """
    Numba-optimized convex combination of two models.

    Args:
        model1_matrix: First model matrix (nSA, nS)
        model2_matrix: Second model matrix (nSA, nS)
        coeff: Combination coefficient (0 to 1)

    Returns:
        Combined model matrix
    """
    if coeff < 0 or coeff > 1:
        raise ValueError("Coefficient must be between 0 and 1")

    return coeff * model1_matrix + (1.0 - coeff) * model2_matrix


@njit(cache=True)
def policy_equiv_check_numba(policy1_matrix, policy2_matrix):
    """
    Numba-optimized equivalence check for policies.

    Args:
        policy1_matrix: First policy matrix (nS, nSA)
        policy2_matrix: Second policy matrix (nS, nSA)

    Returns:
        bool: True if policies are equivalent
    """
    nS = policy1_matrix.shape[0]
    nSA = policy1_matrix.shape[1]

    for s in range(nS):
        for sa in range(nSA):
            if policy1_matrix[s, sa] != policy2_matrix[s, sa]:
                return False
    return True


@njit(cache=True)
def model_equiv_check_numba(model1_matrix, model2_matrix):
    """
    Numba-optimized equivalence check for models.

    Args:
        model1_matrix: First model matrix (nSA, nS)
        model2_matrix: Second model matrix (nSA, nS)

    Returns:
        bool: True if models are equivalent
    """
    nSA = model1_matrix.shape[0]
    nS = model1_matrix.shape[1]

    for sa in range(nSA):
        for s in range(nS):
            if model1_matrix[sa, s] != model2_matrix[sa, s]:
                return False
    return True


@njit(cache=True, fastmath=True, parallel=True)
def greedy_policy_numba(Q, nS, nA):
    """
    Numba-optimized greedy policy computation.

    Args:
        Q: Q-function values (nSA,)
        nS: Number of states
        nA: Number of actions

    Returns:
        policy_matrix: Greedy policy matrix (nS, nSA)
    """
    nSA = nS * nA
    policy_matrix = np.zeros((nS, nSA), dtype=np.float64)

    for s in prange(nS):
        # Find the best action for state s
        best_value = -np.inf
        best_action = 0

        for a in range(nA):
            sa = s * nA + a
            if Q[sa] > best_value:
                best_value = Q[sa]
                best_action = a

        # Set deterministic policy
        best_sa = s * nA + best_action
        policy_matrix[s, best_sa] = 1.0

    return policy_matrix


@njit(cache=True, fastmath=True, parallel=True)
def greedy_model_numba(U, delta_mu_pi, nS, nA):
    """
    Numba-optimized greedy model selection.

    Args:
        U: U-function values (nSA, nS)
        delta_mu_pi: State-action distribution (nSA,)
        nS: Number of states
        nA: Number of actions

    Returns:
        model_matrix: Greedy model matrix (nSA, nS)
    """
    nSA = nS * nA
    model_matrix = np.zeros((nSA, nS), dtype=np.float64)

    for sa in prange(nSA):
        # Weighted U values for this state-action
        weighted_U = U[sa] * delta_mu_pi[sa]

        # Find the best next state
        best_value = -np.inf
        best_state = 0

        for s in range(nS):
            if weighted_U[s] > best_value:
                best_value = weighted_U[s]
                best_state = s

        # Set deterministic transition
        model_matrix[sa, best_state] = 1.0

    return model_matrix