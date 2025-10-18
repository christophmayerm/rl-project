"""
Optimized evaluator module with Numba JIT compilation.
Provides drop-in replacements for the original evaluator functions with significant speedup.
"""

import numpy as np
from optimized.numba_evaluator import (
    compute_q_function_numba,
    compute_v_function_numba,
    compute_u_function_numba,
    compute_performance_numba,
    compute_discounted_s_distribution_numba,
    compute_discounted_sa_distribution_numba,
    compute_policy_er_advantage_numba,
    compute_model_er_advantage_numba
)


def compute_q_function(policy, model, reward, gamma, nS=None, nA=None, horizon=None, use_numba=True):
    """
    Compute Q-function with optional Numba optimization.

    Args:
        policy: Policy object with get_matrix() method
        model: Model object with get_matrix() method
        reward: Reward object with get_matrix() method
        gamma: Discount factor
        nS: Number of states
        nA: Number of actions
        horizon: Horizon for iterative computation (None for exact)
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        Q: Q-function values
    """
    if not use_numba:
        # Fall back to original implementation
        import utils.evaluator as orig_evaluator
        return orig_evaluator.compute_q_function(policy, model, reward, gamma, nS, nA, horizon)

    # Extract matrices
    R_sas = reward.get_matrix()
    P = model.get_matrix()
    pi = policy.get_matrix()

    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])

    # Call Numba-optimized function
    return compute_q_function_numba(pi, P, R_sas, gamma, nS, nA, horizon)


def compute_v_function(policy, model, reward, gamma, nS=None, nA=None, horizon=None, use_numba=True):
    """
    Compute V-function with optional Numba optimization.

    Args:
        policy: Policy object with get_matrix() method
        model: Model object with get_matrix() method
        reward: Reward object with get_matrix() method
        gamma: Discount factor
        nS: Number of states
        nA: Number of actions
        horizon: Horizon for iterative computation (None for exact)
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        V: V-function values
    """
    if not use_numba:
        # Fall back to original implementation
        import utils.evaluator as orig_evaluator
        return orig_evaluator.compute_v_function(policy, model, reward, gamma, nS, nA, horizon)

    # Extract matrices
    R_sas = reward.get_matrix()
    P = model.get_matrix()
    pi = policy.get_matrix()

    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])

    # Call Numba-optimized function
    return compute_v_function_numba(pi, P, R_sas, gamma, nS, nA, horizon)


def compute_u_function(policy, model, reward, gamma, nS=None, nA=None, horizon=None, use_numba=True):
    """
    Compute U-function with optional Numba optimization.

    Args:
        policy: Policy object with get_matrix() method
        model: Model object with get_matrix() method
        reward: Reward object with get_matrix() method
        gamma: Discount factor
        nS: Number of states
        nA: Number of actions
        horizon: Horizon for U-function computation
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        U: U-function values
    """
    if not use_numba:
        # Fall back to original implementation
        import utils.evaluator as orig_evaluator
        return orig_evaluator.compute_u_function(policy, model, reward, gamma, nS, nA, horizon)

    # Extract matrices
    pi = policy.get_matrix()
    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])

    R_sas = reward.get_matrix()
    P = model.get_matrix()

    # Call Numba-optimized function
    return compute_u_function_numba(pi, P, R_sas, gamma, nS, nA, horizon)


def compute_performance(mu, reward, policy, model, gamma, horizon=None, nS=None, nA=None, use_numba=True):
    """
    Compute expected discounted return with optional Numba optimization.

    Args:
        mu: Initial state distribution
        reward: Reward object with get_matrix() method
        policy: Policy object with get_matrix() method
        model: Model object with get_matrix() method
        gamma: Discount factor
        horizon: Horizon for V-function computation
        nS: Number of states
        nA: Number of actions
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        J: Expected discounted return
    """
    if not use_numba:
        # Fall back to original implementation
        import utils.evaluator as orig_evaluator
        return orig_evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

    # Extract matrices
    pi = policy.get_matrix()
    P = model.get_matrix()
    R_sas = reward.get_matrix()

    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])

    # Call Numba-optimized function
    return compute_performance_numba(mu, pi, P, R_sas, gamma, nS, nA, horizon)


def compute_discounted_s_distribution(mu, policy, model, gamma, horizon=None, nS=None, nA=None, use_numba=True):
    """
    Compute discounted state distribution with optional Numba optimization.

    Args:
        mu: Initial state distribution
        policy: Policy object with get_matrix() method
        model: Model object with get_matrix() method
        gamma: Discount factor
        horizon: Horizon for iterative computation (None for exact)
        nS: Number of states
        nA: Number of actions
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        d_mu: Discounted state distribution
    """
    if not use_numba:
        # Fall back to original implementation
        import utils.evaluator as orig_evaluator
        return orig_evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)

    # Extract matrices
    pi = policy.get_matrix()
    P = model.get_matrix()

    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])

    # Call Numba-optimized function
    return compute_discounted_s_distribution_numba(mu, pi, P, gamma, nS, nA, horizon)


def compute_discounted_sa_distribution(mu, policy, model, gamma, horizon=None, nS=None, nA=None, d_mu=None, use_numba=True):
    """
    Compute discounted state-action distribution with optional Numba optimization.

    Args:
        mu: Initial state distribution
        policy: Policy object with get_matrix() method
        model: Model object with get_matrix() method
        gamma: Discount factor
        horizon: Horizon for computation
        nS: Number of states
        nA: Number of actions
        d_mu: Pre-computed state distribution (optional)
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        delta_mu: Discounted state-action distribution
    """
    if not use_numba:
        # Fall back to original implementation
        import utils.evaluator as orig_evaluator
        return orig_evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)

    # Extract matrices
    pi = policy.get_matrix()
    P = model.get_matrix()

    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])

    # Call Numba-optimized function
    return compute_discounted_sa_distribution_numba(mu, pi, P, gamma, nS, nA, horizon, d_mu)


def compute_policy_er_advantage(target, policy, Q, d_mu_pi, use_numba=True):
    """
    Compute expected relative advantage for policies with optional Numba optimization.

    Args:
        target: Target policy object with get_matrix() method
        policy: Current policy object with get_matrix() method
        Q: Q-function values
        d_mu_pi: State distribution under current policy
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        er_advantage: Expected relative advantage
    """
    if not use_numba:
        # Fall back to original implementation
        import utils.evaluator as orig_evaluator
        return orig_evaluator.compute_policy_er_advantage(target, policy, Q, d_mu_pi)

    # Extract matrices
    target_matrix = target.get_matrix()
    policy_matrix = policy.get_matrix()

    # Call Numba-optimized function
    return compute_policy_er_advantage_numba(target_matrix, policy_matrix, Q, d_mu_pi)


def compute_model_er_advantage(target, model, U, delta_mu_pi, use_numba=True):
    """
    Compute expected relative advantage for models with optional Numba optimization.

    Args:
        target: Target model object with get_matrix() method
        model: Current model object with get_matrix() method
        U: U-function values
        delta_mu_pi: State-action distribution
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        er_advantage: Expected relative advantage
    """
    if not use_numba:
        # Fall back to original implementation
        import utils.evaluator as orig_evaluator
        return orig_evaluator.compute_model_er_advantage(target, model, U, delta_mu_pi)

    # Extract matrices
    target_matrix = target.get_matrix()
    model_matrix = model.get_matrix()

    # Call Numba-optimized function
    return compute_model_er_advantage_numba(target_matrix, model_matrix, U, delta_mu_pi)