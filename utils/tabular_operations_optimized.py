"""
Optimized tabular operations module with Numba JIT compilation.
Provides drop-in replacements for distance computations and convex combinations.
"""

import numpy as np
from utils import tabular_factory
from optimized.numba_functions import (
    policy_sup_tv_distance_numba,
    policy_mean_tv_distance_numba,
    model_sup_tv_distance_numba,
    model_mean_tv_distance_numba,
    policy_convex_combination_numba,
    model_convex_combination_numba,
    policy_equiv_check_numba,
    model_equiv_check_numba
)


def policy_convex_combination(policy1, policy2, coeff, use_numba=True):
    """
    Compute convex combination of two policies with optional Numba optimization.

    Args:
        policy1: First policy object with get_matrix() method
        policy2: Second policy object with get_matrix() method
        coeff: Combination coefficient (0 to 1)
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        Combined policy object
    """
    if not use_numba:
        # Fall back to original implementation
        from utils.tabular_operations import policy_convex_combination as orig_func
        return orig_func(policy1, policy2, coeff)

    if coeff < 0 or coeff > 1:
        raise ValueError("Coefficient must be between 0 and 1")

    # Extract matrices
    policy1_matrix = policy1.get_matrix()
    policy2_matrix = policy2.get_matrix()

    # Call Numba-optimized function
    matrix = policy_convex_combination_numba(policy1_matrix, policy2_matrix, coeff)

    # Return wrapped policy object
    return tabular_factory.policy_from_matrix(matrix)


def model_convex_combination(original_model, model1, model2, coeff, use_numba=True):
    """
    Compute convex combination of two models with optional Numba optimization.

    Args:
        original_model: Original model for structure
        model1: First model object with get_matrix() method
        model2: Second model object with get_matrix() method
        coeff: Combination coefficient (0 to 1)
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        Combined model object
    """
    if not use_numba:
        # Fall back to original implementation
        from utils.tabular_operations import model_convex_combination as orig_func
        return orig_func(original_model, model1, model2, coeff)

    if coeff < 0 or coeff > 1:
        raise ValueError("Coefficient must be between 0 and 1")

    # Extract matrices
    model1_matrix = model1.get_matrix()
    model2_matrix = model2.get_matrix()

    # Call Numba-optimized function
    matrix = model_convex_combination_numba(model1_matrix, model2_matrix, coeff)

    # Return wrapped model object
    return tabular_factory.model_from_matrix(matrix, original_model)


def model_convex_combination_set(original_model, model_set, current_model, x, use_numba=True):
    """
    Compute convex combination of a set of models with optional Numba optimization.

    Args:
        original_model: Original model for structure
        model_set: List of model objects
        current_model: Current model object
        x: Weights for combination
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        Combined model object
    """
    if not use_numba:
        # Fall back to original implementation
        from utils.tabular_operations import model_convex_combination_set as orig_func
        return orig_func(original_model, model_set, current_model, x)

    n_models = len(model_set)

    # Get the matrix of the current model
    current_model_matrix = current_model.get_matrix()

    # Compute the weighted combination
    model_set_combination = np.zeros_like(current_model_matrix)
    for i in range(n_models):
        model_set_combination += x[i] * model_set[i].get_matrix()

    # Compute the update
    matrix = model_set_combination + (1.0 - np.sum(x)) * current_model_matrix

    return tabular_factory.model_from_matrix(matrix, original_model)


def policy_sup_tv_distance(policy1, policy2, use_numba=True):
    """
    Compute supremum total variation distance between policies with optional Numba optimization.

    Args:
        policy1: First policy object with get_matrix() method
        policy2: Second policy object with get_matrix() method
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        distance: Supremum TV distance
    """
    if not use_numba:
        # Fall back to original implementation
        from utils.tabular_operations import policy_sup_tv_distance as orig_func
        return orig_func(policy1, policy2)

    # Extract matrices
    policy1_matrix = policy1.get_matrix()
    policy2_matrix = policy2.get_matrix()

    # Call Numba-optimized function
    return policy_sup_tv_distance_numba(policy1_matrix, policy2_matrix)


def policy_mean_tv_distance(policy1, policy2, d_mu_pi, use_numba=True):
    """
    Compute mean total variation distance between policies with optional Numba optimization.

    Args:
        policy1: First policy object with get_matrix() method
        policy2: Second policy object with get_matrix() method
        d_mu_pi: State distribution
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        distance: Mean TV distance
    """
    if not use_numba:
        # Fall back to original implementation
        from utils.tabular_operations import policy_mean_tv_distance as orig_func
        return orig_func(policy1, policy2, d_mu_pi)

    # Extract matrices
    policy1_matrix = policy1.get_matrix()
    policy2_matrix = policy2.get_matrix()

    # Call Numba-optimized function
    return policy_mean_tv_distance_numba(policy1_matrix, policy2_matrix, d_mu_pi)


def model_sup_tv_distance(P1_sa, P2_sa, use_numba=True):
    """
    Compute supremum total variation distance between models with optional Numba optimization.

    Args:
        P1_sa: First model object with get_matrix() method
        P2_sa: Second model object with get_matrix() method
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        distance: Supremum TV distance
    """
    if not use_numba:
        # Fall back to original implementation
        from utils.tabular_operations import model_sup_tv_distance as orig_func
        return orig_func(P1_sa, P2_sa)

    # Extract matrices
    P1_sa_matrix = P1_sa.get_matrix()
    P2_sa_matrix = P2_sa.get_matrix()

    # Call Numba-optimized function
    return model_sup_tv_distance_numba(P1_sa_matrix, P2_sa_matrix)


def model_mean_tv_distance(P1_sa, P2_sa, delta_mu, use_numba=True):
    """
    Compute mean total variation distance between models with optional Numba optimization.

    Args:
        P1_sa: First model object with get_matrix() method
        P2_sa: Second model object with get_matrix() method
        delta_mu: State-action distribution
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        distance: Mean TV distance
    """
    if not use_numba:
        # Fall back to original implementation
        from utils.tabular_operations import model_mean_tv_distance as orig_func
        return orig_func(P1_sa, P2_sa, delta_mu)

    # Extract matrices
    P1_sa_matrix = P1_sa.get_matrix()
    P2_sa_matrix = P2_sa.get_matrix()

    # Call Numba-optimized function
    return model_mean_tv_distance_numba(P1_sa_matrix, P2_sa_matrix, delta_mu)


def policy_equiv_check(policy1, policy2, use_numba=True):
    """
    Check equivalence of two policies with optional Numba optimization.

    Args:
        policy1: First policy object with get_matrix() method
        policy2: Second policy object with get_matrix() method
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        bool: True if policies are equivalent
    """
    if not use_numba:
        # Fall back to original implementation
        from utils.tabular_operations import policy_equiv_check as orig_func
        return orig_func(policy1, policy2)

    # Extract matrices
    policy1_matrix = policy1.get_matrix()
    policy2_matrix = policy2.get_matrix()

    # Call Numba-optimized function
    return policy_equiv_check_numba(policy1_matrix, policy2_matrix)


def model_equiv_check(model1, model2, use_numba=True):
    """
    Check equivalence of two models with optional Numba optimization.

    Args:
        model1: First model object with get_matrix() method
        model2: Second model object with get_matrix() method
        use_numba: Whether to use Numba optimization (default: True)

    Returns:
        bool: True if models are equivalent
    """
    if not use_numba:
        # Fall back to original implementation
        from utils.tabular_operations import model_equiv_check as orig_func
        return orig_func(model1, model2)

    # Extract matrices
    model1_matrix = model1.get_matrix()
    model2_matrix = model2.get_matrix()

    # Call Numba-optimized function
    return model_equiv_check_numba(model1_matrix, model2_matrix)