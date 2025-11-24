import numpy as np
import numpy.linalg as la
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

# Global cache for computed values
_Q_CACHE = {}
_U_CACHE = {}
_V_CACHE = {}
_D_MU_CACHE = {}
_DELTA_MU_CACHE = {}

def clear_cache():
    """Clear all evaluator caches - call after policy/model updates"""
    global _Q_CACHE, _U_CACHE, _V_CACHE, _D_MU_CACHE, _DELTA_MU_CACHE
    _Q_CACHE.clear()
    _U_CACHE.clear()
    _V_CACHE.clear()
    _D_MU_CACHE.clear()
    _DELTA_MU_CACHE.clear()


def compute_q_function(policy, model, reward, gamma, nS=None, nA=None, horizon=None):
    """Cached Numba-optimized Q-function computation"""
    # Create cache key
    cache_key = (id(policy), id(model), id(reward), gamma, horizon)
    if cache_key in _Q_CACHE:
        return _Q_CACHE[cache_key]
    
    # Get matrices
    R_sas = reward.get_matrix()
    P = model.get_matrix()
    pi = policy.get_matrix()
    
    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])
    
    # Use Numba-optimized version
    Q = compute_q_function_numba(pi, P, R_sas, gamma, nS, nA, horizon)
    
    _Q_CACHE[cache_key] = Q
    return Q


def compute_v_function(policy, model, reward, gamma, nS=None, nA=None, horizon=None):
    """Cached Numba-optimized V-function computation"""
    cache_key = (id(policy), id(model), id(reward), gamma, horizon)
    if cache_key in _V_CACHE:
        return _V_CACHE[cache_key]
    
    R_sas = reward.get_matrix()
    P = model.get_matrix()
    pi = policy.get_matrix()
    
    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])
    
    # Use Numba-optimized version
    V = compute_v_function_numba(pi, P, R_sas, gamma, nS, nA, horizon)
    
    _V_CACHE[cache_key] = V
    return V


def compute_u_function(policy, model, reward, gamma, nS=None, nA=None, horizon=None):
    """Cached Numba-optimized U-function computation"""
    cache_key = (id(policy), id(model), id(reward), gamma, horizon)
    if cache_key in _U_CACHE:
        return _U_CACHE[cache_key]
    
    pi = policy.get_matrix()
    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])
    
    R_sas = reward.get_matrix()
    P = model.get_matrix()
    
    # Use Numba-optimized version
    U = compute_u_function_numba(pi, P, R_sas, gamma, nS, nA, horizon)
    
    _U_CACHE[cache_key] = U
    return U


def compute_performance(mu, reward, policy, model, gamma, horizon=None, nS=None, nA=None):
    """Numba-optimized performance computation"""
    pi = policy.get_matrix()
    P = model.get_matrix()
    R_sas = reward.get_matrix()
    
    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])
    
    return compute_performance_numba(mu, pi, P, R_sas, gamma, nS, nA, horizon)


def compute_discounted_s_distribution(mu, policy, model, gamma, horizon=None, nS=None, nA=None):
    """Cached Numba-optimized state distribution computation"""
    cache_key = (id(mu.tobytes()), id(policy), id(model), gamma, horizon)
    if cache_key in _D_MU_CACHE:
        return _D_MU_CACHE[cache_key]
    
    pi = policy.get_matrix()
    P = model.get_matrix()
    
    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])
    
    d_mu = compute_discounted_s_distribution_numba(mu, pi, P, gamma, nS, nA, horizon)
    
    _D_MU_CACHE[cache_key] = d_mu
    return d_mu


def compute_discounted_sa_distribution(mu, policy, model, gamma, horizon=None, nS=None, nA=None, d_mu=None):
    """Cached Numba-optimized state-action distribution computation"""
    pi = policy.get_matrix()
    P = model.get_matrix()
    
    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])
    
    # Use provided d_mu or compute it
    if d_mu is None:
        d_mu = compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
    
    delta_mu = compute_discounted_sa_distribution_numba(mu, pi, P, gamma, nS, nA, horizon, d_mu)
    return delta_mu


def compute_policy_er_advantage(target, policy, Q, d_mu_pi):
    """Numba-optimized policy advantage computation"""
    target_matrix = target.get_matrix()
    policy_matrix = policy.get_matrix()
    
    return compute_policy_er_advantage_numba(target_matrix, policy_matrix, Q, d_mu_pi)


def compute_model_er_advantage(target, model, U, delta_mu_pi):
    """Numba-optimized model advantage computation"""
    target_matrix = target.get_matrix()
    model_matrix = model.get_matrix()
    
    return compute_model_er_advantage_numba(target_matrix, model_matrix, U, delta_mu_pi)