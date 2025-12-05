"""
Federated Safe Policy-Model Iteration (F-SPMI)

This package implements the federated learning extension of SPMI for
Configurable Markov Decision Processes (Conf-MDPs).

The key components are:
- FederatedAgent: Local agent that collects trajectories and computes local statistics
- FederatedServer: Central server that aggregates statistics and computes safe updates
- FSPMI: Main federated algorithm orchestrating the learning process
"""

from .agent import FederatedAgent, LocalStatistics
from .server import FederatedServer, GlobalStatistics
from .f_spmi import FSPMI
from .estimators import (
    estimate_q_function_mc,
    estimate_v_function_mc,
    estimate_u_function_mc,
    estimate_discounted_s_distribution,
    estimate_discounted_sa_distribution
)
from .aggregators import (
    federated_average,
    weighted_federated_average,
    robust_federated_average
)

__all__ = [
    'FederatedAgent',
    'FederatedServer',
    'FSPMI',
    'LocalStatistics',
    'GlobalStatistics',
    'estimate_q_function_mc',
    'estimate_v_function_mc',
    'estimate_u_function_mc',
    'estimate_discounted_s_distribution',
    'estimate_discounted_sa_distribution',
    'federated_average',
    'weighted_federated_average',
    'robust_federated_average'
]
