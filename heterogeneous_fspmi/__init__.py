"""
Heterogeneous Federated Safe Policy-Model Iteration (H-FSPMI)

A genuine federated learning extension of SPMI that provides:
1. TRUE parallel execution across agents
2. Heterogeneous environment variants per agent
3. Robust policies that generalize across environment variations

Key classes:
- HeterogeneousFSPMI: Main algorithm implementation
- HeterogeneousConfig: Configuration for experiments
- EnvironmentVariant: Specification for each agent's environment

Usage:
    from heterogeneous_fspmi import (
        HeterogeneousFSPMI,
        HeterogeneousConfig,
        EnvironmentVariant,
        create_dynamics_heterogeneous_fspmi,
        create_robustness_heterogeneous_fspmi,
    )
    
    # Quick start with dynamics variants
    hfspmi = create_dynamics_heterogeneous_fspmi(
        track_file="T1",
        k_values=[0.3, 0.5, 0.7],
        episodes_per_agent=100,
        n_iterations=100
    )
    
    final_policy, final_model = hfspmi.run(initial_policy, initial_model)
"""

from .heterogeneous_fspmi import (
    HeterogeneousFSPMI,
    HeterogeneousConfig,
    EnvironmentVariant,
    LocalStatistics,
    GlobalStatistics,
    FSPMILogger,
    create_dynamics_heterogeneous_fspmi,
    create_robustness_heterogeneous_fspmi,
)

__all__ = [
    'HeterogeneousFSPMI',
    'HeterogeneousConfig', 
    'EnvironmentVariant',
    'LocalStatistics',
    'GlobalStatistics',
    'FSPMILogger',
    'create_dynamics_heterogeneous_fspmi',
    'create_robustness_heterogeneous_fspmi',
]