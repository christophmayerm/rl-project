"""
Federated Agent for Safe Policy-Model Iteration.

Each agent:
1. Executes the current global policy-model pair to collect trajectories
2. Computes local statistics (Q, δ_μ, d_μ estimates)
3. Sends local statistics to the central server
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from .estimators import (
    Trajectory,
    estimate_q_function_mc,
    estimate_v_function_mc,
    estimate_u_function_mc,
    estimate_discounted_s_distribution,
    estimate_discounted_sa_distribution,
    compute_confidence_bound
)


@dataclass
class LocalStatistics:
    """
    Local statistics computed by a federated agent.

    These are the "gained knowledge" sent to the central server,
    as described in the F-SPMI formulation.
    """
    agent_id: int
    Q_hat: np.ndarray                    # Q-function estimate
    Q_counts: np.ndarray                 # Sample counts per (s,a)
    V_hat: np.ndarray                    # V-function estimate
    U_hat: np.ndarray                    # U-function estimate
    d_mu_hat: np.ndarray                 # γ-discounted state distribution
    delta_mu_hat: np.ndarray             # γ-discounted state-action distribution
    n_episodes: int                      # Number of episodes collected
    n_samples: int                       # Total number of (s,a,r,s') samples
    avg_episode_length: float            # Average episode length
    avg_return: float                    # Average episode return
    confidence_radius: float = 0.0       # Confidence bound on estimates

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'agent_id': self.agent_id,
            'Q_hat': self.Q_hat.tolist(),
            'Q_counts': self.Q_counts.tolist(),
            'V_hat': self.V_hat.tolist(),
            'U_hat': self.U_hat.tolist(),
            'd_mu_hat': self.d_mu_hat.tolist(),
            'delta_mu_hat': self.delta_mu_hat.tolist(),
            'n_episodes': self.n_episodes,
            'n_samples': self.n_samples,
            'avg_episode_length': self.avg_episode_length,
            'avg_return': self.avg_return,
            'confidence_radius': self.confidence_radius
        }


class FederatedAgent:
    """
    A federated learning agent that collects local statistics.

    In the F-SPMI framework, each agent j:
    1. Receives the current global policy π_k and model P_k
    2. Executes episodes using (P_k, π_k)
    3. Computes local estimates Q̂_j, δ̂_μ,j
    4. Sends these to the central server
    """

    def __init__(
        self,
        agent_id: int,
        nS: int,
        nA: int,
        gamma: float,
        horizon: int,
        seed: Optional[int] = None
    ):
        """
        Initialize a federated agent.

        Args:
            agent_id: Unique identifier for this agent
            nS: Number of states in the MDP
            nA: Number of actions in the MDP
            gamma: Discount factor
            horizon: Episode horizon
            seed: Random seed for reproducibility
        """
        self.agent_id = agent_id
        self.nS = nS
        self.nA = nA
        self.nSA = nS * nA
        self.gamma = gamma
        self.horizon = horizon

        # Random number generator
        self.rng = np.random.RandomState(seed)

        # Storage for trajectories
        self.trajectories: List[Trajectory] = []

    def reset(self):
        """Clear stored trajectories"""
        self.trajectories = []

    def execute_policy(
        self,
        mdp,
        policy,
        model,
        n_episodes: int,
        reset_trajectories: bool = True
    ) -> List[Trajectory]:
        """
        Execute the policy-model pair to collect trajectories.

        Args:
            mdp: The MDP environment
            policy: Current policy (TabularPolicy object)
            model: Current model (TabularModel object)
            n_episodes: Number of episodes to collect
            reset_trajectories: Whether to clear previous trajectories

        Returns:
            List of collected trajectories
        """
        if reset_trajectories:
            self.reset()

        for ep in range(n_episodes):
            trajectory = self._rollout(mdp, policy, model)
            self.trajectories.append(trajectory)

        return self.trajectories

    def _rollout(self, mdp, policy, model) -> Trajectory:
        """
        Execute a single episode.

        Args:
            mdp: The MDP environment
            policy: Policy to execute
            model: Model (for configurable environment)

        Returns:
            Trajectory object containing the episode
        """
        states = []
        actions = []
        rewards = []
        next_states = []

        # Reset environment and get initial state
        # Handle different Gymnasium/Gym API versions
        if hasattr(mdp, '_reset'):
            state = mdp._reset()
        else:
            state = mdp.reset()

        # Handle tuple returns (new Gymnasium API returns (obs, info))
        if isinstance(state, tuple):
            state = state[0]

        # Handle array states
        if hasattr(state, '__len__') and not isinstance(state, (int, np.integer)):
            if len(state) > 0:
                state = state[0] if hasattr(state, '__getitem__') else state
            else:
                state = 0  # Fallback

        done = False
        t = 0

        while not done and t < self.horizon:
            # Get action from policy
            state_int = int(state) if state is not None else 0
            policy_probs = policy.get_rep()[state_int]
            action = self.rng.choice(self.nA, p=policy_probs)

            # Take action in environment
            # Handle different Gymnasium/Gym API versions
            if hasattr(mdp, '_step'):
                result = mdp._step(action)
            else:
                result = mdp.step(action)

            # Handle different return formats
            if len(result) == 5:  # Gymnasium style
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            elif len(result) == 4:  # Old Gym style
                next_state, reward, done, info = result
            else:
                next_state, reward = result[:2]
                done = t >= self.horizon - 1

            # Handle array/tuple next_state
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            if hasattr(next_state, '__len__') and not isinstance(next_state, (int, np.integer)):
                if len(next_state) > 0:
                    next_state = next_state[0] if hasattr(next_state, '__getitem__') else next_state
                else:
                    next_state = 0

            # Store transition
            states.append(state_int)
            actions.append(int(action))
            rewards.append(float(reward))
            next_states.append(int(next_state))

            state = next_state
            t += 1

        return Trajectory(
            states=np.array(states, dtype=np.int32),
            actions=np.array(actions, dtype=np.int32),
            rewards=np.array(rewards, dtype=np.float64),
            next_states=np.array(next_states, dtype=np.int32)
        )

    def compute_local_statistics(
        self,
        trajectories: Optional[List[Trajectory]] = None,
        confidence_delta: float = 0.05
    ) -> LocalStatistics:
        """
        Compute local statistics from collected trajectories.

        This computes the "gained knowledge" that will be sent to the server:
        - Q̂_j: Local Q-function estimate
        - δ̂_μ,j: Local γ-discounted state-action distribution

        Args:
            trajectories: Trajectories to use (if None, uses stored trajectories)
            confidence_delta: Confidence level for bounds

        Returns:
            LocalStatistics object containing all local estimates
        """
        if trajectories is None:
            trajectories = self.trajectories

        if not trajectories:
            raise ValueError("No trajectories available for computing statistics")

        # Estimate Q-function
        Q_hat, Q_counts = estimate_q_function_mc(
            trajectories, self.gamma, self.nS, self.nA, method='first_visit'
        )

        # Estimate V-function
        V_hat, V_counts = estimate_v_function_mc(
            trajectories, self.gamma, self.nS, method='first_visit'
        )

        # Estimate U-function
        U_hat = estimate_u_function_mc(
            trajectories, self.gamma, self.nS, self.nA, V_hat
        )

        # Estimate discounted state distribution
        d_mu_hat, d_samples = estimate_discounted_s_distribution(
            trajectories, self.gamma, self.nS, self.horizon
        )

        # Estimate discounted state-action distribution
        delta_mu_hat, delta_samples = estimate_discounted_sa_distribution(
            trajectories, self.gamma, self.nS, self.nA, self.horizon
        )

        # Compute statistics
        n_episodes = len(trajectories)
        n_samples = sum(len(t) for t in trajectories)
        avg_episode_length = n_samples / n_episodes if n_episodes > 0 else 0

        # Compute average return
        returns = []
        for traj in trajectories:
            G = 0.0
            for t in reversed(range(len(traj))):
                G = traj.rewards[t] + self.gamma * G
            returns.append(G)
        avg_return = np.mean(returns) if returns else 0.0

        # Confidence radius
        confidence_radius = compute_confidence_bound(
            n_samples, confidence_delta, value_range=1.0 / (1 - self.gamma)
        )

        return LocalStatistics(
            agent_id=self.agent_id,
            Q_hat=Q_hat,
            Q_counts=Q_counts,
            V_hat=V_hat,
            U_hat=U_hat,
            d_mu_hat=d_mu_hat,
            delta_mu_hat=delta_mu_hat,
            n_episodes=n_episodes,
            n_samples=n_samples,
            avg_episode_length=avg_episode_length,
            avg_return=avg_return,
            confidence_radius=confidence_radius
        )

    def collect_and_compute(
        self,
        mdp,
        policy,
        model,
        n_episodes: int,
        confidence_delta: float = 0.05
    ) -> LocalStatistics:
        """
        Convenience method: collect trajectories and compute statistics.

        Args:
            mdp: The MDP environment
            policy: Current policy
            model: Current model
            n_episodes: Number of episodes to collect
            confidence_delta: Confidence level for bounds

        Returns:
            LocalStatistics object
        """
        self.execute_policy(mdp, policy, model, n_episodes)
        return self.compute_local_statistics(confidence_delta=confidence_delta)


class ParallelAgentExecutor:
    """
    Utility class for executing multiple agents in parallel.

    Note: For true parallelism, consider using multiprocessing or
    distributed computing frameworks like Ray.
    """

    def __init__(self, agents: List[FederatedAgent]):
        self.agents = agents
        self.n_agents = len(agents)

    def execute_all(
        self,
        mdp,
        policy,
        model,
        n_episodes_per_agent: int,
        parallel: bool = False
    ) -> List[LocalStatistics]:
        """
        Execute all agents and collect their local statistics.

        Args:
            mdp: The MDP environment (or list of MDPs for parallel)
            policy: Current global policy
            model: Current global model
            n_episodes_per_agent: Episodes per agent
            parallel: Whether to use parallel execution

        Returns:
            List of LocalStatistics from each agent
        """
        local_stats = []

        if parallel:
            # For true parallelism, implement using multiprocessing
            # This is a placeholder for sequential execution
            import warnings
            warnings.warn("Parallel execution not fully implemented, using sequential")

        for agent in self.agents:
            stats = agent.collect_and_compute(
                mdp, policy, model, n_episodes_per_agent
            )
            local_stats.append(stats)

        return local_stats
