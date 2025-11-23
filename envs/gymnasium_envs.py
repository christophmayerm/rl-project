"""
Gymnasium Environment Wrappers for SPMI/SA-PMI
Converts continuous/large Gymnasium environments to tabular MDPs
"""

import numpy as np
import gymnasium as gym
from utils.tabular import TabularPolicy


class TabularizedGymEnv:
    """
    Base class for converting Gymnasium environments to tabular MDPs
    Uses discretization for continuous spaces
    """
    
    def __init__(self, env_name, n_bins=10, horizon=100, gamma=0.99):
        """
        :param env_name: Gymnasium environment name (e.g., 'CartPole-v1')
        :param n_bins: Number of bins for discretizing continuous states
        :param horizon: Episode horizon
        :param gamma: Discount factor
        """
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.n_bins = n_bins
        self.horizon = horizon
        self.gamma = gamma
        
        # Get action space
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.nA = self.env.action_space.n
        else:
            raise NotImplementedError("Only discrete action spaces supported")
        
        # Discretize state space
        self._setup_state_discretization()
        
        # Build tabular MDP
        self._build_tabular_mdp()
    
    def _setup_state_discretization(self):
        """Setup state space discretization"""
        if isinstance(self.env.observation_space, gym.spaces.Box):
            self.obs_space = self.env.observation_space
            self.n_dims = self.obs_space.shape[0]
            
            # Create bins for each dimension
            self.bins = []
            for i in range(self.n_dims):
                low = self.obs_space.low[i]
                high = self.obs_space.high[i]
                
                # Handle infinite bounds
                if np.isinf(low):
                    low = -10.0
                if np.isinf(high):
                    high = 10.0
                
                self.bins.append(np.linspace(low, high, self.n_bins + 1))
            
            # Total number of discrete states
            self.nS = self.n_bins ** self.n_dims
            
        elif isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.nS = self.env.observation_space.n
        else:
            raise NotImplementedError("Only Box and Discrete observation spaces supported")
    
    def _discretize_state(self, obs):
        """Convert continuous observation to discrete state index"""
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            return obs
        
        # Discretize each dimension
        indices = []
        for i in range(self.n_dims):
            idx = np.digitize(obs[i], self.bins[i]) - 1
            idx = np.clip(idx, 0, self.n_bins - 1)
            indices.append(idx)
        
        # Convert multi-dimensional index to single state index
        state = 0
        for i, idx in enumerate(indices):
            state += idx * (self.n_bins ** i)
        
        return state
    
    def _build_tabular_mdp(self):
        """
        Build tabular MDP by sampling transitions
        Note: This is an approximation. For better results, use model-free methods.
        """
        print(f"Building tabular MDP for {self.env_name}...")
        print(f"  States: {self.nS}, Actions: {self.nA}")
        
        # Initialize transition probabilities
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        
        # Sample transitions for each state-action pair
        n_samples = 100  # Samples per state-action
        
        for s in range(min(self.nS, 1000)):  # Limit for very large state spaces
            if s % 100 == 0:
                print(f"  Sampling state {s}/{min(self.nS, 1000)}...")
            
            for a in range(self.nA):
                # Estimate transitions by sampling
                next_states = {}
                total_reward = 0.0
                
                for _ in range(n_samples):
                    # Reset to approximate state s
                    obs, _ = self.env.reset()
                    
                    # Take action a
                    next_obs, reward, terminated, truncated, _ = self.env.step(a)
                    done = terminated or truncated
                    
                    # Discretize next state
                    s_next = self._discretize_state(next_obs)
                    
                    # Accumulate
                    if s_next not in next_states:
                        next_states[s_next] = 0
                    next_states[s_next] += 1
                    total_reward += reward
                
                # Convert counts to probabilities
                avg_reward = total_reward / n_samples
                for s_next, count in next_states.items():
                    prob = count / n_samples
                    self.P[s][a].append((prob, s_next, avg_reward, False))
        
        # Initial state distribution (uniform for simplicity)
        self.mu = np.ones(self.nS) / self.nS
        
        print(f"✓ Tabular MDP built for {self.env_name}")
    
    def get_uniform_policy(self):
        """Return uniform random policy"""
        policy_rep = {s: np.ones(self.nA) / self.nA for s in range(self.nS)}
        return TabularPolicy(policy_rep, self.nS, self.nA)
    
    def set_model(self, P):
        """Update transition model"""
        self.P = P


class CartPoleTabular(TabularizedGymEnv):
    """CartPole-v1 as tabular MDP"""
    
    def __init__(self, n_bins=8, horizon=200):
        super().__init__('CartPole-v1', n_bins=n_bins, horizon=horizon, gamma=0.99)


class MountainCarTabular(TabularizedGymEnv):
    """MountainCar-v0 as tabular MDP"""
    
    def __init__(self, n_bins=10, horizon=200):
        super().__init__('MountainCar-v0', n_bins=n_bins, horizon=horizon, gamma=0.99)


class AcrobotTabular(TabularizedGymEnv):
    """Acrobot-v1 as tabular MDP"""
    
    def __init__(self, n_bins=6, horizon=500):
        super().__init__('Acrobot-v1', n_bins=n_bins, horizon=horizon, gamma=0.99)


class LunarLanderTabular(TabularizedGymEnv):
    """LunarLander-v2 as tabular MDP (discrete actions)"""
    
    def __init__(self, n_bins=8, horizon=1000):
        super().__init__('LunarLander-v2', n_bins=n_bins, horizon=horizon, gamma=0.99)


# Simplified direct wrappers for small discrete environments
class FrozenLakeEnv:
    """FrozenLake-v1 - Already discrete, no discretization needed"""
    
    def __init__(self, map_size='4x4', is_slippery=True, horizon=100):
        """
        :param map_size: '4x4' or '8x8'
        :param is_slippery: If True, agent has probability of moving perpendicular
        """
        map_name = f"FrozenLake-v1"
        self.env = gym.make(map_name, map_name=map_size, is_slippery=is_slippery)
        
        self.nS = self.env.observation_space.n
        self.nA = self.env.action_space.n
        self.horizon = horizon
        self.gamma = 0.99
        
        # Build P from environment's P dictionary
        self._build_from_gym_p()
    
    def _build_from_gym_p(self):
        """Convert Gym's P format to our format"""
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        
        # FrozenLake provides P directly
        for s in range(self.nS):
            for a in range(self.nA):
                transitions = self.env.unwrapped.P[s][a]
                for prob, s_next, reward, done in transitions:
                    self.P[s][a].append((prob, s_next, reward, done))
        
        # Initial state distribution
        self.mu = np.zeros(self.nS)
        self.mu[0] = 1.0  # FrozenLake always starts at state 0
        
        print(f"✓ FrozenLake environment: {self.nS} states, {self.nA} actions")
    

    def get_uniform_policy(self):
        """Return uniform random policy"""
        policy_rep = {s: np.ones(self.nA) / self.nA for s in range(self.nS)}
        return TabularPolicy(policy_rep, self.nS, self.nA)
    
    def set_model(self, P):
        """Update transition model"""
        self.P = P


class TaxiEnv:
    """Taxi-v3 - Already discrete"""
    
    def __init__(self, horizon=200):
        self.env = gym.make('Taxi-v3')
        self.nS = self.env.observation_space.n
        self.nA = self.env.action_space.n
        self.horizon = horizon
        self.gamma = 0.99
        
        self._build_from_gym_p()
    
    def _build_from_gym_p(self):
        """Convert Gym's P format to our format"""
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        
        for s in range(self.nS):
            for a in range(self.nA):
                transitions = self.env.unwrapped.P[s][a]
                for prob, s_next, reward, done in transitions:
                    self.P[s][a].append((prob, s_next, reward, done))
        
        # Uniform initial state distribution
        self.mu = np.ones(self.nS) / self.nS
        
        print(f"✓ Taxi environment: {self.nS} states, {self.nA} actions")
    
    def get_uniform_policy(self):
        """Return uniform random policy"""
        policy_rep = {s: np.ones(self.nA) / self.nA for s in range(self.nS)}
        return TabularPolicy(policy_rep, self.nS, self.nA)
    
    def set_model(self, P):
        """Update transition model"""
        self.P = P


# Factory function
def create_gym_env(env_name, **kwargs):
    """
    Factory function to create Gymnasium environments
    
    :param env_name: Name of environment
    :param kwargs: Environment-specific parameters
    """
    env_map = {
        'frozen_lake_4x4': lambda: FrozenLakeEnv(map_size='4x4', **kwargs),
        'frozen_lake_8x8': lambda: FrozenLakeEnv(map_size='8x8', **kwargs),
        'taxi': lambda: TaxiEnv(**kwargs),
        'cartpole': lambda: CartPoleTabular(**kwargs),
        'mountain_car': lambda: MountainCarTabular(**kwargs),
        'acrobot': lambda: AcrobotTabular(**kwargs),
        'lunar_lander': lambda: LunarLanderTabular(**kwargs),
    }
    
    if env_name not in env_map:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(env_map.keys())}")
    
    return env_map[env_name]()