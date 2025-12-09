"""
Tabular Q-Learning for Configurable MDPs
Model-free learning that actually works on FrozenLake
"""

import numpy as np
from utils.tabular import TabularPolicy
from collections import defaultdict


class QLearning:
    """
    Tabular Q-Learning with epsilon-greedy exploration
    """
    
    def __init__(
        self, 
        conf_mdp,
        alpha=0.1,           # Learning rate
        gamma=0.99,          # Discount factor
        epsilon=1.0,         # START WITH 100% EXPLORATION
        epsilon_decay=0.999,  # SLOWER DECAY
        epsilon_min=0.01,
        episodes=50000,
        eval_frequency=1000,
        optimistic_init=1.0  # NEW: Optimistic Q-values
    ):
        self.mdp = conf_mdp
        self.env = conf_mdp.env
        self.nS = conf_mdp.nS
        self.nA = conf_mdp.nA
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.eval_frequency = eval_frequency
        self.optimistic_init = optimistic_init
        
        # Q-table with OPTIMISTIC INITIALIZATION
        self.Q = defaultdict(lambda: np.ones(self.nA) * optimistic_init)
        
        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0  # Track successful episodes
        
        # For compatibility with experiment runner
        self.logger = type('obj', (object,), {
            'evaluations': [],
            'iteration': 0,
            'close': lambda *args: None
        })()
        
        # Access underlying gym environment
        if hasattr(self.env, 'env'):
            self.gym_env = self.env.env  # Wrapped environment
        else:
            self.gym_env = self.env  # Already gym environment
        
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.nA)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        # Q(s,a) ← Q(s,a) + α[target - Q(s,a)]
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
    
    def evaluate_policy(self, n_episodes=100):
        """Evaluate current greedy policy"""
        total_rewards = []
        
        for _ in range(n_episodes):
            state, _ = self.gym_env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 200:
                action = self.get_action(state, training=False)  # Greedy
                next_state, reward, terminated, truncated, _ = self.gym_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                steps += 1
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def train(self):
        """Train Q-learning agent"""
        env_name = getattr(self.gym_env, 'spec', None)
        env_name = env_name.id if env_name and hasattr(env_name, 'id') else "Unknown"
        
        print(f"\n{'='*70}")
        print(f"🎯 Q-Learning on {env_name}")
        print(f"{'='*70}")
        print(f"Episodes: {self.episodes}")
        print(f"Alpha: {self.alpha}, Gamma: {self.gamma}")
        print(f"Epsilon: {self.epsilon} → {self.epsilon_min} (decay: {self.epsilon_decay})")
        print(f"Optimistic Init: {self.optimistic_init}")
        print(f"{'='*70}\n")
        
        # NEW: Evaluate initial random policy
        print("Evaluating initial random policy...")
        initial_perf = self.evaluate_random_policy(n_episodes=100)
        self.logger.evaluations.append(initial_perf)
        print(f"Initial random performance: {initial_perf:.4f}\n")
        
        first_success = None
        
        for episode in range(self.episodes):
            state, _ = self.gym_env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 200:
                action = self.get_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.gym_env.step(action)
                done = terminated or truncated
                
                self.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                steps += 1
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            
            # Track first success
            if episode_reward > 0:
                self.success_count += 1
                if first_success is None:
                    first_success = episode
                    print(f"🎉 First success at episode {episode}!")
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Evaluate periodically
            if (episode + 1) % self.eval_frequency == 0:
                eval_perf = self.evaluate_policy(n_episodes=100)
                self.logger.evaluations.append(eval_perf)
                
                avg_reward = np.mean(self.episode_rewards[-self.eval_frequency:])
                success_rate = self.success_count / self.eval_frequency
                self.success_count = 0  # Reset counter
                
                print(f"Episode {episode+1:5d}/{self.episodes}: "
                    f"Avg Reward: {avg_reward:.4f}, "
                    f"Success Rate: {success_rate:.4f}, "
                    f"Eval Performance: {eval_perf:.4f}, "
                    f"Epsilon: {self.epsilon:.4f}")
        
        # Final evaluation
        final_perf = self.evaluate_policy(n_episodes=1000)
        self.logger.evaluations.append(final_perf)
        print(f"\n{'='*70}")
        print(f"✓ Training Complete")
        print(f"{'='*70}")
        if first_success:
            print(f"First success: Episode {first_success}")
        print(f"Final Performance (1000 episodes): {final_perf:.4f}")
        print(f"{'='*70}\n")
        
        # Store final episode as "iteration"
        self.logger.iteration = self.episodes
        
        return self.get_policy()

    def evaluate_random_policy(self, n_episodes=100):
        """Evaluate a purely random policy (before any learning)"""
        total_rewards = []
        
        for _ in range(n_episodes):
            state, _ = self.gym_env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 200:
                action = np.random.randint(0, self.nA)  # Pure random
                next_state, reward, terminated, truncated, _ = self.gym_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                steps += 1
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def get_policy(self):
        """Extract deterministic policy from Q-table"""
        policy_rep = {}
        
        for s in range(self.nS):
            best_action = np.argmax(self.Q[s])
            action_probs = np.zeros(self.nA)
            action_probs[best_action] = 1.0
            policy_rep[s] = action_probs
        
        return TabularPolicy(policy_rep, self.nS, self.nA)