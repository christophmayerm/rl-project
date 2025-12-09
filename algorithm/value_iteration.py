"""
Value Iteration and Policy Iteration for Configurable MDPs
Simple, fast, and actually converges on FrozenLake 8x8
"""

import numpy as np
from utils.tabular import TabularPolicy, TabularReward
from utils import evaluator


class ValueIteration:
    """
    Standard Value Iteration - no safety constraints
    Optimal for known environment models
    """
    
    def __init__(self, conf_mdp, eps=1e-6, max_iter=10000):
        self.mdp = conf_mdp
        self.gamma = conf_mdp.gamma
        self.horizon = conf_mdp.horizon
        self.eps = eps
        self.max_iter = max_iter
        self.nS = conf_mdp.nS
        self.nA = conf_mdp.nA
        
        # For compatibility with experiment runner
        self.logger = type('obj', (object,), {
            'evaluations': [],
            'iteration': 0,
            'close': lambda *args: None  # FIX: Accept any arguments
            })()
        
    def solve(self, initial_model=None):
        """Run value iteration to find optimal policy"""
        if initial_model is None:
            model = self.mdp.get_model()
        else:
            model = initial_model
            
        V = np.zeros(self.nS)
        
        env_name = getattr(self.mdp.env, 'spec', None)
        env_name = env_name.id if env_name and hasattr(env_name, 'id') else "Unknown"
        
        print(f"\n{'='*70}")
        print(f"🎯 Value Iteration on {env_name}")
        print(f"{'='*70}")
        print(f"States: {self.nS}, Actions: {self.nA}")
        print(f"Gamma: {self.gamma}, Horizon: {self.horizon}")
        print(f"Convergence threshold: {self.eps}")
        print(f"{'='*70}\n")
        
        iteration = 0
        converged = False
        model_matrix = model.get_matrix()
        
        # Bellman iteration
        while not converged and iteration < self.max_iter:
            V_old = V.copy()
            
            for s in range(self.nS):
                q_values = np.zeros(self.nA)
                
                for a in range(self.nA):
                    # Q(s,a) = sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
                    # Get transition probabilities and compute expected return
                    transitions = self.mdp.P[s][a]
                    q_value = 0.0
                    
                    for prob, next_state, reward, done in transitions:
                        q_value += prob * (reward + self.gamma * V_old[next_state])
                    
                    q_values[a] = q_value
                
                V[s] = np.max(q_values)
            
            delta = np.max(np.abs(V - V_old))
            converged = delta < self.eps
            
            # Evaluate performance every 100 iterations
            if iteration % 100 == 0:
                # Create deterministic policy from current V
                policy_rep = {}
                for s in range(self.nS):
                    q_values = np.zeros(self.nA)
                    for a in range(self.nA):
                        transitions = self.mdp.P[s][a]
                        q_value = 0.0
                        for prob, next_state, reward, done in transitions:
                            q_value += prob * (reward + self.gamma * V[next_state])
                        q_values[a] = q_value
                    best_action = np.argmax(q_values)
                    
                    # Create probability vector: all zeros except 1.0 for best action
                    action_probs = np.zeros(self.nA)
                    action_probs[best_action] = 1.0
                    policy_rep[s] = action_probs
                
                policy = TabularPolicy(policy_rep, self.nS, self.nA)
                
                # Compute performance using evaluator
                from utils.tabular import TabularReward
                reward_obj = TabularReward(self.mdp.env.P, self.nS, self.nA)
                perf = evaluator.compute_performance(
                    self.mdp.mu, reward_obj, policy, model,
                    self.gamma, self.horizon, self.nS, self.nA
                )
                self.logger.evaluations.append(perf)
                print(f"Iteration {iteration:5d}: delta={delta:.6f}, V_max={np.max(V):.6f}, Performance={perf:.6f}")
            
            iteration += 1
        
        print(f"\n✓ Converged in {iteration} iterations (delta={delta:.6f})\n")
        
        # Extract greedy policy from final value function
        policy_rep = {}
        for s in range(self.nS):
            q_values = np.zeros(self.nA)
            
            for a in range(self.nA):
                transitions = self.mdp.P[s][a]
                q_value = 0.0
                for prob, next_state, reward, done in transitions:
                    q_value += prob * (reward + self.gamma * V[next_state])
                q_values[a] = q_value
            
            # Greedy action selection - create probability vector
            best_action = np.argmax(q_values)
            action_probs = np.zeros(self.nA)
            action_probs[best_action] = 1.0
            policy_rep[s] = action_probs
        
        policy = TabularPolicy(policy_rep, self.nS, self.nA)
        
        # Store final iteration count
        self.logger.iteration = iteration
        
        return policy, V, iteration


class PolicyIteration:
    """
    Policy Iteration - often faster than VI for small state spaces
    """
    
    def __init__(self, conf_mdp, eps=1e-6, max_iter=1000):
        self.mdp = conf_mdp
        self.gamma = conf_mdp.gamma
        self.horizon = conf_mdp.horizon
        self.eps = eps
        self.max_iter = max_iter
        self.nS = conf_mdp.nS
        self.nA = conf_mdp.nA
        
        # For compatibility with experiment runner
        self.logger = type('obj', (object,), {
            'evaluations': [],
            'iteration': 0,
            'close': lambda *args: None
        })()
        
    def policy_evaluation(self, policy, model, max_eval_iter=1000):
        """Evaluate a policy to get its value function"""
        V = np.zeros(self.nS)
        policy_matrix = policy.get_matrix()
        
        for _ in range(max_eval_iter):
            V_old = V.copy()
            
            for s in range(self.nS):
                v = 0
                for a in range(self.nA):
                    prob_action = policy_matrix[s, s*self.nA + a]
                    if prob_action > 0:
                        # Get transitions for this action
                        transitions = self.mdp.P[s][a]
                        
                        for prob_trans, next_state, reward, done in transitions:
                            v += prob_action * prob_trans * (reward + self.gamma * V_old[next_state])
                
                V[s] = v
            
            if np.max(np.abs(V - V_old)) < self.eps:
                break
                
        return V
    
    def policy_improvement(self, V, model):
        """Improve policy greedily with respect to value function"""
        policy_rep = {}
        
        for s in range(self.nS):
            q_values = np.zeros(self.nA)
            
            for a in range(self.nA):
                transitions = self.mdp.P[s][a]
                q_value = 0.0
                
                for prob, next_state, reward, done in transitions:
                    q_value += prob * (reward + self.gamma * V[next_state])
                
                q_values[a] = q_value
            
            best_action = np.argmax(q_values)
            action_probs = np.zeros(self.nA)
            action_probs[best_action] = 1.0
            policy_rep[s] = action_probs
        
        return TabularPolicy(policy_rep, self.nS, self.nA)
    
    def solve(self, initial_policy=None, initial_model=None):
        """
        Run policy iteration to find optimal policy
        
        Returns:
            policy: Optimal policy
            V: Value function
            iterations: Number of iterations until convergence
        """
        if initial_model is None:
            model = self.mdp.get_model()
        else:
            model = initial_model
        
        if initial_policy is None:
            # Start with uniform random policy
            policy_rep = {s: np.ones(self.nA) / self.nA for s in range(self.nS)}
            policy = TabularPolicy(policy_rep, self.nS, self.nA)
        else:
            policy = initial_policy
        
        env_name = getattr(self.mdp.env, 'spec', None)
        env_name = env_name.id if env_name and hasattr(env_name, 'id') else "Unknown"
        
        print(f"\n{'='*70}")
        print(f"🎯 Policy Iteration on {env_name}")
        print(f"{'='*70}")
        print(f"States: {self.nS}, Actions: {self.nA}")
        print(f"{'='*70}\n")
        
        iteration = 0
        policy_stable = False
        
        while not policy_stable and iteration < self.max_iter:
            # Policy Evaluation
            V = self.policy_evaluation(policy, model)
            
            # Policy Improvement
            new_policy = self.policy_improvement(V, model)
            
            # Check if policy changed
            policy_stable = np.allclose(policy.get_matrix(), new_policy.get_matrix())
            
            policy = new_policy
            iteration += 1
            
            # Evaluate performance
            from utils.tabular import TabularReward
            reward_obj = TabularReward(self.mdp.env.P, self.nS, self.nA)
            J = evaluator.compute_performance(
                self.mdp.mu, reward_obj, policy, model, 
                self.gamma, self.horizon, self.nS, self.nA
            )
            self.logger.evaluations.append(J)
            print(f"Iteration {iteration:3d}: Performance={J:.6f}")
        
        print(f"\n✓ Converged in {iteration} iterations\n")
        
        # Store final iteration count
        self.logger.iteration = iteration
        
        return policy, V, iteration