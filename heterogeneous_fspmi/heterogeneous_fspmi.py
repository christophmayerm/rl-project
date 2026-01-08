"""
Heterogeneous Federated Safe Policy-Model Iteration (H-FSPMI)

This implementation provides GENUINE federated learning benefits by:
1. Running agents in TRUE PARALLEL using multiprocessing
2. Each agent interacts with a DIFFERENT environment variant
3. Learns a robust policy that generalizes across environment variations

Environment heterogeneity options:
- Different dynamics (k parameter)
- Different failure probabilities (pfail)
- Different reward weights
- Different obstacle configurations (via hazard zones)

This addresses the core criticism: "Why federation if all agents see the same MDP?"
Answer: They DON'T. Each agent sees a different environment variant.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
import copy
import warnings


# ============================================================================
# ENVIRONMENT VARIANT CONFIGURATION
# ============================================================================

@dataclass
class EnvironmentVariant:
    """Configuration for a single environment variant."""
    variant_id: int
    track_file: str = "T1"
    k: float = 0.5                      # Dynamics interpolation [0,1]
    pfail: float = 0.0                  # Failure probability
    reward_weight: Optional[List[float]] = None  # [goal, offroad, zero_speed, low_speed, high_speed]
    hazard_positions: Optional[List[Tuple[int, int]]] = None  # (x,y) positions with extra penalty
    hazard_penalty: float = -0.5        # Reward penalty for hazard zones
    description: str = ""
    
    def __post_init__(self):
        if self.reward_weight is None:
            self.reward_weight = [1, 0, 0, 0, 0]  # Default: only goal matters


@dataclass
class HeterogeneousConfig:
    """Configuration for heterogeneous F-SPMI experiment."""
    variants: List[EnvironmentVariant]
    episodes_per_agent: int = 100
    n_iterations: int = 100
    gamma: float = 0.9
    horizon: int = 20
    aggregation_method: str = 'weighted'  # 'uniform', 'weighted', 'robust'
    use_parallel: bool = True
    n_workers: Optional[int] = None  # None = use all CPUs
    
    # Learning speed options
    update_mode: str = 'standard'  # 'standard', 'alternating', 'policy_only', 'model_only'
    target_policy_type: str = 'greedy'  # 'greedy' or 'softmax'
    softmax_temperature: float = 1.0  # Temperature for softmax target policy
    min_step_size: float = 0.0  # Minimum step size (0 = no floor, >0 = force progress)
    
    @classmethod
    def create_dynamics_variants(cls, track_file: str = "T1", 
                                  k_values: List[float] = [0.3, 0.5, 0.7],
                                  **kwargs) -> 'HeterogeneousConfig':
        """Create config with different dynamics parameters."""
        variants = [
            EnvironmentVariant(
                variant_id=i,
                track_file=track_file,
                k=k,
                description=f"k={k} ({'low-speed favored' if k < 0.5 else 'high-speed favored' if k > 0.5 else 'balanced'})"
            )
            for i, k in enumerate(k_values)
        ]
        return cls(variants=variants, **kwargs)
    
    @classmethod
    def create_robustness_variants(cls, track_file: str = "T1",
                                    pfail_values: List[float] = [0.0, 0.05, 0.1],
                                    **kwargs) -> 'HeterogeneousConfig':
        """Create config with different failure probabilities."""
        variants = [
            EnvironmentVariant(
                variant_id=i,
                track_file=track_file,
                pfail=pf,
                description=f"pfail={pf} ({'safe' if pf == 0 else 'risky'})"
            )
            for i, pf in enumerate(pfail_values)
        ]
        return cls(variants=variants, **kwargs)
    
    @classmethod
    def create_mixed_variants(cls, track_file: str = "T1", **kwargs) -> 'HeterogeneousConfig':
        """Create config with mixed heterogeneity (dynamics + failure)."""
        variants = [
            EnvironmentVariant(0, track_file, k=0.3, pfail=0.0, description="low-speed, safe"),
            EnvironmentVariant(1, track_file, k=0.5, pfail=0.0, description="balanced, safe"),
            EnvironmentVariant(2, track_file, k=0.7, pfail=0.0, description="high-speed, safe"),
            EnvironmentVariant(3, track_file, k=0.5, pfail=0.05, description="balanced, moderate risk"),
            EnvironmentVariant(4, track_file, k=0.5, pfail=0.1, description="balanced, high risk"),
        ]
        return cls(variants=variants, **kwargs)


# ============================================================================
# TRAJECTORY AND STATISTICS DATACLASSES  
# ============================================================================

@dataclass
class Trajectory:
    """A trajectory from environment interaction."""
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    
    def __len__(self):
        return len(self.states)


@dataclass
class LocalStatistics:
    """Statistics computed by a single agent."""
    agent_id: int
    variant_id: int
    Q_hat: np.ndarray
    Q_counts: np.ndarray
    V_hat: np.ndarray
    U_hat: np.ndarray
    d_mu_hat: np.ndarray
    delta_mu_hat: np.ndarray
    n_episodes: int
    n_samples: int
    avg_return: float
    variant_description: str = ""


@dataclass 
class GlobalStatistics:
    """Aggregated global statistics."""
    Q_global: np.ndarray
    U_global: np.ndarray
    d_mu_global: np.ndarray
    delta_mu_global: np.ndarray
    total_samples: int
    n_agents: int
    Q_counts: np.ndarray


# ============================================================================
# PARALLEL WORKER FUNCTIONS (must be at module level for pickling)
# ============================================================================

def _create_environment(variant: EnvironmentVariant, gamma: float, horizon: int):
    """Create environment from variant config. Called in worker process."""
    # Import here to avoid issues with multiprocessing
    import sys
    import os
    
    # Add parent directory to path if needed
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from envs.racetrack_simulator import RaceTrackConfigurableEnv
    
    env = RaceTrackConfigurableEnv(
        track_file=variant.track_file,
        initial_configuration=[variant.k, 1-variant.k, 0, 0],
        reward_weight=variant.reward_weight,
        pfail=variant.pfail,
        horizon=horizon
    )
    env.gamma = gamma
    
    # Apply hazard zones if specified
    if variant.hazard_positions:
        _apply_hazard_zones(env, variant.hazard_positions, variant.hazard_penalty)
    
    return env


def _apply_hazard_zones(env, hazard_positions: List[Tuple[int, int]], penalty: float):
    """Apply penalty to hazard zone states by modifying reward."""
    for (x, y) in hazard_positions:
        # Find all states at this position (all velocity combinations)
        for vx in env.vel:
            for vy in env.vel:
                try:
                    s = env._s_to_i(x, y, vx, vy)
                    env.R[s] += penalty
                except:
                    pass  # Position not valid


def _worker_collect_statistics(args) -> LocalStatistics:
    """
    Worker function for parallel data collection.
    
    This function runs in a separate process, creates its own environment,
    collects trajectories, and computes local statistics.
    """
    (variant_dict, policy_rep, model_rep, n_episodes, 
     gamma, horizon, nS, nA, seed) = args
    
    # Reconstruct variant from dict (for pickling)
    variant = EnvironmentVariant(**variant_dict)
    
    # Create environment in this process
    env = _create_environment(variant, gamma, horizon)
    
    # Set random seed for reproducibility
    rng = np.random.RandomState(seed)
    
    # Collect trajectories
    trajectories = []
    for ep in range(n_episodes):
        traj = _rollout(env, policy_rep, nA, horizon, rng)
        trajectories.append(traj)
    
    # Compute statistics
    Q_hat, Q_counts = _estimate_q_function(trajectories, gamma, nS, nA)
    V_hat, _ = _estimate_v_function(trajectories, gamma, nS)
    U_hat = _estimate_u_function(trajectories, gamma, nS, nA, V_hat)
    d_mu_hat = _estimate_d_mu(trajectories, gamma, nS, horizon)
    delta_mu_hat = _estimate_delta_mu(trajectories, gamma, nS, nA, horizon)
    
    # Compute average return
    returns = []
    for traj in trajectories:
        G = 0.0
        for t in reversed(range(len(traj))):
            G = traj.rewards[t] + gamma * G
        returns.append(G)
    avg_return = np.mean(returns)
    
    return LocalStatistics(
        agent_id=variant.variant_id,
        variant_id=variant.variant_id,
        Q_hat=Q_hat,
        Q_counts=Q_counts,
        V_hat=V_hat,
        U_hat=U_hat,
        d_mu_hat=d_mu_hat,
        delta_mu_hat=delta_mu_hat,
        n_episodes=n_episodes,
        n_samples=sum(len(t) for t in trajectories),
        avg_return=avg_return,
        variant_description=variant.description
    )


def _rollout(env, policy_rep: Dict, nA: int, horizon: int, rng) -> Trajectory:
    """Execute single episode."""
    states, actions, rewards, next_states = [], [], [], []
    
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    if hasattr(state, '__len__') and not isinstance(state, (int, np.integer)):
        state = int(state[0]) if len(state) > 0 else 0
    
    done = False
    t = 0
    
    while not done and t < horizon:
        state_int = int(state)
        policy_probs = policy_rep[state_int]
        action = rng.choice(nA, p=policy_probs)
        
        result = env._step(action)
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            next_state, reward = result[:2]
            done = t >= horizon - 1
        
        if isinstance(next_state, (list, np.ndarray)):
            next_state = int(next_state[0]) if len(next_state) > 0 else 0
        
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


def _estimate_q_function(trajectories: List[Trajectory], gamma: float, 
                         nS: int, nA: int) -> Tuple[np.ndarray, np.ndarray]:
    """First-visit Monte Carlo Q estimation."""
    nSA = nS * nA
    Q_sum = np.zeros(nSA)
    Q_count = np.zeros(nSA)
    
    for traj in trajectories:
        T = len(traj)
        if T == 0:
            continue
        
        G = 0.0
        returns_from_t = np.zeros(T)
        for t in reversed(range(T)):
            G = traj.rewards[t] + gamma * G
            returns_from_t[t] = G
        
        visited = set()
        for t in range(T):
            s, a = int(traj.states[t]), int(traj.actions[t])
            sa = s * nA + a
            if sa not in visited:
                visited.add(sa)
                Q_sum[sa] += returns_from_t[t]
                Q_count[sa] += 1
    
    Q_hat = np.divide(Q_sum, Q_count, out=np.zeros_like(Q_sum), where=Q_count > 0)
    return Q_hat, Q_count


def _estimate_v_function(trajectories: List[Trajectory], gamma: float,
                         nS: int) -> Tuple[np.ndarray, np.ndarray]:
    """First-visit Monte Carlo V estimation."""
    V_sum = np.zeros(nS)
    V_count = np.zeros(nS)
    
    for traj in trajectories:
        T = len(traj)
        if T == 0:
            continue
        
        G = 0.0
        returns_from_t = np.zeros(T)
        for t in reversed(range(T)):
            G = traj.rewards[t] + gamma * G
            returns_from_t[t] = G
        
        visited = set()
        for t in range(T):
            s = int(traj.states[t])
            if s not in visited:
                visited.add(s)
                V_sum[s] += returns_from_t[t]
                V_count[s] += 1
    
    V_hat = np.divide(V_sum, V_count, out=np.zeros_like(V_sum), where=V_count > 0)
    return V_hat, V_count


def _estimate_u_function(trajectories: List[Trajectory], gamma: float,
                         nS: int, nA: int, V_hat: np.ndarray) -> np.ndarray:
    """Estimate U(s,a,s') = R(s,a,s') + gamma * V(s')."""
    nSA = nS * nA
    R_sum = np.zeros((nSA, nS))
    R_count = np.zeros((nSA, nS))
    
    for traj in trajectories:
        for t in range(len(traj)):
            s, a = int(traj.states[t]), int(traj.actions[t])
            r, s_next = traj.rewards[t], int(traj.next_states[t])
            sa = s * nA + a
            R_sum[sa, s_next] += r
            R_count[sa, s_next] += 1
    
    R_hat = np.divide(R_sum, R_count, out=np.zeros_like(R_sum), where=R_count > 0)
    U_hat = R_hat + gamma * V_hat[np.newaxis, :]
    return U_hat


def _estimate_d_mu(trajectories: List[Trajectory], gamma: float,
                   nS: int, horizon: int) -> np.ndarray:
    """Estimate discounted state distribution."""
    d_mu = np.zeros(nS)
    total_weight = 0.0
    
    for traj in trajectories:
        for t in range(min(len(traj), horizon)):
            s = int(traj.states[t])
            weight = gamma ** t
            d_mu[s] += weight
            total_weight += weight
    
    if total_weight > 0:
        d_mu /= total_weight
        d_mu *= (1 - gamma) / (1 - gamma ** horizon + 1e-10)
    
    return d_mu


def _estimate_delta_mu(trajectories: List[Trajectory], gamma: float,
                       nS: int, nA: int, horizon: int) -> np.ndarray:
    """Estimate discounted state-action distribution."""
    nSA = nS * nA
    delta_mu = np.zeros(nSA)
    total_weight = 0.0
    
    for traj in trajectories:
        for t in range(min(len(traj), horizon)):
            s, a = int(traj.states[t]), int(traj.actions[t])
            sa = s * nA + a
            weight = gamma ** t
            delta_mu[sa] += weight
            total_weight += weight
    
    if total_weight > 0:
        delta_mu /= total_weight
        delta_mu *= (1 - gamma) / (1 - gamma ** horizon + 1e-10)
    
    return delta_mu


# ============================================================================
# MAIN HETEROGENEOUS F-SPMI CLASS
# ============================================================================

class HeterogeneousFSPMI:
    """
    Heterogeneous Federated Safe Policy-Model Iteration.
    
    Key differences from standard F-SPMI:
    1. Each agent interacts with a DIFFERENT environment variant
    2. True PARALLEL execution using multiprocessing
    3. Learns a policy that is ROBUST across environment variations
    
    This provides genuine federation benefits:
    - Wall-clock speedup (parallel execution)
    - Robustness (heterogeneous environments)
    - Generalization (policy works across variants)
    """
    
    def __init__(self, config: HeterogeneousConfig):
        self.config = config
        self.n_agents = len(config.variants)
        
        # Create reference environment to get dimensions
        ref_variant = config.variants[0]
        self.ref_env = _create_environment(ref_variant, config.gamma, config.horizon)
        self.nS = self.ref_env.nS
        self.nA = self.ref_env.nA
        self.gamma = config.gamma
        self.horizon = config.horizon
        
        # Compute delta_q for bound
        self.delta_q = (1.0 - self.gamma ** self.horizon) / (1 - self.gamma)
        
        # Logger
        self.logger = FSPMILogger()
        
    def run(self, initial_policy, initial_model, 
            policy_chooser=None, model_chooser=None,
            verbose: bool = True) -> Tuple[Any, Any]:
        """
        Run heterogeneous F-SPMI.
        
        Args:
            initial_policy: Starting TabularPolicy
            initial_model: Starting TabularModel
            policy_chooser: Optional target selection strategy
            model_chooser: Optional target selection strategy
            verbose: Print progress
            
        Returns:
            (final_policy, final_model)
        """
        self.logger.reset()
        
        policy = initial_policy
        model = initial_model
        
        # Set model on reference environment
        self.ref_env.set_model(model.get_rep())
        
        n_workers = self.config.n_workers or min(cpu_count(), self.n_agents)
        
        for iteration in range(self.config.n_iterations):
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}/{self.config.n_iterations}")
            
            # ================================================
            # Step 1: Parallel data collection across variants
            # ================================================
            local_stats_list = self._collect_parallel(policy, iteration)
            
            # ================================================
            # Step 2: Aggregate statistics
            # ================================================
            global_stats = self._aggregate_statistics(local_stats_list)
            
            # ================================================
            # Step 3: Target selection
            # ================================================
            target_policy = self._choose_target_policy(
                policy, global_stats.d_mu_global, global_stats.Q_global,
                policy_chooser
            )
            target_model = self._choose_target_model(
                model, global_stats.delta_mu_global, global_stats.U_global,
                model_chooser
            )
            
            # ================================================
            # Step 4: Compute safe update
            # ================================================
            debug_this_iter = verbose and iteration == 0
            if debug_this_iter:
                print(f"  [DEBUG] Q range: [{global_stats.Q_global.min():.4f}, {global_stats.Q_global.max():.4f}]")
                print(f"  [DEBUG] d_mu non-zero entries: {np.sum(global_stats.d_mu_global > 1e-6)}/{self.nS}")
            
            alpha_star, beta_star, bound_value, p_adv, m_adv = self._compute_safe_update(
                policy, model, target_policy, target_model, global_stats, 
                debug=debug_this_iter, iteration=iteration
            )
            
            # ================================================
            # Step 5: Update policy and model
            # ================================================
            if alpha_star > 0:
                policy = self._policy_combination(alpha_star, target_policy, policy)
            
            if beta_star > 0:
                model = self._model_combination(beta_star, target_model, model)
                self.ref_env.set_model(model.get_rep())
            
            # ================================================
            # Step 6: Logging
            # ================================================
            avg_return = np.mean([s.avg_return for s in local_stats_list])
            total_samples = sum(s.n_samples for s in local_stats_list)
            
            # Compute true performance on reference environment
            from utils.tabular import TabularReward
            import utils.evaluator as evaluator
            reward = TabularReward(self.ref_env.P, self.nS, self.nA)
            true_perf = evaluator.compute_performance(
                self.ref_env.mu, reward, policy, model, 
                self.gamma, self.horizon, self.nS, self.nA
            )
            
            self.logger.log(
                iteration=iteration,
                performance=avg_return,
                true_performance=true_perf,
                alpha=alpha_star,
                beta=beta_star,
                bound=bound_value,
                p_adv=p_adv,
                m_adv=m_adv,
                samples=total_samples,
                avg_return=avg_return
            )
            
            if verbose and iteration % 10 == 0:
                variant_returns = {s.variant_description: s.avg_return 
                                   for s in local_stats_list}
                print(f"  True perf: {true_perf:.4f}, α*: {alpha_star:.4f}, β*: {beta_star:.4f}")
                print(f"  Policy adv: {p_adv:.6f}, Model adv: {m_adv:.6f}")
                print(f"  Per-variant returns: {variant_returns}")
            
            # Check convergence
            convergence = 1e-6 / (1 - self.gamma)
            if p_adv <= convergence and m_adv <= convergence:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        return policy, model
    
    def _collect_parallel(self, policy, iteration: int) -> List[LocalStatistics]:
        """Collect data from all agents in parallel."""
        policy_rep = policy.get_rep()
        model_rep = self.ref_env.P  # Each worker will set its own k
        
        # Prepare arguments for each worker
        args_list = [
            (
                {  # Convert variant to dict for pickling
                    'variant_id': v.variant_id,
                    'track_file': v.track_file,
                    'k': v.k,
                    'pfail': v.pfail,
                    'reward_weight': v.reward_weight,
                    'hazard_positions': v.hazard_positions,
                    'hazard_penalty': v.hazard_penalty,
                    'description': v.description,
                },
                policy_rep,
                model_rep,
                self.config.episodes_per_agent,
                self.gamma,
                self.horizon,
                self.nS,
                self.nA,
                iteration * 1000 + v.variant_id  # Unique seed per iteration/agent
            )
            for v in self.config.variants
        ]
        
        if self.config.use_parallel and self.n_agents > 1:
            n_workers = self.config.n_workers or min(cpu_count(), self.n_agents)
            with Pool(n_workers) as pool:
                local_stats_list = pool.map(_worker_collect_statistics, args_list)
        else:
            # Sequential fallback
            local_stats_list = [_worker_collect_statistics(args) for args in args_list]
        
        return local_stats_list
    
    def _aggregate_statistics(self, local_stats: List[LocalStatistics]) -> GlobalStatistics:
        """Aggregate local statistics using weighted averaging."""
        N = len(local_stats)
        
        # Extract arrays
        local_Q = [s.Q_hat for s in local_stats]
        local_Q_counts = [s.Q_counts for s in local_stats]
        local_U = [s.U_hat for s in local_stats]
        local_d_mu = [s.d_mu_hat for s in local_stats]
        local_delta_mu = [s.delta_mu_hat for s in local_stats]
        local_samples = [s.n_samples for s in local_stats]
        
        total_samples = sum(local_samples)
        
        if self.config.aggregation_method == 'weighted':
            # Weighted by per-(s,a) counts for Q
            Q_weighted_sum = np.zeros_like(local_Q[0])
            Q_count_total = np.zeros_like(local_Q_counts[0])
            
            for j in range(N):
                Q_weighted_sum += local_Q[j] * local_Q_counts[j]
                Q_count_total += local_Q_counts[j]
            
            Q_global = np.divide(Q_weighted_sum, Q_count_total,
                                 out=np.zeros_like(Q_weighted_sum),
                                 where=Q_count_total > 0)
            
            # Weighted by sample counts for distributions
            weights = np.array(local_samples) / (total_samples + 1e-10)
            d_mu_global = sum(w * d for w, d in zip(weights, local_d_mu))
            delta_mu_global = sum(w * d for w, d in zip(weights, local_delta_mu))
            U_global = sum(w * u for w, u in zip(weights, local_U))
            
        else:  # uniform
            Q_global = np.mean(local_Q, axis=0)
            d_mu_global = np.mean(local_d_mu, axis=0)
            delta_mu_global = np.mean(local_delta_mu, axis=0)
            U_global = np.mean(local_U, axis=0)
            Q_count_total = np.sum(local_Q_counts, axis=0)
        
        return GlobalStatistics(
            Q_global=Q_global,
            U_global=U_global,
            d_mu_global=d_mu_global,
            delta_mu_global=delta_mu_global,
            total_samples=total_samples,
            n_agents=N,
            Q_counts=Q_count_total
        )
    
    def _choose_target_policy(self, current, d_mu, Q, chooser=None):
        """Choose target policy (greedy or softmax w.r.t. Q)."""
        if chooser is not None:
            _, _, _, target = chooser.choose(current, d_mu, Q)
            return target
        
        if self.config.target_policy_type == 'softmax':
            # Softmax policy - smoother, smaller distances
            temp = self.config.softmax_temperature
            target_rep = {s: np.zeros(self.nA) for s in range(self.nS)}
            for s in range(self.nS):
                q_array = Q[s * self.nA:(s + 1) * self.nA]
                # Subtract max for numerical stability
                q_shifted = q_array - np.max(q_array)
                exp_q = np.exp(q_shifted / temp)
                target_rep[s] = exp_q / (np.sum(exp_q) + 1e-10)
        else:
            # Greedy policy
            target_rep = {s: np.zeros(self.nA) for s in range(self.nS)}
            for s in range(self.nS):
                q_array = Q[s * self.nA:(s + 1) * self.nA]
                max_val = np.max(q_array)
                greedy_actions = np.where(np.abs(q_array - max_val) < 1e-10)[0]
                target_rep[s][greedy_actions] = 1.0 / len(greedy_actions)
        
        from utils.tabular import TabularPolicy
        return TabularPolicy(target_rep, self.nS, self.nA)
    
    def _choose_target_model(self, current, delta_mu, U, chooser=None):
        """Choose target model (greedy w.r.t. U)."""
        if chooser is not None:
            _, _, _, target = chooser.choose(current, delta_mu, U)
            return target
        
        # Greedy model
        greedy_rep = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for s in range(self.nS):
            for a in range(self.nA):
                sa = s * self.nA + a
                u_array = U[sa]
                max_val = np.max(u_array)
                greedy_states = np.where(np.abs(u_array - max_val) < 1e-10)[0]
                probs = np.zeros(self.nS)
                probs[greedy_states] = 1.0 / len(greedy_states)
                greedy_rep[s][a] = list(zip(probs, range(self.nS)))
        
        from utils.tabular import TabularModel
        return TabularModel(greedy_rep, self.nS, self.nA)
    
    def _compute_safe_update(self, policy, model, target_policy, target_model,
                             global_stats: GlobalStatistics, debug: bool = False,
                             iteration: int = 0) -> Tuple[float, float, float, float, float]:
        """Compute optimal step sizes using decoupled bound."""
        Q = global_stats.Q_global
        U = global_stats.U_global
        d_mu = global_stats.d_mu_global
        delta_mu = global_stats.delta_mu_global
        
        # Get matrices
        pi = policy.get_matrix()
        pi_target = target_policy.get_matrix()
        P = model.get_matrix()
        P_target = target_model.get_matrix()
        
        # Compute advantages
        A_pi = np.dot(pi_target - pi, Q)
        p_adv = np.dot(d_mu, A_pi)
        
        A_P = np.sum((P_target - P) * U, axis=1)
        m_adv = np.dot(delta_mu, A_P)
        
        # Compute distances
        pi_diff = np.abs(pi_target - pi)
        p_dist_per_s = np.sum(pi_diff, axis=1)
        p_dist_sup = np.max(p_dist_per_s)
        p_dist_mean = np.dot(d_mu, p_dist_per_s)
        
        P_diff = np.abs(P_target - P)
        m_dist_per_sa = np.sum(P_diff, axis=1)
        m_dist_sup = np.max(m_dist_per_sa)
        m_dist_mean = np.dot(delta_mu, m_dist_per_sa)
        
        # Compute optimal step sizes (from SPMI paper Table 1)
        eps = 1e-24
        gamma = self.gamma
        
        # α₀: optimal α when β=0
        alpha0_raw = ((1 - gamma) * p_adv) / (self.delta_q * gamma * p_dist_sup * p_dist_mean + eps)
        
        # β₀: optimal β when α=0  
        beta0_raw = ((1 - gamma) * m_adv) / (self.delta_q * (gamma ** 2) * m_dist_sup * m_dist_mean + eps)
        
        # α₁: optimal α when β=1
        alpha1_raw = alpha0_raw - 0.5 * (
            m_dist_mean / (p_dist_mean + eps) + m_dist_sup / (p_dist_sup + eps)
        )
        
        # β₁: optimal β when α=1
        beta1_raw = beta0_raw - 0.5 / gamma * (
            p_dist_mean / (m_dist_mean + eps) + p_dist_sup / (m_dist_sup + eps)
        )
        
        if debug:
            print(f"  [STEP DEBUG] p_adv={p_adv:.6f}, m_adv={m_adv:.6f}")
            print(f"  [STEP DEBUG] p_dist_sup={p_dist_sup:.4f}, p_dist_mean={p_dist_mean:.6f}")
            print(f"  [STEP DEBUG] m_dist_sup={m_dist_sup:.4f}, m_dist_mean={m_dist_mean:.6f}")
            print(f"  [STEP DEBUG] delta_q={self.delta_q:.4f}, gamma={gamma:.2f}")
            print(f"  [STEP DEBUG] alpha0_raw={alpha0_raw:.6f}, beta0_raw={beta0_raw:.6f}")
            print(f"  [STEP DEBUG] alpha1_raw={alpha1_raw:.6f}, beta1_raw={beta1_raw:.6f}")
        
        # Clip to [0, 1]
        alpha0 = np.clip(alpha0_raw, 0.0, 1.0)
        alpha1 = np.clip(alpha1_raw, 0.0, 1.0)
        beta0 = np.clip(beta0_raw, 0.0, 1.0)
        beta1 = np.clip(beta1_raw, 0.0, 1.0)
        
        # Evaluate bound at candidates
        def bound(a, b):
            advantage = a * p_adv + b * m_adv
            penalty = (gamma / (1 - gamma) * self.delta_q / 2) * (
                (a ** 2) * p_dist_sup * p_dist_mean +
                gamma * (b ** 2) * m_dist_sup * m_dist_mean +
                a * b * p_dist_sup * m_dist_mean +
                a * b * p_dist_mean * m_dist_sup
            )
            return advantage - penalty
        
        # Select candidate set based on update mode
        update_mode = self.config.update_mode
        
        if update_mode == 'alternating':
            # Alternate between policy-only and model-only updates
            if iteration % 2 == 0:
                candidates = [(alpha0, 0.0)]  # Policy update only
            else:
                candidates = [(0.0, beta0)]   # Model update only
        elif update_mode == 'policy_only':
            candidates = [(alpha0, 0.0)]
        elif update_mode == 'model_only':
            candidates = [(0.0, beta0)]
        else:  # standard
            # Full candidate set from SPMI paper
            candidates = [
                (alpha0, 0.0),      # Only policy update
                (0.0, beta0),       # Only model update
                (alpha1, 1.0),      # Policy update with full model update
                (1.0, beta1),       # Full policy update with model update
            ]
        
        best_bound = float('-inf')
        alpha_star, beta_star = 0.0, 0.0
        
        if debug:
            print(f"  [BOUND DEBUG] Update mode: {update_mode}, Evaluating candidates:")
        
        for a, b in candidates:
            if a >= 0 and b >= 0:  # Only valid candidates
                b_val = bound(a, b)
                if debug:
                    print(f"    (α={a:.6f}, β={b:.6f}) -> bound={b_val:.8f}")
                if b_val > best_bound:
                    best_bound = b_val
                    alpha_star = a
                    beta_star = b
        
        # Apply minimum step size if configured
        min_step = self.config.min_step_size
        if min_step > 0:
            if p_adv > 0 and alpha_star < min_step and alpha_star > 0:
                alpha_star = min_step
            if m_adv > 0 and beta_star < min_step and beta_star > 0:
                beta_star = min_step
            # For alternating mode, apply min step to the active component
            if update_mode == 'alternating':
                if iteration % 2 == 0 and p_adv > 0:
                    alpha_star = max(alpha_star, min_step)
                elif iteration % 2 == 1 and m_adv > 0:
                    beta_star = max(beta_star, min_step)
        
        return alpha_star, beta_star, best_bound, p_adv, m_adv
    
    def _policy_combination(self, alpha, target, current):
        """Convex combination of policies."""
        from utils.tabular_operations import policy_convex_combination
        return policy_convex_combination(target, current, alpha)
    
    def _model_combination(self, beta, target, current):
        """Convex combination of models."""
        from utils.tabular_operations import model_convex_combination
        return model_convex_combination(self.ref_env.P, target, current, beta)


# ============================================================================
# LOGGER
# ============================================================================

@dataclass
class FSPMILogger:
    """Logger for algorithm metrics."""
    iterations: List[int] = field(default_factory=list)
    true_performances: List[float] = field(default_factory=list)
    performances: List[float] = field(default_factory=list)
    alphas: List[float] = field(default_factory=list)
    betas: List[float] = field(default_factory=list)
    bounds: List[float] = field(default_factory=list)
    policy_advantages: List[float] = field(default_factory=list)
    model_advantages: List[float] = field(default_factory=list)
    total_samples: List[int] = field(default_factory=list)
    avg_returns: List[float] = field(default_factory=list)
    
    def reset(self):
        self.__init__()
    
    def log(self, iteration, performance, true_performance, alpha, beta,
            bound, p_adv, m_adv, samples, avg_return):
        self.iterations.append(iteration)
        self.performances.append(performance)
        self.true_performances.append(true_performance)
        self.alphas.append(alpha)
        self.betas.append(beta)
        self.bounds.append(bound)
        self.policy_advantages.append(p_adv)
        self.model_advantages.append(m_adv)
        self.total_samples.append(samples)
        self.avg_returns.append(avg_return)
    
    def save(self, filepath: str):
        """Save to CSV."""
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'performance_mc', 'performance_true', 'alpha', 'beta',
                'bound', 'policy_advantage', 'model_advantage', 'total_samples', 'avg_return'
            ])
            for i in range(len(self.iterations)):
                writer.writerow([
                    self.iterations[i], self.performances[i], self.true_performances[i],
                    self.alphas[i], self.betas[i], self.bounds[i],
                    self.policy_advantages[i], self.model_advantages[i],
                    self.total_samples[i], self.avg_returns[i]
                ])


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_dynamics_heterogeneous_fspmi(track_file: str = "T1",
                                         k_values: List[float] = [0.3, 0.5, 0.7],
                                         episodes_per_agent: int = 100,
                                         n_iterations: int = 100,
                                         **kwargs) -> HeterogeneousFSPMI:
    """Create H-FSPMI with different dynamics parameters."""
    config = HeterogeneousConfig.create_dynamics_variants(
        track_file=track_file,
        k_values=k_values,
        episodes_per_agent=episodes_per_agent,
        n_iterations=n_iterations,
        **kwargs
    )
    return HeterogeneousFSPMI(config)


def create_robustness_heterogeneous_fspmi(track_file: str = "T1",
                                           pfail_values: List[float] = [0.0, 0.05, 0.1],
                                           episodes_per_agent: int = 100,
                                           n_iterations: int = 100,
                                           **kwargs) -> HeterogeneousFSPMI:
    """Create H-FSPMI with different failure probabilities."""
    config = HeterogeneousConfig.create_robustness_variants(
        track_file=track_file,
        pfail_values=pfail_values,
        episodes_per_agent=episodes_per_agent,
        n_iterations=n_iterations,
        **kwargs
    )
    return HeterogeneousFSPMI(config)