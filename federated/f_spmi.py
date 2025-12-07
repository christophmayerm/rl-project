"""
Federated Safe Policy-Model Iteration (F-SPMI)

Main algorithm implementing the federated extension of SPMI for
Configurable Markov Decision Processes (Conf-MDPs).

This operationalizes the "sample-based version of SPMI" envisioned by
Metelli et al. (2018) using a federated architecture for robust, parallel evaluation.

Algorithm Overview:
1. N parallel agents execute the current global policy-model pair (P_k, π_k)
2. Each agent gathers local statistics (Q̂_j, δ̂_μ,j)
3. Server aggregates via federated averaging
4. Server computes the Decoupled Bound and safe update
5. Global policy and model are updated with monotonic improvement guarantees
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import copy

import utils.evaluator as evaluator
from .agent import FederatedAgent, LocalStatistics
from .server import FederatedServer, GlobalStatistics, UpdateResult
from utils.tabular import TabularPolicy, TabularModel
from utils.tabular_operations import policy_convex_combination, model_convex_combination
from utils.tabular import TabularReward


@dataclass
class FSPMILogger:
    """Logger for F-SPMI algorithm metrics"""
    iterations: List[int] = field(default_factory=list)
    true_performances: List[float] = field(default_factory=list)
    performances: List[float] = field(default_factory=list)  # MC estimate
    alphas: List[float] = field(default_factory=list)
    betas: List[float] = field(default_factory=list)
    bounds: List[float] = field(default_factory=list)
    policy_advantages: List[float] = field(default_factory=list)
    model_advantages: List[float] = field(default_factory=list)
    total_samples: List[int] = field(default_factory=list)
    avg_returns: List[float] = field(default_factory=list)

    def reset(self):
        self.__init__()

    def log(
        self,
        iteration: int,
        performance: float,
        true_performance: float,
        alpha: float,
        beta: float,
        bound: float,
        p_adv: float,
        m_adv: float,
        samples: int,
        avg_return: float
    ):
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
        """Save log to CSV file"""
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'performance_mc', 'performance_true', 'alpha', 'beta', 'bound',
                'policy_advantage', 'model_advantage', 'total_samples', 'avg_return'
            ])
            for i in range(len(self.iterations)):
                writer.writerow([
                    self.iterations[i],
                    self.performances[i],
                    self.true_performances[i],
                    self.alphas[i],
                    self.betas[i],
                    self.bounds[i],
                    self.policy_advantages[i],
                    self.model_advantages[i],
                    self.total_samples[i],
                    self.avg_returns[i]
                ])


class FSPMI:
    """
    Federated Safe Policy-Model Iteration.

    This class orchestrates the federated learning process for Conf-MDPs,
    coordinating between multiple agents and a central server.

    The algorithm maintains the monotonic performance improvement guarantees
    of SPMI while leveraging parallel exploration to accelerate convergence.
    """

    def __init__(
        self,
        conf_mdp,
        n_agents: int,
        episodes_per_round: int,
        eps: float = 0.0,
        gamma: Optional[float] = None,
        horizon: Optional[int] = None,
        delta_q: Optional[float] = None,
        max_rounds: int = 1000,
        policy_chooser=None,
        model_chooser=None,
        aggregation_method: str = 'weighted',
        persistent: bool = True,
        verbose: bool = True
    ):
        """
        Initialize F-SPMI algorithm.

        Args:
            conf_mdp: Configurable MDP environment
            n_agents: Number of parallel agents
            episodes_per_round: Episodes per agent per round
            eps: Convergence threshold
            gamma: Discount factor (if None, uses mdp.gamma)
            horizon: Episode horizon (if None, uses mdp.horizon)
            delta_q: ΔQ value for bound (if None, computed from γ and H)
            max_rounds: Maximum number of communication rounds
            policy_chooser: Policy target selection strategy
            model_chooser: Model target selection strategy
            aggregation_method: 'uniform', 'weighted', or 'robust'
            persistent: Whether to use persistent target selection
            verbose: Whether to print progress
        """
        self.mdp = conf_mdp
        self.nS = conf_mdp.nS
        self.nA = conf_mdp.nA
        self.n_agents = n_agents
        self.episodes_per_round = episodes_per_round
        self.eps = eps
        self.max_rounds = max_rounds
        self.persistent = persistent
        self.verbose = verbose

        # Use MDP parameters if not specified
        self.gamma = gamma if gamma is not None else conf_mdp.gamma
        self.horizon = horizon if horizon is not None else conf_mdp.horizon

        # Compute delta_q if not specified
        if delta_q is None:
            self.delta_q = (1.0 - self.gamma ** self.horizon) / (1 - self.gamma)
        else:
            self.delta_q = delta_q

        # Initialize agents
        self.agents = [
            FederatedAgent(
                agent_id=i,
                nS=self.nS,
                nA=self.nA,
                gamma=self.gamma,
                horizon=self.horizon,
                seed=i * 1000  # Different seeds for diversity
            )
            for i in range(n_agents)
        ]

        # Initialize server
        self.server = FederatedServer(
            nS=self.nS,
            nA=self.nA,
            gamma=self.gamma,
            horizon=self.horizon,
            delta_q=self.delta_q,
            aggregation_method=aggregation_method
        )

        # Target choosers
        self.policy_chooser = policy_chooser
        self.model_chooser = model_chooser

        # Logger
        self.logger = FSPMILogger()

    def run(
        self,
        initial_policy: TabularPolicy,
        initial_model: TabularModel
    ) -> Tuple[TabularPolicy, TabularModel]:
        """
        Run the F-SPMI algorithm.

        Args:
            initial_policy: Starting policy
            initial_model: Starting model

        Returns:
            (final_policy, final_model) tuple
        """
        self.logger.reset()

        policy = initial_policy
        model = initial_model
        target_policy_old = None
        target_model_old = None

        # Convergence threshold
        convergence = self.eps / (1 - self.gamma)
        mu = self.mdp.mu

        for round_k in range(self.max_rounds):
            if self.verbose and round_k % 10 == 0:
                print(f"Round {round_k}/{self.max_rounds}")

            # ============================================
            # Step 1: Parallel data collection by agents
            # ============================================
            local_stats_list = []
            for agent in self.agents:
                stats = agent.collect_and_compute(
                    self.mdp, policy, model, self.episodes_per_round
                )
                local_stats_list.append(stats)

            # ============================================
            # Step 2: Server aggregation
            # ============================================
            global_stats = self.server.aggregate_statistics(local_stats_list)

            # ============================================
            # Step 3: Target selection
            # ============================================
            target_policy = self._choose_target_policy(
                policy, global_stats.d_mu_global, global_stats.Q_global
            )
            target_model = self._choose_target_model(
                model, global_stats.delta_mu_global, global_stats.U_global
            )

            # Build candidate lists (for persistent choice)
            target_policies = [target_policy]
            target_models = [target_model]

            if self.persistent and target_policy_old is not None:
                if not self._policy_equiv(target_policy, target_policy_old):
                    if not self._policy_equiv(policy, target_policy_old):
                        target_policies.append(target_policy_old)

            if self.persistent and target_model_old is not None:
                if not self._model_equiv(target_model, target_model_old):
                    if not self._model_equiv(model, target_model_old):
                        target_models.append(target_model_old)

            # ============================================
            # Step 4: Compute safe update
            # ============================================
            update_result, best_target_policy, best_target_model = \
                self.server.compute_safe_update_with_multiple_targets(
                    policy, model,
                    target_policies, target_models,
                    global_stats
                )

            alpha_star = update_result.alpha_star
            beta_star = update_result.beta_star

            # ============================================
            # Step 5: Update policy and model
            # ============================================
            if alpha_star > 0:
                policy = self._policy_combination(alpha_star, best_target_policy, policy)

            if beta_star > 0:
                model = self._model_combination(beta_star, best_target_model, model)
                self.mdp.set_model(model.get_rep())

            # ============================================
            # Step 6: Logging
            # ============================================
            avg_return = np.mean([s.avg_return for s in local_stats_list])
            total_samples = sum(s.n_samples for s in local_stats_list)

            # Estimate performance from average returns and compute true performance
            performance_mc = avg_return
            reward = TabularReward(self.mdp.P, self.nS, self.nA)
            performance_true = evaluator.compute_performance(
                mu, reward, policy, model, self.gamma, self.horizon, self.nS, self.nA
            )

            self.logger.log(
                iteration=round_k,
                performance=performance_mc,
                true_performance=performance_true,
                alpha=alpha_star,
                beta=beta_star,
                bound=update_result.bound_value,
                p_adv=update_result.policy_advantage,
                m_adv=update_result.model_advantage,
                samples=total_samples,
                avg_return=avg_return
            )

            if self.verbose and round_k % 10 == 0:
                print(f"  Performance (MC/true): {performance_mc:.4f}/{performance_true:.4f}, α*: {alpha_star:.4f}, β*: {beta_star:.4f}")

            # Update old targets for persistent choice
            target_policy_old = best_target_policy
            target_model_old = best_target_model

            # ============================================
            # Step 7: Check convergence
            # ============================================
            p_adv = update_result.policy_advantage
            m_adv = update_result.model_advantage

            if p_adv <= convergence and m_adv <= convergence:
                if self.verbose:
                    print(f"Converged at round {round_k}")
                break

        return policy, model

    def _choose_target_policy(
        self,
        current_policy: TabularPolicy,
        d_mu: np.ndarray,
        Q: np.ndarray
    ) -> TabularPolicy:
        """
        Choose target policy using the policy chooser.

        If no chooser is provided, uses greedy policy selection.
        """
        if self.policy_chooser is not None:
            _, _, _, target = self.policy_chooser.choose(current_policy, d_mu, Q)
            return target
        else:
            # Greedy policy selection
            return self._greedy_policy(Q)

    def _choose_target_model(
        self,
        current_model: TabularModel,
        delta_mu: np.ndarray,
        U: np.ndarray
    ) -> TabularModel:
        """
        Choose target model using the model chooser.

        If no chooser is provided, uses greedy model selection.
        """
        if self.model_chooser is not None:
            _, _, _, target = self.model_chooser.choose(current_model, delta_mu, U)
            return target
        else:
            # Greedy model selection
            return self._greedy_model(U)

    def _greedy_policy(self, Q: np.ndarray) -> TabularPolicy:
        """Compute greedy policy with respect to Q-function"""
        greedy_rep = {s: np.zeros(self.nA) for s in range(self.nS)}

        for s in range(self.nS):
            q_array = Q[s * self.nA:(s + 1) * self.nA]
            max_val = np.max(q_array)
            greedy_actions = np.where(np.abs(q_array - max_val) < 1e-10)[0]
            greedy_rep[s][greedy_actions] = 1.0 / len(greedy_actions)

        return TabularPolicy(greedy_rep, self.nS, self.nA)

    def _greedy_model(self, U: np.ndarray) -> TabularModel:
        """Compute greedy model with respect to U-function"""
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

        return TabularModel(greedy_rep, self.nS, self.nA)

    def _policy_combination(
        self,
        alpha: float,
        target: TabularPolicy,
        current: TabularPolicy
    ) -> TabularPolicy:
        """Convex combination of policies: α·target + (1-α)·current"""
        return policy_convex_combination(target, current, alpha)

    def _model_combination(
        self,
        beta: float,
        target: TabularModel,
        current: TabularModel
    ) -> TabularModel:
        """Convex combination of models: β·target + (1-β)·current"""
        return model_convex_combination(self.mdp.P, target, current, beta)

    def _policy_equiv(self, p1: TabularPolicy, p2: TabularPolicy) -> bool:
        """Check if two policies are equivalent"""
        return np.array_equal(p1.get_matrix(), p2.get_matrix())

    def _model_equiv(self, m1: TabularModel, m2: TabularModel) -> bool:
        """Check if two models are equivalent"""
        return np.array_equal(m1.get_matrix(), m2.get_matrix())


class FSPMIVariants:
    """
    Factory class for F-SPMI algorithm variants.

    Provides implementations analogous to SPMI variants:
    - F-SPMI: Full federated SPMI
    - F-SPMI-sup: Using looser bound (sup distances)
    - F-SPMI-alt: Alternated policy/model updates
    """

    @staticmethod
    def create_fspmi(conf_mdp, n_agents, episodes_per_round, **kwargs) -> FSPMI:
        """Create standard F-SPMI instance"""
        return FSPMI(conf_mdp, n_agents, episodes_per_round, **kwargs)

    @staticmethod
    def create_fspmi_robust(conf_mdp, n_agents, episodes_per_round, **kwargs) -> FSPMI:
        """Create F-SPMI with robust aggregation"""
        return FSPMI(
            conf_mdp, n_agents, episodes_per_round,
            aggregation_method='robust',
            **kwargs
        )

    @staticmethod
    def create_fspmi_uniform(conf_mdp, n_agents, episodes_per_round, **kwargs) -> FSPMI:
        """Create F-SPMI with uniform aggregation"""
        return FSPMI(
            conf_mdp, n_agents, episodes_per_round,
            aggregation_method='uniform',
            **kwargs
        )
