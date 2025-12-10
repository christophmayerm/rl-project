import numpy as np
from utils import evaluator
from utils.tabular import TabularPolicy
from algorithm.curriculum_scheduler import CurriculumScheduler

from utils.tabular_operations import policy_sup_tv_distance, policy_mean_tv_distance


class PolicyChooser(object):

    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA

    def choose(self, policy, d_mu_pi, Q):
        pass

class GreedyPolicyChooser(PolicyChooser):

    def choose(self, policy, d_mu_pi, Q):
        # GREEDY POLICY COMPUTATION
        target_policy_rep = self.greedy_policy(Q)
        # instantiation of a target policy object
        target_policy = TabularPolicy(target_policy_rep, self.nS, self.nA)

        # EXPECTED RELATIVE ADVANTAGE COMPUTATION
        er_advantage = evaluator.compute_policy_er_advantage(target_policy, policy, Q, d_mu_pi)

        # POLICY DISTANCE COMPUTATIONS
        distance_sup = policy_sup_tv_distance(target_policy, policy)
        distance_mean = policy_mean_tv_distance(target_policy, policy, d_mu_pi)

        return er_advantage, distance_sup, distance_mean, target_policy

    def greedy_policy(self, Q, tol=0.0):

        greedy_policy_rep = {s: [] for s in range(self.nS)}
        # loop to give maximum probability to the greedy action,
        # if more than one is greedy then uniform on the greedy actions
        for s in range(self.nS):
            q_array = Q[s*self.nA : (s+1)*self.nA]
            probabilities = np.zeros(self.nA)

            # uniform if more than one greedy
            max = np.max(q_array)
            a = np.argwhere(np.abs(q_array - max) <= tol).flatten()
            probabilities[a] = 1. / len(a)
            greedy_policy_rep[s] = probabilities

        return greedy_policy_rep

class SetPolicyChooser(PolicyChooser):

    def __init__(self, policy_set, nS, nA):
        self.policy_set = policy_set
        self.n_policies = len(self.policy_set)
        super(SetPolicyChooser, self).__init__(nS, nA)

    def choose(self, policy, d_mu_pi, Q):
        er_advantages = np.zeros(self.n_policies)

        for i in range(self.n_policies):
            er_advantages[i] = evaluator.compute_policy_er_advantage(self.policy_set[i], policy, Q, d_mu_pi)

        index = np.argmax(er_advantages)
        target_policy = self.policy_set[index]
        er_advantage = er_advantages[index]

        # POLICY DISTANCE COMPUTATIONS
        distance_sup = policy_sup_tv_distance(target_policy, policy)
        distance_mean = policy_mean_tv_distance(target_policy, policy, d_mu_pi)

        return er_advantage, distance_sup, distance_mean, target_policy

class DoNotCreateTransitionsGreedyPolicyChooser(PolicyChooser):
    def __init__(self, original_policy, nS, nA):
        self.original_policy = original_policy
        super(DoNotCreateTransitionsGreedyPolicyChooser, self).__init__(nS, nA)

    def choose(self, policy, d_mu_pi, Q):
        target_policy_rep = self.dnct_greedy_policy(Q)
        target_policy = TabularPolicy(target_policy_rep, self.nS, self.nA)

        # EXPECTED RELATIVE ADVANTAGE COMPUTATION
        er_advantage = evaluator.compute_policy_er_advantage(target_policy,
                                                             policy, Q, d_mu_pi)

        # POLICY DISTANCE COMPUTATIONS
        distance_sup = policy_sup_tv_distance(target_policy, policy)
        distance_mean = policy_mean_tv_distance(target_policy, policy, d_mu_pi)

        return er_advantage, distance_sup, distance_mean, target_policy

    def dnct_greedy_policy(self, Q, tol=0.0):
        greedy_policy_rep = {s: [] for s in range(self.nS)}

        for s in range(self.nS):

            q_array = np.copy(Q[s*self.nA:(s+1)*self.nA])

            li = self.original_policy[s]
            for i, elem in enumerate(li):
                if elem == 0.:
                    q_array[i] = -np.inf

            probabilities = np.zeros(self.nA)

            # uniform if more than one greedy
            max = np.max(q_array)
            s1 = np.argwhere(np.abs(q_array - max) <= tol).flatten()
            probabilities[s1] = 1. / len(s1)
            greedy_policy_rep[s] = probabilities

        return greedy_policy_rep



class SAPMIPolicyChooser(PolicyChooser):
    """
    Safe Adversarial Policy-Model Iteration (SA-PMI) Policy Chooser.
    
    Creates robust policy targets by blending:
    - Nominal greedy policy (exploits current Q-function)
    - Conservative/robust policy (hedges against uncertainty)
    
    The curriculum controls the blend, gradually introducing robustness constraints.
    """
    
    def __init__(
        self, 
        nS: int,
        nA: int,
        base_chooser: PolicyChooser = None,
        curriculum_scheduler: CurriculumScheduler = None,
        robustness_temperature: float = 0.3,
        entropy_bonus: float = 0.1
    ):
        super().__init__(nS, nA)
        
        if base_chooser is None:
            base_chooser = GreedyPolicyChooser(nS, nA)
        
        self.base_chooser = base_chooser
        self.curriculum = curriculum_scheduler
        self.robustness_temperature = robustness_temperature
        self.entropy_bonus = entropy_bonus
        self.iteration = 0
        
        # Track metrics for analysis
        self.robustness_weights_history = []
        self.nominal_advantages_history = []
        self.robust_advantages_history = []
        self.blended_advantages_history = []
        
    def choose(self, policy, d_mu, Q):
        """Choose policy target with curriculum-based robustness."""
        self.iteration += 1
        
        # Get curriculum weight λ(t) ∈ [0,1]
        if self.curriculum is not None:
            robustness_weight = self.curriculum.get_adversarial_weight(self.iteration)
        else:
            robustness_weight = 0.0
        
        self.robustness_weights_history.append(robustness_weight)
        
        # Step 1: Get nominal greedy target from base chooser
        er_adv_nom, dist_sup_nom, dist_mean_nom, target_policy_nom = \
            self.base_chooser.choose(policy, d_mu, Q)
        
        self.nominal_advantages_history.append(er_adv_nom)
        
        if robustness_weight < 1e-6:
            # Pure nominal learning (curriculum start)
            self.robust_advantages_history.append(er_adv_nom)
            self.blended_advantages_history.append(er_adv_nom)
            return er_adv_nom, dist_sup_nom, dist_mean_nom, target_policy_nom
        
        # Step 2: Compute robust/conservative policy
        target_policy_robust = self._robust_policy(Q, policy, d_mu)
        
        # Compute robust advantage
        er_adv_robust = evaluator.compute_policy_er_advantage(
            target_policy_robust, policy, Q, d_mu
        )
        self.robust_advantages_history.append(er_adv_robust)
        
        # Step 3: Blend nominal and robust targets via curriculum
        target_policy_blended = self._blend_policies(
            target_policy_nom, 
            target_policy_robust, 
            robustness_weight
        )
        
        # Step 4: Recompute metrics for blended target
        er_adv = evaluator.compute_policy_er_advantage(
            target_policy_blended, policy, Q, d_mu
        )
        dist_sup = policy_sup_tv_distance(target_policy_blended, policy)
        dist_mean = policy_mean_tv_distance(target_policy_blended, policy, d_mu)
        
        self.blended_advantages_history.append(er_adv)
        
        return er_adv, dist_sup, dist_mean, target_policy_blended
    
    def _robust_policy(self, Q, current_policy, d_mu):
        """Create robust policy using softmax + entropy regularization."""
        robust_rep = {}
        
        for s in range(self.nS):
            sa_start = s * self.nA
            sa_end = (s + 1) * self.nA
            q_values = Q[sa_start:sa_end]
            
            # Softmax with temperature
            q_scaled = q_values / (self.robustness_temperature + 1e-10)
            q_scaled = q_scaled - np.max(q_scaled)  # numerical stability
            exp_q = np.exp(q_scaled)
            softmax_probs = exp_q / (np.sum(exp_q) + 1e-10)
            
            # Add entropy regularization (blend with uniform)
            uniform_probs = np.ones(self.nA) / self.nA
            robust_probs = (1 - self.entropy_bonus) * softmax_probs + \
                          self.entropy_bonus * uniform_probs
            
            # Ensure normalization
            robust_probs = robust_probs / np.sum(robust_probs)
            
            # CRITICAL: Store as 1D numpy array
            robust_rep[s] = np.array(robust_probs, dtype=np.float64)
        
        return TabularPolicy(robust_rep, self.nS, self.nA)
    
    def _blend_policies(self, policy_nominal, policy_robust, weight):
        """Convex combination: weight·π_robust + (1-weight)·π_nominal"""
        blended_rep = {}
        
        # Get the raw representations (dictionaries)
        nom_rep = policy_nominal.get_rep()
        rob_rep = policy_robust.get_rep()
        
        for s in range(self.nS):
            # Get action probabilities for state s
            nom_probs = np.array(nom_rep[s], dtype=np.float64)
            rob_probs = np.array(rob_rep[s], dtype=np.float64)
            
            # Blend the probabilities
            blended_probs = weight * rob_probs + (1.0 - weight) * nom_probs
            
            # Ensure normalization
            blended_probs = blended_probs / (np.sum(blended_probs) + 1e-10)
            
            # CRITICAL: Store as 1D numpy array to match TabularPolicy expectations
            blended_rep[s] = np.array(blended_probs, dtype=np.float64)
        
        return TabularPolicy(blended_rep, self.nS, self.nA)
    
    def save_sapmi_policy_metrics(self, filepath_prefix):
        """Save SA-PMI policy-specific metrics for analysis."""
        np.savetxt(
            f"{filepath_prefix}_robustness_weights.csv", 
            np.array(self.robustness_weights_history), 
            delimiter=";"
        )
        np.savetxt(
            f"{filepath_prefix}_nominal_policy_advantages.csv", 
            np.array(self.nominal_advantages_history), 
            delimiter=";"
        )
        np.savetxt(
            f"{filepath_prefix}_robust_policy_advantages.csv", 
            np.array(self.robust_advantages_history), 
            delimiter=";"
        )
        np.savetxt(
            f"{filepath_prefix}_blended_policy_advantages.csv", 
            np.array(self.blended_advantages_history), 
            delimiter=";"
        )