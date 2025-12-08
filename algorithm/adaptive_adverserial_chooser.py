"""
Adaptive Adversarial Model Chooser
Works with AdaptiveCurriculumScheduler to implement safe adversarial training
"""

import numpy as np
from typing import Tuple, Optional
from utils.tabular import TabularModel, policy_sup_tv_distance, policy_mean_tv_distance


class AdaptiveAdversarialChooser:
    """
    Adversarial model chooser with adaptive budget integration
    
    Chooses worst-case model within dynamically adjusted budget constraint
    """
    
    def __init__(
        self,
        nS: int,
        nA: int,
        curriculum_scheduler: Optional[object] = None,
    ):
        """
        Initialize adaptive adversarial chooser
        
        Args:
            nS: Number of states
            nA: Number of actions
            curriculum_scheduler: AdaptiveCurriculumScheduler instance (optional)
        """
        self.nS = nS
        self.nA = nA
        self.curriculum_scheduler = curriculum_scheduler
        
        # Track adversarial model history for analysis
        self.adversarial_history = []
        
    def choose_adversarial(
        self,
        model_coop: TabularModel,
        nominal_model: TabularModel,
        delta_mu: np.ndarray,
        U: np.ndarray,
        iteration: int,
        logger: Optional[object] = None,
        override_budget: Optional[float] = None,
    ) -> Tuple[TabularModel, float, float, float, float]:
        """
        Choose adversarial model that minimizes expected return within budget
        
        Args:
            model_coop: Current cooperative model M_c
            nominal_model: Nominal/baseline model M_0
            delta_mu: State-action distribution
            U: Advantage function U(s,a,s')
            iteration: Current iteration
            logger: Logger with performance history
            override_budget: Override curriculum budget (for adaptive scheduling)
            
        Returns:
            (adversarial_model, er_disadvantage, dist_sup, dist_mean, budget_used)
        """
        
        # Get adaptive budget if scheduler is available
        if override_budget is not None:
            budget = override_budget
        elif self.curriculum_scheduler is not None and logger is not None:
            # This is handled externally in sa_pmi_v2
            raise ValueError("Use override_budget when curriculum_scheduler is set")
        else:
            # Fallback: no adversarial perturbation
            budget = 0.0
        
        # If budget is zero, return cooperative model (no adversarial perturbation)
        if budget <= 1e-10:
            return (
                model_coop,
                0.0,  # No disadvantage
                0.0,  # No distance
                0.0,  # No distance
                0.0   # No budget used
            )
        
        # ADVERSARIAL MODEL SELECTION
        # For each (s,a), perturb transition distribution to minimize expected U
        
        model_adv_rep = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        total_perturbation = 0.0
        num_sa_pairs = 0
        
        for s in range(self.nS):
            for a in range(self.nA):
                # Get current cooperative transitions
                coop_transitions = model_coop.get_rep()[s][a]
                
                # Get nominal transitions for reference
                nominal_transitions = nominal_model.get_rep()[s][a]
                
                # Build transition probability distributions
                coop_probs = np.zeros(self.nS)
                nominal_probs = np.zeros(self.nS)
                
                for prob, s_next in coop_transitions:
                    coop_probs[s_next] += prob
                
                for prob, s_next in nominal_transitions:
                    nominal_probs[s_next] += prob
                
                # Normalize (should already be normalized, but ensure)
                coop_probs = coop_probs / (coop_probs.sum() + 1e-10)
                nominal_probs = nominal_probs / (nominal_probs.sum() + 1e-10)
                
                # Compute advantage for each next state
                advantages = U[s, a, :]
                
                # GREEDY ADVERSARIAL PERTURBATION within budget
                # Shift probability mass from high-advantage to low-advantage states
                
                # Available budget for this (s,a) pair
                sa_budget = budget
                
                # Current TV distance from nominal
                current_tv = 0.5 * np.abs(coop_probs - nominal_probs).sum()
                
                # Maximum allowed perturbation
                max_perturbation = sa_budget - current_tv
                
                if max_perturbation > 1e-10:
                    # Perturb: move probability from high-advantage to low-advantage
                    adv_probs = coop_probs.copy()
                    
                    # Sort states by advantage
                    sorted_indices = np.argsort(advantages)
                    
                    # Transfer probability from high to low advantage states
                    remaining_budget = max_perturbation
                    
                    # Iterate from highest to lowest advantage
                    for i in range(self.nS - 1, -1, -1):
                        if remaining_budget <= 0:
                            break
                        
                        s_high = sorted_indices[i]
                        if adv_probs[s_high] < 1e-6:
                            continue
                        
                        # Find lowest advantage state with room for probability
                        for j in range(self.nS):
                            s_low = sorted_indices[j]
                            if s_low == s_high:
                                continue
                            
                            # Amount to transfer
                            transfer = min(
                                adv_probs[s_high] * 0.5,  # Don't take all from high
                                remaining_budget,
                                1.0 - adv_probs[s_low]  # Don't exceed 1.0 for low
                            )
                            
                            if transfer > 1e-6:
                                adv_probs[s_high] -= transfer
                                adv_probs[s_low] += transfer
                                remaining_budget -= transfer
                                break
                    
                    # Ensure probabilities are valid
                    adv_probs = np.maximum(adv_probs, 0)
                    adv_probs = adv_probs / (adv_probs.sum() + 1e-10)
                    
                    # Compute actual perturbation
                    perturbation = 0.5 * np.abs(adv_probs - nominal_probs).sum()
                    total_perturbation += perturbation
                    num_sa_pairs += 1
                    
                else:
                    # Already at or beyond budget limit
                    adv_probs = coop_probs.copy()
                
                # Build transition list for this (s,a)
                adv_transitions = [(float(p), s_next) for s_next, p in enumerate(adv_probs) if p > 1e-10]
                model_adv_rep[s][a] = adv_transitions
        
        # Create adversarial model
        model_adv = TabularModel(model_adv_rep, self.nS, self.nA)
        
        # Compute metrics
        er_disadvantage = self.compute_model_er_disadvantage(model_adv, model_coop, U, delta_mu)
        dist_sup = policy_sup_tv_distance(model_adv, nominal_model)
        dist_mean = policy_mean_tv_distance(model_adv, nominal_model, delta_mu)
        
        # Average perturbation per (s,a) pair
        avg_perturbation = total_perturbation / max(num_sa_pairs, 1)
        
        # Store history
        self.adversarial_history.append({
            'iteration': iteration,
            'budget': budget,
            'er_disadvantage': er_disadvantage,
            'dist_sup': dist_sup,
            'dist_mean': dist_mean,
            'avg_perturbation': avg_perturbation
        })
        
        return model_adv, er_disadvantage, dist_sup, dist_mean, budget
    
    def compute_model_er_disadvantage(
        self,
        model_adv: TabularModel,
        model_coop: TabularModel,
        U: np.ndarray,
        delta_mu: np.ndarray
    ) -> float:
        """
        Compute expected disadvantage of adversarial model vs cooperative model
        
        This is the negative of the advantage (how much worse is model_adv)
        """
        disadvantage = 0.0
        
        for s in range(self.nS):
            for a in range(self.nA):
                # Get transition distributions
                adv_trans = {s_next: prob for prob, s_next in model_adv.get_rep()[s][a]}
                coop_trans = {s_next: prob for prob, s_next in model_coop.get_rep()[s][a]}
                
                # Expected U under each model
                expected_U_adv = sum(adv_trans.get(s_next, 0) * U[s, a, s_next] 
                                    for s_next in range(self.nS))
                expected_U_coop = sum(coop_trans.get(s_next, 0) * U[s, a, s_next]
                                     for s_next in range(self.nS))
                
                # Disadvantage weighted by visitation
                disadvantage += delta_mu[s, a] * (expected_U_coop - expected_U_adv)
        
        return max(disadvantage, 0.0)  # Ensure non-negative
    
    def get_adversarial_summary(self) -> dict:
        """Get summary statistics of adversarial model history"""
        if not self.adversarial_history:
            return {}
        
        budgets = [h['budget'] for h in self.adversarial_history]
        disadvantages = [h['er_disadvantage'] for h in self.adversarial_history]
        dist_means = [h['dist_mean'] for h in self.adversarial_history]
        
        return {
            'total_adversarial_iterations': len(self.adversarial_history),
            'avg_budget': np.mean(budgets),
            'avg_disadvantage': np.mean(disadvantages),
            'avg_dist_mean': np.mean(dist_means),
            'max_dist_mean': max(dist_means) if dist_means else 0.0,
        }