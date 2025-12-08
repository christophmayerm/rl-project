import numpy as np
from utils.tabular import TabularModel
from utils.tabular_operations import model_sup_tv_distance, model_mean_tv_distance
from utils.evaluator import compute_model_er_advantage


class AdversarialModelChooser(object):
    """
    Chooses adversarial model configurations to challenge the agent
    within a curriculum-scheduled budget constraint.
    """
    
    def __init__(self, nS, nA, budget_schedule='linear', B_min=0.01, B_max=0.3, K_warmup=50):
        """
        :param nS: number of states
        :param nA: number of actions
        :param budget_schedule: 'linear', 'exponential', 'smooth_exponential', or 'step'
        :param B_min: minimum adversarial budget (for curriculum warmup)
        :param B_max: maximum adversarial budget
        :param K_warmup: iterations to reach full adversarial strength
        """
        self.nS = nS
        self.nA = nA
        self.budget_schedule = budget_schedule
        self.B_min = B_min
        self.B_max = B_max
        self.K_warmup = K_warmup
        self.current_iteration = 0
        
    def get_budget(self, iteration=None):
        """Compute curriculum budget B(k)"""
        k = iteration if iteration is not None else self.current_iteration
        
        if self.budget_schedule == 'linear':
            progress = min(1.0, k / self.K_warmup)
            budget = self.B_min + (self.B_max - self.B_min) * progress
            
        elif self.budget_schedule == 'exponential':
            # IMPROVED: Smoother exponential that reaches B_max earlier
            # Use progress^2 to create a curve that accelerates in the middle
            progress = min(1.0, k / self.K_warmup)
            # Quadratic interpolation creates smoother transition
            budget = self.B_min + (self.B_max - self.B_min) * (progress ** 2)
            
        elif self.budget_schedule == 'smooth_exponential':
            # ALTERNATIVE: Sigmoid-like curve for very smooth transitions
            progress = min(1.0, k / self.K_warmup)
            # Maps [0,1] to smoother curve using tanh
            # This reaches ~95% of B_max at 80% progress
            alpha = 5.0  # Controls steepness (higher = steeper)
            smooth_progress = 0.5 * (1 + np.tanh(alpha * (progress - 0.5)))
            budget = self.B_min + (self.B_max - self.B_min) * smooth_progress
            
        elif self.budget_schedule == 'sqrt':
            # NEW: Square root schedule - faster initial growth, slower later
            # Good for building robustness early
            progress = min(1.0, k / self.K_warmup)
            budget = self.B_min + (self.B_max - self.B_min) * np.sqrt(progress)
            
        elif self.budget_schedule == 'cosine':
            # NEW: Cosine annealing schedule (smooth S-curve)
            progress = min(1.0, k / self.K_warmup)
            cosine_progress = 0.5 * (1 - np.cos(np.pi * progress))
            budget = self.B_min + (self.B_max - self.B_min) * cosine_progress
            
        elif self.budget_schedule == 'step':
            budget = self.B_max if k >= self.K_warmup else self.B_min
            
        else:
            raise ValueError(f"Unknown budget schedule: {self.budget_schedule}")
            
        return budget
    
    def choose_adversarial(self, model_coop, model_nominal, delta_mu, U, iteration=None):
        """
        Choose adversarial model that minimizes expected relative advantage
        within budget constraint.
        
        :param model_coop: current cooperative model M_c
        :param model_nominal: nominal/baseline model M_nominal
        :param delta_mu: discounted state-action distribution
        :param U: state-action-nextstate value function (nSA x nS)
        :param iteration: current iteration (for curriculum)
        :return: (adversarial_model, er_disadvantage, dist_sup, dist_mean, budget)
        """
        # Get current budget from curriculum
        budget = self.get_budget(iteration)
        
        # Early exit if budget is essentially zero
        if budget < 1e-10:
            # Return cooperative model as-is (no adversarial perturbation)
            er_disadvantage = 0.0
            dist_sup = 0.0
            dist_mean = 0.0
            self.current_iteration += 1
            return model_coop, er_disadvantage, dist_sup, dist_mean, budget
        
        # Get model representation
        model_coop_rep = model_coop.get_rep()
        model_coop_matrix = model_coop.get_matrix()
        
        # Create adversarial model representation
        model_adv_rep = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        
        # For each state-action, perturb towards worst next state
        for s in range(self.nS):
            for a in range(self.nA):
                sa = s * self.nA + a
                
                # Get current transition probabilities
                P_current = model_coop_matrix[sa, :]
                
                # Get value of each next state (from U matrix)
                # U[sa, s'] represents the value of transitioning to s' from (s,a)
                state_values = U[sa, :]
                
                # Find worst next state (minimum value)
                worst_state = np.argmin(state_values)
                
                # Create adversarial distribution
                P_adv = P_current.copy()
                
                # Calculate maximum perturbation allowed by budget
                # Budget constrains the L1 distance: ||P_adv - P_current||_1 <= 2*budget
                max_shift = min(budget, 1.0 - P_adv[worst_state])
                
                if max_shift > 1e-10:
                    # Shift probability mass towards worst state
                    remaining_mass = np.sum(P_adv) - P_adv[worst_state]
                    
                    if remaining_mass > 1e-10:
                        # Proportionally reduce other states
                        scale_factor = max(0.0, 1.0 - max_shift / remaining_mass)
                        for s_p in range(self.nS):
                            if s_p != worst_state:
                                P_adv[s_p] *= scale_factor
                        P_adv[worst_state] += max_shift
                
                # Ensure valid probability distribution
                P_adv = np.clip(P_adv, 0, 1)
                if P_adv.sum() > 0:
                    P_adv /= P_adv.sum()
                else:
                    P_adv = P_current.copy()  # Fallback to current
                
                # FIXED: Only include non-zero probabilities (match initial model format)
                model_adv_rep[s][a] = [(P_adv[s_next], s_next) 
                                       for s_next in range(self.nS) 
                                       if P_adv[s_next] > 1e-10]
                
                # Ensure we have at least one transition
                if len(model_adv_rep[s][a]) == 0:
                    # Fallback: use cooperative model's transitions
                    model_adv_rep[s][a] = model_coop_rep[s][a]
        
        # Create adversarial model object
        model_adv = TabularModel(model_adv_rep, self.nS, self.nA)
        
        # Compute expected relative disadvantage (negative advantage)
        # Adversarial model REDUCES performance, so we expect negative advantage
        er_advantage = compute_model_er_advantage(model_adv, model_coop, U, delta_mu)
        er_disadvantage = -er_advantage  # Make it positive for interpretability
        
        # Compute dissimilarities
        dist_sup = model_sup_tv_distance(model_adv, model_coop)
        dist_mean = model_mean_tv_distance(model_adv, model_coop, delta_mu)
        
        self.current_iteration += 1
        
        return model_adv, er_disadvantage, dist_sup, dist_mean, budget


class GreedyAdversarialChooser(AdversarialModelChooser):
    """
    Simplified greedy adversarial chooser.
    For each (s,a), shifts probability mass to the globally worst state.
    Faster but less sophisticated than full adversarial optimization.
    """
    
    def choose_adversarial(self, model_coop, model_nominal, delta_mu, U, iteration=None):
        """
        Greedy version: for each (s,a), shift probability to globally worst next state
        """
        budget = self.get_budget(iteration)
        
        # Early exit if budget is essentially zero
        if budget < 1e-10:
            er_disadvantage = 0.0
            dist_sup = 0.0
            dist_mean = 0.0
            self.current_iteration += 1
            return model_coop, er_disadvantage, dist_sup, dist_mean, budget
        
        model_coop_rep = model_coop.get_rep()
        model_coop_matrix = model_coop.get_matrix()
        model_adv_rep = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        
        for s in range(self.nS):
            for a in range(self.nA):
                sa = s * self.nA + a
                P_current = model_coop_matrix[sa, :]
                
                # Find worst next state based on value function
                state_values = U[sa, :]
                worst_state = np.argmin(state_values)
                
                # Shift budget amount to worst state
                P_adv = P_current.copy()
                max_shift = min(budget, 1.0 - P_adv[worst_state])
                
                if max_shift > 1e-10:
                    remaining = P_adv.sum() - P_adv[worst_state]
                    if remaining > 1e-10:
                        scale = max(0.0, 1.0 - max_shift / remaining)
                        for s_p in range(self.nS):
                            if s_p != worst_state:
                                P_adv[s_p] *= scale
                        P_adv[worst_state] += max_shift
                
                P_adv = np.clip(P_adv, 0, 1)
                if P_adv.sum() > 0:
                    P_adv /= P_adv.sum()
                else:
                    P_adv = P_current.copy()
                
                # FIXED: Only include non-zero probabilities
                model_adv_rep[s][a] = [(P_adv[s_next], s_next) 
                                       for s_next in range(self.nS) 
                                       if P_adv[s_next] > 1e-10]
                
                # Ensure we have at least one transition
                if len(model_adv_rep[s][a]) == 0:
                    model_adv_rep[s][a] = model_coop_rep[s][a]
        
        model_adv = TabularModel(model_adv_rep, self.nS, self.nA)
        
        er_advantage = compute_model_er_advantage(model_adv, model_coop, U, delta_mu)
        er_disadvantage = -er_advantage
        dist_sup = model_sup_tv_distance(model_adv, model_coop)
        dist_mean = model_mean_tv_distance(model_adv, model_coop, delta_mu)
        
        self.current_iteration += 1
        
        return model_adv, er_disadvantage, dist_sup, dist_mean, budget