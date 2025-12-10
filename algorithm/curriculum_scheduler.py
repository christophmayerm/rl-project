"""
Curriculum Learning Schedulers for SA-PMI

Manages the gradual introduction of adversarial training by controlling
the blend weight λ(t) between nominal and adversarial model selection.
"""

import numpy as np
from abc import ABC, abstractmethod


class CurriculumScheduler(ABC):
    """
    Base class for curriculum learning schedulers.
    
    Controls adversarial weight λ(t) ∈ [0,1] where:
    - λ=0: Pure nominal learning
    - λ=1: Pure adversarial learning
    - 0<λ<1: Blended approach
    """
    
    @abstractmethod
    def get_adversarial_weight(self, iteration: int) -> float:
        """
        Get adversarial blending weight for current iteration.
        
        Args:
            iteration: Current training iteration
            
        Returns:
            float in [0,1] controlling adversarial influence
        """
        pass
    
    def reset(self):
        """Reset scheduler state (for multi-run experiments)"""
        pass


class ConstantCurriculumScheduler(CurriculumScheduler):
    """
    Constant adversarial weight throughout training.
    
    Useful for ablation studies and baseline comparisons.
    """
    
    def __init__(self, weight: float = 0.0):
        """
        Args:
            weight: Fixed adversarial weight (default: 0.0 = no adversarial)
        """
        self.weight = np.clip(weight, 0.0, 1.0)
    
    def get_adversarial_weight(self, iteration: int) -> float:
        return self.weight


class LinearCurriculumScheduler(CurriculumScheduler):
    """
    Linear ramp-up of adversarial weight.
    
    Gradually increases adversarial influence from start_weight to end_weight
    over the interval [start_iter, end_iter].
    """
    
    def __init__(
        self, 
        start_iter: int = 0, 
        end_iter: int = 500, 
        start_weight: float = 0.0, 
        end_weight: float = 0.5
    ):
        """
        Args:
            start_iter: Iteration to begin ramping up adversarial weight
            end_iter: Iteration to reach maximum adversarial weight
            start_weight: Initial adversarial weight
            end_weight: Final adversarial weight
        """
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.start_weight = np.clip(start_weight, 0.0, 1.0)
        self.end_weight = np.clip(end_weight, 0.0, 1.0)
        
        if end_iter <= start_iter:
            raise ValueError("end_iter must be greater than start_iter")
    
    def get_adversarial_weight(self, iteration: int) -> float:
        if iteration < self.start_iter:
            return self.start_weight
        elif iteration >= self.end_iter:
            return self.end_weight
        else:
            progress = (iteration - self.start_iter) / (self.end_iter - self.start_iter)
            return self.start_weight + progress * (self.end_weight - self.start_weight)


class ExponentialCurriculumScheduler(CurriculumScheduler):
    """
    Exponential ramp-up of adversarial weight.
    
    Uses: λ(t) = max_weight · (1 - exp(-growth_rate · t))
    
    Provides smooth, accelerating increase in adversarial difficulty.
    """
    
    def __init__(self, growth_rate: float = 0.01, max_weight: float = 0.5):
        """
        Args:
            growth_rate: Rate of exponential growth (higher = faster ramp-up)
            max_weight: Asymptotic maximum adversarial weight
        """
        self.growth_rate = growth_rate
        self.max_weight = np.clip(max_weight, 0.0, 1.0)
        
        if growth_rate <= 0:
            raise ValueError("growth_rate must be positive")
    
    def get_adversarial_weight(self, iteration: int) -> float:
        return self.max_weight * (1.0 - np.exp(-self.growth_rate * iteration))


class CosineCurriculumScheduler(CurriculumScheduler):
    """
    Cosine annealing schedule for adversarial weight.
    
    Uses: λ(t) = 0.5 · max_weight · (1 - cos(π · t / T))
    
    Provides smooth S-curve transition from nominal to adversarial.
    """
    
    def __init__(
        self, 
        start_iter: int = 0, 
        end_iter: int = 500, 
        max_weight: float = 0.5
    ):
        """
        Args:
            start_iter: Iteration to begin curriculum
            end_iter: Iteration to complete curriculum
            max_weight: Maximum adversarial weight
        """
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.max_weight = np.clip(max_weight, 0.0, 1.0)
        
        if end_iter <= start_iter:
            raise ValueError("end_iter must be greater than start_iter")
    
    def get_adversarial_weight(self, iteration: int) -> float:
        if iteration < self.start_iter:
            return 0.0
        elif iteration >= self.end_iter:
            return self.max_weight
        else:
            progress = (iteration - self.start_iter) / (self.end_iter - self.start_iter)
            return 0.5 * self.max_weight * (1.0 - np.cos(np.pi * progress))


class StepCurriculumScheduler(CurriculumScheduler):
    """
    Step-wise increase in adversarial weight.
    
    Increases adversarial weight in discrete steps at specified milestones.
    Useful for ablation studies and staged training.
    """
    
    def __init__(self, milestones: list, weights: list):
        """
        Args:
            milestones: List of iteration milestones [m1, m2, ..., mn]
            weights: List of weights [w0, w1, ..., wn] where w0 is initial weight
        
        Example:
            milestones=[100, 300], weights=[0.0, 0.2, 0.5]
            → λ=0.0 for t<100, λ=0.2 for 100≤t<300, λ=0.5 for t≥300
        """
        if len(weights) != len(milestones) + 1:
            raise ValueError("weights must have one more element than milestones")
        
        self.milestones = sorted(milestones)
        self.weights = [np.clip(w, 0.0, 1.0) for w in weights]
    
    def get_adversarial_weight(self, iteration: int) -> float:
        for i, milestone in enumerate(self.milestones):
            if iteration < milestone:
                return self.weights[i]
        return self.weights[-1]


class AdaptiveCurriculumScheduler(CurriculumScheduler):
    """
    Performance-based adaptive curriculum.
    
    Adjusts adversarial weight based on training progress:
    - Increases weight when performance is improving well
    - Decreases weight when training struggles
    
    Requires explicit update() calls with performance metrics.
    """
    
    def __init__(
        self, 
        initial_weight: float = 0.0,
        min_weight: float = 0.0,
        max_weight: float = 0.5,
        increase_threshold: float = 0.01,
        decrease_threshold: float = -0.005,
        adjustment_rate: float = 0.05
    ):
        """
        Args:
            initial_weight: Starting adversarial weight
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
            increase_threshold: Performance improvement to increase weight
            decrease_threshold: Performance drop to decrease weight
            adjustment_rate: Step size for weight adjustments
        """
        self.current_weight = np.clip(initial_weight, 0.0, 1.0)
        self.min_weight = np.clip(min_weight, 0.0, 1.0)
        self.max_weight = np.clip(max_weight, 0.0, 1.0)
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.adjustment_rate = adjustment_rate
        self.last_performance = None
    
    def get_adversarial_weight(self, iteration: int) -> float:
        return self.current_weight
    
    def update(self, current_performance: float):
        """
        Update curriculum based on performance.
        
        Args:
            current_performance: Current policy performance metric
        """
        if self.last_performance is not None:
            performance_delta = current_performance - self.last_performance
            
            if performance_delta >= self.increase_threshold:
                # Good progress → increase adversarial difficulty
                self.current_weight = min(
                    self.max_weight, 
                    self.current_weight + self.adjustment_rate
                )
            elif performance_delta <= self.decrease_threshold:
                # Struggling → decrease adversarial difficulty
                self.current_weight = max(
                    self.min_weight, 
                    self.current_weight - self.adjustment_rate
                )
        
        self.last_performance = current_performance
    
    def reset(self):
        """Reset adaptive state"""
        self.last_performance = None


class SigmoidCurriculumScheduler(CurriculumScheduler):
    """
    Smooth sigmoid transition - good for small envs.
    More gradual than cosine, faster than exponential.
    
    λ(t) = max_weight / (1 + exp(-steepness * (t - midpoint)))
    """
    def __init__(self, midpoint=150, steepness=0.02, max_weight=0.5):
        self.midpoint = midpoint
        self.steepness = steepness
        self.max_weight = np.clip(max_weight, 0.0, 1.0)
    
    def get_adversarial_weight(self, iteration):
        return self.max_weight / (1.0 + np.exp(-self.steepness * (iteration - self.midpoint)))


class PolynomialCurriculumScheduler(CurriculumScheduler):
    """
    Polynomial growth - controllable acceleration.
    
    λ(t) = max_weight * ((t - start) / (end - start))^power
    
    power=1: linear
    power=2: quadratic (slow start, fast finish)
    power=0.5: square root (fast start, slow finish)
    """
    def __init__(self, start_iter=0, end_iter=300, max_weight=0.5, power=2.0):
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.max_weight = np.clip(max_weight, 0.0, 1.0)
        self.power = power
        
    def get_adversarial_weight(self, iteration):
        if iteration < self.start_iter:
            return 0.0
        elif iteration >= self.end_iter:
            return self.max_weight
        else:
            progress = (iteration - self.start_iter) / (self.end_iter - self.start_iter)
            return self.max_weight * (progress ** self.power)


class WarmRestartCurriculumScheduler(CurriculumScheduler):
    """
    Periodic curriculum with warm restarts - for exploration.
    Cycles between low and high adversarial influence.
    """
    def __init__(self, cycle_length=100, min_weight=0.0, max_weight=0.5, 
                 restart_multiplier=1.5):
        self.cycle_length = cycle_length
        self.min_weight = np.clip(min_weight, 0.0, 1.0)
        self.max_weight = np.clip(max_weight, 0.0, 1.0)
        self.restart_multiplier = restart_multiplier
    
    def get_adversarial_weight(self, iteration):
        # Cosine annealing within current cycle
        current_cycle_length = self.cycle_length * (self.restart_multiplier ** (iteration // self.cycle_length))
        t_cur = iteration % current_cycle_length
        weight = self.min_weight + 0.5 * (self.max_weight - self.min_weight) * \
                 (1 + np.cos(np.pi * t_cur / current_cycle_length))
        return weight


class EarlyPeakCurriculumScheduler(CurriculumScheduler):
    """
    Peak early then plateau - for environments that need initial robustness.
    Ramps up quickly, then maintains high adversarial weight.
    """
    def __init__(self, peak_iter=100, max_weight=0.5, decay_rate=0.05):
        self.peak_iter = peak_iter
        self.max_weight = np.clip(max_weight, 0.0, 1.0)
        self.decay_rate = decay_rate
    
    def get_adversarial_weight(self, iteration):
        if iteration < self.peak_iter:
            # Rapid increase
            return self.max_weight * (1.0 - np.exp(-0.05 * iteration))
        else:
            # Slight decay to plateau
            decay = np.exp(-self.decay_rate * (iteration - self.peak_iter))
            return self.max_weight * (0.7 + 0.3 * decay)


class TwoStageCurriculumScheduler(CurriculumScheduler):
    """
    Two-stage training: exploration phase then exploitation phase.
    Stage 1: Low adversarial (learn basics)
    Stage 2: High adversarial (robust refinement)
    """
    def __init__(self, transition_iter=200, stage1_weight=0.1, stage2_weight=0.6,
                 transition_length=50):
        self.transition_iter = transition_iter
        self.stage1_weight = np.clip(stage1_weight, 0.0, 1.0)
        self.stage2_weight = np.clip(stage2_weight, 0.0, 1.0)
        self.transition_length = transition_length
    
    def get_adversarial_weight(self, iteration):
        if iteration < self.transition_iter:
            return self.stage1_weight
        elif iteration >= self.transition_iter + self.transition_length:
            return self.stage2_weight
        else:
            # Smooth transition
            progress = (iteration - self.transition_iter) / self.transition_length
            return self.stage1_weight + progress * (self.stage2_weight - self.stage1_weight)