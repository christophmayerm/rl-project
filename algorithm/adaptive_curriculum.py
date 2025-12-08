"""
Adaptive Curriculum Learning System for SA-PMI
Implements performance-based, safety-constrained adversarial training
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any
import warnings


@dataclass
class CurriculumState:
    """Tracks curriculum learning state and decisions"""
    iteration: int
    phase: str  # 'warmup', 'gradual', 'robust', 'adaptive'
    current_budget: float
    base_budget: float
    safety_scale: float
    performance_trend: float
    stability_score: float
    tv_distance_mean: float
    tv_distance_max: float
    reason: str  # Why this budget was chosen
    

class AdaptiveCurriculumScheduler:
    """
    Adaptive curriculum scheduler that adjusts adversarial budget based on:
    1. Learning progress (performance trend)
    2. Training stability (performance variance)
    3. Model divergence (TV distance)
    4. Training phase (multi-stage curriculum)
    """
    
    def __init__(
        self,
        B_max: float,
        B_min: float = 0.0,
        # Performance-based adaptation
        performance_window: int = 100,
        convergence_trigger: float = 0.001,
        # Safety thresholds
        tv_safe_threshold: float = 0.65,
        tv_danger_threshold: float = 0.75,
        stability_threshold: float = 0.05,
        # Multi-phase curriculum
        warmup_phase_ratio: float = 0.4,
        gradual_phase_ratio: float = 0.4,
        robust_phase_ratio: float = 0.2,
        # Adaptive parameters
        stability_sensitivity: float = 5.0,
        progress_sensitivity: float = 10.0,
        min_iterations_before_adversarial: int = 500,
        # Budget adaptation rates
        budget_increase_rate: float = 1.05,
        budget_decrease_rate: float = 0.8,
        emergency_budget_scale: float = 0.3,
    ):
        """
        Initialize adaptive curriculum scheduler
        
        Args:
            B_max: Maximum adversarial budget
            B_min: Minimum adversarial budget (initial)
            performance_window: Window size for computing performance metrics
            convergence_trigger: Performance improvement threshold to start adversarial training
            tv_safe_threshold: TV distance threshold for safety warnings
            tv_danger_threshold: TV distance threshold for emergency budget reduction
            stability_threshold: Performance std threshold for stability assessment
            warmup_phase_ratio: Fraction of training for pure cooperative learning
            gradual_phase_ratio: Fraction for gradual adversarial introduction
            robust_phase_ratio: Fraction for full adversarial training
            stability_sensitivity: How much variance affects budget (higher = more sensitive)
            progress_sensitivity: How much progress affects budget (higher = more sensitive)
            min_iterations_before_adversarial: Minimum iterations before adversarial training
            budget_increase_rate: Multiplicative factor for increasing budget
            budget_decrease_rate: Multiplicative factor for decreasing budget
            emergency_budget_scale: Budget scale in emergency situations
        """
        # Validate inputs
        assert 0 <= B_min < B_max, f"Must have 0 <= B_min ({B_min}) < B_max ({B_max})"
        assert warmup_phase_ratio + gradual_phase_ratio + robust_phase_ratio <= 1.0
        assert 0 < tv_safe_threshold < tv_danger_threshold
        
        # Budget parameters
        self.B_max = B_max
        self.B_min = B_min
        
        # Performance tracking
        self.performance_window = performance_window
        self.convergence_trigger = convergence_trigger
        self.performance_history = deque(maxlen=performance_window * 2)
        self.budget_history = deque(maxlen=performance_window)
        
        # Safety thresholds
        self.tv_safe_threshold = tv_safe_threshold
        self.tv_danger_threshold = tv_danger_threshold
        self.stability_threshold = stability_threshold
        
        # Multi-phase curriculum
        self.warmup_phase_ratio = warmup_phase_ratio
        self.gradual_phase_ratio = gradual_phase_ratio
        self.robust_phase_ratio = robust_phase_ratio
        
        # Adaptive parameters
        self.stability_sensitivity = stability_sensitivity
        self.progress_sensitivity = progress_sensitivity
        self.min_iterations_before_adversarial = min_iterations_before_adversarial
        
        # Budget adaptation rates
        self.budget_increase_rate = budget_increase_rate
        self.budget_decrease_rate = budget_decrease_rate
        self.emergency_budget_scale = emergency_budget_scale
        
        # State tracking
        self.adversarial_active = False
        self.adversarial_start_iteration = None
        self.current_phase = 'warmup'
        self.emergency_mode = False
        self.emergency_start = None
        self.consecutive_stable_iterations = 0
        self.consecutive_unstable_iterations = 0
        
        # History for analysis
        self.state_history = []
        
    def should_start_adversarial(self, iteration: int, logger: Any) -> bool:
        """
        Determine if adversarial training should begin
        
        Returns True when:
        1. Minimum iterations requirement met
        2. Cooperative learning has converged (improvement rate low)
        3. Training is stable (low variance)
        """
        if iteration < self.min_iterations_before_adversarial:
            return False
            
        if len(logger.evaluations) < self.performance_window:
            return False
        
        # Compute recent performance metrics
        recent = logger.evaluations[-self.performance_window:]
        improvement_rate = (recent[-1] - recent[0]) / len(recent)
        performance_std = np.std(recent)
        
        # Check convergence: improvement rate below threshold
        converged = improvement_rate < self.convergence_trigger
        
        # Check stability: variance is low
        stable = performance_std < self.stability_threshold
        
        return converged and stable
    
    def compute_performance_metrics(self, logger: Any) -> Dict[str, float]:
        """Compute performance-based metrics for curriculum adaptation"""
        if len(logger.evaluations) < 10:
            return {
                'improvement_rate': 0.0,
                'stability': 1.0,
                'trend_slope': 0.0,
                'recent_mean': logger.evaluations[-1] if logger.evaluations else 0.0
            }
        
        recent = logger.evaluations[-self.performance_window:] if len(logger.evaluations) >= self.performance_window \
                 else logger.evaluations
        
        # Improvement rate (change per iteration)
        improvement_rate = (recent[-1] - recent[0]) / len(recent)
        
        # Stability (inverse of normalized std)
        performance_std = np.std(recent)
        performance_mean = np.mean(recent)
        stability = 1.0 / (1.0 + self.stability_sensitivity * (performance_std / (performance_mean + 1e-6)))
        
        # Trend slope (linear regression)
        x = np.arange(len(recent))
        trend_slope = np.polyfit(x, recent, 1)[0]
        
        return {
            'improvement_rate': improvement_rate,
            'stability': stability,
            'trend_slope': trend_slope,
            'recent_mean': np.mean(recent[-10:]),
            'performance_std': performance_std
        }
    
    def compute_phase_budget(self, iteration: int, max_iterations: int) -> float:
        """
        Compute base budget based on training phase (multi-stage curriculum)
        
        Phase 1 (warmup): 0% budget - pure cooperative learning
        Phase 2 (gradual): 0% → 30% budget - gentle adversarial introduction  
        Phase 3 (robust): 30% → 100% budget - full adversarial training
        """
        if not self.adversarial_active:
            return 0.0
        
        # Progress since adversarial training started
        adv_iterations = iteration - self.adversarial_start_iteration
        
        # Estimate remaining iterations
        estimated_remaining = max_iterations - self.adversarial_start_iteration
        if estimated_remaining <= 0:
            return self.B_max
            
        progress = adv_iterations / estimated_remaining
        
        # Phase 1: Warmup (pure cooperative)
        if progress < self.warmup_phase_ratio:
            self.current_phase = 'warmup'
            return 0.0
        
        # Phase 2: Gradual adversarial introduction
        elif progress < (self.warmup_phase_ratio + self.gradual_phase_ratio):
            self.current_phase = 'gradual'
            phase_progress = (progress - self.warmup_phase_ratio) / self.gradual_phase_ratio
            # Smooth transition: 0 → 0.3 * B_max
            return 0.3 * self.B_max * phase_progress
        
        # Phase 3: Full robust training
        else:
            self.current_phase = 'robust'
            phase_progress = (progress - self.warmup_phase_ratio - self.gradual_phase_ratio) / \
                           (self.robust_phase_ratio if self.robust_phase_ratio > 0 else 0.2)
            phase_progress = min(phase_progress, 1.0)
            # Smooth transition: 0.3 * B_max → B_max
            return 0.3 * self.B_max + 0.7 * self.B_max * phase_progress
    
    def compute_adaptive_scale(self, performance_metrics: Dict[str, float]) -> tuple[float, str]:
        """
        Compute adaptive scaling factor based on learning dynamics
        
        Returns:
            (scale_factor, reason) where scale_factor in [0.5, 1.5]
        """
        stability = performance_metrics['stability']
        trend_slope = performance_metrics['trend_slope']
        improvement_rate = performance_metrics['improvement_rate']
        
        # Stability factor: low stability → reduce budget
        if stability < 0.5:
            stability_factor = 0.8
            reason_parts = [f"low stability ({stability:.2f})"]
        elif stability < 0.7:
            stability_factor = 0.9
            reason_parts = [f"moderate stability ({stability:.2f})"]
        else:
            stability_factor = 1.0
            reason_parts = []
        
        # Progress factor: negative trend → reduce budget, positive → increase
        if trend_slope < -0.001:
            progress_factor = 0.7
            reason_parts.append(f"declining performance ({trend_slope:.4f})")
        elif trend_slope < 0:
            progress_factor = 0.85
            reason_parts.append(f"stagnating ({trend_slope:.4f})")
        elif trend_slope < 0.001:
            progress_factor = 1.0
            reason_parts.append(f"stable improvement")
        else:
            progress_factor = 1.1
            reason_parts.append(f"strong improvement ({trend_slope:.4f})")
        
        # Combined adaptive scale
        adaptive_scale = stability_factor * progress_factor
        adaptive_scale = np.clip(adaptive_scale, 0.5, 1.5)
        
        reason = "adaptive: " + ", ".join(reason_parts) if reason_parts else "adaptive: normal"
        
        return adaptive_scale, reason
    
    def compute_safety_scale(
        self, 
        model_tv_mean: float, 
        model_tv_max: float,
        performance_metrics: Dict[str, float]
    ) -> tuple[float, str]:
        """
        Compute safety scaling based on model divergence
        
        Returns:
            (safety_scale, reason) where safety_scale in [0.3, 1.0]
        """
        reasons = []
        
        # Check TV distance thresholds
        if model_tv_mean > self.tv_danger_threshold:
            tv_scale = self.emergency_budget_scale
            reasons.append(f"⚠️ DANGER: TV mean={model_tv_mean:.3f} > {self.tv_danger_threshold}")
            self.emergency_mode = True
            self.consecutive_unstable_iterations += 1
            self.consecutive_stable_iterations = 0
            
            if self.emergency_start is None:
                self.emergency_start = len(self.state_history)
                
        elif model_tv_mean > self.tv_safe_threshold:
            tv_scale = 0.7
            reasons.append(f"⚠️ WARNING: TV mean={model_tv_mean:.3f} > {self.tv_safe_threshold}")
            self.consecutive_unstable_iterations += 1
            self.consecutive_stable_iterations = 0
            
        else:
            tv_scale = 1.0
            self.consecutive_stable_iterations += 1
            self.consecutive_unstable_iterations = 0
            
            # Exit emergency mode after sustained stability
            if self.emergency_mode and self.consecutive_stable_iterations > 50:
                reasons.append("✅ Exiting emergency mode - stability restored")
                self.emergency_mode = False
                self.emergency_start = None
        
        # Check performance instability (separate from TV distance)
        if performance_metrics['stability'] < 0.3:
            instability_scale = 0.8
            reasons.append(f"unstable performance (stability={performance_metrics['stability']:.2f})")
        else:
            instability_scale = 1.0
        
        # Combined safety scale
        safety_scale = min(tv_scale, instability_scale)
        
        reason = "safety: " + ", ".join(reasons) if reasons else "safety: ok"
        
        return safety_scale, reason
    
    def get_budget(
        self,
        iteration: int,
        max_iterations: int,
        logger: Any,
        model_tv_mean: float = 0.0,
        model_tv_max: float = 0.0
    ) -> tuple[float, CurriculumState]:
        """
        Compute adaptive adversarial budget with safety constraints
        
        Args:
            iteration: Current iteration
            max_iterations: Maximum iterations for training
            logger: Logger with performance history
            model_tv_mean: Mean TV distance between adversarial and cooperative model
            model_tv_max: Max TV distance between adversarial and cooperative model
            
        Returns:
            (budget, state) where:
                - budget: Float in [0, B_max]
                - state: CurriculumState with detailed curriculum decisions
        """
        # Update performance history
        if len(logger.evaluations) > 0:
            self.performance_history.append(logger.evaluations[-1])
        
        # Check if adversarial training should start
        if not self.adversarial_active:
            should_start = self.should_start_adversarial(iteration, logger)
            if should_start:
                self.adversarial_active = True
                self.adversarial_start_iteration = iteration
                print(f"\n🎯 Iteration {iteration}: Activating adversarial curriculum learning")
                print(f"   Performance trend stable, beginning gradual adversarial introduction")
        
        # If adversarial not active, return zero budget
        if not self.adversarial_active:
            state = CurriculumState(
                iteration=iteration,
                phase='pre_adversarial',
                current_budget=0.0,
                base_budget=0.0,
                safety_scale=1.0,
                performance_trend=0.0,
                stability_score=1.0,
                tv_distance_mean=0.0,
                tv_distance_max=0.0,
                reason="Cooperative learning phase - building foundation"
            )
            self.state_history.append(state)
            return 0.0, state
        
        # Compute performance metrics
        perf_metrics = self.compute_performance_metrics(logger)
        
        # STEP 1: Compute phase-based base budget (multi-stage curriculum)
        base_budget = self.compute_phase_budget(iteration, max_iterations)
        
        # STEP 2: Apply adaptive scaling (performance-based)
        adaptive_scale, adaptive_reason = self.compute_adaptive_scale(perf_metrics)
        
        # STEP 3: Apply safety constraints (divergence-based)
        safety_scale, safety_reason = self.compute_safety_scale(
            model_tv_mean, 
            model_tv_max,
            perf_metrics
        )
        
        # STEP 4: Compute final budget
        final_budget = base_budget * adaptive_scale * safety_scale
        
        # STEP 5: Enforce hard limits
        final_budget = np.clip(final_budget, self.B_min, self.B_max)
        
        # STEP 6: Smooth budget changes (prevent oscillation)
        if len(self.budget_history) > 0:
            prev_budget = self.budget_history[-1]
            max_change = 0.2 * self.B_max  # Max 20% change per iteration
            
            if abs(final_budget - prev_budget) > max_change:
                if final_budget > prev_budget:
                    final_budget = prev_budget + max_change
                    smooth_reason = f"smoothed: capped increase to {max_change:.4f}"
                else:
                    final_budget = prev_budget - max_change
                    smooth_reason = f"smoothed: capped decrease to {max_change:.4f}"
            else:
                smooth_reason = "no smoothing needed"
        else:
            smooth_reason = "first adversarial iteration"
        
        self.budget_history.append(final_budget)
        
        # Build reason string
        full_reason = f"phase: {self.current_phase} | {adaptive_reason} | {safety_reason} | {smooth_reason}"
        
        # Create state object
        state = CurriculumState(
            iteration=iteration,
            phase=self.current_phase,
            current_budget=final_budget,
            base_budget=base_budget,
            safety_scale=safety_scale,
            performance_trend=perf_metrics['trend_slope'],
            stability_score=perf_metrics['stability'],
            tv_distance_mean=model_tv_mean,
            tv_distance_max=model_tv_max,
            reason=full_reason
        )
        
        self.state_history.append(state)
        
        # Warnings
        if self.emergency_mode:
            if iteration % 10 == 0:  # Print every 10 iterations in emergency
                print(f"⚠️  [Iter {iteration}] EMERGENCY MODE: Budget reduced to {final_budget:.4f} "
                      f"(TV distance={model_tv_mean:.3f})")
        
        return final_budget, state
    
    def get_curriculum_summary(self) -> Dict[str, Any]:
        """Generate summary statistics of curriculum learning"""
        if not self.state_history:
            return {}
        
        states = self.state_history
        
        # Phase distribution
        phase_counts = {}
        for state in states:
            phase_counts[state.phase] = phase_counts.get(state.phase, 0) + 1
        
        # Budget statistics
        budgets = [s.current_budget for s in states if s.current_budget > 0]
        
        # Safety events
        safety_events = sum(1 for s in states if s.safety_scale < 1.0)
        emergency_events = sum(1 for s in states if s.safety_scale < 0.5)
        
        # Performance correlation
        if len(budgets) > 10:
            budget_perf_corr = np.corrcoef(
                [s.current_budget for s in states[-len(budgets):]],
                [s.performance_trend for s in states[-len(budgets):]]
            )[0, 1]
        else:
            budget_perf_corr = 0.0
        
        summary = {
            'total_iterations': len(states),
            'adversarial_start': self.adversarial_start_iteration,
            'phase_distribution': phase_counts,
            'budget_stats': {
                'mean': np.mean(budgets) if budgets else 0.0,
                'std': np.std(budgets) if budgets else 0.0,
                'min': min(budgets) if budgets else 0.0,
                'max': max(budgets) if budgets else 0.0,
            },
            'safety_events': safety_events,
            'emergency_events': emergency_events,
            'emergency_mode_active': self.emergency_mode,
            'budget_performance_correlation': budget_perf_corr,
            'consecutive_stable_iterations': self.consecutive_stable_iterations,
        }
        
        return summary
    
    def plot_curriculum_history(self, save_path: Optional[str] = None):
        """Plot detailed curriculum learning history"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            warnings.warn("matplotlib not available for plotting")
            return
        
        if not self.state_history:
            print("No curriculum history to plot")
            return
        
        states = self.state_history
        iterations = [s.iteration for s in states]
        budgets = [s.current_budget for s in states]
        base_budgets = [s.base_budget for s in states]
        safety_scales = [s.safety_scale for s in states]
        stability_scores = [s.stability_score for s in states]
        tv_means = [s.tv_distance_mean for s in states]
        performance_trends = [s.performance_trend for s in states]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Budget evolution
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(iterations, budgets, 'b-', linewidth=2, label='Final Budget', alpha=0.8)
        ax1.plot(iterations, base_budgets, 'g--', linewidth=1.5, label='Base Budget (Phase)', alpha=0.6)
        ax1.axhline(self.B_max, color='r', linestyle=':', label=f'B_max={self.B_max}')
        ax1.axhline(self.tv_safe_threshold * self.B_max, color='orange', linestyle=':', 
                   label=f'Safety Threshold')
        
        # Mark phase transitions
        phases = [s.phase for s in states]
        phase_changes = [i for i in range(1, len(phases)) if phases[i] != phases[i-1]]
        for pc in phase_changes:
            ax1.axvline(iterations[pc], color='gray', linestyle='--', alpha=0.3)
            ax1.text(iterations[pc], ax1.get_ylim()[1]*0.95, phases[pc], 
                    rotation=90, va='top', fontsize=8)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Budget')
        ax1.set_title('Adaptive Budget Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Safety scaling
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(iterations, safety_scales, 'r-', linewidth=2, label='Safety Scale')
        ax2.axhline(1.0, color='g', linestyle='--', alpha=0.5, label='No constraint')
        ax2.axhline(self.emergency_budget_scale, color='r', linestyle=':', 
                   label=f'Emergency ({self.emergency_budget_scale})')
        ax2.fill_between(iterations, 0, safety_scales, alpha=0.3, color='red',
                        where=[s < 1.0 for s in safety_scales], label='Constraint Active')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Safety Scale')
        ax2.set_title('Safety Constraint Activation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. TV Distance tracking
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(iterations, tv_means, 'purple', linewidth=2, label='TV Distance (mean)')
        ax3.axhline(self.tv_safe_threshold, color='orange', linestyle='--', 
                   label=f'Safe Threshold ({self.tv_safe_threshold})')
        ax3.axhline(self.tv_danger_threshold, color='red', linestyle='--',
                   label=f'Danger Threshold ({self.tv_danger_threshold})')
        ax3.fill_between(iterations, self.tv_safe_threshold, self.tv_danger_threshold, 
                        alpha=0.2, color='orange', label='Warning Zone')
        ax3.fill_between(iterations, self.tv_danger_threshold, max(tv_means + [self.tv_danger_threshold]),
                        alpha=0.3, color='red', label='Danger Zone')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('TV Distance')
        ax3.set_title('Model Divergence Monitoring')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Stability score
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(iterations, stability_scores, 'green', linewidth=2, label='Stability Score')
        ax4.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Low Stability')
        ax4.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Very Low Stability')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Stability Score')
        ax4.set_title('Learning Stability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance trend
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(iterations, performance_trends, 'blue', linewidth=2, label='Performance Trend')
        ax5.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax5.fill_between(iterations, 0, performance_trends, 
                        where=[pt > 0 for pt in performance_trends],
                        alpha=0.3, color='green', label='Improving')
        ax5.fill_between(iterations, performance_trends, 0,
                        where=[pt < 0 for pt in performance_trends],
                        alpha=0.3, color='red', label='Declining')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Performance Slope')
        ax5.set_title('Performance Trend')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Adaptive Curriculum Learning: Complete Diagnostic View', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Curriculum history saved to {save_path}")
        else:
            plt.show()
        
        plt.close()