"""
Example script demonstrating SA-PMI with adversarial curriculum learning
using the TeacherStudent environment - FIXED VERSION
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.spmi import SPMI
from algorithm.policy_chooser import GreedyPolicyChooser
from algorithm.model_chooser import GreedyModelChooser
from utils.tabular import TabularPolicy, TabularModel
from envs.student_teacher import TeacherStudentEnv


def create_teacher_student_env():
    """
    Create TeacherStudent environment with simple parameters.
    """
    print("Creating TeacherStudent environment...")
    env = TeacherStudentEnv(
        n_literals=2,           # Small for testing
        max_value=1,            # Binary literals
        max_update=1,
        max_literals_in_examples=2,
        horizon=10
    )
    
    print(f"Environment created:")
    print(f"  - States: {env.nS}")
    print(f"  - Actions: {env.nA}")
    print(f"  - Gamma: {env.gamma}")
    print(f"  - Horizon: {env.horizon}")
    
    return env


def create_initial_policy_model(env):
    """
    Create initial uniform policy and extract model from environment.
    """
    nS, nA = env.nS, env.nA
    
    # Get uniform policy from environment
    initial_policy = env.get_uniform_policy()
    
    # Extract model representation from environment
    model_rep = {s: {a: [] for a in range(nA)} for s in range(nS)}
    for s in range(nS):
        for a in range(nA):
            transitions = env.P[s][a]
            for prob, s_next, reward, done in transitions:
                if prob > 0:  # Only include non-zero probabilities
                    model_rep[s][a].append((prob, s_next))
    
    initial_model = TabularModel(model_rep, nS, nA)
    
    return initial_policy, initial_model


class ConfigurableMDP:
    """
    Wrapper to make DiscreteEnv compatible with SPMI.
    """
    def __init__(self, env):
        self.env = env
        self.nS = env.nS
        self.nA = env.nA
        self.gamma = env.gamma
        self.horizon = env.horizon
        self.mu = env.mu
        self.P = env.P
        
    def set_model(self, model_rep):
        """Update the environment's transition model."""
        # Convert TabularModel representation to environment format
        if hasattr(model_rep, 'get_rep'):
            model_dict = model_rep.get_rep()
        else:
            model_dict = model_rep
            
        # Reconstruct full P with rewards from original environment
        new_P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        
        for s in range(self.nS):
            for a in range(self.nA):
                # Get original transitions to extract rewards
                original_transitions = self.env.P[s][a]
                reward_map = {t[1]: t[2] for t in original_transitions}
                
                # Get new transition probabilities
                if isinstance(model_dict, dict):
                    new_transitions = model_dict[s][a]
                else:
                    # If it's a list format from TabularModel
                    new_transitions = model_dict[s][a]
                
                # Reconstruct with rewards - handle both 2-tuple and 4-tuple formats
                for transition in new_transitions:
                    if len(transition) == 2:
                        # Format: (prob, s_next)
                        prob, s_next = transition
                        reward = reward_map.get(s_next, 0.0)
                        done = False
                    elif len(transition) == 4:
                        # Format: (prob, s_next, reward, done)
                        prob, s_next, reward, done = transition
                    else:
                        raise ValueError(f"Unexpected transition format: {transition}")
                    
                    new_P[s][a].append((prob, s_next, reward, done))
        
        self.P = new_P
        self.env.set_model(new_P)


def run_comparison():
    """
    Run comparison between standard SPMI and SA-PMI on TeacherStudent.
    """
    print("=" * 70)
    print("SA-PMI DEMONSTRATION: TeacherStudent Environment")
    print("=" * 70)
    
    # Create environment
    env = create_teacher_student_env()
    initial_policy, initial_model = create_initial_policy_model(env)
    
    # Wrap in ConfigurableMDP
    conf_mdp = ConfigurableMDP(env)
    
    # ==========================================
    # EXPERIMENT 1: Standard SPMI (Baseline)
    # ==========================================
    print("\n\n" + "="*70)
    print("EXPERIMENT 1: Standard SPMI (Baseline)")
    print("="*70)
    
    # Create fresh environment for SPMI
    env_spmi = create_teacher_student_env()
    conf_mdp_spmi = ConfigurableMDP(env_spmi)
    initial_policy_spmi, initial_model_spmi = create_initial_policy_model(env_spmi)
    
    spmi_baseline = SPMI(
        conf_mdp=conf_mdp_spmi,
        eps=0.000001,
        policy_chooser=GreedyPolicyChooser(conf_mdp_spmi.nS, conf_mdp_spmi.nA),
        model_chooser=GreedyModelChooser(conf_mdp_spmi.nS, conf_mdp_spmi.nA),
        max_iter=10000,
        persistent=True
    )
    
    print("\nRunning SPMI...")
    policy_spmi, model_spmi = spmi_baseline.spmi(initial_policy_spmi, initial_model_spmi)
    
    print("\n--- SPMI Results ---")
    print(f"Final Performance: {spmi_baseline.logger.evaluations[-1]:.4f}")
    print(f"Iterations: {spmi_baseline.logger.iteration}")
    print(f"Performance trajectory: {[f'{x:.4f}' for x in spmi_baseline.logger.evaluations[:5]]}...")
    
    # ==========================================
    # EXPERIMENT 2: SA-PMI with Linear Curriculum (FIXED)
    # ==========================================
    print("\n\n" + "="*70)
    print("EXPERIMENT 2: SA-PMI with Linear Curriculum (FIXED)")
    print("="*70)
    
    # Create fresh environment for SA-PMI
    env_sapmi = create_teacher_student_env()
    conf_mdp_sapmi = ConfigurableMDP(env_sapmi)
    initial_policy_sapmi, initial_model_sapmi = create_initial_policy_model(env_sapmi)
    
    sa_pmi_linear = SPMI(
        conf_mdp=conf_mdp_sapmi,
        eps=0.000001,
        policy_chooser=GreedyPolicyChooser(conf_mdp_sapmi.nS, conf_mdp_sapmi.nA),
        model_chooser=GreedyModelChooser(conf_mdp_sapmi.nS, conf_mdp_sapmi.nA),
        max_iter=10000,
        persistent=True,
        # SA-PMI specific parameters - MUCH GENTLER
        curriculum_schedule='linear',
        B_min=0.0,       # Start with NO adversarial perturbation
        B_max=0.05,      # Max 5% perturbation (was 20%!)
        K_warmup=10000     # Ramp up slowly over 180 iterations (was 120)
    )
    
    print("\nRunning SA-PMI (Linear Curriculum)...")
    policy_sapmi_lin, model_coop_lin, model_adv_lin = sa_pmi_linear.sa_pmi(
        initial_policy_sapmi, initial_model_sapmi
    )
    
    print("\n--- SA-PMI (Linear) Results ---")
    print(f"Final Performance (worst-case): {sa_pmi_linear.logger.evaluations[-1]:.4f}")
    print(f"Iterations: {sa_pmi_linear.logger.iteration}")
    print(f"Final Adversarial Budget: {sa_pmi_linear.logger.adversarial_budgets[-1]:.4f}")
    print(f"Performance trajectory: {[f'{x:.4f}' for x in sa_pmi_linear.logger.evaluations[:5]]}...")
    print(f"Budget trajectory: {[f'{x:.4f}' for x in sa_pmi_linear.logger.adversarial_budgets[:5]]}...")
    
    # ==========================================
    # EXPERIMENT 3: SA-PMI with Exponential Curriculum (FIXED)
    # ==========================================
    print("\n\n" + "="*70)
    print("EXPERIMENT 3: SA-PMI with Exponential Curriculum (FIXED)")
    print("="*70)
    
    # Create fresh environment for SA-PMI exponential
    env_sapmi_exp = create_teacher_student_env()
    conf_mdp_sapmi_exp = ConfigurableMDP(env_sapmi_exp)
    initial_policy_sapmi_exp, initial_model_sapmi_exp = create_initial_policy_model(env_sapmi_exp)
    
    sa_pmi_exp = SPMI(
        conf_mdp=conf_mdp_sapmi_exp,
        eps=0.000001,
        policy_chooser=GreedyPolicyChooser(conf_mdp_sapmi_exp.nS, conf_mdp_sapmi_exp.nA),
        model_chooser=GreedyModelChooser(conf_mdp_sapmi_exp.nS, conf_mdp_sapmi_exp.nA),
        max_iter=10000,
        persistent=True,
        curriculum_schedule='exponential',
        B_min=0.0,       
        B_max=0.05,      # Reduce from 0.10 → 0.05 (same as linear!)
        K_warmup=10000     # Extend to full duration (was 180)
    )
    print("\nRunning SA-PMI (Exponential Curriculum)...")
    policy_sapmi_exp, model_coop_exp, model_adv_exp = sa_pmi_exp.sa_pmi(
        initial_policy_sapmi_exp, initial_model_sapmi_exp
    )
    
    print("\n--- SA-PMI (Exponential) Results ---")
    print(f"Final Performance (worst-case): {sa_pmi_exp.logger.evaluations[-1]:.4f}")
    print(f"Iterations: {sa_pmi_exp.logger.iteration}")
    print(f"Final Adversarial Budget: {sa_pmi_exp.logger.adversarial_budgets[-1]:.4f}")
    print(f"Performance trajectory: {[f'{x:.4f}' for x in sa_pmi_exp.logger.evaluations[:5]]}...")
    print(f"Budget trajectory: {[f'{x:.4f}' for x in sa_pmi_exp.logger.adversarial_budgets[:5]]}...")
    
    # ==========================================
    # SAVE RESULTS
    # ==========================================
    print("\n\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    spmi_baseline.logger.save('results', 'teacher_student_spmi_baseline.csv')
    sa_pmi_linear.logger.save('results', 'teacher_student_sa_pmi_linear.csv')
    sa_pmi_exp.logger.save('results', 'teacher_student_sa_pmi_exponential.csv')
    
    print("Results saved to 'results/' directory")
    print("  - teacher_student_spmi_baseline.csv")
    print("  - teacher_student_sa_pmi_linear.csv")
    print("  - teacher_student_sa_pmi_exponential.csv")
    
    # ==========================================
    # PLOT COMPARISON
    # ==========================================
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SA-PMI vs SPMI: TeacherStudent Environment (FIXED)', fontsize=16, fontweight='bold')
        
        # Plot 1: Performance over iterations
        ax = axes[0, 0]
        ax.plot(spmi_baseline.logger.iterations, spmi_baseline.logger.evaluations, 
                'b-', label='SPMI', linewidth=2, marker='o', markersize=4)
        ax.plot(sa_pmi_linear.logger.iterations, sa_pmi_linear.logger.evaluations, 
                'r--', label='SA-PMI (Linear)', linewidth=2, marker='s', markersize=4)
        ax.plot(sa_pmi_exp.logger.iterations, sa_pmi_exp.logger.evaluations, 
                'g-.', label='SA-PMI (Exp)', linewidth=2, marker='^', markersize=4)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Performance (J)', fontsize=11)
        ax.set_title('Performance Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Adversarial budget schedule
        ax = axes[0, 1]
        ax.plot(sa_pmi_linear.logger.iterations, sa_pmi_linear.logger.adversarial_budgets,
                'r-', label='Linear Schedule', linewidth=2, marker='s', markersize=4)
        ax.plot(sa_pmi_exp.logger.iterations, sa_pmi_exp.logger.adversarial_budgets,
                'g-', label='Exponential Schedule', linewidth=2, marker='^', markersize=4)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Adversarial Budget B(k)', fontsize=11)
        ax.set_title('Curriculum Schedule', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Adversarial disadvantage
        ax = axes[1, 0]
        ax.plot(sa_pmi_linear.logger.iterations, sa_pmi_linear.logger.adversarial_disadvantages,
                'r-', label='Linear', linewidth=2, marker='s', markersize=4)
        ax.plot(sa_pmi_exp.logger.iterations, sa_pmi_exp.logger.adversarial_disadvantages,
                'g-', label='Exponential', linewidth=2, marker='^', markersize=4)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Adversarial Disadvantage', fontsize=11)
        ax.set_title('Adversarial Impact on Performance', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Bound values
        ax = axes[1, 1]
        ax.plot(spmi_baseline.logger.iterations, spmi_baseline.logger.bound,
                'b-', label='SPMI Bound', linewidth=2, marker='o', markersize=4)
        ax.plot(sa_pmi_linear.logger.iterations, sa_pmi_linear.logger.bound,
                'r--', label='SA-PMI Bound (Linear)', linewidth=2, marker='s', markersize=4)
        ax.plot(sa_pmi_exp.logger.iterations, sa_pmi_exp.logger.bound,
                'g-.', label='SA-PMI Bound (Exp)', linewidth=2, marker='^', markersize=4)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Lower Bound Value', fontsize=11)
        ax.set_title('Performance Bound Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/teacher_student_comparison_fixed.png', dpi=300, bbox_inches='tight')
        print("\nComparison plot saved to 'results/teacher_student_comparison_fixed.png'")
        plt.show()
        
    except ImportError as e:
        print(f"\nMatplotlib not available: {e}")
        print("Skipping plots.")
    except Exception as e:
        print(f"\nError generating plots: {e}")
        print("Results still saved to CSV files.")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\n{'Algorithm':<25} {'Final Perf':<15} {'Iterations':<12} {'Adv Budget':<12}")
    print("-" * 70)
    print(f"{'SPMI (Baseline)':<25} {spmi_baseline.logger.evaluations[-1]:<15.4f} {spmi_baseline.logger.iteration:<12} {'N/A':<12}")
    print(f"{'SA-PMI (Linear)':<25} {sa_pmi_linear.logger.evaluations[-1]:<15.4f} {sa_pmi_linear.logger.iteration:<12} {sa_pmi_linear.logger.adversarial_budgets[-1]:<12.4f}")
    print(f"{'SA-PMI (Exponential)':<25} {sa_pmi_exp.logger.evaluations[-1]:<15.4f} {sa_pmi_exp.logger.iteration:<12} {sa_pmi_exp.logger.adversarial_budgets[-1]:<12.4f}")
    
    return {
        'spmi': (policy_spmi, model_spmi, spmi_baseline.logger),
        'sa_pmi_linear': (policy_sapmi_lin, model_coop_lin, model_adv_lin, sa_pmi_linear.logger),
        'sa_pmi_exp': (policy_sapmi_exp, model_coop_exp, model_adv_exp, sa_pmi_exp.logger)
    }


if __name__ == '__main__':
    print("\n" + "="*70)
    print("STARTING SA-PMI EXPERIMENTS (FIXED VERSION)")
    print("="*70)
    
    try:
        results = run_comparison()
        print("\n" + "="*70)
        print("✓ EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*70)
    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ ERROR: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()