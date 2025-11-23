"""
Example script demonstrating SA-PMI with enhanced logging
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.spmi import SPMI
from algorithm.policy_chooser import GreedyPolicyChooser
from algorithm.model_chooser import GreedyModelChooser
from utils.tabular import TabularPolicy, TabularModel
from envs.student_teacher import TeacherStudentEnv


def create_teacher_student_env():
    """Create TeacherStudent environment"""
    print("Creating TeacherStudent environment...")
    env = TeacherStudentEnv(
        n_literals=2,
        max_value=1,
        max_update=1,
        max_literals_in_examples=2,
        horizon=10
    )
    
    print(f"Environment: {env.nS} states, {env.nA} actions")
    return env


def create_initial_policy_model(env):
    """Create initial uniform policy and model"""
    nS, nA = env.nS, env.nA
    initial_policy = env.get_uniform_policy()
    
    model_rep = {s: {a: [] for a in range(nA)} for s in range(nS)}
    for s in range(nS):
        for a in range(nA):
            transitions = env.P[s][a]
            for prob, s_next, reward, done in transitions:
                if prob > 0:
                    model_rep[s][a].append((prob, s_next))
    
    initial_model = TabularModel(model_rep, nS, nA)
    return initial_policy, initial_model


class ConfigurableMDP:
    """Wrapper for SPMI compatibility"""
    def __init__(self, env):
        self.env = env
        self.nS = env.nS
        self.nA = env.nA
        self.gamma = env.gamma
        self.horizon = env.horizon
        self.mu = env.mu
        self.P = env.P
        
    def set_model(self, model_rep):
        """Update transition model"""
        if hasattr(model_rep, 'get_rep'):
            model_dict = model_rep.get_rep()
        else:
            model_dict = model_rep
            
        new_P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        
        for s in range(self.nS):
            for a in range(self.nA):
                original_transitions = self.env.P[s][a]
                reward_map = {t[1]: t[2] for t in original_transitions}
                new_transitions = model_dict[s][a]
                
                for transition in new_transitions:
                    if len(transition) == 2:
                        prob, s_next = transition
                        reward = reward_map.get(s_next, 0.0)
                        done = False
                    elif len(transition) == 4:
                        prob, s_next, reward, done = transition
                    else:
                        raise ValueError(f"Unexpected transition format: {transition}")
                    
                    new_P[s][a].append((prob, s_next, reward, done))
        
        self.P = new_P
        self.env.set_model(new_P)


def run_comparison(use_wandb=False, wandb_project="sa-pmi-demo", verbose=2):
    """
    Run SPMI vs SA-PMI comparison with enhanced logging
    
    :param use_wandb: Enable Weights & Biases logging
    :param wandb_project: W&B project name
    :param verbose: Verbosity level (0=silent, 1=minimal, 2=normal, 3=detailed)
    """
    print("=" * 70)
    print("SA-PMI DEMONSTRATION")
    print("=" * 70)
    
    # Create environment
    env = create_teacher_student_env()
    
    # ==========================================
    # EXPERIMENT 1: Standard SPMI (Baseline)
    # ==========================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: Standard SPMI (Baseline)")
    print("="*70)
    
    env_spmi = create_teacher_student_env()
    conf_mdp_spmi = ConfigurableMDP(env_spmi)
    initial_policy_spmi, initial_model_spmi = create_initial_policy_model(env_spmi)
    
    # Configure wandb for SPMI
    wandb_config = None
    if use_wandb:
        wandb_config = {
            'project': wandb_project,
            'name': 'SPMI-Baseline',
            'config': {
                'algorithm': 'SPMI',
                'max_iter': 10000,
                'eps': 0.000001,
                'environment': 'TeacherStudent',
                'n_literals': 2
            },
            'tags': ['baseline', 'spmi']
        }
    
    spmi_baseline = SPMI(
        conf_mdp=conf_mdp_spmi,
        eps=0.000001,
        policy_chooser=GreedyPolicyChooser(conf_mdp_spmi.nS, conf_mdp_spmi.nA),
        model_chooser=GreedyModelChooser(conf_mdp_spmi.nS, conf_mdp_spmi.nA),
        max_iter=10000,
        persistent=True
    )
    
    # Configure logger
    spmi_baseline.logger.verbose = verbose
    spmi_baseline.logger.log_interval = 500
    spmi_baseline.logger.use_wandb = use_wandb
    if use_wandb:
        spmi_baseline.logger._init_wandb(wandb_config)
    spmi_baseline.logger.set_max_iter(10000)
    
    print("\nRunning SPMI...")
    policy_spmi, model_spmi = spmi_baseline.spmi(initial_policy_spmi, initial_model_spmi)
    
    spmi_baseline.logger.print_final_summary()
    spmi_baseline.logger.close()
    
    # ==========================================
    # EXPERIMENT 2: SA-PMI with Linear Curriculum
    # ==========================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: SA-PMI (Linear Curriculum)")
    print("="*70)
    
    env_sapmi = create_teacher_student_env()
    conf_mdp_sapmi = ConfigurableMDP(env_sapmi)
    initial_policy_sapmi, initial_model_sapmi = create_initial_policy_model(env_sapmi)
    
    # Configure wandb for SA-PMI Linear
    wandb_config_linear = None
    if use_wandb:
        wandb_config_linear = {
            'project': wandb_project,
            'name': 'SA-PMI-Linear',
            'config': {
                'algorithm': 'SA-PMI',
                'curriculum': 'linear',
                'max_iter': 10000,
                'eps': 0.000001,
                'B_min': 0.0,
                'B_max': 0.05,
                'K_warmup': 10000,
                'environment': 'TeacherStudent',
                'n_literals': 2
            },
            'tags': ['adversarial', 'sa-pmi', 'linear']
        }
    
    sa_pmi_linear = SPMI(
        conf_mdp=conf_mdp_sapmi,
        eps=0.000001,
        policy_chooser=GreedyPolicyChooser(conf_mdp_sapmi.nS, conf_mdp_sapmi.nA),
        model_chooser=GreedyModelChooser(conf_mdp_sapmi.nS, conf_mdp_sapmi.nA),
        max_iter=10000,
        persistent=True,
        curriculum_schedule='linear',
        B_min=0.0,
        B_max=0.05,
        K_warmup=10000
    )
    
    # Configure logger
    sa_pmi_linear.logger.verbose = verbose
    sa_pmi_linear.logger.log_interval = 500
    sa_pmi_linear.logger.use_wandb = use_wandb
    if use_wandb:
        sa_pmi_linear.logger._init_wandb(wandb_config_linear)
    sa_pmi_linear.logger.set_max_iter(10000)
    
    print("\nRunning SA-PMI (Linear)...")
    policy_sapmi_lin, model_coop_lin, model_adv_lin = sa_pmi_linear.sa_pmi(
        initial_policy_sapmi, initial_model_sapmi
    )
    
    sa_pmi_linear.logger.print_final_summary()
    sa_pmi_linear.logger.close()
    
    # ==========================================
    # EXPERIMENT 3: SA-PMI with Exponential Curriculum
    # ==========================================
    print("\n" + "="*70)
    print("EXPERIMENT 3: SA-PMI (Exponential Curriculum)")
    print("="*70)
    
    env_sapmi_exp = create_teacher_student_env()
    conf_mdp_sapmi_exp = ConfigurableMDP(env_sapmi_exp)
    initial_policy_sapmi_exp, initial_model_sapmi_exp = create_initial_policy_model(env_sapmi_exp)
    
    # Configure wandb for SA-PMI Exponential
    wandb_config_exp = None
    if use_wandb:
        wandb_config_exp = {
            'project': wandb_project,
            'name': 'SA-PMI-Exponential',
            'config': {
                'algorithm': 'SA-PMI',
                'curriculum': 'exponential',
                'max_iter': 10000,
                'eps': 0.000001,
                'B_min': 0.001,
                'B_max': 0.05,
                'K_warmup': 10000,
                'environment': 'TeacherStudent',
                'n_literals': 2
            },
            'tags': ['adversarial', 'sa-pmi', 'exponential']
        }
    
    sa_pmi_exp = SPMI(
        conf_mdp=conf_mdp_sapmi_exp,
        eps=0.000001,
        policy_chooser=GreedyPolicyChooser(conf_mdp_sapmi_exp.nS, conf_mdp_sapmi_exp.nA),
        model_chooser=GreedyModelChooser(conf_mdp_sapmi_exp.nS, conf_mdp_sapmi_exp.nA),
        max_iter=10000,
        persistent=True,
        curriculum_schedule='exponential',
        B_min=0.001,
        B_max=0.05,
        K_warmup=10000
    )
    
    # Configure logger
    sa_pmi_exp.logger.verbose = verbose
    sa_pmi_exp.logger.log_interval = 500
    sa_pmi_exp.logger.use_wandb = use_wandb
    if use_wandb:
        sa_pmi_exp.logger._init_wandb(wandb_config_exp)
    sa_pmi_exp.logger.set_max_iter(10000)
    
    print("\nRunning SA-PMI (Exponential)...")
    policy_sapmi_exp, model_coop_exp, model_adv_exp = sa_pmi_exp.sa_pmi(
        initial_policy_sapmi_exp, initial_model_sapmi_exp
    )
    
    sa_pmi_exp.logger.print_final_summary()
    sa_pmi_exp.logger.close()
    
    # ==========================================
    # SAVE RESULTS
    # ==========================================
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    spmi_baseline.logger.save('results', 'teacher_student_spmi_baseline.csv')
    sa_pmi_linear.logger.save('results', 'teacher_student_sa_pmi_linear.csv')
    sa_pmi_exp.logger.save('results', 'teacher_student_sa_pmi_exponential.csv')
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SA-PMI experiments')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--project', type=str, default='sa-pmi-demo', help='W&B project name')
    parser.add_argument('--verbose', type=int, default=2, choices=[0,1,2,3],
                       help='Verbosity: 0=silent, 1=minimal, 2=normal, 3=detailed')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("STARTING SA-PMI EXPERIMENTS")
    print("="*70)
    print(f"Verbosity Level: {args.verbose}")
    print(f"Weights & Biases: {'Enabled' if args.wandb else 'Disabled'}")
    if args.wandb:
        print(f"W&B Project: {args.project}")
    print("="*70)
    
    try:
        results = run_comparison(use_wandb=args.wandb, wandb_project=args.project, verbose=args.verbose)
        print("\n" + "="*70)
        print("✓ EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*70)
    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ ERROR: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()