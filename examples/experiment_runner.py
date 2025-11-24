"""
Comprehensive Experiment Runner for SA-PMI Research
Supports: sweeps, multiple seeds, multiple environments, ablations
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from itertools import product
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.spmi import SPMI
from algorithm.policy_chooser import GreedyPolicyChooser
from algorithm.model_chooser import GreedyModelChooser
from utils.tabular import TabularPolicy, TabularModel
from envs.student_teacher import TeacherStudentEnv
from envs.gymnasium_envs import create_gym_env


class ExperimentConfig:
    """Configuration for experiments"""
    
    # Environment-specific recommended settings (research-grade)
    RECOMMENDED_SETTINGS = {
        'teacher_student_small': {
            'max_iter': 10000,
            'eps': 1e-7,
            'K_warmup': 7500,   # 75% of max_iter
            'verbose': 1,
            'log_interval': 250,
        },
        'teacher_student_medium': {
            'max_iter': 12000,
            'eps': 1e-7,
            'K_warmup': 9000,   # 75% of max_iter
            'verbose': 1,
            'log_interval': 300,
        },
        'frozen_lake_4x4': {
            'max_iter': 15000,
            'eps': 1e-7,
            'K_warmup': 11250,  # 75% of max_iter
            'verbose': 1,
            'log_interval': 375,
        },
        'frozen_lake_8x8': {
            'max_iter': 20000,
            'eps': 1e-8,
            'K_warmup': 15000,  # 75% of max_iter
            'verbose': 1,
            'log_interval': 500,
        },
        'taxi': {
            'max_iter': 20000,
            'eps': 1e-8,
            'K_warmup': 15000,  # 75% of max_iter
            'verbose': 1,
            'log_interval': 500,
        },
        'cartpole': {
            'max_iter': 25000,
            'eps': 1e-8,
            'K_warmup': 18750,  # 75% of max_iter
            'verbose': 1,
            'log_interval': 625,
        },
        'mountain_car': {
            'max_iter': 30000,
            'eps': 1e-9,
            'K_warmup': 22500,  # 75% of max_iter
            'verbose': 1,
            'log_interval': 750,
        },
        'acrobot': {
            'max_iter': 30000,
            'eps': 1e-9,
            'K_warmup': 22500,  # 75% of max_iter
            'verbose': 1,
            'log_interval': 750,
        }
    }
    
    # Default base configuration (fallback)
    BASE_CONFIG = {
        'max_iter': 20000,
        'eps': 1e-7,
        'verbose': 1,
        'log_interval': 500,
    }
    
    @staticmethod
    def get_base_config(environment_name):
        """Get environment-specific configuration"""
        if environment_name in ExperimentConfig.RECOMMENDED_SETTINGS:
            return ExperimentConfig.RECOMMENDED_SETTINGS[environment_name].copy()
        else:
            print(f"Warning: No recommended settings for {environment_name}, using defaults")
            return ExperimentConfig.BASE_CONFIG.copy()
    
    # Algorithm configurations
    ALGORITHMS = {
        'spmi': {
            'name': 'SPMI-Baseline',
            'curriculum_schedule': None,
            'tags': ['baseline', 'spmi']
        },
        'sa_pmi_sqrt': {
            'name': 'SA-PMI-Sqrt',
            'curriculum_schedule': 'sqrt',
            'B_min': 0.0,
            'B_max': 0.05,
            'tags': ['adversarial', 'sa-pmi', 'sqrt']
        },
        'sa_pmi_cosine': {
            'name': 'SA-PMI-Cosine',
            'curriculum_schedule': 'cosine',
            'B_min': 0.0,
            'B_max': 0.05,
            'tags': ['adversarial', 'sa-pmi', 'cosine']
        },
        'sa_pmi_exponential_improved': {
            'name': 'SA-PMI-Exponential-v2',
            'curriculum_schedule': 'exponential',
            'B_min': 0.0,
            'B_max': 0.05,
            'tags': ['adversarial', 'sa-pmi', 'exponential-v2']
        },
        'sa_pmi_smooth': {
            'name': 'SA-PMI-Smooth',
            'curriculum_schedule': 'smooth_exponential',
            'B_min': 0.0,
            'B_max': 0.05,
            'tags': ['adversarial', 'sa-pmi', 'smooth']
        },
        'sa_pmi_linear': {
            'name': 'SA-PMI-Linear',
            'curriculum_schedule': 'linear',
            'B_min': 0.0,
            'B_max': 0.05,
            'tags': ['adversarial', 'sa-pmi', 'linear']
        }
    }
    
    # Environment configurations
    ENVIRONMENTS = {
        # Teacher-Student (Original toy environment)
        'teacher_student_small': {
            'type': 'teacher_student',
            'n_literals': 2,
            'max_value': 1,
            'max_update': 1,
            'max_literals_in_examples': 2,
            'horizon': 10
        },
        'teacher_student_medium': {
            'type': 'teacher_student',
            'n_literals': 3,
            'max_value': 2,
            'max_update': 1,
            'max_literals_in_examples': 3,
            'horizon': 15
        },
        
        # Gymnasium Classic Control (Small discrete environments)
        'frozen_lake_4x4': {
            'type': 'gymnasium',
            'env_name': 'frozen_lake_4x4',
            'is_slippery': True,
            'horizon': 100
        },
        'frozen_lake_8x8': {
            'type': 'gymnasium',
            'env_name': 'frozen_lake_8x8',
            'is_slippery': True,
            'horizon': 200
        },
        'taxi': {
            'type': 'gymnasium',
            'env_name': 'taxi',
            'horizon': 200
        },
        
        # Gymnasium Classic Control (Tabularized continuous)
        'cartpole': {
            'type': 'gymnasium',
            'env_name': 'cartpole',
            'n_bins': 8,
            'horizon': 200
        },
        'mountain_car': {
            'type': 'gymnasium',
            'env_name': 'mountain_car',
            'n_bins': 10,
            'horizon': 200
        },
        'acrobot': {
            'type': 'gymnasium',
            'env_name': 'acrobot',
            'n_bins': 6,
            'horizon': 500
        }
    }
    
    # Hyperparameter sweep configurations (auto-adjust K_warmup based on environment)
    SWEEP_CONFIGS = {
        'sa_pmi_curriculum_sweep': {
            'B_max': [0.05],
            'B_min': [0.0],
            'curriculum_schedule': ['linear', 'exponential', 'sqrt', 'cosine', 'smooth_exponential']
            # K_warmup will be auto-set from RECOMMENDED_SETTINGS
        },
        'sa_pmi_budget_sweep': {
            'B_max': [0.01, 0.03, 0.05, 0.07, 0.1],
            'B_min': [0.0],
            'curriculum_schedule': ['linear', 'exponential', 'sqrt', 'cosine', 'smooth_exponential']
        },
        'sa_pmi_warmup_sweep': {
            'B_max': [0.05],
            'B_min': [0.0],
            'K_warmup_ratio': [0.50, 0.625, 0.75, 0.875],  # Ratios of max_iter
            'curriculum_schedule': ['linear', 'exponential', 'sqrt', 'cosine', 'smooth_exponential']
        },
        'sa_pmi_bmin_sweep': {
            'B_max': [0.05],
            'B_min': [0.0, 0.001, 0.005, 0.01],
            'curriculum_schedule': ['linear', 'exponential', 'sqrt', 'cosine', 'smooth_exponential']
        }
    }


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


def create_env(env_config):
    """Create environment from config"""
    env_type = env_config.get('type', 'teacher_student')
    
    if env_type == 'teacher_student':
        config = {k: v for k, v in env_config.items() if k != 'type'}
        env = TeacherStudentEnv(**config)
    elif env_type == 'gymnasium':
        config = {k: v for k, v in env_config.items() if k != 'type'}
        env = create_gym_env(**config)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
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


def run_single_experiment(algo_name, algo_config, env_name, env_config, 
                         seed, base_config, wandb_project):
    """
    Run a single experiment with specific configuration
    
    Returns: dict with results
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Create environment
    env = create_env(env_config)
    conf_mdp = ConfigurableMDP(env)
    initial_policy, initial_model = create_initial_policy_model(env)
    
    # Get environment-specific base config if not provided
    if base_config is None:
        base_config = ExperimentConfig.get_base_config(env_name)
        print(f"Using recommended settings for {env_name}:")
        print(f"  max_iter: {base_config['max_iter']}")
        print(f"  eps: {base_config['eps']}")
        if 'K_warmup' in base_config:
            print(f"  K_warmup: {base_config['K_warmup']}")
    
    # Determine K_warmup for adversarial algorithms
    K_warmup = None
    if 'K_warmup' in algo_config:
        K_warmup = algo_config['K_warmup']
    elif 'K_warmup_ratio' in algo_config:
        K_warmup = int(base_config['max_iter'] * algo_config['K_warmup_ratio'])
    elif 'K_warmup' in base_config:
        # Use environment's recommended K_warmup
        K_warmup = base_config['K_warmup']
    else:
        # Default: 75% of max_iter
        K_warmup = int(base_config['max_iter'] * 0.75)
    
    # Prepare wandb config
    full_config = {
        **base_config,
        **algo_config,
        'environment': env_name,
        'env_config': env_config,
        'seed': seed,
        'algorithm': algo_name
    }
    
    if K_warmup is not None:
        full_config['K_warmup'] = K_warmup
    
    wandb_config = {
        'project': wandb_project,
        'name': f"{algo_config['name']}_{env_name}_seed{seed}",
        'config': full_config,
        'tags': algo_config.get('tags', []) + [env_name, f'seed_{seed}'],
        'group': f"{algo_name}_{env_name}",
        'job_type': 'experiment'
    }
    
    # Create algorithm instance
    is_adversarial = algo_config.get('curriculum_schedule') is not None
    
    if is_adversarial:
        algorithm = SPMI(
            conf_mdp=conf_mdp,
            eps=base_config['eps'],
            policy_chooser=GreedyPolicyChooser(conf_mdp.nS, conf_mdp.nA),
            model_chooser=GreedyModelChooser(conf_mdp.nS, conf_mdp.nA),
            max_iter=base_config['max_iter'],
            persistent=True,
            curriculum_schedule=algo_config['curriculum_schedule'],
            B_min=algo_config.get('B_min', 0.0),
            B_max=algo_config.get('B_max', 0.05),
            K_warmup=K_warmup
        )
    else:
        algorithm = SPMI(
            conf_mdp=conf_mdp,
            eps=base_config['eps'],
            policy_chooser=GreedyPolicyChooser(conf_mdp.nS, conf_mdp.nA),
            model_chooser=GreedyModelChooser(conf_mdp.nS, conf_mdp.nA),
            max_iter=base_config['max_iter'],
            persistent=True
        )
    
    # Configure logger
    algorithm.logger.verbose = base_config['verbose']
    algorithm.logger.log_interval = base_config['log_interval']
    algorithm.logger.use_wandb = True
    algorithm.logger._init_wandb(wandb_config)
    algorithm.logger.set_max_iter(base_config['max_iter'])
    
    # Run algorithm
    print(f"\n{'='*70}")
    print(f"Running: {algo_name} on {env_name} (seed={seed})")
    if is_adversarial:
        print(f"  Schedule: {algo_config['curriculum_schedule']}")
        print(f"  K_warmup: {K_warmup} ({K_warmup/base_config['max_iter']*100:.1f}% of max_iter)")
    print(f"{'='*70}")
    
    if is_adversarial:
        policy, model_coop, model_adv = algorithm.sa_pmi(initial_policy, initial_model)
    else:
        policy, model = algorithm.spmi(initial_policy, initial_model)
        model_coop, model_adv = model, None
    
    # Extract results
    results = {
        'algorithm': algo_name,
        'environment': env_name,
        'seed': seed,
        'final_performance': algorithm.logger.evaluations[-1],
        'initial_performance': algorithm.logger.evaluations[0],
        'improvement': algorithm.logger.evaluations[-1] - algorithm.logger.evaluations[0],
        'iterations': algorithm.logger.iteration,
        'final_alpha': algorithm.logger.alfas[-1] if algorithm.logger.alfas else None,
        'final_beta': algorithm.logger.betas[-1] if algorithm.logger.betas else None,
        'max_iter': base_config['max_iter'],
        'eps': base_config['eps'],
    }
    
    if is_adversarial:
        results['K_warmup'] = K_warmup
        if algorithm.logger.adversarial_budgets:
            results['final_adversarial_budget'] = algorithm.logger.adversarial_budgets[-1]
    
    # Close logger
    algorithm.logger.close()
    
    return results


def run_benchmark(algorithms=None, environments=None, seeds=None, 
                 wandb_project="sa-pmi-benchmark", base_config=None):
    """
    Run full benchmark across algorithms, environments, and seeds
    """
    if algorithms is None:
        algorithms = list(ExperimentConfig.ALGORITHMS.keys())
    if environments is None:
        environments = list(ExperimentConfig.ENVIRONMENTS.keys())
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    
    print("\n" + "="*70)
    print("BENCHMARK CONFIGURATION")
    print("="*70)
    print(f"Algorithms: {algorithms}")
    print(f"Environments: {environments}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(algorithms) * len(environments) * len(seeds)}")
    print(f"W&B Project: {wandb_project}")
    print("="*70)
    
    all_results = []
    
    for algo_name in algorithms:
        algo_config = ExperimentConfig.ALGORITHMS[algo_name]
        
        for env_name in environments:
            env_config = ExperimentConfig.ENVIRONMENTS[env_name]
            
            for seed in seeds:
                try:
                    results = run_single_experiment(
                        algo_name, algo_config, env_name, env_config,
                        seed, base_config, wandb_project
                    )
                    all_results.append(results)
                    
                except Exception as e:
                    print(f"\n✗ ERROR in {algo_name}/{env_name}/seed{seed}: {e}")
                    import traceback
                    traceback.print_exc()
    
    save_benchmark_summary(all_results, wandb_project)
    
    return all_results


def run_hyperparameter_sweep(sweep_name, algorithm='sa_pmi_linear', 
                             environment='teacher_student_small',
                             seeds=None, wandb_project="sa-pmi-sweep"):
    """Run hyperparameter sweep"""
    if seeds is None:
        seeds = [0, 1, 2]
    
    sweep_config = ExperimentConfig.SWEEP_CONFIGS[sweep_name]
    base_algo_config = ExperimentConfig.ALGORITHMS.get(algorithm, {}).copy()
    env_config = ExperimentConfig.ENVIRONMENTS[environment]
    
    # Get environment-specific base config
    base_config = ExperimentConfig.get_base_config(environment)
    
    # Generate all hyperparameter combinations
    param_names = list(sweep_config.keys())
    param_values = [sweep_config[name] for name in param_names]
    combinations = list(product(*param_values))
    
    print("\n" + "="*70)
    print(f"HYPERPARAMETER SWEEP: {sweep_name}")
    print("="*70)
    print(f"Algorithm: {algorithm}")
    print(f"Environment: {environment}")
    print(f"Base Config: max_iter={base_config['max_iter']}, eps={base_config['eps']}")
    print(f"Parameters: {param_names}")
    print(f"Combinations: {len(combinations)}")
    print(f"Seeds per combination: {len(seeds)}")
    print(f"Total runs: {len(combinations) * len(seeds)}")
    print("="*70)
    
    all_results = []
    
    for combination in combinations:
        # Create modified config
        modified_config = base_algo_config.copy()
        for param_name, param_value in zip(param_names, combination):
            if param_name == 'K_warmup_ratio':
                # Convert ratio to actual K_warmup
                modified_config['K_warmup'] = int(base_config['max_iter'] * param_value)
            else:
                modified_config[param_name] = param_value
        
        # Add sweep identifier to name
        param_str = "_".join([f"{k}={v}" for k, v in zip(param_names, combination)])
        modified_config['name'] = f"{algorithm}_sweep_{param_str}"
        modified_config['tags'] = modified_config.get('tags', []) + ['sweep', sweep_name]
        
        for seed in seeds:
            try:
                results = run_single_experiment(
                    algorithm, modified_config, environment, env_config,
                    seed, base_config, wandb_project
                )
                # Add sweep parameters to results
                for param_name, param_value in zip(param_names, combination):
                    results[param_name] = param_value
                all_results.append(results)
                
            except Exception as e:
                print(f"\n✗ ERROR in sweep {param_str}/seed{seed}: {e}")
    
    save_sweep_summary(all_results, sweep_name, wandb_project)
    
    return all_results


def save_benchmark_summary(results, project_name):
    """Save benchmark results summary"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_summary_{project_name}_{timestamp}.json"
    
    os.makedirs('results/benchmarks', exist_ok=True)
    filepath = os.path.join('results/benchmarks', filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Benchmark summary saved to: {filepath}")
    
    print_summary_statistics(results)


def save_sweep_summary(results, sweep_name, project_name):
    """Save sweep results summary"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sweep_{sweep_name}_{timestamp}.json"
    
    os.makedirs('results/sweeps', exist_ok=True)
    filepath = os.path.join('results/sweeps', filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Sweep summary saved to: {filepath}")


def print_summary_statistics(results):
    """Print summary statistics from results"""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY STATISTICS")
    print("="*70)
    
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for r in results:
        key = (r['algorithm'], r['environment'])
        grouped[key].append(r['final_performance'])
    
    print(f"\n{'Algorithm':<30} {'Environment':<25} {'Mean±Std':<20} {'Min':<10} {'Max':<10}")
    print("-" * 95)
    
    for (algo, env), perfs in sorted(grouped.items()):
        mean = np.mean(perfs)
        std = np.std(perfs)
        min_val = np.min(perfs)
        max_val = np.max(perfs)
        print(f"{algo:<30} {env:<25} {mean:.4f}±{std:.4f}      {min_val:.4f}    {max_val:.4f}")
    
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SA-PMI experiments')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'benchmark', 'sweep'],
                       help='Experiment mode')
    parser.add_argument('--project', type=str, default='sa-pmi-experiments',
                       help='W&B project name')
    
    # Single run arguments
    parser.add_argument('--algorithm', type=str, 
                       choices=list(ExperimentConfig.ALGORITHMS.keys()),
                       help='Algorithm for single run')
    parser.add_argument('--environment', type=str,
                       choices=list(ExperimentConfig.ENVIRONMENTS.keys()),
                       help='Environment for single run')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for single run')
    
    # Benchmark arguments
    parser.add_argument('--algorithms', nargs='+',
                       help='Algorithms for benchmark (default: all)')
    parser.add_argument('--environments', nargs='+',
                       help='Environments for benchmark (default: all)')
    parser.add_argument('--seeds', nargs='+', type=int,
                       help='Seeds for benchmark (default: [0,1,2,3,4])')
    
    # Sweep arguments
    parser.add_argument('--sweep', type=str,
                       choices=list(ExperimentConfig.SWEEP_CONFIGS.keys()),
                       help='Sweep configuration name')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'single':
            if not args.algorithm or not args.environment:
                parser.error("--algorithm and --environment required for single mode")
            
            algo_config = ExperimentConfig.ALGORITHMS[args.algorithm]
            env_config = ExperimentConfig.ENVIRONMENTS[args.environment]
            base_config = ExperimentConfig.get_base_config(args.environment)
            base_config['verbose'] = 2  # More verbose for single runs
            
            results = run_single_experiment(
                args.algorithm, algo_config, args.environment, env_config,
                args.seed, base_config, args.project
            )
            print("\n" + "="*70)
            print("SINGLE RUN RESULTS")
            print("="*70)
            for key, value in results.items():
                print(f"{key}: {value}")
            print("="*70)
        
        elif args.mode == 'benchmark':
            results = run_benchmark(
                algorithms=args.algorithms,
                environments=args.environments,
                seeds=args.seeds,
                wandb_project=args.project
            )
            print("\n✓ BENCHMARK COMPLETED!")
        
        elif args.mode == 'sweep':
            if not args.sweep:
                parser.error("--sweep required for sweep mode")
            
            results = run_hyperparameter_sweep(
                sweep_name=args.sweep,
                algorithm=args.algorithm or 'sa_pmi_linear',
                environment=args.environment or 'teacher_student_small',
                seeds=args.seeds or [0, 1, 2],
                wandb_project=args.project
            )
            print("\n✓ SWEEP COMPLETED!")
    
    except Exception as e:
        print(f"\n✗ EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)