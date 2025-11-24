"""
SA-PMI Experiment Runner with YAML Configuration
Flexible CLI with per-run parameter overrides
"""

import numpy as np
import sys
import os
import json
import yaml
import argparse
from datetime import datetime
from itertools import product
import wandb
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.spmi import SPMI
from algorithm.policy_chooser import GreedyPolicyChooser
from algorithm.model_chooser import GreedyModelChooser
from utils.tabular import TabularPolicy, TabularModel
from envs.student_teacher import TeacherStudentEnv
from envs.gymnasium_envs import create_gym_env

import cProfile
import pstats

class ExperimentRunner:
    """Main experiment runner with YAML configuration"""
    
    def __init__(self, config_path='config/adversarial_experiments.yaml'):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.environments = self.config['environments']
        self.algorithms = self.config['algorithms']
        self.sweeps = self.config['sweeps']
        self.defaults = self.config['defaults']
        
    def get_env_config(self, env_name):
        """Get environment configuration"""
        if env_name not in self.environments:
            raise ValueError(f"Unknown environment: {env_name}. Available: {list(self.environments.keys())}")
        return self.environments[env_name]
    
    def get_algo_config(self, algo_name):
        """Get algorithm configuration"""
        if algo_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algo_name}. Available: {list(self.algorithms.keys())}")
        return self.algorithms[algo_name]
    
    def create_env(self, env_config):
        """Create environment from config"""
        env_type = env_config['type']
        params = env_config['params']
        
        if env_type == 'teacher_student':
            return TeacherStudentEnv(**params)
        elif env_type == 'gymnasium':
            return create_gym_env(**params)
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
    
    def create_initial_policy_model(self, env):
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
    
    def run_single(self, env_name, algo_name, seed=0, 
                   max_iter=None, eps=None, K_warmup_ratio=None,
                   B_max=None, B_min=None, wandb_project=None):
        """
        Run single experiment with optional parameter overrides
        
        Args:
            env_name: Environment name from config
            algo_name: Algorithm name from config
            seed: Random seed
            max_iter: Override max_iter from config
            eps: Override eps from config
            K_warmup_ratio: Override K_warmup_ratio
            B_max: Override B_max
            B_min: Override B_min
            wandb_project: W&B project name
        """
        np.random.seed(seed)
        
        # Get configs
        env_config = self.get_env_config(env_name)
        algo_config = self.get_algo_config(algo_name)
        
        # Apply overrides
        training_config = env_config['training'].copy()
        if max_iter is not None:
            training_config['max_iter'] = max_iter
            print(f"⚙️  Overriding max_iter: {max_iter}")
        if eps is not None:
            training_config['eps'] = eps
            print(f"⚙️  Overriding eps: {eps}")
        if K_warmup_ratio is not None:
            training_config['K_warmup_ratio'] = K_warmup_ratio
            print(f"⚙️  Overriding K_warmup_ratio: {K_warmup_ratio}")
        
        adversarial_config = env_config.get('adversarial', {}).copy()
        if B_max is not None:
            adversarial_config['B_max'] = B_max
            print(f"⚙️  Overriding B_max: {B_max}")
        if B_min is not None:
            adversarial_config['B_min'] = B_min
            print(f"⚙️  Overriding B_min: {B_min}")
        
        # Compute K_warmup
        K_warmup = int(training_config['max_iter'] * training_config['K_warmup_ratio'])
        
        # Create environment
        env = self.create_env(env_config)
        from algorithm.spmi import SPMI
        
        # Wrap in ConfigurableMDP
        class ConfigurableMDP:
            def __init__(self, env):
                self.env = env
                self.nS = env.nS
                self.nA = env.nA
                self.gamma = env.gamma
                self.horizon = env.horizon
                self.mu = env.mu
                self.P = env.P
                
            def set_model(self, model_rep):
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
        
        conf_mdp = ConfigurableMDP(env)
        initial_policy, initial_model = self.create_initial_policy_model(env)
        
        # Print configuration
        print(f"\n{'='*70}")
        print(f"🚀 Running: {algo_config['name']} on {env_name} (seed={seed})")
        print(f"{'='*70}")
        print(f"Environment:")
        print(f"  Type: {env_config['type']}")
        print(f"  Difficulty: {env_config.get('difficulty', 'unknown')}")
        print(f"  States: {conf_mdp.nS}, Actions: {conf_mdp.nA}")
        print(f"\nTraining:")
        print(f"  max_iter: {training_config['max_iter']}")
        print(f"  eps: {training_config['eps']}")
        print(f"  K_warmup: {K_warmup} ({training_config['K_warmup_ratio']*100:.1f}%)")
        
        is_adversarial = algo_config['curriculum_schedule'] is not None
        if is_adversarial:
            print(f"\nAdversarial:")
            print(f"  Schedule: {algo_config['curriculum_schedule']}")
            print(f"  B_max: {adversarial_config['B_max']}")
            print(f"  B_min: {adversarial_config['B_min']}")
        print(f"{'='*70}\n")

        profiler = cProfile.Profile()
        profiler.enable()

        # Create algorithm
        if is_adversarial:
            algorithm = SPMI(
                conf_mdp=conf_mdp,
                eps=training_config['eps'],
                policy_chooser=GreedyPolicyChooser(conf_mdp.nS, conf_mdp.nA),
                model_chooser=GreedyModelChooser(conf_mdp.nS, conf_mdp.nA),
                max_iter=training_config['max_iter'],
                persistent=True,
                curriculum_schedule=algo_config['curriculum_schedule'],
                B_min=adversarial_config['B_min'],
                B_max=adversarial_config['B_max'],
                K_warmup=K_warmup
            )
        else:
            algorithm = SPMI(
                conf_mdp=conf_mdp,
                eps=training_config['eps'],
                policy_chooser=GreedyPolicyChooser(conf_mdp.nS, conf_mdp.nA),
                model_chooser=GreedyModelChooser(conf_mdp.nS, conf_mdp.nA),
                max_iter=training_config['max_iter'],
                persistent=True
            )

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 slowest functions
        
        # Configure logger
        algorithm.logger.progress_style = training_config.get('progress_style', 'rich')
        algorithm.logger.verbose = training_config.get('verbose', 1)
        algorithm.logger.log_interval = training_config['log_interval']
        
        if wandb_project and self.defaults.get('use_wandb', True):
            algorithm.logger.use_wandb = True
            
            # Build adversarial config dict for wandb
            wandb_adversarial_config = adversarial_config if is_adversarial else {}
            
            wandb_config = {
                'project': wandb_project,
                'name': f"{algo_config['name']}_{env_name}_seed{seed}",
                'config': {
                    'environment': env_name,
                    'algorithm': algo_name,
                    'seed': seed,
                    **training_config,
                    **wandb_adversarial_config
                },
                'tags': algo_config['tags'] + [env_name, f'seed_{seed}'],
                'group': f"{algo_name}_{env_name}"
            }
            algorithm.logger._init_wandb(wandb_config)
        
        algorithm.logger.set_max_iter(training_config['max_iter'])

        profiler = cProfile.Profile()
        profiler.enable()
        # Run!
        if is_adversarial:
            policy, model_coop, model_adv = algorithm.sa_pmi(initial_policy, initial_model)
        else:
            policy, model = algorithm.spmi(initial_policy, initial_model)

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 slowest functions

        # Extract results
        results = {
            'algorithm': algo_name,
            'environment': env_name,
            'seed': seed,
            'final_performance': algorithm.logger.evaluations[-1],
            'initial_performance': algorithm.logger.evaluations[0],
            'improvement': algorithm.logger.evaluations[-1] - algorithm.logger.evaluations[0],
            'iterations': algorithm.logger.iteration,
            **training_config,
            **(adversarial_config if is_adversarial else {})
        }
        
        # Close logger
        algorithm.logger.close()
        
        # Print results
        print(f"\n{'='*70}")
        print(f"📊 RESULTS")
        print(f"{'='*70}")
        print(f"Initial performance: {results['initial_performance']:.6f}")
        print(f"Final performance:   {results['final_performance']:.6f}")
        print(f"Improvement:         {results['improvement']:.6f}")
        print(f"Iterations:          {results['iterations']}")
        
        # Performance validation
        if env_name in self.config.get('expected_performance', {}):
            expected = self.config['expected_performance'][env_name]
            final_perf = results['final_performance']
            
            if final_perf < expected['random'] * 1.5:
                print(f"\n⚠️  WARNING: Performance ({final_perf:.4f}) barely exceeds random ({expected['random']:.4f})")
                print(f"   Consider: increasing max_iter, adjusting B_max, or using deterministic variant")
            elif final_perf >= expected['good']:
                print(f"\n✅ GOOD: Performance ({final_perf:.4f}) exceeds good threshold ({expected['good']:.4f})")
            else:
                print(f"\n📈 Performance ({final_perf:.4f}) between random and good thresholds")
        
        print(f"{'='*70}\n")
        
        return results
    
    def run_sweep(self, sweep_name, environments=None, seeds=None, wandb_project=None):
        """Run hyperparameter sweep"""
        if sweep_name not in self.sweeps:
            raise ValueError(f"Unknown sweep: {sweep_name}. Available: {list(self.sweeps.keys())}")
        
        sweep_config = self.sweeps[sweep_name]
        
        # Use sweep-specific environments if specified
        if environments is None:
            environments = sweep_config.get('environments', list(self.environments.keys()))
        if seeds is None:
            seeds = self.defaults['seeds']
        
        parameters = sweep_config['parameters']
        param_names = list(parameters.keys())
        param_values = [parameters[name] for name in param_names]
        combinations = list(product(*param_values))
        
        print(f"\n{'='*70}")
        print(f"🔬 HYPERPARAMETER SWEEP: {sweep_name}")
        print(f"{'='*70}")
        print(f"Description: {sweep_config.get('description', 'N/A')}")
        print(f"Environments: {environments}")
        print(f"Parameters: {param_names}")
        print(f"Combinations: {len(combinations)}")
        print(f"Seeds: {seeds}")
        print(f"Total runs: {len(combinations) * len(environments) * len(seeds)}")
        print(f"{'='*70}\n")
        
        all_results = []
        
        for env_name in environments:
            for combination in combinations:
                # Build parameter dict
                params = dict(zip(param_names, combination))
                
                # Extract parameters for run_single
                schedule = params.pop('curriculum_schedule', None)
                if schedule:
                    # Find algorithm with this schedule
                    algo_name = next((k for k, v in self.algorithms.items() 
                                     if v['curriculum_schedule'] == schedule), 'spmi')
                else:
                    algo_name = 'spmi'
                
                for seed in seeds:
                    try:
                        results = self.run_single(
                            env_name=env_name,
                            algo_name=algo_name,
                            seed=seed,
                            wandb_project=wandb_project or f"{sweep_name}-sweep",
                            **params
                        )
                        results['sweep'] = sweep_name
                        results.update(params)
                        if schedule:
                            results['curriculum_schedule'] = schedule
                        all_results.append(results)
                        
                    except Exception as e:
                        print(f"\n❌ ERROR: {env_name}/{combination}/seed{seed}: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('results/sweeps', exist_ok=True)
        filepath = f"results/sweeps/sweep_{sweep_name}_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ Sweep results saved to: {filepath}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description='SA-PMI Experiment Runner with YAML Configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Single run with defaults from config
        python experiment_runner.py single --env frozen_lake_8x8 --algo sa_pmi_cosine --seed 0
        
        # Override parameters for difficult environment
        python experiment_runner.py single --env frozen_lake_8x8 --algo sa_pmi_cosine \\
            --max_iter 40000 --eps 1e-9 --K_warmup_ratio 0.85 --B_max 0.03
        
        # Run sweep
        python experiment_runner.py sweep --sweep curriculum_comparison --envs frozen_lake_8x8_det taxi
        
        # Frozen Lake 8x8 tuning sweep
        python experiment_runner.py sweep --sweep frozen_lake_tuning --project frozen-lake-tuning
                """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Experiment mode')
    
    # Single run parser
    single_parser = subparsers.add_parser('single', help='Run single experiment')
    single_parser.add_argument('--env', required=True, help='Environment name from config')
    single_parser.add_argument('--algo', required=True, help='Algorithm name from config')
    single_parser.add_argument('--seed', type=int, default=0, help='Random seed')
    single_parser.add_argument('--max_iter', type=int, help='Override max iterations')
    single_parser.add_argument('--eps', type=float, help='Override convergence threshold')
    single_parser.add_argument('--K_warmup_ratio', type=float, help='Override K_warmup ratio (0-1)')
    single_parser.add_argument('--B_max', type=float, help='Override max adversarial budget')
    single_parser.add_argument('--B_min', type=float, help='Override min adversarial budget')
    single_parser.add_argument('--project', help='W&B project name')
    single_parser.add_argument('--config', default='config/adversarial_experiments.yaml', help='Config file path')  # UPDATED
    
    # Sweep parser
    sweep_parser = subparsers.add_parser('sweep', help='Run hyperparameter sweep')
    sweep_parser.add_argument('--sweep', required=True, help='Sweep name from config')
    sweep_parser.add_argument('--envs', nargs='+', help='Environment names (default: from sweep config)')
    sweep_parser.add_argument('--seeds', nargs='+', type=int, help='Seeds (default: from config)')
    sweep_parser.add_argument('--project', help='W&B project name')
    sweep_parser.add_argument('--config', default='config/adversarial_experiments.yaml', help='Config file path')  # UPDATED
    
    # List parser
    list_parser = subparsers.add_parser('list', help='List available environments and algorithms')
    list_parser.add_argument('--config', default='config/adversarial_experiments.yaml', help='Config file path')  # UPDATED
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return
    
    # Initialize runner with updated default
    runner = ExperimentRunner(args.config if hasattr(args, 'config') else 'config/adversarial_experiments.yaml')  # UPDATED
    
    if args.mode == 'single':
        runner.run_single(
            env_name=args.env,
            algo_name=args.algo,
            seed=args.seed,
            max_iter=args.max_iter,
            eps=args.eps,
            K_warmup_ratio=args.K_warmup_ratio,
            B_max=args.B_max,
            B_min=args.B_min,
            wandb_project=args.project
        )
    
    elif args.mode == 'sweep':
        runner.run_sweep(
            sweep_name=args.sweep,
            environments=args.envs,
            seeds=args.seeds,
            wandb_project=args.project
        )
    
    elif args.mode == 'list':
        print("\n📋 Available Environments:")
        for name, config in runner.environments.items():
            difficulty = config.get('difficulty', '?')
            print(f"  • {name:<30} (difficulty: {difficulty})")
        
        print("\n🤖 Available Algorithms:")
        for name, config in runner.algorithms.items():
            schedule = config.get('curriculum_schedule', 'None')
            print(f"  • {name:<30} (schedule: {schedule})")
        
        print("\n🔬 Available Sweeps:")
        for name, config in runner.sweeps.items():
            desc = config.get('description', 'No description')
            print(f"  • {name:<30} - {desc}")


if __name__ == '__main__':
    main()