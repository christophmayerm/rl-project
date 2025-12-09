"""
SA-PMI Experiment Runner with YAML Configuration
Now supports: SPMI, SA-PMI variants, Value Iteration, Policy Iteration, Q-Learning
Flexible CLI with per-run parameter overrides
Integrated with IterationMetricsEvaluator for auxiliary metrics
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
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.spmi import SPMI
from algorithm.policy_chooser import GreedyPolicyChooser
from algorithm.model_chooser import GreedyModelChooser
from algorithm.value_iteration import ValueIteration, PolicyIteration  # NEW
from algorithm.q_learning import QLearning  # NEW
from utils.tabular import TabularPolicy, TabularModel, TabularReward
from envs.student_teacher import TeacherStudentEnv
from envs.gymnasium_envs import create_gym_env
from utils.iteration_metrics_evaluator import IterationMetricsEvaluator, EvaluationOptions
from utils import evaluator

import cProfile
import pstats


class ExperimentRunner:
    """Main experiment runner with YAML configuration and metrics evaluation"""
    
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
    
    def create_env(self, env_config, seed=None):
        """Create environment from config"""
        env_type = env_config['type']
        params = env_config['params']
        
        if env_type == 'teacher_student':
            env = TeacherStudentEnv(**params)
        elif env_type == 'gymnasium':
            env = create_gym_env(**params)
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
        
        # Set seed AFTER environment creation
        if seed is not None and hasattr(env, 'seed'):
            env.seed(seed)
        
        return env
    
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
    
    def setup_metrics_evaluator(self, conf_mdp, nominal_model, output_dir=None, 
                                enable_metrics=True):
        """
        Setup IterationMetricsEvaluator for auxiliary metrics tracking
        
        Args:
            conf_mdp: ConfigurableMDP instance
            nominal_model: Nominal model for divergence tracking
            output_dir: Directory for saving visualizations
            enable_metrics: Whether to enable metrics evaluation
        
        Returns:
            IterationMetricsEvaluator instance or None if disabled
        """
        if not enable_metrics:
            return None
        
        # Check if environment supports visualization
        has_position_viz = hasattr(conf_mdp.env, 'lin') and hasattr(conf_mdp.env, 'nrow')
        
        eval_options = EvaluationOptions(
            state_coverage=True,
            state_entropy=True,
            reward_diversity=True,
            novelty_yield=True,
            model_divergence=True,  # Track adversarial model divergence
            store_visitations=has_position_viz,  # Only store if we can visualize
            coverage_threshold=None  # Auto-computed based on state distribution
        )
        
        # Setup live visualization if supported and output dir provided
        live_vis_path = None
        if has_position_viz and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            live_vis_path = os.path.join(output_dir, "position_visitation_live.png")
        
        metrics_evaluator = IterationMetricsEvaluator(
            mdp=conf_mdp,
            options=eval_options,
            reference_model=nominal_model,
            live_vis_path=live_vis_path,
            live_vis_normalize=True
        )
        
        return metrics_evaluator
    
    def run_single(self, env_name, algo_name, seed=0, 
                   max_iter=None, eps=None, K_warmup_ratio=None,
                   B_max=None, B_min=None, wandb_project=None,
                   enable_metrics=True, save_metrics=True,
                   # NEW: Q-Learning specific parameters
                   alpha=None, epsilon=None, epsilon_decay=None, episodes=None):
        """
        Run single experiment with optional parameter overrides
        
        Args:
            env_name: Environment name from config
            algo_name: Algorithm name from config
            seed: Random seed
            max_iter: Override max_iter from config
            eps: Override eps from config (convergence threshold for VI/PI/SPMI)
            K_warmup_ratio: Override K_warmup_ratio
            B_max: Override B_max
            B_min: Override B_min
            wandb_project: W&B project name
            enable_metrics: Enable auxiliary metrics evaluation
            save_metrics: Save metrics to CSV
            alpha: Q-Learning learning rate
            epsilon: Q-Learning exploration rate
            epsilon_decay: Q-Learning epsilon decay
            episodes: Q-Learning number of episodes
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
        env = self.create_env(env_config, seed=seed)
        
        # Wrap in ConfigurableMDP
        # In experiment_runner.py, around line 160
        class ConfigurableMDP:
            def __init__(self, env):
                self.env = env
                self.nS = env.nS
                self.nA = env.nA
                self.gamma = env.gamma
                self.horizon = env.horizon
                self.mu = env.mu
                self.P = env.P
                
                # NEW: Add env_name for compatibility
                if hasattr(env, 'spec') and hasattr(env.spec, 'id'):
                    self.env_name = env.spec.id
                elif hasattr(env, 'name'):
                    self.env_name = env.name
                else:
                    self.env_name = "Unknown"
                
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
            
            def get_model(self):
                """Get current model from environment"""
                model_rep = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
                for s in range(self.nS):
                    for a in range(self.nA):
                        transitions = self.P[s][a]
                        for prob, s_next, reward, done in transitions:
                            if prob > 0:
                                model_rep[s][a].append((prob, s_next))
                return TabularModel(model_rep, self.nS, self.nA)
        
        conf_mdp = ConfigurableMDP(env)
        initial_policy, initial_model = self.create_initial_policy_model(env)
        
        # Setup output directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/runs/{algo_name}_{env_name}_seed{seed}_{timestamp}"
        
        # Setup metrics evaluator
        nominal_model = copy.deepcopy(initial_model)
        metrics_evaluator = self.setup_metrics_evaluator(
            conf_mdp=conf_mdp,
            nominal_model=nominal_model,
            output_dir=output_dir,
            enable_metrics=enable_metrics
        )
        
        # Determine algorithm type
        algo_type = algo_config.get('type', 'spmi')  # 'spmi', 'value_iteration', 'policy_iteration', 'q_learning'
        is_adversarial = algo_config.get('curriculum_schedule') is not None
        
        # Print configuration
        print(f"\n{'='*70}")
        print(f"🚀 Running: {algo_config['name']} on {env_name} (seed={seed})")
        print(f"{'='*70}")
        print(f"Environment:")
        print(f"  Type: {env_config['type']}")
        print(f"  Difficulty: {env_config.get('difficulty', 'unknown')}")
        print(f"  States: {conf_mdp.nS}, Actions: {conf_mdp.nA}")
        print(f"\nAlgorithm Type: {algo_type}")
        
        if algo_type in ['spmi', 'sa_pmi']:
            print(f"\nTraining:")
            print(f"  max_iter: {training_config['max_iter']}")
            print(f"  eps: {training_config['eps']}")
            print(f"  K_warmup: {K_warmup} ({training_config['K_warmup_ratio']*100:.1f}%)")
            
            if is_adversarial:
                print(f"\nAdversarial:")
                print(f"  Schedule: {algo_config['curriculum_schedule']}")
                print(f"  B_max: {adversarial_config['B_max']}")
                print(f"  B_min: {adversarial_config['B_min']}")
        
        elif algo_type in ['value_iteration', 'policy_iteration']:
            print(f"\nTraining:")
            print(f"  max_iter: {training_config['max_iter']}")
            print(f"  eps: {training_config['eps']}")
        
        elif algo_type == 'q_learning':
            # Q-Learning specific parameters
            qlearn_episodes = episodes or training_config.get('episodes', 50000)
            qlearn_alpha = alpha or 0.1
            qlearn_epsilon = epsilon or 0.1
            qlearn_epsilon_decay = epsilon_decay or 0.9995
            
            print(f"\nQ-Learning:")
            print(f"  episodes: {qlearn_episodes}")
            print(f"  alpha (learning rate): {qlearn_alpha}")
            print(f"  epsilon (exploration): {qlearn_epsilon}")
            print(f"  epsilon_decay: {qlearn_epsilon_decay}")
        
        if enable_metrics:
            print(f"\n📊 Metrics Evaluation:")
            print(f"  State coverage: ✓")
            print(f"  State entropy: ✓")
            print(f"  Reward diversity: ✓")
            print(f"  Novelty yield: ✓")
            print(f"  Model divergence: ✓")
            if metrics_evaluator and metrics_evaluator.live_vis_path:
                print(f"  Live visualization: {metrics_evaluator.live_vis_path}")
        
        print(f"{'='*70}\n")

        # Profile training
        profiler = cProfile.Profile()
        profiler.enable()
        
        # ==========================================
        # CREATE AND RUN ALGORITHM
        # ==========================================
        
        if algo_type in ['value_iteration', 'policy_iteration']:
            # Value Iteration or Policy Iteration
            if algo_type == 'value_iteration':
                algorithm = ValueIteration(
                    conf_mdp=conf_mdp,
                    eps=training_config['eps'],
                    max_iter=training_config['max_iter']
                )
                policy, V, iterations = algorithm.solve(initial_model=initial_model)
            else:  # policy_iteration
                algorithm = PolicyIteration(
                    conf_mdp=conf_mdp,
                    eps=training_config['eps'],
                    max_iter=training_config['max_iter']
                )
                policy, V, iterations = algorithm.solve(
                    initial_policy=initial_policy,
                    initial_model=initial_model
                )
            
            # Evaluate final performance
            reward = TabularReward(conf_mdp.env.P, conf_mdp.nS, conf_mdp.nA)
            final_perf = evaluator.compute_performance(
                conf_mdp.mu, reward, policy, conf_mdp.get_model(),
                conf_mdp.gamma, conf_mdp.horizon, conf_mdp.nS, conf_mdp.nA
            )
            
            # Store in algorithm.logger for compatibility
            algorithm.logger.evaluations.append(final_perf)
            algorithm.logger.iteration = iterations
            
        elif algo_type == 'q_learning':
            # Q-Learning
            algorithm = QLearning(
                conf_mdp=conf_mdp,
                alpha=qlearn_alpha,
                gamma=conf_mdp.gamma,
                epsilon=qlearn_epsilon,
                epsilon_decay=qlearn_epsilon_decay,
                episodes=qlearn_episodes,
                eval_frequency=training_config.get('eval_frequency', 1000)
            )
            policy = algorithm.train()
            
        else:  # SPMI or SA-PMI variants
            # Create SPMI algorithm with metrics evaluator
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
                    K_warmup=K_warmup,
                    metrics_evaluator=metrics_evaluator
                )
            else:
                algorithm = SPMI(
                    conf_mdp=conf_mdp,
                    eps=training_config['eps'],
                    policy_chooser=GreedyPolicyChooser(conf_mdp.nS, conf_mdp.nA),
                    model_chooser=GreedyModelChooser(conf_mdp.nS, conf_mdp.nA),
                    max_iter=training_config['max_iter'],
                    persistent=True,
                    metrics_evaluator=metrics_evaluator
                )

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
                        'enable_metrics': enable_metrics,
                        **training_config,
                        **wandb_adversarial_config
                    },
                    'tags': algo_config['tags'] + [env_name, f'seed_{seed}'],
                    'group': f"{algo_name}_{env_name}"
                }
                algorithm.logger._init_wandb(wandb_config)
            
            algorithm.logger.set_max_iter(training_config['max_iter'])
            
            # Run!
            if is_adversarial:
                policy, model_coop, model_adv = algorithm.sa_pmi(
                    initial_policy, 
                    initial_model,
                    nominal_model=nominal_model
                )
            else:
                policy, model = algorithm.spmi(initial_policy, initial_model)

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Only print profiling stats if verbose
        if training_config.get('verbose', 1) >= 2:
            print("\n" + "="*70)
            print("🔍 PROFILING STATS (Top 20)")
            print("="*70)
            stats.print_stats(20)

        # Save metrics if enabled AND metrics were actually collected
        if metrics_evaluator and save_metrics and metrics_evaluator.history:
            os.makedirs(output_dir, exist_ok=True)
            metrics_evaluator.save(output_dir, "auxiliary_metrics.csv")
            print(f"\n📊 Auxiliary metrics saved to: {output_dir}/auxiliary_metrics.csv")
        elif metrics_evaluator and save_metrics and not metrics_evaluator.history:
            print(f"\n⚠️  No metrics collected (evaluation may have been skipped)")
            
            # Save final position visitation if available
            if metrics_evaluator.visitation_history:
                try:
                    final_vis_path = os.path.join(output_dir, "position_visitation_final.png")
                    metrics_evaluator.visualize_position_visitation(
                        final_vis_path, 
                        iteration=-1, 
                        normalize=True
                    )
                    print(f"📊 Final visitation map saved to: {final_vis_path}")
                except Exception as e:
                    print(f"⚠️  Could not save final visitation map: {e}")

        # Extract results
        results = {
            'algorithm': algo_name,
            'algorithm_type': algo_type,
            'environment': env_name,
            'seed': seed,
            'final_performance': algorithm.logger.evaluations[-1] if algorithm.logger.evaluations else 0.0,
            'initial_performance': algorithm.logger.evaluations[0] if algorithm.logger.evaluations else 0.0,
            'improvement': (algorithm.logger.evaluations[-1] - algorithm.logger.evaluations[0]) if len(algorithm.logger.evaluations) > 0 else 0.0,
            'iterations': algorithm.logger.iteration,
            'output_dir': output_dir,
            **training_config,
            **(adversarial_config if is_adversarial else {})
        }
        
        # Add final metrics summary if available
        if metrics_evaluator and metrics_evaluator.history:
            final_metrics = metrics_evaluator.history[-1]
            results['final_state_coverage'] = final_metrics.get('state_coverage', None)
            results['final_state_entropy'] = final_metrics.get('state_entropy', None)
            results['final_reward_mean'] = final_metrics.get('reward_mean', None)
            results['final_novelty_yield'] = final_metrics.get('novelty_yield', None)
            if is_adversarial:
                results['final_model_tv_mean'] = final_metrics.get('model_tv_mean', None)
                results['final_model_tv_max'] = final_metrics.get('model_tv_max', None)
        
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
        
        # Print auxiliary metrics if available
        if enable_metrics and metrics_evaluator and metrics_evaluator.history:
            print(f"\n📈 Final Auxiliary Metrics:")
            print(f"  State coverage:    {results.get('final_state_coverage', 'N/A')}")
            print(f"  State entropy:     {results.get('final_state_entropy', 'N/A')}")
            print(f"  Reward mean:       {results.get('final_reward_mean', 'N/A')}")
            print(f"  Novelty yield:     {results.get('final_novelty_yield', 'N/A')}")
            if is_adversarial:
                print(f"  Model TV (mean):   {results.get('final_model_tv_mean', 'N/A')}")
                print(f"  Model TV (max):    {results.get('final_model_tv_max', 'N/A')}")
        
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
    
    def run_sweep(self, sweep_name, environments=None, seeds=None, wandb_project=None,
                  enable_metrics=True):
        """Run hyperparameter sweep with metrics evaluation"""
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
        print(f"Metrics evaluation: {'Enabled' if enable_metrics else 'Disabled'}")
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
                            enable_metrics=enable_metrics,
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

    def run_benchmark(self, env_name, algo_names=None, seeds=None, 
                    max_iter=None, eps=None, K_warmup_ratio=None,
                    B_max=None, B_min=None, wandb_project=None,
                    enable_metrics=True):
        """
        Run benchmark: multiple algorithms with multiple seeds on single environment
        
        Args:
            env_name: Environment name from config
            algo_names: List of algorithm names (default: all from config)
            seeds: List of seeds (default: from config)
            enable_metrics: Enable auxiliary metrics evaluation
            ... (other override params)
        """
        if algo_names is None:
            algo_names = list(self.algorithms.keys())
        if seeds is None:
            seeds = self.defaults['seeds']
        
        print(f"\n{'='*70}")
        print(f"🏆 BENCHMARK: {env_name}")
        print(f"{'='*70}")
        print(f"Algorithms: {algo_names}")
        print(f"Seeds: {seeds}")
        print(f"Total runs: {len(algo_names) * len(seeds)}")
        print(f"Metrics evaluation: {'Enabled' if enable_metrics else 'Disabled'}")
        print(f"{'='*70}\n")
        
        all_results = []
        
        for algo_name in algo_names:
            for seed in seeds:
                try:
                    print(f"\n{'='*70}")
                    print(f"Running: {algo_name} (seed={seed})")
                    print(f"{'='*70}\n")
                    
                    results = self.run_single(
                        env_name=env_name,
                        algo_name=algo_name,
                        seed=seed,
                        max_iter=max_iter,
                        eps=eps,
                        K_warmup_ratio=K_warmup_ratio,
                        B_max=B_max,
                        B_min=B_min,
                        wandb_project=wandb_project or f"benchmark_{env_name}",
                        enable_metrics=enable_metrics
                    )
                    
                    results['benchmark_env'] = env_name
                    all_results.append(results)
                    
                except Exception as e:
                    print(f"\n❌ ERROR: {algo_name}/seed{seed}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Save benchmark results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('results/benchmarks', exist_ok=True)
        filepath = f"results/benchmarks/benchmark_{env_name}_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary statistics
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARK SUMMARY: {env_name}")
        print(f"{'='*70}\n")
        
        import pandas as pd
        df = pd.DataFrame(all_results)
        
        agg_dict = {
            'final_performance': ['mean', 'std', 'min', 'max'],
            'improvement': ['mean', 'std'],
            'iterations': ['mean', 'std']
        }
        
        # Add metrics columns if available
        if enable_metrics:
            metric_cols = ['final_state_coverage', 'final_state_entropy', 
                          'final_novelty_yield', 'final_model_tv_mean']
            for col in metric_cols:
                if col in df.columns:
                    agg_dict[col] = ['mean', 'std']
        
        summary = df.groupby('algorithm').agg(agg_dict).round(6)
        
        print(summary)
        print(f"\n{'='*70}")
        print(f"✅ Benchmark results saved to: {filepath}")
        print(f"{'='*70}\n")
        
        return all_results, summary


def main():
    parser = argparse.ArgumentParser(
        description='Experiment Runner: SPMI, SA-PMI, Value Iteration, Policy Iteration, Q-Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SPMI/SA-PMI runs
  python examples/experiment_runner.py single --env teacher_student_small --algo sa_pmi_cosine --seed 0
  
  # Value Iteration (fast, optimal for known models)
  python examples/experiment_runner.py single --env frozen_lake_8x8 --algo value_iteration --seed 0
  
  # Policy Iteration (even faster)
  python examples/experiment_runner.py single --env frozen_lake_8x8 --algo policy_iteration --seed 0
  
  # Q-Learning (model-free)
  python examples/experiment_runner.py single --env frozen_lake_8x8 --algo q_learning --seed 0 --episodes 50000
  
  # Benchmark classical vs adversarial algorithms
  python examples/experiment_runner.py benchmark --env frozen_lake_8x8 \\
      --algos spmi sa_pmi_linear value_iteration policy_iteration q_learning
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Experiment mode')
    
    # Single run parser
    single_parser = subparsers.add_parser('single', help='Run single experiment')
    single_parser.add_argument('--env', required=True, help='Environment name from config')
    single_parser.add_argument('--algo', required=True, help='Algorithm name from config')
    single_parser.add_argument('--seed', type=int, default=0, help='Random seed')
    single_parser.add_argument('--max_iter', type=int, help='Override max iterations (VI/PI/SPMI)')
    single_parser.add_argument('--eps', type=float, help='Override convergence threshold (VI/PI/SPMI)')
    single_parser.add_argument('--K_warmup_ratio', type=float, help='Override K_warmup ratio (SPMI)')
    single_parser.add_argument('--B_max', type=float, help='Override max adversarial budget (SA-PMI)')
    single_parser.add_argument('--B_min', type=float, help='Override min adversarial budget (SA-PMI)')
    # Q-Learning specific
    single_parser.add_argument('--alpha', type=float, help='Q-Learning learning rate')
    single_parser.add_argument('--epsilon', type=float, help='Q-Learning exploration rate')
    single_parser.add_argument('--epsilon_decay', type=float, help='Q-Learning epsilon decay')
    single_parser.add_argument('--episodes', type=int, help='Q-Learning number of episodes')
    # General
    single_parser.add_argument('--project', help='W&B project name')
    single_parser.add_argument('--no-metrics', action='store_true', help='Disable auxiliary metrics evaluation')
    single_parser.add_argument('--config', default='config/adversarial_experiments.yaml', help='Config file path')
    
    # Sweep parser
    sweep_parser = subparsers.add_parser('sweep', help='Run hyperparameter sweep')
    sweep_parser.add_argument('--sweep', required=True, help='Sweep name from config')
    sweep_parser.add_argument('--envs', nargs='+', help='Environment names (default: from sweep config)')
    sweep_parser.add_argument('--seeds', nargs='+', type=int, help='Seeds (default: from config)')
    sweep_parser.add_argument('--project', help='W&B project name')
    sweep_parser.add_argument('--no-metrics', action='store_true', help='Disable auxiliary metrics evaluation')
    sweep_parser.add_argument('--config', default='config/adversarial_experiments.yaml', help='Config file path')

    # Benchmark parser
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark: multiple algos, multiple seeds')
    benchmark_parser.add_argument('--env', required=True, help='Environment name from config')
    benchmark_parser.add_argument('--algos', nargs='+', help='Algorithm names (default: all)')
    benchmark_parser.add_argument('--seeds', nargs='+', type=int, help='Seeds (default: from config)')
    benchmark_parser.add_argument('--max_iter', type=int, help='Override max iterations')
    benchmark_parser.add_argument('--eps', type=float, help='Override convergence threshold')
    benchmark_parser.add_argument('--K_warmup_ratio', type=float, help='Override K_warmup ratio (SPMI)')
    benchmark_parser.add_argument('--B_max', type=float, help='Override max adversarial budget')
    benchmark_parser.add_argument('--B_min', type=float, help='Override min adversarial budget')
    benchmark_parser.add_argument('--project', help='W&B project name')
    benchmark_parser.add_argument('--no-metrics', action='store_true', help='Disable auxiliary metrics evaluation')
    benchmark_parser.add_argument('--config', default='config/adversarial_experiments.yaml', help='Config file path')
    
    # List parser
    list_parser = subparsers.add_parser('list', help='List available environments and algorithms')
    list_parser.add_argument('--config', default='config/adversarial_experiments.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return
    
    # Initialize runner
    runner = ExperimentRunner(args.config if hasattr(args, 'config') else 'config/adversarial_experiments.yaml')
    
    # Determine if metrics should be enabled
    enable_metrics = not getattr(args, 'no_metrics', False)
    
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
            wandb_project=args.project,
            enable_metrics=enable_metrics,
            # Q-Learning params
            alpha=getattr(args, 'alpha', None),
            epsilon=getattr(args, 'epsilon', None),
            epsilon_decay=getattr(args, 'epsilon_decay', None),
            episodes=getattr(args, 'episodes', None)
        )
    
    elif args.mode == 'sweep':
        runner.run_sweep(
            sweep_name=args.sweep,
            environments=args.envs,
            seeds=args.seeds,
            wandb_project=args.project,
            enable_metrics=enable_metrics
        )

    elif args.mode == 'benchmark':
        runner.run_benchmark(
            env_name=args.env,
            algo_names=args.algos,
            seeds=args.seeds,
            max_iter=args.max_iter,
            eps=args.eps,
            K_warmup_ratio=args.K_warmup_ratio,
            B_max=args.B_max,
            B_min=args.B_min,
            wandb_project=args.project,
            enable_metrics=enable_metrics
        )
    
    elif args.mode == 'list':
        print("\n📋 Available Environments:")
        for name, config in runner.environments.items():
            difficulty = config.get('difficulty', '?')
            print(f"  • {name:<30} (difficulty: {difficulty})")
        
        print("\n🤖 Available Algorithms:")
        for name, config in runner.algorithms.items():
            algo_type = config.get('type', 'spmi')
            schedule = config.get('curriculum_schedule', 'None')
            print(f"  • {name:<30} (type: {algo_type}, schedule: {schedule})")
        
        print("\n🔬 Available Sweeps:")
        for name, config in runner.sweeps.items():
            desc = config.get('description', 'No description')
            print(f"  • {name:<30} - {desc}")


if __name__ == '__main__':
    main()