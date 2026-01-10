"""
Unified SPMI Experiment Runner
===============================
Runs all SPMI variants (Standard, SA-PMI, Federated, Federated SA-PMI)
with configurable parameters and WandB integration.

Usage:
    python run_experiments.py --config config.yaml
    python run_experiments.py --config config.yaml --experiment sapmi
    python run_experiments.py --config config.yaml --no-wandb
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import copy
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Import SPMI components
from algorithm.spmi import SPMI
from algorithm.policy_chooser import GreedyPolicyChooser, SAPMIPolicyChooser
from algorithm.model_chooser import (
    GPModelChooser, 
    SetModelChooser
)
from algorithm.curriculum_scheduler import (
    LinearCurriculumScheduler,
    ExponentialCurriculumScheduler,
    CosineCurriculumScheduler,
    ConstantCurriculumScheduler
)
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy
# from envs.student_teacher import TeacherStudentEnv
from envs.racetrack_simulator import RaceTrackConfigurableEnv
from federated import FSPMI

# Optional WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run: pip install wandb")


class ExperimentRunner:
    """Unified runner for all SPMI experiment variants"""
    
    def __init__(self, config_path: str, use_wandb: bool = True, n_seeds: int = 1):
        """Initialize experiment runner with configuration"""
        self.config = self._load_config(config_path)
        self.use_wandb = use_wandb and WANDB_AVAILABLE and self.config['wandb']['enabled']
        self.n_seeds = n_seeds
        # Setup output directory
        self.output_dir = Path(self.config['output']['dir']) / f'racetrack4_{self.config["environment"]["params"]["track_file"]}' / self.config['model_chooser']['type'] / time.strftime("%Y%m%d-%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment
        self.mdp = self._init_environment()
        
        # Initialize policy and model
        self.uniform_policy = UniformPolicy(self.mdp)
        self.original_model = copy.deepcopy(self.mdp.P)
        self.initial_model = TabularModel(self.mdp.P, self.mdp.nS, self.mdp.nA)
        self.initial_policy = TabularPolicy(
            self.uniform_policy.get_rep(), 
            self.mdp.nS, 
            self.mdp.nA
        )

        self.model_set = [TabularModel(self.mdp.P_highspeed_noboost, self.mdp.nS, self.mdp.nA),
                    TabularModel(self.mdp.P_lowspeed_noboost, self.mdp.nS, self.mdp.nA),
                    TabularModel(self.mdp.P_highspeed_boost, self.mdp.nS, self.mdp.nA),
                    TabularModel(self.mdp.P_lowspeed_boost, self.mdp.nS, self.mdp.nA)]
        
        # Results storage
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_environment(self) -> RaceTrackConfigurableEnv:
        """Initialize the MDP environment"""
        env_params = self.config['environment']['params']
        print(f"\nInitializing {self.config['environment']['name']} environment...")
        
        mdp = RaceTrackConfigurableEnv(**env_params)
        
        print(f"State space size: {mdp.nS}")
        print(f"Action space size: {mdp.nA}")
        print(f"Discount factor: {mdp.gamma}")
        print(f"Horizon: {mdp.horizon}")
        
        return mdp
    
    def _create_model_chooser(self):
        """Create model chooser based on config"""
        print(f"\nCreating Model Chooser {self.config['model_chooser']['type']}...")

        if self.config['model_chooser']['type'] == 'gp':
            gp_config = self.config['model_chooser']['gp']
            print(gp_config)
            chooser = GPModelChooser(
                self.model_set,
                self.mdp.nS,
                self.mdp.nA,
                init_model_vector=self.config['environment']['params']['initial_configuration'],
                beta=gp_config['beta'],
                original_model=self.original_model,
                gp_update_frequency=gp_config['gp_update_frequency']
            )
            return chooser
        else:  # greedy
            return SetModelChooser(
                self.model_set,
                self.mdp.nS,
                self.mdp.nA
            )

    
    def run_standard_spmi(self) -> Dict[str, Any]:
        """Run standard SPMI"""
        if not self.config['experiments']['standard_spmi']['enabled']:
            print("\nSkipping Standard SPMI (disabled in config)")
            return {}
        
        print("\n" + "=" * 80)
        print("Running Standard SPMI")
        print("=" * 80)
        
        exp_config = self.config['experiments']['standard_spmi']
        
        # self.mdp.set_model(copy.deepcopy(self.original_model))
        # if hasattr(self, 'init_model_vector'):
        #     self.mdp.model_vector = np.array(self.init_model_vector)
        
        policy_chooser = GreedyPolicyChooser(self.mdp.nS, self.mdp.nA)
        model_chooser = self._create_model_chooser()
        
        spmi = SPMI(
            self.mdp,
            eps=0.0,
            policy_chooser=policy_chooser,
            model_chooser=model_chooser,
            max_iter=exp_config['max_iter'],
            persistent=self.config['model_chooser']['type'] != 'gp',
            delta_q=1
        )
        
        # WandB logging
        if self.use_wandb:
            wandb.init(
                project=self.config['wandb']['project'],
                entity=self.config['wandb']['entity'],
                name="standard_spmi",
                tags=self.config['wandb']['tags'] + ["standard"],
                config={
                    "algorithm": "SPMI",
                    "variant": "standard",
                    "max_iter": exp_config['max_iter'],
                    "model_chooser": self.config['model_chooser']['type']
                },
                reinit=True
            )
        
        start_time = time.time()
        final_policy, final_model = spmi.spmi(
            copy.deepcopy(self.initial_policy),
            copy.deepcopy(self.initial_model)
        )
        elapsed = time.time() - start_time
        
        # Log to WandB
        if self.use_wandb:
            for i, perf in enumerate(spmi.logger.evaluations):
                wandb.log({"iteration": i, "performance": perf})
            wandb.log({
                "final_performance": spmi.logger.evaluations[-1],
                "total_iterations": spmi.logger.iteration,
                "elapsed_time": elapsed
            })
            wandb.finish()
        
        # Save results
        spmi.logger.save(str(self.output_dir), 'standard_spmi.csv')
        if self.config['output']['save_gp_metrics'] and hasattr(model_chooser, 'save_gp_times'):
            model_chooser.save_gp_times(str(self.output_dir / "standard_spmi"))
        
        print(f"Completed in {elapsed:.2f}s | Iterations: {spmi.logger.iteration} | "
              f"Performance: {spmi.logger.evaluations[-1]:.4f}")
        
        return {
            'spmi': spmi,
            'final_policy': final_policy,
            'final_model': final_model,
            'elapsed_time': elapsed
        }
    
    def run_federated_spmi(self, n_seeds: int = 1) -> Dict[str, Any]:
        """Run Federated SPMI with multiple seeds for confidence intervals"""
        if not self.config['experiments']['federated_spmi']['enabled']:
            print("\nSkipping Federated SPMI (disabled in config)")
            return {}
        
        print("\n" + "=" * 80)
        print(f"Running Federated SPMI ({n_seeds} seed(s))")
        print("=" * 80)
        
        exp_config = self.config['experiments']['federated_spmi']
        results = {}
        
        for agent_config in exp_config['configurations']:
            config_key = f"n{agent_config['n_agents']}_eps{agent_config['episodes_per_round']}"
            print(f"\n{config_key}")
            print("-" * 60)
            
            seed_results = []
            
            for seed in range(n_seeds):
                if n_seeds > 1:
                    print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)
                
                # Reset MDP
                self.mdp.set_model(copy.deepcopy(self.original_model))
                
                policy_chooser = GreedyPolicyChooser(self.mdp.nS, self.mdp.nA)
                model_chooser = self._create_model_chooser()
                
                fspmi = FSPMI(
                    conf_mdp=self.mdp,
                    n_agents=agent_config['n_agents'],
                    episodes_per_round=agent_config['episodes_per_round'],
                    eps=0.0,
                    max_rounds=agent_config['max_rounds'],
                    policy_chooser=policy_chooser,
                    model_chooser=model_chooser,
                    aggregation_method='weighted',
                    persistent=self.config['model_chooser']['type'] != 'gp',
                    delta_q=1,
                    verbose=(n_seeds == 1)  # Only verbose for single seed
                )
                
                # KEY: Set different seed for each agent in each run
                for i, agent in enumerate(fspmi.agents):
                    agent.rng = np.random.RandomState(seed * 10000 + i)
                
                start_time = time.time()
                final_policy, final_model = fspmi.run(
                    copy.deepcopy(self.initial_policy),
                    copy.deepcopy(self.initial_model)
                )
                elapsed = time.time() - start_time
                
                if n_seeds > 1:
                    perf = fspmi.logger.true_performances[-1] if fspmi.logger.true_performances else float('nan')
                    print(f"Perf: {perf:.4f}, Time: {elapsed:.1f}s")
                
                # Save individual seed result
                filename = f"fspmi_{config_key}_seed{seed}.csv"
                fspmi.logger.save(str(self.output_dir / filename))
                
                seed_results.append({
                    'fspmi': fspmi,
                    'final_policy': final_policy,
                    'final_model': final_model,
                    'elapsed': elapsed
                })
            
            # Save aggregated results if multiple seeds
            if n_seeds > 1:
                self._save_aggregated_results(config_key, seed_results)
            
            results[config_key] = seed_results
        
        return results


    def _save_aggregated_results(self, config_key: str, seed_results: List[Dict]):
        """Save mean ± std across seeds, including alpha and beta"""
        import pandas as pd
        
        # Find minimum length across seeds
        min_len = min(len(sr['fspmi'].logger.true_performances) for sr in seed_results)
        
        # Stack results
        true_perfs = np.array([sr['fspmi'].logger.true_performances[:min_len] for sr in seed_results])
        mc_perfs = np.array([sr['fspmi'].logger.performances[:min_len] for sr in seed_results])
        bounds = np.array([sr['fspmi'].logger.bounds[:min_len] for sr in seed_results])
        alphas = np.array([sr['fspmi'].logger.alphas[:min_len] for sr in seed_results])
        betas = np.array([sr['fspmi'].logger.betas[:min_len] for sr in seed_results])
        
        # Cumulative samples
        cum_samples = []
        for sr in seed_results:
            cumsum = np.cumsum(sr['fspmi'].logger.total_samples[:min_len])
            cum_samples.append(cumsum)
        cum_samples = np.array(cum_samples)
        
        # Create dataframe with all metrics
        df = pd.DataFrame({
            'iteration': range(min_len),
            'true_perf_mean': np.mean(true_perfs, axis=0),
            'true_perf_std': np.std(true_perfs, axis=0),
            'mc_perf_mean': np.mean(mc_perfs, axis=0),
            'mc_perf_std': np.std(mc_perfs, axis=0),
            'bound_mean': np.mean(bounds, axis=0),
            'bound_std': np.std(bounds, axis=0),
            'alpha_mean': np.mean(alphas, axis=0),
            'alpha_std': np.std(alphas, axis=0),
            'beta_mean': np.mean(betas, axis=0),
            'beta_std': np.std(betas, axis=0),
            'cum_samples_mean': np.mean(cum_samples, axis=0),
        })
        
        filename = f"fspmi_{config_key}_aggregated.csv"
        df.to_csv(self.output_dir / filename, index=False)
        print(f"  Saved: {filename}")
    
    
    def run_all(self, experiments: Optional[List[str]] = None):
        """Run all enabled experiments or specified experiments
        
        Args:
            experiments: List of experiment names to run. If None, runs all enabled.
                        Options: ['standard_spmi', 'federated_spmi']
        """
        print("=" * 80)
        print("UNIFIED SPMI EXPERIMENT RUNNER")
        print("=" * 80)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"WandB enabled: {self.use_wandb}")
        
        if experiments is None:
            experiments = ['standard_spmi', 'federated_spmi']
        
        # Run experiments
        if 'standard_spmi' in experiments:
            self.results['standard_spmi'] = self.run_standard_spmi()
        
        if 'federated_spmi' in experiments:
            self.results['federated_spmi'] = self.run_federated_spmi(n_seeds=self.n_seeds)
    
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print experiment summary"""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        
        print(f"\n{'Method':<50} {'Iterations/Rounds':<15} {'Final Performance':<20}")
        print("-" * 85)
        
        # Standard SPMI
        if 'standard_spmi' in self.results and self.results['standard_spmi']:
            spmi = self.results['standard_spmi']['spmi']
            perf = spmi.logger.evaluations[-1] if spmi.logger.evaluations else float('nan')
            print(f"{'Standard SPMI':<50} {spmi.logger.iteration:<15} {perf:.4f}")
        
        
        # Federated SPMI
        if 'federated_spmi' in self.results and self.results['federated_spmi']:
            for name, result in self.results['federated_spmi'].items():
                fspmi = result['fspmi']
                perf = fspmi.logger.true_performances[-1] if fspmi.logger.true_performances else float('nan')
                method_name = f"F-SPMI ({name})"
                print(f"{method_name:<50} {len(fspmi.logger.iterations):<15} {perf:.4f}")
        
        print("\n" + "=" * 80)
        print(f"✓ All experiments completed!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Unified SPMI Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments with default config
  python simulations/run_experiments.py
  
  # Use custom config
  python simulations/run_experiments.py --config my_config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml'
    )
    
    parser.add_argument(
        '--experiment',
        nargs='+',
        choices=['standard_spmi', 'sapmi', 'federated_spmi', 'federated_sapmi'],
        help='Specific experiments to run (default: all enabled in config)'
    )
    
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable WandB logging'
    )

    parser.add_argument(
        '--max_iter',
        type=int,
        help='Override max_iter for standard_spmi and sapmi experiments.'
    )
    
    parser.add_argument(
        '--max_rounds',
        type=int,
        help='Override max_rounds for federated_spmi and federated_sapmi configurations.'
    )
    
    parser.add_argument(
            '--model_chooser',
            choices=['gp', 'greedy', 'gb'],
            help="Override the model chooser type. 'gb' is an alias for 'greedy'."
        )
    
    # Add to argument parser
    parser.add_argument(
        '--n_agents',
        type=int,
        help='Override n_agents for all federated configs'
    )

    parser.add_argument(
        '--eps_per_round',
        type=int,
        help='Override episodes_per_round for all federated configs'
    )
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=1,
        help='Number of random seeds to run (for confidence intervals)'
    )
    parser.add_argument(
        '--track',
        type=str,
        choices=['T1', 'T2', 'T3', 'T4'],
        help='Override track file'
    )
    # -------------------------------
    
    args = parser.parse_args()


    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found!")
        print("Please create a config.yaml file or specify a valid config path.")
        sys.exit(1)
    
    # Initialize and run experiments
    runner = ExperimentRunner(
        config_path=args.config,
        use_wandb=not args.no_wandb,
        n_seeds=args.n_seeds
    )
    
    # --- Apply Overrides to Runner Config ---
    # Then in the override section:
    if args.n_agents is not None:
        print(f"Applying N_AGENTS override: {args.n_agents}")
        for config in runner.config['experiments']['federated_spmi']['configurations']:
            config['n_agents'] = args.n_agents

    if args.eps_per_round is not None:
        print(f"Applying EPS_PER_ROUND override: {args.eps_per_round}")
        for config in runner.config['experiments']['federated_spmi']['configurations']:
            config['episodes_per_round'] = args.eps_per_round
            
    if args.model_chooser is not None:
            # Handle alias: 'gb' -> 'greedy'
            mc_type = 'greedy' if args.model_chooser == 'gb' else args.model_chooser
            
            print(f"Applying MODEL_CHOOSER override: {mc_type}")
            runner.config['model_chooser']['type'] = mc_type

            # Optional: If switching to greedy, ensure parameters exist
            if mc_type == 'greedy' and 'greedy' not in runner.config['model_chooser']:
                runner.config['model_chooser']['greedy'] = {'do_not_create_transitions': True}

    if args.max_iter is not None:
        print(f"Applying MAX_ITER override: {args.max_iter}")
        runner.config['experiments']['standard_spmi']['max_iter'] = args.max_iter
    
    if args.max_rounds is not None:
        print(f"Applying MAX_ROUNDS override: {args.max_rounds}")
        # Federated SPMI
        for config in runner.config['experiments']['federated_spmi']['configurations']:
            config['max_rounds'] = args.max_rounds

    if args.track is not None:
        print(f"Applying TRACK override: {args.track}")
        runner.config['environment']['params']['track_file'] = args.track

# ----  ------------------------------------------
    
    runner.run_all(experiments=args.experiment)


if __name__ == '__main__':
    main()