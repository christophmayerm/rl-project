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
    
    def __init__(self, config_path: str, use_wandb: bool = True):
        """Initialize experiment runner with configuration"""
        self.config = self._load_config(config_path)
        self.use_wandb = use_wandb and WANDB_AVAILABLE and self.config['wandb']['enabled']
        
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
        
        # mdp = TeacherStudentEnv(**env_params)
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
    
    def _create_curriculum_scheduler(self, curriculum_config: Dict[str, Any]):
        """Create curriculum scheduler from config"""
        curr_type = curriculum_config['type']
        
        if curr_type == 'constant':
            return ConstantCurriculumScheduler(weight=curriculum_config['weight'])
        
        elif curr_type == 'linear':
            return LinearCurriculumScheduler(
                start_iter=curriculum_config['start_iter'],
                end_iter=curriculum_config['end_iter'],
                start_weight=curriculum_config['start_weight'],
                end_weight=curriculum_config['end_weight']
            )
        
        elif curr_type == 'exponential':
            return ExponentialCurriculumScheduler(
                growth_rate=curriculum_config['growth_rate'],
                max_weight=curriculum_config['max_weight']
            )
        
        elif curr_type == 'cosine':
            return CosineCurriculumScheduler(
                start_iter=curriculum_config['start_iter'],
                end_iter=curriculum_config['end_iter'],
                max_weight=curriculum_config['max_weight']
            )
        
        else:
            raise ValueError(f"Unknown curriculum type: {curr_type}")
    
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
    
    def run_sapmi(self) -> Dict[str, Any]:
        """Run SA-PMI with all configured curricula"""
        if not self.config['experiments']['sapmi']['enabled']:
            print("\nSkipping SA-PMI (disabled in config)")
            return {}
        
        print("\n" + "=" * 80)
        print("Running SA-PMI with Policy Robustness")
        print("=" * 80)
        
        exp_config = self.config['experiments']['sapmi']
        results = {}
        
        for curriculum_config in exp_config['curricula']:
            print(f"\n{curriculum_config['description']}")
            print("-" * 60)
            
            self.mdp.set_model(copy.deepcopy(self.original_model))
            if hasattr(self, 'init_model_vector'):
                self.mdp.model_vector = np.array(self.init_model_vector)
            
            # Create policy chooser with curriculum
            base_policy_chooser = GreedyPolicyChooser(self.mdp.nS, self.mdp.nA)
            curriculum_scheduler = self._create_curriculum_scheduler(curriculum_config)
            
            policy_chooser = SAPMIPolicyChooser(
                nS=self.mdp.nS,
                nA=self.mdp.nA,
                base_chooser=base_policy_chooser,
                curriculum_scheduler=curriculum_scheduler,
                robustness_temperature=curriculum_config['robustness_temperature'],
                entropy_bonus=curriculum_config['entropy_bonus']
            )
            
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
                    name=f"sapmi_{curriculum_config['name']}",
                    tags=self.config['wandb']['tags'] + ["sapmi", curriculum_config['type']],
                    config={
                        "algorithm": "SA-PMI",
                        "variant": "policy_robust",
                        "curriculum": curriculum_config['name'],
                        "curriculum_type": curriculum_config['type'],
                        **curriculum_config
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
            filename = f"sapmi_policy_{curriculum_config['name']}.csv"
            spmi.logger.save(str(self.output_dir), filename)
            
            if self.config['output']['save_sapmi_metrics']:
                spmi.policy_chooser.save_sapmi_policy_metrics(
                    str(self.output_dir / f"sapmi_policy_{curriculum_config['name']}")
                )
            
            if self.config['output']['save_gp_metrics'] and hasattr(model_chooser, 'save_gp_times'):
                model_chooser.save_gp_times(
                    str(self.output_dir / f"sapmi_policy_{curriculum_config['name']}")
                )
            
            print(f"Completed in {elapsed:.2f}s | Iterations: {spmi.logger.iteration} | "
                  f"Performance: {spmi.logger.evaluations[-1]:.4f}")
            
            results[curriculum_config['name']] = {
                'spmi': spmi,
                'final_policy': final_policy,
                'final_model': final_model,
                'elapsed_time': elapsed,
                'config': curriculum_config
            }
        
        return results
    
    def run_federated_spmi(self) -> Dict[str, Any]:
        """Run Federated SPMI with all configurations"""
        if not self.config['experiments']['federated_spmi']['enabled']:
            print("\nSkipping Federated SPMI (disabled in config)")
            return {}
        
        print("\n" + "=" * 80)
        print("Running Federated SPMI")
        print("=" * 80)
        
        exp_config = self.config['experiments']['federated_spmi']
        results = {}
        
        for agent_config in exp_config['configurations']:
            print(f"\nN={agent_config['n_agents']} agents, "
                  f"{agent_config['episodes_per_round']} episodes/round")
            print("-" * 60)
            
            self.mdp.set_model(copy.deepcopy(self.original_model))
            if hasattr(self, 'init_model_vector'):
                self.mdp.model_vector = np.array(self.init_model_vector)
            
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
                verbose=True
            )
            
            # WandB logging
            if self.use_wandb:
                wandb.init(
                    project=self.config['wandb']['project'],
                    entity=self.config['wandb']['entity'],
                    name=f"fspmi_n{agent_config['n_agents']}",
                    tags=self.config['wandb']['tags'] + ["federated"],
                    config={
                        "algorithm": "F-SPMI",
                        "variant": "federated",
                        **agent_config
                    },
                    reinit=True
                )
            
            start_time = time.time()
            final_policy, final_model = fspmi.run(
                copy.deepcopy(self.initial_policy),
                copy.deepcopy(self.initial_model)
            )
            elapsed = time.time() - start_time
            
            # Log to WandB
            if self.use_wandb:
                for i, perf in enumerate(fspmi.logger.true_performances):
                    wandb.log({
                        "round": i,
                        "performance": perf,
                        "avg_return": fspmi.logger.avg_returns[i] if i < len(fspmi.logger.avg_returns) else None
                    })
                wandb.log({
                    "final_performance": fspmi.logger.true_performances[-1],
                    "total_rounds": len(fspmi.logger.iterations),
                    "total_samples": sum(fspmi.logger.total_samples),
                    "elapsed_time": elapsed
                })
                wandb.finish()
            
            # Save results
            filename = f"fspmi_n{agent_config['n_agents']}.csv"
            fspmi.logger.save(str(self.output_dir / filename))
            
            if self.config['output']['save_gp_metrics'] and hasattr(model_chooser, 'save_gp_times'):
                model_chooser.save_gp_times(
                    str(self.output_dir / f"fspmi_n{agent_config['n_agents']}")
                )
            
            perf = fspmi.logger.true_performances[-1] if fspmi.logger.true_performances else float('nan')
            print(f"Completed in {elapsed:.2f}s | Rounds: {len(fspmi.logger.iterations)} | "
                  f"Performance: {perf:.4f}")
            
            results[f"n{agent_config['n_agents']}"] = {
                'fspmi': fspmi,
                'final_policy': final_policy,
                'final_model': final_model,
                'elapsed_time': elapsed,
                'config': agent_config
            }
        
        return results
    
    def run_federated_sapmi(self) -> Dict[str, Any]:
        """Run Federated SA-PMI with all configurations"""
        if not self.config['experiments']['federated_sapmi']['enabled']:
            print("\nSkipping Federated SA-PMI (disabled in config)")
            return {}
        
        print("\n" + "=" * 80)
        print("Running Federated SA-PMI")
        print("=" * 80)
        
        exp_config = self.config['experiments']['federated_sapmi']
        results = {}
        
        for n_agents in exp_config['agent_counts']:
            for curriculum_config in exp_config['curricula']:
                print(f"\nN={n_agents} agents, {curriculum_config['description']}")
                print("-" * 60)
                
                self.mdp.set_model(copy.deepcopy(self.original_model))
                if hasattr(self, 'init_model_vector'):
                    self.mdp.model_vector = np.array(self.init_model_vector)
                
                # Create policy chooser with curriculum
                base_policy_chooser = GreedyPolicyChooser(self.mdp.nS, self.mdp.nA)
                curriculum_scheduler = self._create_curriculum_scheduler(curriculum_config)
                
                policy_chooser = SAPMIPolicyChooser(
                    nS=self.mdp.nS,
                    nA=self.mdp.nA,
                    base_chooser=base_policy_chooser,
                    curriculum_scheduler=curriculum_scheduler,
                    robustness_temperature=curriculum_config['robustness_temperature'],
                    entropy_bonus=curriculum_config['entropy_bonus']
                )
                
                model_chooser = self._create_model_chooser()
                
                fspmi = FSPMI(
                    conf_mdp=self.mdp,
                    n_agents=n_agents,
                    episodes_per_round=exp_config['episodes_per_round'],
                    eps=0.0,
                    max_rounds=exp_config['max_rounds'],
                    policy_chooser=policy_chooser,
                    model_chooser=model_chooser,
                    aggregation_method='weighted',
                    persistent=self.config['model_chooser']['type'] != 'gp',
                    delta_q=1,
                    verbose=True
                )
                
                # WandB logging
                if self.use_wandb:
                    wandb.init(
                        project=self.config['wandb']['project'],
                        entity=self.config['wandb']['entity'],
                        name=f"fsapmi_{curriculum_config['name']}_n{n_agents}",
                        tags=self.config['wandb']['tags'] + ["federated", "sapmi", curriculum_config['type']],
                        config={
                            "algorithm": "F-SA-PMI",
                            "variant": "federated_policy_robust",
                            "n_agents": n_agents,
                            "curriculum": curriculum_config['name'],
                            "curriculum_type": curriculum_config['type'],
                            **curriculum_config
                        },
                        reinit=True
                    )
                
                start_time = time.time()
                final_policy, final_model = fspmi.run(
                    copy.deepcopy(self.initial_policy),
                    copy.deepcopy(self.initial_model)
                )
                elapsed = time.time() - start_time
                
                # Log to WandB
                if self.use_wandb:
                    for i, perf in enumerate(fspmi.logger.true_performances):
                        wandb.log({
                            "round": i,
                            "performance": perf,
                            "avg_return": fspmi.logger.avg_returns[i] if i < len(fspmi.logger.avg_returns) else None
                        })
                    wandb.log({
                        "final_performance": fspmi.logger.true_performances[-1],
                        "total_rounds": len(fspmi.logger.iterations),
                        "total_samples": sum(fspmi.logger.total_samples),
                        "elapsed_time": elapsed
                    })
                    wandb.finish()
                
                # Save results
                filename = f"fsapmi_{curriculum_config['name']}_n{n_agents}.csv"
                fspmi.logger.save(str(self.output_dir / filename))
                
                if self.config['output']['save_gp_metrics'] and hasattr(model_chooser, 'save_gp_times'):
                    model_chooser.save_gp_times(
                        str(self.output_dir / filename.replace('.csv', ''))
                    )
                
                perf = fspmi.logger.true_performances[-1] if fspmi.logger.true_performances else float('nan')
                print(f"Completed in {elapsed:.2f}s | Rounds: {len(fspmi.logger.iterations)} | "
                      f"Performance: {perf:.4f}")
                
                key = f"{curriculum_config['name']}_n{n_agents}"
                results[key] = {
                    'fspmi': fspmi,
                    'final_policy': final_policy,
                    'final_model': final_model,
                    'elapsed_time': elapsed,
                    'config': {**curriculum_config, 'n_agents': n_agents}
                }
        
        return results
    
    def run_all(self, experiments: Optional[List[str]] = None):
        """Run all enabled experiments or specified experiments
        
        Args:
            experiments: List of experiment names to run. If None, runs all enabled.
                        Options: ['standard_spmi', 'sapmi', 'federated_spmi', 'federated_sapmi']
        """
        print("=" * 80)
        print("UNIFIED SPMI EXPERIMENT RUNNER")
        print("=" * 80)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"WandB enabled: {self.use_wandb}")
        
        if experiments is None:
            experiments = ['standard_spmi', 'sapmi', 'federated_spmi', 'federated_sapmi']
        
        # Run experiments
        if 'standard_spmi' in experiments:
            self.results['standard_spmi'] = self.run_standard_spmi()
        
        if 'sapmi' in experiments:
            self.results['sapmi'] = self.run_sapmi()
        
        if 'federated_spmi' in experiments:
            self.results['federated_spmi'] = self.run_federated_spmi()
        
        if 'federated_sapmi' in experiments:
            self.results['federated_sapmi'] = self.run_federated_sapmi()
        
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
        
        # SA-PMI
        if 'sapmi' in self.results and self.results['sapmi']:
            for name, result in self.results['sapmi'].items():
                spmi = result['spmi']
                perf = spmi.logger.evaluations[-1] if spmi.logger.evaluations else float('nan')
                method_name = f"SA-PMI ({name})"
                print(f"{method_name:<50} {spmi.logger.iteration:<15} {perf:.4f}")
        
        # Federated SPMI
        if 'federated_spmi' in self.results and self.results['federated_spmi']:
            for name, result in self.results['federated_spmi'].items():
                fspmi = result['fspmi']
                perf = fspmi.logger.true_performances[-1] if fspmi.logger.true_performances else float('nan')
                method_name = f"F-SPMI ({name})"
                print(f"{method_name:<50} {len(fspmi.logger.iterations):<15} {perf:.4f}")
        
        # Federated SA-PMI
        if 'federated_sapmi' in self.results and self.results['federated_sapmi']:
            for name, result in self.results['federated_sapmi'].items():
                fspmi = result['fspmi']
                perf = fspmi.logger.true_performances[-1] if fspmi.logger.true_performances else float('nan')
                method_name = f"F-SA-PMI ({name})"
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
  
  # Run only specific experiments
  python simulations/run_experiments.py --experiment sapmi federated_spmi
  
  # Disable WandB and set max iterations to 5000
  python simulations/run_experiments.py --no-wandb --max_iter 5000
  
  # Override robustness temperature for all SA-PMI variants
  python simulations/run_experiments.py --rob_temp 0.5
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
        '--rob_temp',
        type=float,
        help='Override the robustness_temperature for ALL SA-PMI and F-SA-PMI curricula.'
    )
    
    parser.add_argument(
        '--all_curricula',
        type=float,
        help='Override ALL SA-PMI and F-SA-PMI curricula to use a CONSTANT weight. Sets curriculum type to "constant" and weight to this value.'
    )
    parser.add_argument(
            '--model_chooser',
            choices=['gp', 'greedy', 'gb'],
            help="Override the model chooser type. 'gb' is an alias for 'greedy'."
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
        use_wandb=not args.no_wandb
    )
    
    # --- Apply Overrides to Runner Config ---
    
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
        runner.config['experiments']['sapmi']['max_iter'] = args.max_iter
    
    if args.max_rounds is not None:
        print(f"Applying MAX_ROUNDS override: {args.max_rounds}")
        # Federated SPMI
        for config in runner.config['experiments']['federated_spmi']['configurations']:
            config['max_rounds'] = args.max_rounds
        # Federated SA-PMI (assuming it uses the top-level max_rounds)
        runner.config['experiments']['federated_sapmi']['max_rounds'] = args.max_rounds

    
    if args.rob_temp is not None:
        print(f"Applying ROBUSTNESS_TEMPERATURE override: {args.rob_temp}")
        for exp_key in ['sapmi', 'federated_sapmi']:
            if runner.config['experiments'].get(exp_key) and runner.config['experiments'][exp_key].get('curricula'):
                for curriculum in runner.config['experiments'][exp_key]['curricula']:
                    curriculum['robustness_temperature'] = args.rob_temp

    if args.all_curricula is not None:
        print(f"Applying ALL_CURRICULA (CONSTANT WEIGHT) override: {args.all_curricula}")
        for exp_key in ['sapmi', 'federated_sapmi']:
            if runner.config['experiments'].get(exp_key) and runner.config['experiments'][exp_key].get('curricula'):
                for curriculum in runner.config['experiments'][exp_key]['curricula']:
                    # Force to constant and set weight
                    curriculum['type'] = 'constant'
                    curriculum['weight'] = args.all_curricula
                    # Optional: update description
                    curriculum['description'] = f"Command Line Override: Constant {args.all_curricula}"
                    curriculum['name'] = f"cmd_const_{args.all_curricula}"
                    # Remove dynamic fields for clarity
                    for key in ['start_iter', 'end_iter', 'start_weight', 'end_weight', 'growth_rate', 'max_weight']:
                        curriculum.pop(key, None)

    # ----------------------------------------------
    
    runner.run_all(experiments=args.experiment)


if __name__ == '__main__':
    main()