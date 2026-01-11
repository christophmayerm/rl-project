"""
Unified SPMI Experiment Runner
===============================
Runs all SPMI variants (Standard, Federated)
with configurable parameters and WandB integration.

Usage:
    python run_experiments.py --config config.yaml
    python run_experiments.py --config config.yaml --experiment federated_spmi
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
from algorithm.policy_chooser import GreedyPolicyChooser
from algorithm.model_chooser import (
    GPModelChooser, 
    SetModelChooser
)
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy
from envs.racetrack_simulator import RaceTrackConfigurableEnv
from envs.racetrack_simulator_obstacles import RaceTrackWithObstacles
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
        self.output_dir = Path(self.config['output']['dir']) / \
            f'racetrack_{self.config["environment"]["params"]["track_file"]}' / \
            self.config['model_chooser']['type'] / \
            time.strftime("%Y%m%d-%H%M%S")
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

        self.model_set = [
            TabularModel(self.mdp.P_highspeed_noboost, self.mdp.nS, self.mdp.nA),
            TabularModel(self.mdp.P_lowspeed_noboost, self.mdp.nS, self.mdp.nA),
            TabularModel(self.mdp.P_highspeed_boost, self.mdp.nS, self.mdp.nA),
            TabularModel(self.mdp.P_lowspeed_boost, self.mdp.nS, self.mdp.nA)
        ]
        
        # Results storage
        self.results = {}
    
    def _get_experiment_name(self, algorithm: str, model_chooser: str, 
                             track: str, extra: str = "") -> str:
        """Generate a descriptive experiment name for WandB"""
        name_parts = [algorithm, model_chooser, track]
        if extra:
            name_parts.append(extra)
        return "_".join(name_parts)
    
    def _get_run_group(self) -> str:
        """Get WandB group name for grouping related runs"""
        track = self.config['environment']['params']['track_file']
        model_chooser = self.config['model_chooser']['type']
        timestamp = time.strftime("%Y%m%d-%H%M")
        return f"{track}_{model_chooser}_{timestamp}"
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_environment(self) -> RaceTrackConfigurableEnv:
        """Initialize the MDP environment"""
        env_params = self.config['environment']['params']
        print(f"\nInitializing {self.config['environment']['name']} environment...")

        # mdp = TeacherStudentEnv(**env_params)
        use_obstacles = (
            self.config['environment']['name'] in {"racetrack_obstacles", "racetrack_with_obstacles"}
            or "obstacle_positions" in env_params
            or "obstacle_at_iteration" in env_params
        )
        mdp = RaceTrackWithObstacles(**env_params) if use_obstacles else RaceTrackConfigurableEnv(**env_params)
        
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

    def _reset_obstacles(self) -> None:
        if hasattr(self.mdp, "reset_obstacles"):
            self.mdp.reset_obstacles()
    

    def run_standard_spmi(self) -> Dict[str, Any]:
        """Run standard SPMI"""
        if not self.config['experiments']['standard_spmi']['enabled']:
            print("\nSkipping Standard SPMI (disabled in config)")
            return {}
        
        print("\n" + "=" * 80)
        print("Running Standard SPMI")
        print("=" * 80)

        self._reset_obstacles()
        
        exp_config = self.config['experiments']['standard_spmi']
        track = self.config['environment']['params']['track_file']
        model_chooser_type = self.config['model_chooser']['type']
        
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
        
        # WandB logging with better naming
        if self.use_wandb:
            run_name = self._get_experiment_name(
                algorithm="SPMI",
                model_chooser=model_chooser_type.upper(),
                track=track,
                extra=f"iter{exp_config['max_iter']}"
            )
            wandb.init(
                project=self.config['wandb']['project'],
                entity=self.config['wandb']['entity'],
                name=run_name,
                group=self._get_run_group(),
                tags=self.config['wandb']['tags'] + [
                    "spmi", 
                    model_chooser_type,
                    track
                ],
                config={
                    "algorithm": "SPMI",
                    "variant": "standard",
                    "max_iter": exp_config['max_iter'],
                    "model_chooser": model_chooser_type,
                    "track": track,
                    "nS": self.mdp.nS,
                    "nA": self.mdp.nA,
                    "gamma": self.mdp.gamma,
                    "horizon": self.mdp.horizon,
                    "pfail": self.config['environment']['params'].get('pfail', 0.0)
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
                wandb.log({
                    "iteration": i,
                    "performance": perf,
                    "alpha": spmi.logger.alfas[i] if i < len(spmi.logger.alfas) else None,
                    "beta": spmi.logger.betas[i] if i < len(spmi.logger.betas) else None
                })
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

            self._reset_obstacles()
            
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
    
    def run_federated_spmi(self, n_seeds: int = 1) -> Dict[str, Any]:
        """Run Federated SPMI with multiple seeds for confidence intervals"""
        if not self.config['experiments']['federated_spmi']['enabled']:
            print("\nSkipping Federated SPMI (disabled in config)")
            return {}
        
        print("\n" + "=" * 80)
        print(f"Running Federated SPMI ({n_seeds} seed(s))")
        print("=" * 80)
        
        exp_config = self.config['experiments']['federated_spmi']
        track = self.config['environment']['params']['track_file']
        model_chooser_type = self.config['model_chooser']['type']
        results = {}
        
        for agent_config in exp_config['configurations']:
            n_agents = agent_config['n_agents']
            eps_per_round = agent_config['episodes_per_round']
            max_rounds = agent_config['max_rounds']
            
            config_key = f"n{n_agents}_eps{eps_per_round}"
            print(f"\n{config_key}")
            print("-" * 60)
            
            seed_results = []
            
            # Initialize WandB for this configuration (logs all seeds together)
            if self.use_wandb:
                run_name = self._get_experiment_name(
                    algorithm="FSPMI",
                    model_chooser=model_chooser_type.upper(),
                    track=track,
                    extra=f"N{n_agents}_eps{eps_per_round}_seeds{n_seeds}"
                )
                wandb.init(
                    project=self.config['wandb']['project'],
                    entity=self.config['wandb']['entity'],
                    name=run_name,
                    group=self._get_run_group(),
                    tags=self.config['wandb']['tags'] + [
                        "fspmi",
                        "federated",
                        model_chooser_type,
                        track,
                        f"N{n_agents}",
                        f"eps{eps_per_round}"
                    ],
                    config={
                        "algorithm": "F-SPMI",
                        "variant": "federated",
                        "n_agents": n_agents,
                        "episodes_per_round": eps_per_round,
                        "max_rounds": max_rounds,
                        "n_seeds": n_seeds,
                        "model_chooser": model_chooser_type,
                        "track": track,
                        "nS": self.mdp.nS,
                        "nA": self.mdp.nA,
                        "gamma": self.mdp.gamma,
                        "horizon": self.mdp.horizon,
                        "pfail": self.config['environment']['params'].get('pfail', 0.0),
                        "total_episodes_per_round": n_agents * eps_per_round
                    },
                    reinit=True
                )
            
            all_performances = []
            all_alphas = []
            all_betas = []
            
            for seed in range(n_seeds):
                if n_seeds > 1:
                    print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)
                
                # Reset MDP
                self._reset_obstacles()
                self.mdp.set_model(copy.deepcopy(self.original_model))
                
                policy_chooser = GreedyPolicyChooser(self.mdp.nS, self.mdp.nA)
                model_chooser = self._create_model_chooser()
                
                fspmi = FSPMI(
                    conf_mdp=self.mdp,
                    n_agents=n_agents,
                    episodes_per_round=eps_per_round,
                    eps=0.0,
                    max_rounds=max_rounds,
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
                
                perf = fspmi.logger.true_performances[-1] if fspmi.logger.true_performances else float('nan')
                                
                if n_seeds > 1:
                    print(f"Perf: {perf:.4f}, Time: {elapsed:.1f}s")
                
                # Store for aggregation
                all_performances.append(fspmi.logger.true_performances)
                all_alphas.append(fspmi.logger.alphas)
                all_betas.append(fspmi.logger.betas)
                
                # Save individual seed result
                filename = f"fspmi_{config_key}_seed{seed}.csv"
                fspmi.logger.save(str(self.output_dir / filename))
                
                seed_results.append({
                    'fspmi': fspmi,
                    'final_policy': final_policy,
                    'final_model': final_model,
                    'elapsed': elapsed,
                    'seed': seed
                })
            
            # Log aggregated metrics to WandB
            if self.use_wandb:
                min_len = min(len(p) for p in all_performances)
                
                for i in range(min_len):
                    perfs_at_i = [p[i] for p in all_performances]
                    alphas_at_i = [a[i] for a in all_alphas if i < len(a)]
                    betas_at_i = [b[i] for b in all_betas if i < len(b)]
                    
                    log_dict = {
                        "round": i,
                        "performance_mean": np.mean(perfs_at_i),
                        "performance_std": np.std(perfs_at_i),
                        "performance_min": np.min(perfs_at_i),
                        "performance_max": np.max(perfs_at_i),
                    }
                    
                    if alphas_at_i:
                        log_dict["alpha_mean"] = np.mean(alphas_at_i)
                        log_dict["alpha_std"] = np.std(alphas_at_i)
                    if betas_at_i:
                        log_dict["beta_mean"] = np.mean(betas_at_i)
                        log_dict["beta_std"] = np.std(betas_at_i)
                    
                    # Log individual seed performances for comparison
                    for seed_idx, perf_list in enumerate(all_performances):
                        if i < len(perf_list):
                            log_dict[f"performance_seed{seed_idx}"] = perf_list[i]
                    
                    wandb.log(log_dict)
                
                # Final summary
                final_perfs = [p[-1] for p in all_performances]
                wandb.log({
                    "final_performance_mean": np.mean(final_perfs),
                    "final_performance_std": np.std(final_perfs),
                    "final_performance_min": np.min(final_perfs),
                    "final_performance_max": np.max(final_perfs),
                    "total_rounds": min_len,
                    "avg_elapsed_time": np.mean([sr['elapsed'] for sr in seed_results])
                })
                
                # Create summary table
                summary_table = wandb.Table(
                    columns=["seed", "final_performance", "elapsed_time"],
                    data=[[sr['seed'], 
                           sr['fspmi'].logger.true_performances[-1] if sr['fspmi'].logger.true_performances else float('nan'),
                           sr['elapsed']] 
                          for sr in seed_results]
                )
                wandb.log({"seed_results": summary_table})
                
                wandb.finish()
            
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
        
        # Print summary statistics
        final_perfs = true_perfs[:, -1]
        print(f"  Final Performance: {np.mean(final_perfs):.4f} ± {np.std(final_perfs):.4f}")
    
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

                self._reset_obstacles()
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
                        Options: ['standard_spmi', 'federated_spmi']
        """
        print("=" * 80)
        print("UNIFIED SPMI EXPERIMENT RUNNER")
        print("=" * 80)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"WandB enabled: {self.use_wandb}")
        print(f"Track: {self.config['environment']['params']['track_file']}")
        print(f"Model Chooser: {self.config['model_chooser']['type']}")
        print(f"Seeds: {self.n_seeds}")
        
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
        
        track = self.config['environment']['params']['track_file']
        model_chooser = self.config['model_chooser']['type']
        print(f"\nTrack: {track} | Model Chooser: {model_chooser} | Seeds: {self.n_seeds}")
        
        print(f"\n{'Method':<50} {'Iterations/Rounds':<15} {'Final Performance':<20}")
        print("-" * 85)
        
        # Standard SPMI
        if 'standard_spmi' in self.results and self.results['standard_spmi']:
            spmi = self.results['standard_spmi']['spmi']
            perf = spmi.logger.evaluations[-1] if spmi.logger.evaluations else float('nan')
            print(f"{'SPMI':<50} {spmi.logger.iteration:<15} {perf:.4f}")
        
        # Federated SPMI
        if 'federated_spmi' in self.results and self.results['federated_spmi']:
            for name, result_list in self.results['federated_spmi'].items():
                # Get mean ± std if multiple seeds
                if isinstance(result_list, list) and len(result_list) > 1:
                    perfs = [r['fspmi'].logger.true_performances[-1] 
                             for r in result_list 
                             if r['fspmi'].logger.true_performances]
                    perf_str = f"{np.mean(perfs):.4f} ± {np.std(perfs):.4f}"
                    rounds = len(result_list[0]['fspmi'].logger.iterations)
                else:
                    result = result_list[0] if isinstance(result_list, list) else result_list
                    fspmi = result['fspmi']
                    perf = fspmi.logger.true_performances[-1] if fspmi.logger.true_performances else float('nan')
                    perf_str = f"{perf:.4f}"
                    rounds = len(fspmi.logger.iterations)
                
                method_name = f"F-SPMI ({name})"
                print(f"{method_name:<50} {rounds:<15} {perf_str}")
        
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
  python simulations/run_experiments.py --config config/experiment_config.yaml
  
  # F-SPMI vs SPMI with Greedy model chooser
  python simulations/run_experiments.py --config config/experiment_config.yaml \\
      --experiment standard_spmi federated_spmi --model_chooser greedy --n_seeds 5
  
  # F-SPMI vs SPMI with GP model chooser  
  python simulations/run_experiments.py --config config/experiment_config.yaml \\
      --experiment standard_spmi federated_spmi --model_chooser gp --n_seeds 5
  
  # Different tracks
  python simulations/run_experiments.py --config config/experiment_config.yaml \\
      --experiment standard_spmi federated_spmi --track T2 --n_seeds 5
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/experiment_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--experiment',
        nargs='+',
        choices=['standard_spmi', 'federated_spmi'],
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
        help='Override max_iter for standard_spmi'
    )
    
    parser.add_argument(
        '--max_rounds',
        type=int,
        help='Override max_rounds for federated_spmi'
    )
    
    parser.add_argument(
        '--model_chooser',
        choices=['gp', 'greedy', 'gb'],
        help="Override model chooser type. 'gb' is alias for 'greedy'."
    )
    
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
    
    args = parser.parse_args()

    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found!")
        print("Please create a config.yaml file or specify a valid config path.")
        sys.exit(1)
    
    # Initialize runner
    runner = ExperimentRunner(
        config_path=args.config,
        use_wandb=not args.no_wandb,
        n_seeds=args.n_seeds
    )
    
    # --- Apply Overrides ---
    if args.track is not None:
        print(f"Applying TRACK override: {args.track}")
        runner.config['environment']['params']['track_file'] = args.track
        # Reinitialize environment with new track
        runner.mdp = runner._init_environment()
        runner.uniform_policy = UniformPolicy(runner.mdp)
        runner.original_model = copy.deepcopy(runner.mdp.P)
        runner.initial_model = TabularModel(runner.mdp.P, runner.mdp.nS, runner.mdp.nA)
        runner.initial_policy = TabularPolicy(
            runner.uniform_policy.get_rep(), 
            runner.mdp.nS, 
            runner.mdp.nA
        )
        runner.model_set = [
            TabularModel(runner.mdp.P_highspeed_noboost, runner.mdp.nS, runner.mdp.nA),
            TabularModel(runner.mdp.P_lowspeed_noboost, runner.mdp.nS, runner.mdp.nA),
            TabularModel(runner.mdp.P_highspeed_boost, runner.mdp.nS, runner.mdp.nA),
            TabularModel(runner.mdp.P_lowspeed_boost, runner.mdp.nS, runner.mdp.nA)
        ]
        # Update output directory
        runner.output_dir = Path(runner.config['output']['dir']) / \
            f'racetrack_{args.track}' / \
            runner.config['model_chooser']['type'] / \
            time.strftime("%Y%m%d-%H%M%S")
        runner.output_dir.mkdir(parents=True, exist_ok=True)

    if args.n_agents is not None:
        print(f"Applying N_AGENTS override: {args.n_agents}")
        for config in runner.config['experiments']['federated_spmi']['configurations']:
            config['n_agents'] = args.n_agents

    if args.eps_per_round is not None:
        print(f"Applying EPS_PER_ROUND override: {args.eps_per_round}")
        for config in runner.config['experiments']['federated_spmi']['configurations']:
            config['episodes_per_round'] = args.eps_per_round
            
    if args.model_chooser is not None:
        mc_type = 'greedy' if args.model_chooser == 'gb' else args.model_chooser
        print(f"Applying MODEL_CHOOSER override: {mc_type}")
        runner.config['model_chooser']['type'] = mc_type
        # Update output directory
        runner.output_dir = Path(runner.config['output']['dir']) / \
            f'racetrack_{runner.config["environment"]["params"]["track_file"]}' / \
            mc_type / \
            time.strftime("%Y%m%d-%H%M%S")
        runner.output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_iter is not None:
        print(f"Applying MAX_ITER override: {args.max_iter}")
        runner.config['experiments']['standard_spmi']['max_iter'] = args.max_iter
    
    if args.max_rounds is not None:
        print(f"Applying MAX_ROUNDS override: {args.max_rounds}")
        for config in runner.config['experiments']['federated_spmi']['configurations']:
            config['max_rounds'] = args.max_rounds

    # Run experiments
    runner.run_all(experiments=args.experiment)


if __name__ == '__main__':
    main()
