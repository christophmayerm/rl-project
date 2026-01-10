"""
Heterogeneous F-SPMI Experiment Runner
=======================================
Runs heterogeneous federated learning experiments where each agent
operates on a DIFFERENT MDP variant.

Experiment Types:
1. Dynamics heterogeneity (different k values)
2. Robustness heterogeneity (different pfail values)
3. Obstacle heterogeneity (different obstacle configurations)
4. Mixed heterogeneity (combination of above)

Key comparisons:
- Homogeneous F-SPMI vs Heterogeneous F-SPMI
- Robustness evaluation across all variants
- Learning curves and convergence analysis

Usage:
    python run_heterogeneous_experiments.py --config config/hetero_config.yaml
    python run_heterogeneous_experiments.py --experiment dynamics --n_seeds 5
    python run_heterogeneous_experiments.py --experiment robustness --track T1
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
from dataclasses import dataclass, field

# Import components
from algorithm.policy_chooser import GreedyPolicyChooser
from algorithm.model_chooser import SetModelChooser
from utils.uniform_policy import UniformPolicy
from utils.tabular import TabularModel, TabularPolicy, TabularReward
import utils.evaluator as evaluator
from envs.racetrack_simulator import RaceTrackConfigurableEnv

# Import heterogeneous F-SPMI
from heterogeneous_fspmi.heterogeneous_fspmi import (
    HeterogeneousFSPMI,
    HeterogeneousConfig,
    EnvironmentVariant,
    create_dynamics_heterogeneous_fspmi,
    create_robustness_heterogeneous_fspmi,
)

# Import standard F-SPMI for comparison
from federated import FSPMI

# Optional imports
try:
    from envs.racetrack_obstacles_h import RaceTrackWithObstacles, create_obstacle_variants
    OBSTACLES_AVAILABLE = True
except ImportError:
    OBSTACLES_AVAILABLE = False
    print("Warning: racetrack_obstacles not available. Obstacle experiments disabled.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run: pip install wandb")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class HeterogeneousExperimentConfig:
    """Configuration for heterogeneous experiments."""
    # Environment
    track_file: str = "T1"
    gamma: float = 0.9
    horizon: int = 20
    base_pfail: float = 0.07
    
    # Experiment parameters
    n_iterations: int = 500
    episodes_per_agent: int = 100
    n_seeds: int = 5
    
    # Heterogeneity options
    dynamics_k_values: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    pfail_values: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1])
    max_obstacles: int = 3
    n_obstacle_variants: int = 4
    
    # Learning options
    use_parallel: bool = True
    update_mode: str = 'standard'  # 'standard', 'alternating', 'policy_only'
    target_policy_type: str = 'greedy'  # 'greedy', 'softmax'
    softmax_temperature: float = 1.0
    min_step_size: float = 0.0
    step_size_boost: float = 1.0  # Multiply step sizes by this factor
    
    # WandB
    wandb_enabled: bool = True
    wandb_project: str = "heterogeneous-fspmi"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: ["heterogeneous", "robustness"])
    
    # Output
    output_dir: str = "./data/heterogeneous"
    save_per_seed: bool = True


def load_config(config_path: str) -> HeterogeneousExperimentConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return HeterogeneousExperimentConfig(**config_dict)


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class HeterogeneousExperimentRunner:
    """Runner for heterogeneous F-SPMI experiments."""
    
    def __init__(self, config: HeterogeneousExperimentConfig):
        self.config = config
        self.run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.track_file / self.run_timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup WandB
        self.use_wandb = (config.wandb_enabled and WANDB_AVAILABLE)
        
        # Create reference environment
        self.ref_env = self._create_ref_environment()
        self.nS = self.ref_env.nS
        self.nA = self.ref_env.nA
        
        # Initialize policy and model
        self.uniform_policy = UniformPolicy(self.ref_env)
        self.initial_policy = TabularPolicy(
            self.uniform_policy.get_rep(), self.nS, self.nA
        )
        self.initial_model = TabularModel(self.ref_env.P, self.nS, self.nA)
        
        # Model set for choosers
        self.model_set = [
            TabularModel(self.ref_env.P_highspeed_noboost, self.nS, self.nA),
            TabularModel(self.ref_env.P_lowspeed_noboost, self.nS, self.nA),
            TabularModel(self.ref_env.P_highspeed_boost, self.nS, self.nA),
            TabularModel(self.ref_env.P_lowspeed_boost, self.nS, self.nA)
        ]
        
        # Results storage
        self.results = {}
        
        print(f"\n{'='*60}")
        print("Heterogeneous F-SPMI Experiment Runner")
        print(f"{'='*60}")
        print(f"Track: {config.track_file}")
        print(f"State space: {self.nS}, Action space: {self.nA}")
        print(f"Output: {self.output_dir}")
        print(f"WandB: {'enabled' if self.use_wandb else 'disabled'}")
        print(f"Step size boost: {config.step_size_boost}x")
        print(f"{'='*60}\n")
    
    def _create_ref_environment(self) -> RaceTrackConfigurableEnv:
        """Create reference environment."""
        return RaceTrackConfigurableEnv(
            track_file=self.config.track_file,
            initial_configuration=[0.5, 0.5, 0, 0],
            pfail=self.config.base_pfail,
            horizon=self.config.horizon
        )
    
    def _get_wandb_run_name(self, experiment_type: str, extra: str = "") -> str:
        """Generate descriptive WandB run name."""
        parts = ["HFSPMI", self.config.track_file, experiment_type.upper()]
        if extra:
            parts.append(extra)
        return "_".join(parts)
    
    def _get_wandb_group(self, experiment_type: str) -> str:
        """Get WandB group name."""
        return f"{self.config.track_file}_{experiment_type}_{self.run_timestamp}"
    
    def _get_base_wandb_config(self) -> Dict[str, Any]:
        """Get base WandB config."""
        return {
            "track": self.config.track_file,
            "nS": self.nS,
            "nA": self.nA,
            "gamma": self.config.gamma,
            "horizon": self.config.horizon,
            "n_iterations": self.config.n_iterations,
            "episodes_per_agent": self.config.episodes_per_agent,
            "use_parallel": self.config.use_parallel,
            "update_mode": self.config.update_mode,
            "step_size_boost": self.config.step_size_boost,
        }
    
    def _evaluate_on_variants(self, policy: TabularPolicy, model: TabularModel,
                              variants: List[Any]) -> Dict[str, Any]:
        """Evaluate policy on all environment variants."""
        results = {
            'variant_id': [],
            'performance': [],
            'description': []
        }
        
        for i, variant in enumerate(variants):
            # Create environment for this variant
            if hasattr(variant, 'P'):
                env = variant
            else:
                env = RaceTrackConfigurableEnv(
                    track_file=variant.track_file,
                    initial_configuration=[variant.k, 1-variant.k, 0, 0],
                    pfail=variant.pfail,
                    horizon=self.config.horizon
                )
            
            mu = env.mu
            reward = TabularReward(env.P, self.nS, self.nA)
            variant_model = TabularModel(env.P, self.nS, self.nA)
            
            perf = evaluator.compute_performance(
                mu, reward, policy, variant_model,
                self.config.gamma, self.config.horizon, self.nS, self.nA
            )
            
            results['variant_id'].append(i)
            results['performance'].append(perf)
            desc = getattr(variant, 'description', f"variant_{i}")
            results['description'].append(desc)
        
        perfs = np.array(results['performance'])
        results['mean'] = np.mean(perfs)
        results['std'] = np.std(perfs)
        results['min'] = np.min(perfs)
        results['max'] = np.max(perfs)
        results['worst_case'] = np.min(perfs)
        
        return results
    
    # ========================================================================
    # DYNAMICS HETEROGENEITY EXPERIMENT
    # ========================================================================
    
    def run_dynamics_experiment(self) -> Dict[str, Any]:
        """Run experiment with different dynamics (k values)."""
        print("\n" + "="*60)
        print("DYNAMICS HETEROGENEITY EXPERIMENT")
        print("="*60)
        
        k_values = self.config.dynamics_k_values
        n_agents = len(k_values)
        
        print(f"K values: {k_values}")
        print(f"Number of agents: {n_agents}")
        
        variants = [
            EnvironmentVariant(
                variant_id=i,
                track_file=self.config.track_file,
                k=k,
                pfail=self.config.base_pfail,
                description=f"k={k}"
            )
            for i, k in enumerate(k_values)
        ]
        
        results = self._run_heterogeneous_comparison(
            experiment_name="dynamics",
            variants=variants,
            description=f"Dynamics heterogeneity with k={k_values}"
        )
        
        self.results['dynamics'] = results
        return results
    
    # ========================================================================
    # ROBUSTNESS HETEROGENEITY EXPERIMENT  
    # ========================================================================
    
    def run_robustness_experiment(self) -> Dict[str, Any]:
        """Run experiment with different failure probabilities."""
        print("\n" + "="*60)
        print("ROBUSTNESS HETEROGENEITY EXPERIMENT")
        print("="*60)
        
        pfail_values = self.config.pfail_values
        n_agents = len(pfail_values)
        
        print(f"Pfail values: {pfail_values}")
        print(f"Number of agents: {n_agents}")
        
        variants = [
            EnvironmentVariant(
                variant_id=i,
                track_file=self.config.track_file,
                k=0.5,
                pfail=pf,
                description=f"pfail={pf}"
            )
            for i, pf in enumerate(pfail_values)
        ]
        
        results = self._run_heterogeneous_comparison(
            experiment_name="robustness",
            variants=variants,
            description=f"Robustness heterogeneity with pfail={pfail_values}"
        )
        
        self.results['robustness'] = results
        return results
    
    # ========================================================================
    # OBSTACLE HETEROGENEITY EXPERIMENT
    # ========================================================================
    
    def run_obstacle_experiment(self) -> Dict[str, Any]:
        """Run experiment with different obstacle configurations."""
        if not OBSTACLES_AVAILABLE:
            print("\nSkipping obstacle experiment (racetrack_obstacles not available)")
            return {}
        
        print("\n" + "="*60)
        print("OBSTACLE HETEROGENEITY EXPERIMENT")
        print("="*60)
        
        obstacle_variants = create_obstacle_variants(
            track_file=self.config.track_file,
            n_variants=self.config.n_obstacle_variants,
            max_obstacles=self.config.max_obstacles,
            base_pfail=self.config.base_pfail,
            seed=42
        )
        
        print(f"Created {len(obstacle_variants)} obstacle variants")
        for i, v in enumerate(obstacle_variants):
            info = v.get_variant_info()
            print(f"  Variant {i}: {info['n_obstacles']} obstacles")
        
        results = self._run_obstacle_comparison(
            experiment_name="obstacles",
            obstacle_variants=obstacle_variants,
            description=f"Obstacle heterogeneity with max {self.config.max_obstacles} obstacles"
        )
        
        self.results['obstacles'] = results
        return results
    
    # ========================================================================
    # MIXED HETEROGENEITY EXPERIMENT
    # ========================================================================
    
    def run_mixed_experiment(self) -> Dict[str, Any]:
        """Run experiment with mixed heterogeneity (dynamics + pfail)."""
        print("\n" + "="*60)
        print("MIXED HETEROGENEITY EXPERIMENT")
        print("="*60)
        
        variants = [
            EnvironmentVariant(0, self.config.track_file, k=0.3, pfail=0.0, 
                              description="low-speed, deterministic"),
            EnvironmentVariant(1, self.config.track_file, k=0.5, pfail=0.0,
                              description="balanced, deterministic"),
            EnvironmentVariant(2, self.config.track_file, k=0.7, pfail=0.0,
                              description="high-speed, deterministic"),
            EnvironmentVariant(3, self.config.track_file, k=0.5, pfail=0.05,
                              description="balanced, moderate noise"),
            EnvironmentVariant(4, self.config.track_file, k=0.5, pfail=0.1,
                              description="balanced, high noise"),
        ]
        
        print(f"Mixed variants: {len(variants)}")
        for v in variants:
            print(f"  {v.variant_id}: {v.description}")
        
        results = self._run_heterogeneous_comparison(
            experiment_name="mixed",
            variants=variants,
            description="Mixed heterogeneity (dynamics + robustness)"
        )
        
        self.results['mixed'] = results
        return results
    
    # ========================================================================
    # CORE COMPARISON RUNNER
    # ========================================================================
    
    def _run_heterogeneous_comparison(
        self,
        experiment_name: str,
        variants: List[EnvironmentVariant],
        description: str
    ) -> Dict[str, Any]:
        """Run comparison between homogeneous and heterogeneous F-SPMI."""
        n_seeds = self.config.n_seeds
        n_agents = len(variants)
        
        print(f"\nRunning {n_seeds} seeds...")
        
        homo_results = []
        hetero_results = []
        
        if self.use_wandb:
            run_name = self._get_wandb_run_name(experiment_name, f"seeds{n_seeds}")
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=run_name,
                group=self._get_wandb_group(experiment_name),
                tags=self.config.wandb_tags + [experiment_name],
                config={
                    **self._get_base_wandb_config(),
                    "experiment_type": experiment_name,
                    "n_agents": n_agents,
                    "n_seeds": n_seeds,
                    "variants": [v.description for v in variants],
                    "description": description,
                },
                reinit=True
            )
        
        for seed in range(n_seeds):
            print(f"\n--- Seed {seed+1}/{n_seeds} ---")
            
            # 1. Homogeneous F-SPMI
            print(f"  Running Homogeneous F-SPMI (N={n_agents}, base environment)...")
            homo_result = self._run_homogeneous_fspmi(n_agents=n_agents, seed=seed)
            homo_results.append(homo_result)
            
            # 2. Heterogeneous F-SPMI
            print(f"  Running Heterogeneous F-SPMI (N={n_agents}, different variants)...")
            hetero_result = self._run_heterogeneous_fspmi(variants=variants, seed=seed)
            hetero_results.append(hetero_result)
            
            # 3. Evaluate
            print(f"  Evaluating on all variants...")
            homo_eval = self._evaluate_on_variants(
                homo_result['final_policy'], homo_result['final_model'], variants
            )
            hetero_eval = self._evaluate_on_variants(
                hetero_result['final_policy'], hetero_result['final_model'], variants
            )
            
            homo_result['evaluation'] = homo_eval
            hetero_result['evaluation'] = hetero_eval
            
            # Print results
            print(f"  Per-variant performance:")
            for i, desc in enumerate(homo_eval['description']):
                homo_perf = homo_eval['performance'][i]
                hetero_perf = hetero_eval['performance'][i]
                better = "HETERO" if hetero_perf > homo_perf else "HOMO"
                print(f"    {desc}: Homo={homo_perf:.4f}, Hetero={hetero_perf:.4f} ({better})")
            
            print(f"  Summary:")
            print(f"    Homo  - Mean: {homo_eval['mean']:.4f}, Min: {homo_eval['min']:.4f}")
            print(f"    Hetero - Mean: {hetero_eval['mean']:.4f}, Min: {hetero_eval['min']:.4f}")
            
            if self.use_wandb:
                self._log_seed_results_wandb(seed, homo_result, hetero_result, variants)
        
        aggregated = self._aggregate_results(homo_results, hetero_results, variants)
        
        if self.use_wandb:
            self._log_final_results_wandb(aggregated, variants)
            wandb.finish()
        
        self._save_results(experiment_name, homo_results, hetero_results, aggregated)
        self._print_summary(experiment_name, aggregated, variants)
        
        return {
            'homo_results': homo_results,
            'hetero_results': hetero_results,
            'aggregated': aggregated,
            'variants': variants
        }
    
    def _run_homogeneous_fspmi(self, n_agents: int, seed: int) -> Dict[str, Any]:
        """Run homogeneous F-SPMI (all agents on same MDP)."""
        ref_env = self._create_ref_environment()
        
        policy_chooser = GreedyPolicyChooser(self.nS, self.nA)
        model_chooser = SetModelChooser(self.model_set, self.nS, self.nA)
        
        fspmi = FSPMI(
            conf_mdp=ref_env,
            n_agents=n_agents,
            episodes_per_round=self.config.episodes_per_agent,
            eps=0.0,
            max_rounds=self.config.n_iterations,
            policy_chooser=policy_chooser,
            model_chooser=model_chooser,
            aggregation_method='weighted',
            persistent=True,
            delta_q=1,
            verbose=False
        )
        
        for i, agent in enumerate(fspmi.agents):
            agent.rng = np.random.RandomState(seed * 10000 + i)
        
        start_time = time.time()
        final_policy, final_model = fspmi.run(
            copy.deepcopy(self.initial_policy),
            copy.deepcopy(self.initial_model)
        )
        elapsed = time.time() - start_time
        
        return {
            'final_policy': final_policy,
            'final_model': final_model,
            'logger': fspmi.logger,
            'elapsed': elapsed,
            'seed': seed
        }
    
    def _run_heterogeneous_fspmi(self, variants: List[EnvironmentVariant], 
                                  seed: int) -> Dict[str, Any]:
        """Run heterogeneous F-SPMI (each agent on different MDP)."""
        hetero_config = HeterogeneousConfig(
            variants=variants,
            episodes_per_agent=self.config.episodes_per_agent,
            n_iterations=self.config.n_iterations,
            gamma=self.config.gamma,
            horizon=self.config.horizon,
            aggregation_method='weighted',
            use_parallel=self.config.use_parallel,
            update_mode=self.config.update_mode,
            target_policy_type=self.config.target_policy_type,
            softmax_temperature=self.config.softmax_temperature,
            min_step_size=self.config.min_step_size,
        )
        
        hfspmi = HeterogeneousFSPMI(hetero_config)
        
        np.random.seed(seed * 10000)
        
        policy_chooser = GreedyPolicyChooser(self.nS, self.nA)
        model_chooser = SetModelChooser(self.model_set, self.nS, self.nA)
        
        start_time = time.time()
        final_policy, final_model = hfspmi.run(
            copy.deepcopy(self.initial_policy),
            copy.deepcopy(self.initial_model),
            policy_chooser=policy_chooser,
            model_chooser=model_chooser,
            verbose=False
        )
        elapsed = time.time() - start_time
        
        return {
            'final_policy': final_policy,
            'final_model': final_model,
            'logger': hfspmi.logger,
            'elapsed': elapsed,
            'seed': seed
        }
    
    def _run_obstacle_comparison(
        self,
        experiment_name: str,
        obstacle_variants: List,
        description: str
    ) -> Dict[str, Any]:
        """Comparison for obstacle variants."""
        n_seeds = self.config.n_seeds
        n_agents = len(obstacle_variants)
        
        print(f"\nRunning {n_seeds} seeds...")
        
        # CRITICAL: Store original P matrices BEFORE any training
        # Training may modify mdp.P, so we need to preserve the true obstacle dynamics
        original_Ps = [copy.deepcopy(mdp.P) for mdp in obstacle_variants]
        original_mus = [copy.deepcopy(mdp.mu) for mdp in obstacle_variants]
        n_obstacles_list = [mdp.get_variant_info()['n_obstacles'] for mdp in obstacle_variants]
        
        homo_results = []
        hetero_results = []
        
        if self.use_wandb:
            run_name = self._get_wandb_run_name(experiment_name, f"seeds{n_seeds}")
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=run_name,
                group=self._get_wandb_group(experiment_name),
                tags=self.config.wandb_tags + [experiment_name, "obstacles"],
                config={
                    **self._get_base_wandb_config(),
                    "experiment_type": experiment_name,
                    "n_agents": n_agents,
                    "n_seeds": n_seeds,
                    "max_obstacles": self.config.max_obstacles,
                    "description": description,
                },
                reinit=True
            )
        
        for seed in range(n_seeds):
            print(f"\n--- Seed {seed+1}/{n_seeds} ---")
            
            # Reset obstacle variants to original P matrices before each seed
            for i, mdp in enumerate(obstacle_variants):
                mdp.P = copy.deepcopy(original_Ps[i])
            
            print(f"  Running Homogeneous F-SPMI on base track...")
            homo_result = self._run_homogeneous_fspmi(n_agents=n_agents, seed=seed)
            homo_results.append(homo_result)
            
            # Reset again before heterogeneous run
            for i, mdp in enumerate(obstacle_variants):
                mdp.P = copy.deepcopy(original_Ps[i])
            
            print(f"  Running Heterogeneous F-SPMI with obstacle variants...")
            hetero_result = self._run_obstacle_heterogeneous(obstacle_variants, seed)
            hetero_results.append(hetero_result)
            
            print(f"  Evaluating on all variants...")
            # Use ORIGINAL P matrices for evaluation, not the modified ones
            homo_eval = self._evaluate_obstacle_variants_with_original_P(
                homo_result['final_policy'], homo_result['final_model'], 
                original_Ps, original_mus, n_obstacles_list
            )
            hetero_eval = self._evaluate_obstacle_variants_with_original_P(
                hetero_result['final_policy'], hetero_result['final_model'], 
                original_Ps, original_mus, n_obstacles_list
            )
            
            homo_result['evaluation'] = homo_eval
            hetero_result['evaluation'] = hetero_eval
            
            print(f"  Per-variant performance:")
            for i, n_obs in enumerate(n_obstacles_list):
                homo_perf = homo_eval['performance'][i]
                hetero_perf = hetero_eval['performance'][i]
                better = "HETERO" if hetero_perf > homo_perf else "HOMO"
                print(f"    {n_obs} obstacles: Homo={homo_perf:.4f}, Hetero={hetero_perf:.4f} ({better})")
            
            print(f"  Summary:")
            print(f"    Homo  - Mean: {homo_eval['mean']:.4f}, Min: {homo_eval['min']:.4f}")
            print(f"    Hetero - Mean: {hetero_eval['mean']:.4f}, Min: {hetero_eval['min']:.4f}")
        
        aggregated = self._aggregate_obstacle_results(homo_results, hetero_results, obstacle_variants)
        
        if self.use_wandb:
            self._log_obstacle_results_wandb(aggregated, obstacle_variants)
            wandb.finish()
        
        self._save_obstacle_results(experiment_name, homo_results, hetero_results, aggregated)
        self._print_obstacle_summary(experiment_name, aggregated, obstacle_variants)
        
        return {
            'homo_results': homo_results,
            'hetero_results': hetero_results,
            'aggregated': aggregated,
            'variants': obstacle_variants
        }
    
    def _run_obstacle_heterogeneous(self, obstacle_variants: List, seed: int) -> Dict[str, Any]:
        """Run heterogeneous F-SPMI with obstacle variants."""
        from federated.agent import FederatedAgent
        from federated.server import FederatedServer
        
        ref_mdp = obstacle_variants[0]
        
        agents = [
            FederatedAgent(
                agent_id=i, nS=self.nS, nA=self.nA,
                gamma=self.config.gamma, horizon=self.config.horizon,
                seed=seed * 10000 + i
            )
            for i in range(len(obstacle_variants))
        ]
        
        server = FederatedServer(
            nS=self.nS, nA=self.nA,
            gamma=self.config.gamma, horizon=self.config.horizon,
            delta_q=(1 - self.config.gamma ** self.config.horizon) / (1 - self.config.gamma),
            aggregation_method='weighted'
        )
        
        policy = copy.deepcopy(self.initial_policy)
        model = copy.deepcopy(self.initial_model)
        
        performances = []
        alphas = []
        betas = []
        
        start_time = time.time()
        
        for iteration in range(self.config.n_iterations):
            local_stats = []
            for i, (agent, mdp) in enumerate(zip(agents, obstacle_variants)):
                stats = agent.collect_and_compute(mdp, policy, model, self.config.episodes_per_agent)
                local_stats.append(stats)
            
            global_stats = server.aggregate_statistics(local_stats)
            
            target_policy = self._greedy_policy(global_stats.Q_global)
            target_model = self._greedy_model(global_stats.U_global)
            
            update_result = server.compute_safe_update(
                policy, model, target_policy, target_model, global_stats
            )
            
            alpha_star = update_result.alpha_star
            beta_star = update_result.beta_star
            
            if self.config.step_size_boost > 1.0:
                alpha_star = min(1.0, alpha_star * self.config.step_size_boost)
                beta_star = min(1.0, beta_star * self.config.step_size_boost)
            
            if alpha_star > 0:
                from utils.tabular_operations import policy_convex_combination
                policy = policy_convex_combination(target_policy, policy, alpha_star)
            
            if beta_star > 0:
                from utils.tabular_operations import model_convex_combination
                model = model_convex_combination(ref_mdp.P, target_model, model, beta_star)
                # NOTE: Do NOT call mdp.set_model() on obstacle variants!
                # Each variant should keep its TRUE obstacle dynamics for data collection.
                # The 'model' variable is the learned shared model, separate from env dynamics.
            
            reward = TabularReward(ref_mdp.P, self.nS, self.nA)
            true_perf = evaluator.compute_performance(
                ref_mdp.mu, reward, policy, model,
                self.config.gamma, self.config.horizon, self.nS, self.nA
            )
            performances.append(true_perf)
            alphas.append(alpha_star)
            betas.append(beta_star)
        
        elapsed = time.time() - start_time
        
        class SimpleLogger:
            def __init__(self):
                self.true_performances = performances
                self.alphas = alphas
                self.betas = betas
        
        return {
            'final_policy': policy,
            'final_model': model,
            'logger': SimpleLogger(),
            'elapsed': elapsed,
            'seed': seed
        }
    
    def _greedy_policy(self, Q: np.ndarray) -> TabularPolicy:
        """Create greedy policy from Q values."""
        policy_rep = {s: np.zeros(self.nA) for s in range(self.nS)}
        for s in range(self.nS):
            q_array = Q[s * self.nA:(s + 1) * self.nA]
            max_val = np.max(q_array)
            greedy_actions = np.where(np.abs(q_array - max_val) < 1e-10)[0]
            policy_rep[s][greedy_actions] = 1.0 / len(greedy_actions)
        return TabularPolicy(policy_rep, self.nS, self.nA)
    
    def _greedy_model(self, U: np.ndarray) -> TabularModel:
        """Create greedy model from U values."""
        model_rep = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for s in range(self.nS):
            for a in range(self.nA):
                sa = s * self.nA + a
                u_array = U[sa]
                max_val = np.max(u_array)
                greedy_states = np.where(np.abs(u_array - max_val) < 1e-10)[0]
                probs = np.zeros(self.nS)
                probs[greedy_states] = 1.0 / len(greedy_states)
                model_rep[s][a] = list(zip(probs, range(self.nS)))
        return TabularModel(model_rep, self.nS, self.nA)
    
    def _evaluate_obstacle_variants(self, policy, model, obstacle_variants) -> Dict:
        """Evaluate policy on obstacle variants."""
        results = {'variant_id': [], 'performance': [], 'n_obstacles': []}
        
        for i, mdp in enumerate(obstacle_variants):
            mu = mdp.mu
            reward = TabularReward(mdp.P, self.nS, self.nA)
            variant_model = TabularModel(mdp.P, self.nS, self.nA)
            
            perf = evaluator.compute_performance(
                mu, reward, policy, variant_model,
                self.config.gamma, self.config.horizon, self.nS, self.nA
            )
            
            results['variant_id'].append(i)
            results['performance'].append(perf)
            results['n_obstacles'].append(mdp.get_variant_info()['n_obstacles'])
        
        perfs = np.array(results['performance'])
        results['mean'] = np.mean(perfs)
        results['std'] = np.std(perfs)
        results['min'] = np.min(perfs)
        results['max'] = np.max(perfs)
        
        return results
    
    def _evaluate_obstacle_variants_with_original_P(
        self, policy, model, 
        original_Ps: List, original_mus: List, n_obstacles_list: List
    ) -> Dict:
        """
        Evaluate policy on obstacle variants using ORIGINAL P matrices.
        
        This is critical because during training, mdp.P may be modified.
        For proper evaluation, we need to use the original obstacle dynamics.
        """
        results = {'variant_id': [], 'performance': [], 'n_obstacles': []}
        
        for i, (P, mu, n_obs) in enumerate(zip(original_Ps, original_mus, n_obstacles_list)):
            # Use the ORIGINAL P matrix with obstacles, not any modified version
            reward = TabularReward(P, self.nS, self.nA)
            variant_model = TabularModel(P, self.nS, self.nA)
            
            perf = evaluator.compute_performance(
                mu, reward, policy, variant_model,
                self.config.gamma, self.config.horizon, self.nS, self.nA
            )
            
            results['variant_id'].append(i)
            results['performance'].append(perf)
            results['n_obstacles'].append(n_obs)
        
        perfs = np.array(results['performance'])
        results['mean'] = np.mean(perfs)
        results['std'] = np.std(perfs)
        results['min'] = np.min(perfs)
        results['max'] = np.max(perfs)
        
        return results

    # ========================================================================
    # AGGREGATION AND LOGGING
    # ========================================================================
    
    def _aggregate_results(self, homo_results, hetero_results, variants) -> Dict:
        """Aggregate results across seeds."""
        n_seeds = len(homo_results)
        n_variants = len(variants)
        
        homo_perfs = np.zeros((n_seeds, n_variants))
        hetero_perfs = np.zeros((n_seeds, n_variants))
        
        for seed_idx in range(n_seeds):
            homo_perfs[seed_idx] = homo_results[seed_idx]['evaluation']['performance']
            hetero_perfs[seed_idx] = hetero_results[seed_idx]['evaluation']['performance']
        
        return {
            'homo': {
                'per_variant_mean': np.mean(homo_perfs, axis=0),
                'per_variant_std': np.std(homo_perfs, axis=0),
                'overall_mean': np.mean(homo_perfs),
                'overall_std': np.std(homo_perfs),
                'worst_case_mean': np.mean(np.min(homo_perfs, axis=1)),
                'worst_case_std': np.std(np.min(homo_perfs, axis=1)),
            },
            'hetero': {
                'per_variant_mean': np.mean(hetero_perfs, axis=0),
                'per_variant_std': np.std(hetero_perfs, axis=0),
                'overall_mean': np.mean(hetero_perfs),
                'overall_std': np.std(hetero_perfs),
                'worst_case_mean': np.mean(np.min(hetero_perfs, axis=1)),
                'worst_case_std': np.std(np.min(hetero_perfs, axis=1)),
            },
            'improvement': {
                'worst_case': np.mean(np.min(hetero_perfs, axis=1)) - np.mean(np.min(homo_perfs, axis=1)),
                'worst_case_pct': 100 * (
                    np.mean(np.min(hetero_perfs, axis=1)) - np.mean(np.min(homo_perfs, axis=1))
                ) / (np.mean(np.min(homo_perfs, axis=1)) + 1e-10),
            }
        }
    
    def _aggregate_obstacle_results(self, homo_results, hetero_results, obstacle_variants) -> Dict:
        """Aggregate obstacle experiment results."""
        n_seeds = len(homo_results)
        n_variants = len(obstacle_variants)
        
        homo_perfs = np.zeros((n_seeds, n_variants))
        hetero_perfs = np.zeros((n_seeds, n_variants))
        
        for seed_idx in range(n_seeds):
            homo_perfs[seed_idx] = homo_results[seed_idx]['evaluation']['performance']
            hetero_perfs[seed_idx] = hetero_results[seed_idx]['evaluation']['performance']
        
        # Get n_obstacles from the first seed's evaluation (stored during evaluation)
        n_obstacles_list = homo_results[0]['evaluation']['n_obstacles']
        
        return {
            'homo': {
                'per_variant_mean': np.mean(homo_perfs, axis=0),
                'per_variant_std': np.std(homo_perfs, axis=0),
                'overall_mean': np.mean(homo_perfs),
                'worst_case_mean': np.mean(np.min(homo_perfs, axis=1)),
            },
            'hetero': {
                'per_variant_mean': np.mean(hetero_perfs, axis=0),
                'per_variant_std': np.std(hetero_perfs, axis=0),
                'overall_mean': np.mean(hetero_perfs),
                'worst_case_mean': np.mean(np.min(hetero_perfs, axis=1)),
            },
            'n_obstacles': n_obstacles_list
        }
    
    def _log_seed_results_wandb(self, seed, homo_result, hetero_result, variants):
        """Log per-seed results to WandB."""
        homo_eval = homo_result['evaluation']
        hetero_eval = hetero_result['evaluation']
        
        log_dict = {
            f"seed_{seed}/homo_mean": homo_eval['mean'],
            f"seed_{seed}/homo_min": homo_eval['min'],
            f"seed_{seed}/hetero_mean": hetero_eval['mean'],
            f"seed_{seed}/hetero_min": hetero_eval['min'],
        }
        
        for i, desc in enumerate(homo_eval['description']):
            log_dict[f"seed_{seed}/homo_{desc}"] = homo_eval['performance'][i]
            log_dict[f"seed_{seed}/hetero_{desc}"] = hetero_eval['performance'][i]
        
        wandb.log(log_dict)
    
    def _log_final_results_wandb(self, aggregated, variants):
        """Log final aggregated results to WandB."""
        wandb.log({
            "homo_mean": aggregated['homo']['overall_mean'],
            "homo_worst_case": aggregated['homo']['worst_case_mean'],
            "hetero_mean": aggregated['hetero']['overall_mean'],
            "hetero_worst_case": aggregated['hetero']['worst_case_mean'],
            "improvement_worst_case": aggregated['improvement']['worst_case'],
            "improvement_worst_case_pct": aggregated['improvement']['worst_case_pct'],
        })
        
        table_data = []
        for i, v in enumerate(variants):
            table_data.append([
                v.description,
                aggregated['homo']['per_variant_mean'][i],
                aggregated['homo']['per_variant_std'][i],
                aggregated['hetero']['per_variant_mean'][i],
                aggregated['hetero']['per_variant_std'][i],
            ])
        
        table = wandb.Table(
            columns=["variant", "homo_mean", "homo_std", "hetero_mean", "hetero_std"],
            data=table_data
        )
        wandb.log({"per_variant_results": table})
    
    def _log_obstacle_results_wandb(self, aggregated, obstacle_variants):
        """Log obstacle experiment results to WandB."""
        wandb.log({
            "homo_mean": aggregated['homo']['overall_mean'],
            "homo_worst_case": aggregated['homo']['worst_case_mean'],
            "hetero_mean": aggregated['hetero']['overall_mean'],
            "hetero_worst_case": aggregated['hetero']['worst_case_mean'],
        })
        
        table_data = []
        # Use n_obstacles from aggregated results (stored during evaluation)
        for i, n_obs in enumerate(aggregated['n_obstacles']):
            table_data.append([
                n_obs,
                aggregated['homo']['per_variant_mean'][i],
                aggregated['hetero']['per_variant_mean'][i],
            ])
        
        table = wandb.Table(
            columns=["n_obstacles", "homo_perf", "hetero_perf"],
            data=table_data
        )
        wandb.log({"obstacle_results": table})
    
    # ========================================================================
    # SAVING AND PRINTING
    # ========================================================================
    
    def _save_results(self, experiment_name, homo_results, hetero_results, aggregated):
        """Save results to files."""
        import csv
        
        filepath = self.output_dir / f"{experiment_name}_aggregated.csv"
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'homo', 'hetero'])
            writer.writerow(['overall_mean', aggregated['homo']['overall_mean'], 
                           aggregated['hetero']['overall_mean']])
            writer.writerow(['worst_case_mean', aggregated['homo']['worst_case_mean'],
                           aggregated['hetero']['worst_case_mean']])
        
        print(f"  Saved: {filepath}")
    
    def _save_obstacle_results(self, experiment_name, homo_results, hetero_results, aggregated):
        """Save obstacle experiment results."""
        import csv
        
        filepath = self.output_dir / f"{experiment_name}_results.csv"
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['n_obstacles', 'homo_mean', 'homo_std', 'hetero_mean', 'hetero_std'])
            for i, n_obs in enumerate(aggregated['n_obstacles']):
                writer.writerow([
                    n_obs,
                    aggregated['homo']['per_variant_mean'][i],
                    aggregated['homo']['per_variant_std'][i],
                    aggregated['hetero']['per_variant_mean'][i],
                    aggregated['hetero']['per_variant_std'][i],
                ])
        
        print(f"  Saved: {filepath}")
    
    def _print_summary(self, experiment_name, aggregated, variants):
        """Print experiment summary."""
        print(f"\n{'='*60}")
        print(f"SUMMARY: {experiment_name.upper()}")
        print(f"{'='*60}")
        
        print(f"\nPer-variant performance:")
        print(f"{'Variant':<30} {'Homo':<20} {'Hetero':<20} {'Winner':<10}")
        print("-" * 80)
        
        for i, v in enumerate(variants):
            homo_perf = aggregated['homo']['per_variant_mean'][i]
            homo_std = aggregated['homo']['per_variant_std'][i]
            hetero_perf = aggregated['hetero']['per_variant_mean'][i]
            hetero_std = aggregated['hetero']['per_variant_std'][i]
            winner = "HETERO" if hetero_perf > homo_perf else "HOMO"
            print(f"{v.description:<30} {homo_perf:.4f}±{homo_std:.4f}      "
                  f"{hetero_perf:.4f}±{hetero_std:.4f}      {winner}")
        
        print(f"\nOverall:")
        print(f"  Homogeneous  - Mean: {aggregated['homo']['overall_mean']:.4f}, "
              f"Worst-case: {aggregated['homo']['worst_case_mean']:.4f}")
        print(f"  Heterogeneous - Mean: {aggregated['hetero']['overall_mean']:.4f}, "
              f"Worst-case: {aggregated['hetero']['worst_case_mean']:.4f}")
        
        improvement = aggregated['improvement']['worst_case_pct']
        if improvement > 0:
            print(f"\n✓ Heterogeneous improves worst-case by {improvement:.1f}%!")
        else:
            print(f"\n⚠ Heterogeneous worst-case is {-improvement:.1f}% lower")
    
    def _print_obstacle_summary(self, experiment_name, aggregated, obstacle_variants):
        """Print obstacle experiment summary."""
        print(f"\n{'='*60}")
        print(f"SUMMARY: {experiment_name.upper()}")
        print(f"{'='*60}")
        
        print(f"\nPer-variant performance:")
        print(f"{'Obstacles':<15} {'Homo':<15} {'Hetero':<15} {'Winner':<10}")
        print("-" * 55)
        
        # Use n_obstacles from aggregated results
        for i, n_obs in enumerate(aggregated['n_obstacles']):
            homo_perf = aggregated['homo']['per_variant_mean'][i]
            hetero_perf = aggregated['hetero']['per_variant_mean'][i]
            winner = "HETERO" if hetero_perf > homo_perf else "HOMO"
            print(f"{n_obs:<15} {homo_perf:.4f}          {hetero_perf:.4f}          {winner}")
        
        print(f"\nOverall:")
        print(f"  Homogeneous  - Mean: {aggregated['homo']['overall_mean']:.4f}, "
              f"Worst-case: {aggregated['homo']['worst_case_mean']:.4f}")
        print(f"  Heterogeneous - Mean: {aggregated['hetero']['overall_mean']:.4f}, "
              f"Worst-case: {aggregated['hetero']['worst_case_mean']:.4f}")
    
    # ========================================================================
    # RUN ALL
    # ========================================================================
    
    def run_all(self, experiments: Optional[List[str]] = None):
        """Run all specified experiments."""
        if experiments is None:
            experiments = ['dynamics', 'robustness', 'mixed']
            if OBSTACLES_AVAILABLE:
                experiments.append('obstacles')
        
        for exp in experiments:
            if exp == 'dynamics':
                self.run_dynamics_experiment()
            elif exp == 'robustness':
                self.run_robustness_experiment()
            elif exp == 'obstacles':
                self.run_obstacle_experiment()
            elif exp == 'mixed':
                self.run_mixed_experiment()
            else:
                print(f"Unknown experiment: {exp}")
        
        print(f"\n{'='*60}")
        print("ALL EXPERIMENTS COMPLETED")
        print(f"{'='*60}")
        print(f"Results saved to: {self.output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Heterogeneous F-SPMI Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run dynamics heterogeneity experiment
  python run_heterogeneous_experiments.py --experiment dynamics --n_seeds 5
  
  # Run robustness experiment on Track T2
  python run_heterogeneous_experiments.py --experiment robustness --track T2 --n_seeds 5
  
  # Run obstacle experiment with step size boost
  python run_heterogeneous_experiments.py --experiment obstacles --step_boost 50 --n_seeds 3
  
  # Run all experiments
  python run_heterogeneous_experiments.py --experiment all --n_seeds 5
  
  # Use config file
  python run_heterogeneous_experiments.py --config config/hetero_config.yaml
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--experiment', nargs='+', 
                        choices=['dynamics', 'robustness', 'obstacles', 'mixed', 'all'],
                        default=['dynamics'],
                        help='Experiments to run')
    parser.add_argument('--track', type=str, choices=['T1', 'T2', 'T4'], default='T1')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--n_iterations', type=int, default=500)
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per agent per round')
    parser.add_argument('--step_boost', type=float, default=50.0, help='Step size multiplier')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--no-parallel', action='store_true')
    parser.add_argument('--max_obstacles', type=int, default=3)
    parser.add_argument('--k_values', nargs='+', type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument('--pfail_values', nargs='+', type=float, default=[0.0, 0.05, 0.1])
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = HeterogeneousExperimentConfig(
            track_file=args.track,
            n_iterations=args.n_iterations,
            episodes_per_agent=args.episodes,
            n_seeds=args.n_seeds,
            step_size_boost=args.step_boost,
            wandb_enabled=not args.no_wandb,
            use_parallel=not args.no_parallel,
            max_obstacles=args.max_obstacles,
            dynamics_k_values=args.k_values,
            pfail_values=args.pfail_values,
        )
    
    runner = HeterogeneousExperimentRunner(config)
    
    if 'all' in args.experiment:
        experiments = ['dynamics', 'robustness', 'mixed']
        if OBSTACLES_AVAILABLE:
            experiments.append('obstacles')
    else:
        experiments = args.experiment
    
    runner.run_all(experiments)


if __name__ == '__main__':
    main()