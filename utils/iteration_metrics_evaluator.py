import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils import evaluator as default_evaluator
from utils.tabular import TabularReward
import matplotlib.pyplot as plt
import matplotlib


@dataclass
class EvaluationOptions:
    state_coverage: bool = False
    state_entropy: bool = False
    reward_diversity: bool = False
    novelty_yield: bool = False
    store_visitations: bool = False
    coverage_threshold: float | None = None;
    model_divergence: bool = False

    def skip_evaluation(self):
        return not (
            self.state_coverage
            or self.state_entropy
            or self.reward_diversity
            or self.novelty_yield
            or self.model_divergence
        )


class IterationMetricsEvaluator:
    # Evaluator computing aux metrics at each iteration.
    # Optional: visualization of state visitations over time.

    def __init__(self, mdp, options: Optional[EvaluationOptions] = None, evaluator=None,
                 live_vis_path: Optional[str] = None, live_vis_normalize: bool = True,
                 reference_model=None):
        self.mdp = mdp
        self.options = options or EvaluationOptions()
        self.evaluator = evaluator or default_evaluator
        self.history: List[Dict[str, float]] = []
        self.visited_states = np.zeros(self.mdp.nS, dtype=bool)
        self.visitation_history: List[np.ndarray] = []
        self.live_vis_path = live_vis_path
        self.live_vis_normalize = live_vis_normalize
        self.reference_model = reference_model 

    def reset(self):
        self.history.clear()
        self.visited_states[:] = False
        self.visitation_history.clear()

    def evaluate(self, policy, model, reward=None, d_mu=None, delta_mu=None, iteration: Optional[int] = None):
        if not self._should_compute():
            return {}

        reward = reward or TabularReward(self.mdp.P, self.mdp.nS, self.mdp.nA)

        if d_mu is None:
            d_mu = self.evaluator.compute_discounted_s_distribution(
                self.mdp.mu, policy, model, self.mdp.gamma, self.mdp.horizon, self.mdp.nS, self.mdp.nA
            )
        if delta_mu is None:
            delta_mu = self.evaluator.compute_discounted_sa_distribution(
                self.mdp.mu, policy, model, self.mdp.gamma, self.mdp.horizon, self.mdp.nS, self.mdp.nA, d_mu
            )

        if self.options.coverage_threshold is None:
            # while this is dependent on discounted mass and horizon, it's a reasonable default
            self.options.coverage_threshold = 0.05 * d_mu.sum() / self.mdp.nS

        metrics: Dict[str, float] = {}
        if iteration is not None:
            metrics["iteration"] = int(iteration)
        if self.options.state_coverage:
            metrics["state_coverage"] = self._state_coverage(d_mu)
        if self.options.state_entropy:
            metrics["state_entropy"] = self._state_entropy(d_mu)
        if self.options.reward_diversity:
            metrics.update(self._reward_stats(delta_mu, reward, model))
        if self.options.novelty_yield:
            metrics["novelty_yield"] = self._novelty_yield(d_mu)
        if self.options.model_divergence:
            metrics.update(self._model_divergence(delta_mu, model))

        self._update_seen_states(d_mu)
        if self.options.store_visitations:
            self.visitation_history.append(np.array(d_mu, copy=True))
            if self.live_vis_path is not None:
                self.visualize_position_visitation(
                    self.live_vis_path, iteration=iteration, normalize=self.live_vis_normalize
                )

        self.history.append(metrics)

        return metrics

    def _state_coverage(self, d_mu):
        active_states = np.count_nonzero(d_mu > self.options.coverage_threshold)
        return float(active_states) / float(self.mdp.nS)

    def _state_entropy(self, d_mu):
        eps = 1e-24
        probs = np.clip(d_mu, eps, None)
        entropy = float(-np.sum(probs * np.log(probs)))
        normalizer = np.log(self.mdp.nS) if self.mdp.nS > 1 else 1.0
        return entropy / normalizer

    def _reward_stats(self, delta_mu, reward, model):
        weights = np.array(delta_mu, dtype=float)
        weights_sum = weights.sum()
        if weights_sum > 0:
            weights = weights / weights_sum

        reward_sa = (reward.get_matrix() * model.get_matrix()).sum(axis=1)
        mean_reward = float(np.dot(weights, reward_sa))
        variance = float(np.dot(weights, (reward_sa - mean_reward) ** 2))
        return {"reward_mean": mean_reward, "reward_std": float(np.sqrt(max(variance, 0.0)))}

    def _novelty_yield(self, d_mu):
        mass = float(np.sum(d_mu))
        if mass <= 1e-24: # avoid division by zero
            return 0.0
        unseen = np.logical_and(~self.visited_states, d_mu > self.options.coverage_threshold)
        return float(np.sum(d_mu[unseen]) / mass)
    
    def _model_divergence(self, delta_mu, model):
        P_ref = self.reference_model.get_matrix()
        P_cur = model.get_matrix() 

        # TV distance per (s,a): 0.5 * ||P_cur - P_ref||_1
        tv_sa = 0.5 * np.abs(P_cur - P_ref).sum(axis=1)

        w = np.array(delta_mu, dtype=float)
        s = w.sum()
        if s > 0:
            w /= s

        mean_tv = float(np.dot(w, tv_sa))
        max_tv = float(tv_sa.max())

        return {"model_tv_mean": mean_tv, "model_tv_max": max_tv}

    def _update_seen_states(self, d_mu):
        # mark states as visited if their discounted visitation exceeds the threshold
        self.visited_states = np.logical_or(self.visited_states, d_mu > self.options.coverage_threshold)

    def save(self, dir_path: str, file_name: str):
        """Save collected metrics history to CSV (semicolon-separated)."""
        if not self.history:
            raise ValueError("No metrics collected; run evaluate before saving.")

        keys = []
        if any("iteration" in m for m in self.history):
            keys.append("iteration")
        for m in self.history:
            for k in m.keys():
                if k not in keys:
                    keys.append(k)

        os.makedirs(dir_path, exist_ok=True)
        out_path = os.path.join(dir_path, file_name)
        with open(out_path, "w") as f:
            f.write(";".join(keys) + "\n")
            for m in self.history:
                row = [str(m.get(k, "")) for k in keys]
                f.write(";".join(row) + "\n")

    def _should_compute(self):
        opts = self.options
        return opts.state_coverage or opts.state_entropy or opts.reward_diversity or opts.novelty_yield
    

    # ----- VISUALIZER ------
    def position_visitation(self, iteration: Optional[int] = None, normalize: bool = True):
        # aggregate discounted visitation over positions (summing over velocities) for a stored iteration.
        d_mu = self._get_visitation(iteration)
        grid = self._aggregate_position_visitation(d_mu, normalize=normalize)
        return grid

    def visualize_position_visitation(self, file_path: str, iteration: Optional[int] = None,
                                      normalize: bool = True, cmap: str = "viridis"):
        # simple heatmap vis for position
        grid = self.position_visitation(iteration=iteration, normalize=normalize)

        colormap = matplotlib.cm.get_cmap(cmap).copy()
        colormap.set_bad(color="lightgray")

        plt.figure(figsize=(8, 6))
        plt.imshow(grid, origin="upper", cmap=colormap)
        plt.colorbar(label="Discounted visitation" + (" (normalized)" if normalize else ""))
        plt.title("Position visitation" + (f" (iteration {iteration})" if iteration is not None else ""))
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    def _get_visitation(self, iteration: Optional[int]):
        if iteration is None or iteration == -1: # simply last
            return self.visitation_history[-1]
        return self.visitation_history[iteration]

    def _aggregate_position_visitation(self, d_mu, normalize=True):
        # initialize grid with NaN from mdp
        grid = np.full((self.mdp.nrow, self.mdp.ncol), np.nan)
        vel = self.mdp.vel
        for x, y in self.mdp.lin:
            total = 0.0
            for vx in vel:
                for vy in vel:
                    idx = self.mdp._s_to_i(x, y, vx, vy)
                    total += d_mu[idx]
            grid[x, y] = total

        if normalize:
            mass = np.nansum(grid)
            if mass > 0:
                grid = grid / mass
        return grid