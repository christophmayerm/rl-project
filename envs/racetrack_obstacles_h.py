"""
Racetrack environment with configurable obstacles.

Extends RaceTrackConfigurableEnv to support:
1. Dynamic obstacle placement
2. Parameter variations (pfail, speed coefficients)
3. Easy creation of heterogeneous MDP variants for federated learning
"""

import numpy as np
import copy
from typing import List, Tuple, Optional, Dict, Any
from envs.racetrack_simulator import RaceTrackConfigurableEnv


class RaceTrackWithObstacles(RaceTrackConfigurableEnv):
    """
    Racetrack environment with dynamically placed obstacles.
    
    Obstacles are '4' cells that block movement - hitting one resets
    the car to zero velocity at the previous position.
    
    This enables creating heterogeneous MDP variants for federated learning,
    where each agent operates on a different track configuration.
    """
    
    def __init__(
        self,
        track_file: str,
        obstacle_positions: Optional[List[Tuple[int, int]]] = None,
        pfail: float = 0.0,
        speed_noise: float = 0.0,
        initial_configuration=None,
        reward_weight=None,
        reward_fail_abs: float = 0,
        horizon: int = 20,
        variant_id: int = 0
    ):
        """
        Initialize Racetrack with optional obstacles.
        
        Args:
            track_file: Base track CSV file name
            obstacle_positions: List of (x, y) positions to place obstacles
            pfail: Failure probability (higher = more stochastic)
            speed_noise: Noise added to speed-dependent transition probs
            initial_configuration: Model configuration coefficients
            reward_weight: Reward basis weights
            reward_fail_abs: Reward for failure state
            horizon: Episode horizon
            variant_id: Identifier for this variant (for logging)
        """
        self.obstacle_positions = obstacle_positions or []
        self.speed_noise = speed_noise
        self.variant_id = variant_id
        
        # Call parent constructor
        super().__init__(
            track_file=track_file,
            initial_configuration=initial_configuration,
            reward_weight=reward_weight,
            reward_fail_abs=reward_fail_abs,
            pfail=pfail,
            horizon=horizon
        )
        
        # Add obstacles AFTER parent init (modifies self.track)
        if self.obstacle_positions:
            self._add_obstacles(self.obstacle_positions)
            # Rebuild transition model with new obstacles
            self._rebuild_with_obstacles()
    
    def _add_obstacles(self, positions: List[Tuple[int, int]]):
        """
        Add obstacles by modifying transitions (NOT by changing state space).
        
        Obstacles are stored and their effect is applied in _modify_transitions_for_obstacles().
        This keeps nS constant across all variants for federation compatibility.
        
        Args:
            positions: List of (x, y) grid positions for obstacles
        """
        # Store obstacle positions (don't modify track - that changes nS!)
        self._obstacle_coords = []
        for x, y in positions:
            if 0 <= x < self.nrow and 0 <= y < self.ncol:
                # Only valid if on track (not start, goal, or already blocked)
                current = self.track[x, y]
                if current not in ['1', '2', '4', ' ']:
                    self._obstacle_coords.append((x, y))
    
    def _rebuild_with_obstacles(self):
        """
        Modify transitions to account for obstacles.
        
        Instead of removing states (which changes nS), we modify transitions:
        - Any transition landing on obstacle → goes to failure state instead
        
        This keeps state space CONSTANT across all variants.
        """
        if not self._obstacle_coords:
            return
        
        # Convert obstacle (x,y) to state indices (for all velocities)
        obstacle_states = set()
        for obs_x, obs_y in self._obstacle_coords:
            for vx in self.vel:
                for vy in self.vel:
                    try:
                        s_idx = self._s_to_i(obs_x, obs_y, vx, vy)
                        obstacle_states.add(s_idx)
                    except:
                        pass  # Position might not be in lin
        
        self._obstacle_states = obstacle_states
        failure_state = self.nS - 1
        
        # Modify all P matrices: redirect transitions to obstacles → failure
        for P_dict in [self.P_highspeed_noboost, self.P_lowspeed_noboost,
                       self.P_highspeed_boost, self.P_lowspeed_boost]:
            self._redirect_obstacle_transitions(P_dict, obstacle_states, failure_state)
        
        # Rebuild matrix representations
        self.P_highspeed_noboost_sas = self._p_sas(self.P_highspeed_noboost)
        self.P_highspeed_noboost_sa = self._p_sa(self.P_highspeed_noboost_sas)
        self.P_lowspeed_noboost_sas = self._p_sas(self.P_lowspeed_noboost)
        self.P_lowspeed_noboost_sa = self._p_sa(self.P_lowspeed_noboost_sas)
        self.P_highspeed_boost_sas = self._p_sas(self.P_highspeed_boost)
        self.P_highspeed_boost_sa = self._p_sa(self.P_highspeed_boost_sas)
        self.P_lowspeed_boost_sas = self._p_sas(self.P_lowspeed_boost)
        self.P_lowspeed_boost_sa = self._p_sa(self.P_lowspeed_boost_sas)
        
        # Reconfigure the active model
        self.P = self.model_configuration(self.k)
        
        print(f"  Added {len(self._obstacle_coords)} obstacles, "
              f"affects {len(obstacle_states)} states")
    
    def _redirect_obstacle_transitions(
        self, 
        P_dict: Dict, 
        obstacle_states: set, 
        failure_state: int
    ):
        """
        Modify transition dict to redirect obstacle transitions to failure.
        
        For each (s, a) pair, if any transition leads to an obstacle state,
        redirect that probability mass to the failure state.
        """
        for s in range(self.nS):
            for a in range(self.nA):
                transitions = P_dict[s][a]
                new_transitions = []
                redirect_prob = 0.0
                
                for (prob, next_s, reward, done) in transitions:
                    if next_s in obstacle_states:
                        # Redirect to failure state
                        redirect_prob += prob
                    else:
                        new_transitions.append((prob, next_s, reward, done))
                
                # Add accumulated failure probability
                if redirect_prob > 0:
                    # Check if failure state already in transitions
                    found = False
                    for i, (p, ns, r, d) in enumerate(new_transitions):
                        if ns == failure_state:
                            new_transitions[i] = (p + redirect_prob, ns, self.reward_fail_abs, True)
                            found = True
                            break
                    if not found:
                        new_transitions.append((redirect_prob, failure_state, self.reward_fail_abs, True))
                
                P_dict[s][a] = new_transitions
    
    def get_variant_info(self) -> Dict[str, Any]:
        """Return information about this variant for logging."""
        return {
            'variant_id': self.variant_id,
            'n_obstacles': len(self.obstacle_positions),
            'obstacle_positions': self.obstacle_positions,
            'pfail': self.pfail,
            'nS': self.nS,
            'nA': self.nA
        }
    
    def visualize_track(self) -> str:
        """Return ASCII visualization of track with obstacles."""
        lines = []
        for i in range(self.nrow):
            row = ""
            for j in range(self.ncol):
                cell = self.track[i, j]
                if cell == '1':
                    row += 'S'  # Start
                elif cell == '2':
                    row += 'G'  # Goal
                elif cell == '4':
                    row += '█'  # Obstacle/Wall
                elif cell == '5':
                    row += '░'  # Track
                elif cell == '3':
                    row += '▒'  # Offroad
                else:
                    row += ' '
            lines.append(row)
        return '\n'.join(lines)


def create_obstacle_variants(
    track_file: str,
    n_variants: int,
    max_obstacles: int = 6,
    seed: int = 42,
    base_pfail: float = 0.0,
    vary_pfail: bool = False,
    **kwargs
) -> List[RaceTrackWithObstacles]:
    """
    Create multiple track variants with different obstacle configurations.
    
    All variants have the SAME state space (nS, nA) - obstacles modify
    transitions, not state indices. This is critical for federation.
    
    Strategy:
    - Variant 0: Base track (no obstacles)
    - Variant 1-N: Increasing obstacles at semi-random positions
    
    Args:
        track_file: Base track CSV file
        n_variants: Number of variants to create
        max_obstacles: Maximum obstacles in hardest variant
        seed: Random seed for obstacle placement
        base_pfail: Base failure probability
        vary_pfail: Whether to also vary pfail across variants
        **kwargs: Additional args passed to RaceTrackWithObstacles
    
    Returns:
        List of RaceTrackWithObstacles instances (all with same nS, nA)
    """
    rng = np.random.RandomState(seed)
    variants = []
    
    # First, create base environment to get track layout
    base_env = RaceTrackWithObstacles(
        track_file=track_file,
        obstacle_positions=[],
        pfail=base_pfail,
        variant_id=0,
        **kwargs
    )
    
    # Find valid positions for obstacles (on track, not start/goal)
    valid_obstacle_positions = []
    for idx in range(base_env.nlin):
        x, y = base_env.lin[idx]
        cell_type = base_env.track[x, y]
        if cell_type == '5':  # On track (not start/goal)
            valid_obstacle_positions.append((int(x), int(y)))
    
    print(f"Base track: nS={base_env.nS}, nA={base_env.nA}")
    print(f"Found {len(valid_obstacle_positions)} valid obstacle positions")
    
    # Shuffle for random placement
    rng.shuffle(valid_obstacle_positions)
    
    variants.append(base_env)
    print(f"Variant 0: 0 obstacles (base)")
    
    # Create variants with increasing obstacles
    for i in range(1, n_variants):
        # Number of obstacles increases with variant index
        n_obs = int((i / (n_variants - 1)) * max_obstacles) if n_variants > 1 else 0
        n_obs = max(1, n_obs)  # At least 1 obstacle for non-base variants
        n_obs = min(n_obs, len(valid_obstacle_positions))  # Don't exceed available
        
        # Select obstacle positions
        obs_positions = valid_obstacle_positions[:n_obs]
        
        # Optionally vary pfail
        pfail = base_pfail
        if vary_pfail:
            pfail = base_pfail + (i / (n_variants - 1)) * 0.1  # Up to +0.1
        
        variant = RaceTrackWithObstacles(
            track_file=track_file,
            obstacle_positions=list(obs_positions),
            pfail=pfail,
            variant_id=i,
            **kwargs
        )
        variants.append(variant)
        
        print(f"Variant {i}: {n_obs} obstacles, pfail={pfail:.3f}")
    
    # Verify all variants have same state space
    ref_nS, ref_nA = variants[0].nS, variants[0].nA
    for i, v in enumerate(variants):
        assert v.nS == ref_nS and v.nA == ref_nA, \
            f"Variant {i} has wrong dimensions: ({v.nS}, {v.nA}) vs ({ref_nS}, {ref_nA})"
    
    print(f"\n✓ All {n_variants} variants have consistent state space: nS={ref_nS}, nA={ref_nA}")
    
    return variants


def create_parameter_variants(
    track_file: str,
    n_variants: int,
    pfail_range: Tuple[float, float] = (0.0, 0.2),
    seed: int = 42,
    **kwargs
) -> List[RaceTrackWithObstacles]:
    """
    Create variants with same track but different transition parameters.
    
    This creates heterogeneity through stochasticity, not obstacles.
    
    Args:
        track_file: Track CSV file
        n_variants: Number of variants
        pfail_range: (min, max) failure probability
        seed: Random seed
        **kwargs: Additional args
    
    Returns:
        List of RaceTrackWithObstacles instances
    """
    variants = []
    
    for i in range(n_variants):
        # Linear interpolation of pfail
        pfail = pfail_range[0] + (i / max(1, n_variants - 1)) * (pfail_range[1] - pfail_range[0])
        
        variant = RaceTrackWithObstacles(
            track_file=track_file,
            obstacle_positions=[],
            pfail=pfail,
            variant_id=i,
            **kwargs
        )
        variants.append(variant)
        
        print(f"Variant {i}: pfail={pfail:.3f}, nS={variant.nS}")
    
    return variants


# Predefined obstacle configurations for reproducibility
OBSTACLE_CONFIGS = {
    'easy': [
        [],  # No obstacles
        [(3, 4)],  # 1 obstacle
    ],
    'medium': [
        [],
        [(3, 4)],
        [(3, 4), (5, 6)],
        [(3, 4), (5, 6), (4, 5)],
    ],
    'hard': [
        [],
        [(3, 4)],
        [(3, 4), (5, 6)],
        [(3, 4), (5, 6), (4, 5)],
        [(3, 4), (5, 6), (4, 5), (6, 4)],
        [(3, 4), (5, 6), (4, 5), (6, 4), (5, 3)],
    ]
}


def create_predefined_variants(
    track_file: str = 'T1',
    difficulty: str = 'medium',
    **kwargs
) -> List[RaceTrackWithObstacles]:
    """
    Create variants using predefined obstacle configurations.
    
    Args:
        track_file: Track CSV file
        difficulty: 'easy', 'medium', or 'hard'
        **kwargs: Additional args
    
    Returns:
        List of RaceTrackWithObstacles instances
    """
    configs = OBSTACLE_CONFIGS.get(difficulty, OBSTACLE_CONFIGS['medium'])
    variants = []
    
    for i, obs_positions in enumerate(configs):
        variant = RaceTrackWithObstacles(
            track_file=track_file,
            obstacle_positions=obs_positions,
            variant_id=i,
            **kwargs
        )
        variants.append(variant)
        print(f"Variant {i}: {len(obs_positions)} obstacles, nS={variant.nS}")
    
    return variants