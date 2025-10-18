"""
Optimized RaceTrack environment using Numba JIT compilation.
This version significantly speeds up the environment initialization and operations.
"""

import os
import numpy as np
import pandas as pd
from envs.racetrack_simulator import RaceTrackConfigurableEnv
from optimized.numba_racetrack import (
    build_P_matrices_numba,
    p_sas_numba_wrapper,
    p_sa_numba,
    p_sas_to_dict_numba,
    check_valid_state_numba,
    check_valid_path_numba,
    next_state_numba,
    s_to_i_numba,
    reward_state_numba
)


class RaceTrackConfigurableEnvOptimized(RaceTrackConfigurableEnv):
    """
    Optimized version of RaceTrackConfigurableEnv using Numba JIT compilation.
    This class overrides the most computationally intensive methods with Numba-optimized versions.
    """

    def __init__(self, track_file, initial_configuration=None, reward_weight=None,
                 reward_fail_abs=0, pfail=0., horizon=20, use_numba=True):
        """
        Constructor with optional Numba optimization.

        :param track_file: csv file describing the track
        :param initial_configuration: coefficient describing the initial model
        :param reward_weight: input vector to weight the reward basis vector
        :param reward_fail_abs: reward in case of failure
        :param pfail: failure probability baseline
        :param horizon: number of time-steps in a single episode
        :param use_numba: whether to use Numba optimizations (default: True)
        """
        self.use_numba = use_numba

        # Call parent constructor which will call our optimized _build_P
        super().__init__(track_file, initial_configuration, reward_weight,
                        reward_fail_abs, pfail, horizon)

    def _build_P(self):
        """
        Optimized version of _build_P using Numba JIT compilation.
        Builds all 4 vertex transition models significantly faster.
        """
        if not self.use_numba:
            # Fall back to original implementation
            super()._build_P()
            return

        # Prepare data for Numba function
        lin = np.array(self.lin, dtype=np.int32)
        vel = np.array(self.vel, dtype=np.int32)
        nA = self.nA
        nS = self.nS

        # Convert track to flat array for Numba
        track_flat = np.array([ord(c) for row in self.track for c in row], dtype=np.int8)
        track_ncol = self.ncol

        # Default reward weight if not specified
        if self.reward_weight is None:
            reward_weight = np.array([1, 0, 0, 0, 0], dtype=np.float64)
        else:
            reward_weight = np.array(self.reward_weight, dtype=np.float64)

        # Call Numba-optimized function
        P_hs_nb, P_ls_nb, P_hs_b, P_ls_b = build_P_matrices_numba(
            lin, vel, nA, nS,
            track_flat, track_ncol, self.nrow, self.ncol,
            reward_weight, self.reward_fail_abs,
            self.max_psuc, self.min_psuc, self.max_psuc2, self.min_psuc2,
            self.max_pboost, self.min_pboost, self.pfail, self.max_speed,
            self.min_vel, self.max_vel, self.min_vel_nb, self.max_vel_nb, self.nvel
        )

        # Convert back to dictionary format for compatibility
        self.P_highspeed_noboost = self._matrix_to_dict(P_hs_nb, reward_weight, track_flat, track_ncol)
        self.P_lowspeed_noboost = self._matrix_to_dict(P_ls_nb, reward_weight, track_flat, track_ncol)
        self.P_highspeed_boost = self._matrix_to_dict(P_hs_b, reward_weight, track_flat, track_ncol)
        self.P_lowspeed_boost = self._matrix_to_dict(P_ls_b, reward_weight, track_flat, track_ncol)

    def _matrix_to_dict(self, P_matrix, reward_weight, track_flat, track_ncol):
        """
        Convert dense matrix representation to dictionary format.

        Args:
            P_matrix: Dense probability matrix (nS, nA, nS)
            reward_weight: Reward weights
            track_flat: Flattened track array
            track_ncol: Number of columns in track

        Returns:
            Dictionary representation compatible with DiscreteEnv
        """
        P_dict = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        # Convert using Numba function
        transitions_flat = p_sas_to_dict_numba(
            P_matrix, self.nS, self.nA, reward_weight,
            np.array(self.lin, dtype=np.int32), self.nvel, self.min_vel,
            track_flat, track_ncol, self.reward_fail_abs
        )

        # Reconstruct dictionary
        idx = 0
        for s in range(self.nS):
            for a in range(self.nA):
                P_dict[s][a] = list(transitions_flat[idx])
                idx += 1

        return P_dict

    def _p_sas(self, P):
        """
        Optimized version of _p_sas using Numba JIT compilation.
        """
        if not self.use_numba:
            return super()._p_sas(P)

        # If P is already a matrix, return it
        if isinstance(P, np.ndarray) and P.shape == (self.nS, self.nA, self.nS):
            return P

        # Use wrapper function to handle dictionary conversion
        return p_sas_numba_wrapper(P, self.nS, self.nA)

    def _p_sa(self, P_sas):
        """
        Optimized version of _p_sa using Numba JIT compilation.
        """
        if not self.use_numba:
            return super()._p_sa(P_sas)

        # Use Numba-optimized conversion
        return p_sa_numba(P_sas, self.nS, self.nA)

    def _check_valid_state(self, x, y):
        """
        Optimized state validity checking.
        """
        if not self.use_numba:
            return super()._check_valid_state(x, y)

        track_flat = np.array([ord(c) for row in self.track for c in row], dtype=np.int8)
        return check_valid_state_numba(x, y, self.nrow, self.ncol, track_flat, self.ncol)

    def _check_valid_path(self, x1, y1, x2, y2):
        """
        Optimized path validity checking.
        """
        if not self.use_numba:
            return super()._check_valid_path(x1, y1, x2, y2)

        track_flat = np.array([ord(c) for row in self.track for c in row], dtype=np.int8)
        return check_valid_path_numba(x1, y1, x2, y2, self.nrow, self.ncol, track_flat, self.ncol)

    def _next_state(self, x, y, vx, vy, a, outcome):
        """
        Optimized next state computation.
        """
        if not self.use_numba:
            return super()._next_state(x, y, vx, vy, a, outcome)

        track_flat = np.array([ord(c) for row in self.track for c in row], dtype=np.int8)
        return next_state_numba(x, y, vx, vy, a, outcome, self.nrow, self.ncol, track_flat, self.ncol)

    def _s_to_i(self, x, y, vx, vy):
        """
        Optimized state to index conversion.
        """
        if not self.use_numba:
            return super()._s_to_i(x, y, vx, vy)

        lin = np.array(self.lin, dtype=np.int32)
        return s_to_i_numba(x, y, vx, vy, lin, self.nvel, self.min_vel)

    def _reward_state(self, x, y, vx, vy, weight):
        """
        Optimized reward computation.
        """
        if not self.use_numba:
            return super()._reward_state(x, y, vx, vy, weight)

        if weight is None:
            weight = np.array([1, 0, 0, 0, 0], dtype=np.float64)
        else:
            weight = np.array(weight, dtype=np.float64)

        track_flat = np.array([ord(c) for row in self.track for c in row], dtype=np.int8)
        return reward_state_numba(x, y, vx, vy, weight, track_flat, self.ncol, self.reward_fail_abs)