from typing import List, Optional, Sequence

from envs.racetrack_simulator import RaceTrackConfigurableEnv
from utils.tabular import TabularModel


class RaceTrackWithObstacles(RaceTrackConfigurableEnv):
    def __init__(self,
                 track_file,
                 initial_configuration=None,
                 reward_weight=None,
                 reward_fail_abs = 0,
                 pfail = 0.0,
                 horizon = 20,
                 obstacle_positions = None,
                 obstacle_at_iteration = 0):
        
        self.obstacle_positions = obstacle_positions or []
        self.obstacle_at_iteration = obstacle_at_iteration
        self._obstacles_applied = False

        super().__init__(
            track_file=track_file,
            initial_configuration=initial_configuration,
            reward_weight=reward_weight,
            reward_fail_abs=reward_fail_abs,
            pfail=pfail,
            horizon=horizon,
        )

        self._base_track = self.track.copy()

        if self._should_apply_obstacles(0):
            self._apply_obstacles()

    def _should_apply_obstacles(self, iteration):
        return (
            bool(self.obstacle_positions)
            and self.obstacle_at_iteration is not None
            and iteration >= self.obstacle_at_iteration
        )

    def maybe_apply_obstacles(self, iteration):
        if self._obstacles_applied or not self._should_apply_obstacles(iteration):
            return False
        self._apply_obstacles()
        return True

    def _apply_obstacles(self):
        for pos in self.obstacle_positions:
            if len(pos) != 2:
                continue
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.nrow and 0 <= y < self.ncol and self.track[x, y] == "5":
                self.track[x, y] = "4"

        self._rebuild_models()
        self._obstacles_applied = True

    def reset_obstacles(self, rebuild_models = True):
        if self._base_track is None:
            return
        self.track = self._base_track.copy()
        self._obstacles_applied = False
        if rebuild_models:
            self._rebuild_models()

    def _rebuild_models(self):
        self.P_highspeed_noboost = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        self.P_lowspeed_noboost = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        self.P_highspeed_boost = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        self.P_lowspeed_boost = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        self._build_P()

        self.P_highspeed_noboost_sas = self._p_sas(self.P_highspeed_noboost)
        self.P_highspeed_noboost_sa = self._p_sa(self.P_highspeed_noboost_sas)
        self.P_lowspeed_noboost_sas = self._p_sas(self.P_lowspeed_noboost)
        self.P_lowspeed_noboost_sa = self._p_sa(self.P_lowspeed_noboost_sas)
        self.P_highspeed_boost_sas = self._p_sas(self.P_highspeed_boost)
        self.P_highspeed_boost_sa = self._p_sa(self.P_highspeed_boost_sas)
        self.P_lowspeed_boost_sas = self._p_sas(self.P_lowspeed_boost)
        self.P_lowspeed_boost_sa = self._p_sa(self.P_lowspeed_boost_sas)

        self.P = self.model_configuration(self.k)

    def build_model_set(self):
        return [
            TabularModel(self.P_highspeed_noboost, self.nS, self.nA),
            TabularModel(self.P_lowspeed_noboost, self.nS, self.nA),
            TabularModel(self.P_highspeed_boost, self.nS, self.nA),
            TabularModel(self.P_lowspeed_boost, self.nS, self.nA),
        ]
