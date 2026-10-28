"""Staged success scorer, shared by every stack, route and body.

Each control step feeds a body-free state record; the scorer returns the furthest stage reached (0 to 5).
Thresholds come from protocol.yaml so they freeze with the protocol.
"""
from dataclasses import dataclass, field

import numpy as np

STAGES = ["none", "reach", "grasp", "lift", "transport", "place"]


@dataclass
class StepState:
    t: float                    # seconds since episode start
    tcp: np.ndarray             # TCP position (m), world frame
    obj: np.ndarray             # object centre (m)
    obj_vel: float              # object speed (m/s)
    both_jaws_touch: bool       # contact between the object and both jaws (one fixed and one moving on the SO-101)
    jaws_open: bool
    table_z: float
    container_xy: np.ndarray
    container_r: float


@dataclass
class StageScorer:
    cfg: dict                   # protocol["stages"]
    stage: int = 0
    _held_since: float = None
    _lift_since: float = None
    _rest_since: float = None
    log: list = field(default_factory=list)

    def update(self, s: StepState) -> int:
        c = self.cfg
        if self.stage < 1 and np.linalg.norm(s.tcp - s.obj) <= c["1_reach"]["gripper_to_object_m"]:
            self._advance(1, s.t)
        if self.stage >= 1:
            self._held_since = (self._held_since or s.t) if s.both_jaws_touch else None
            if self.stage < 2 and self._held_since is not None and s.t - self._held_since >= c["2_grasp"]["held_s"]:
                self._advance(2, s.t)
        lifted = s.obj[2] - s.table_z >= c["3_lift"]["height_m"]
        if self.stage >= 2:
            self._lift_since = (self._lift_since or s.t) if (lifted and s.both_jaws_touch) else None
            if self.stage < 3 and self._lift_since is not None and s.t - self._lift_since >= c["3_lift"]["hold_s"]:
                self._advance(3, s.t)
        over = np.linalg.norm(s.obj[:2] - s.container_xy) <= s.container_r
        if self.stage == 3 and over and lifted:
            self._advance(4, s.t)
        if self.stage == 4:
            resting = over and s.obj_vel < 0.01 and s.jaws_open and not s.both_jaws_touch
            self._rest_since = (self._rest_since or s.t) if resting else None
            if self._rest_since is not None and s.t - self._rest_since >= c["5_place"]["at_rest_s"]:
                self._advance(5, s.t)
        return self.stage

    def _advance(self, k, t):
        self.stage = k
        self.log.append((STAGES[k], round(t, 3)))

    @property
    def success(self):
        return self.stage == 5
