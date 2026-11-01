"""GPU-trainable version of the classical benchmark task (MuJoCo MJX, MuJoCo Playground interface).

The scene is the classical one, compiled by the same builder (xbody.scene.build_task): Menagerie Franka, 3 cm
cube, the plate behind the obstacle post. Cube positions come from the classical evaluation region. Policies trained
here are scored back in the CPU MuJoCo scene, with the classical seeds and the same StageScorer
(scripts/eval_learned_mjx.py), so learned and classical numbers come out of one harness.

Action (8): joint-target increments for the 7 arm joints (scaled), and an absolute gripper command.
Observation: proprioception, TCP pose, cube pose, plate and post positions relative to the cube and the TCP.
Reward: dense staged shaping (reach, hold, lift, transport over the plate, release on it) with a penalty for
coming near the post. Training reward is a shaping signal only; the benchmark number is the staged success.

impl="warp" (MuJoCo Warp) on NVIDIA GPUs, impl="jax" anywhere (slow, used for CPU smoke tests).
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State

from xbody.bodies import SPECS
from xbody.scene import CLASSICAL, build_task

BODY = "franka"
REGION_X, REGION_Y = (0.18, 0.26), (-0.08, 0.06)   # classical evaluation region (scripts/eval_classical.py)
TABLE_Z, PLATE_R = 0.015, 0.05


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,             # the Menagerie model's own timestep, as in the classical runs
        episode_length=400,       # 8 s
        action_repeat=1,
        action_scale=0.05,
        impl="jax",
        naconmax=64 * 4096,
        naccdmax=64 * 4096,
        njmax=256,
        reward=config_dict.create(reach=1.0, hold=2.0, lift=4.0, transport=6.0, place=10.0, post=2.0,
                                  action_rate=0.01),
    )


class FrankaPickPlacePost(mjx_env.MjxEnv):
    def __init__(self, config: config_dict.ConfigDict = default_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        super().__init__(config, config_overrides)
        spec = SPECS[BODY]
        m = build_task(BODY)
        m.opt.timestep = self.sim_dt
        if self._config.impl == "jax":
            # The JAX backend has no cylinder-mesh collider: for CPU smoke tests only, the post becomes a capsule of
            # the same radius and height. GPU training (impl="warp") keeps the classical cylinder.
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "post")
            r, h = m.geom_size[g, 0], m.geom_size[g, 1]
            m.geom_type[g] = mujoco.mjtGeom.mjGEOM_CAPSULE
            m.geom_size[g, :2] = [r, h - r]
            # Same reason: compiling every arm-link mesh pair runs a CPU out of memory, so the smoke test keeps
            # collisions on the hand and fingers only.
            keep = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ["hand", *spec["jaw_bodies"]]}
            from xbody.bodies import robot_body_ids
            for gi in np.where(np.isin(m.geom_bodyid, list(robot_body_ids(m, spec) - keep)))[0]:
                m.geom_contype[gi] = m.geom_conaffinity[gi] = 0
        self._mj_model = m
        self._mjx_model = mjx.put_model(m, impl=self._config.impl)
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self._arm_q = np.array([m.jnt_qposadr[jid(j)] for j in spec["arm"]])
        self._arm_v = np.array([m.jnt_dofadr[jid(j)] for j in spec["arm"]])
        self._finger_q = np.array([m.jnt_qposadr[jid(f"finger_joint{i}")] for i in (1, 2)])
        self._arm_act = np.array([i for i in range(m.nu) if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT])
        self._grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, spec["gripper_actuator"])
        self._hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, spec["tcp_body"])
        self._cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._cube_q = m.jnt_qposadr[jid("cube_free")]
        self._tcp_off = jp.array(spec["tcp_offset"])
        off = np.array(spec["layout_offset"])
        self._plate = jp.array(m.site_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "plate")][:2])
        self._post = jp.array(np.array(CLASSICAL["post_xy"]) + off)
        self._region_lo = jp.array([REGION_X[0], REGION_Y[0]]) + jp.array(off)
        self._region_hi = jp.array([REGION_X[1], REGION_Y[1]]) + jp.array(off)
        self._lo, self._hi = (jp.array(x) for x in m.actuator_ctrlrange.T)
        q0 = np.zeros(m.nq)
        q0[self._arm_q] = spec["home"]
        q0[self._finger_q] = 0.04
        self._q0 = jp.array(q0)
        c0 = np.zeros(m.nu)
        c0[self._arm_act] = spec["home"]
        c0[self._grip_act] = spec["gripper_open"]
        self._ctrl0 = jp.array(c0)
        self._grip_open = spec["gripper_open"]

    # ---------------------------------------------------------------- interface
    @property
    def xml_path(self) -> str:
        return "xbody.scene.build_task('franka')"

    @property
    def action_size(self) -> int:
        return 8

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # ---------------------------------------------------------------- dynamics
    def reset(self, rng: jax.Array) -> State:
        rng, r_xy = jax.random.split(rng)
        xy = jax.random.uniform(r_xy, (2,), minval=self._region_lo, maxval=self._region_hi)
        q = self._q0.at[self._cube_q:self._cube_q + 3].set(jp.array([xy[0], xy[1], TABLE_Z]))
        q = q.at[self._cube_q + 3:self._cube_q + 7].set(jp.array([1.0, 0, 0, 0]))
        data = mjx_env.make_data(self._mj_model, qpos=q, qvel=jp.zeros(self._mj_model.nv), ctrl=self._ctrl0,
                                 impl=self._mjx_model.impl.value, naconmax=self._config.naconmax,
                                 naccdmax=self._config.naccdmax, njmax=self._config.njmax)
        data = mjx.forward(self._mjx_model, data)
        info = {"rng": rng, "reached": 0.0, "lifted": 0.0, "placed": 0.0, "last_act": jp.zeros(8)}
        metrics = {k: jp.zeros(()) for k in ("reach", "hold", "lift", "transport", "place", "post", "success")}
        return State(data, self._obs(data, info), jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, -1, 1)
        ctrl = state.data.ctrl
        arm = jp.clip(ctrl[self._arm_act] + action[:7] * self._config.action_scale,
                      self._lo[self._arm_act], self._hi[self._arm_act])
        ctrl = ctrl.at[self._arm_act].set(arm)
        ctrl = ctrl.at[self._grip_act].set((action[7] + 1) / 2 * self._grip_open)
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
        terms, info = self._reward(data, state.info, action)
        w = self._config.reward
        reward = (w.reach * terms["reach"] + w.hold * terms["hold"] + w.lift * terms["lift"]
                  + w.transport * terms["transport"] + w.place * terms["place"] - w.post * terms["post"]
                  - w.action_rate * jp.sum((action - state.info["last_act"]) ** 2))
        info = {**info, "last_act": action}
        cube = data.xpos[self._cube]
        done = (jp.any(jp.abs(cube[:2] - self._plate) > 0.6) | (cube[2] < 0.0)
                | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()).astype(float)
        metrics = {**state.metrics, **terms, "success": info["placed"]}
        return State(data, self._obs(data, info), reward, done, metrics, info)

    # ---------------------------------------------------------------- helpers
    def _tcp(self, data):
        R = data.xmat[self._hand].reshape(3, 3)
        return data.xpos[self._hand] + R @ self._tcp_off, R

    def _held(self, data, cube, tcp):
        fingers = data.qpos[self._finger_q]
        closed_on_cube = (fingers > 0.009).all() & (fingers < 0.021).all()   # 3 cm cube between the jaws
        return (jp.linalg.norm(cube - tcp) < 0.025) & closed_on_cube

    def _reward(self, data, info, action):
        tcp, _ = self._tcp(data)
        cube = data.xpos[self._cube]
        d_tc = jp.linalg.norm(cube - tcp)
        reached = jp.maximum(info["reached"], (d_tc < 0.03).astype(float))
        held = self._held(data, cube, tcp).astype(float)
        height = jp.clip((cube[2] - TABLE_Z) / 0.05, 0.0, 1.0)
        lifted = jp.maximum(info["lifted"], held * (height >= 1.0))
        d_plate = jp.linalg.norm(cube[:2] - self._plate)
        over = d_plate < 0.8 * PLATE_R
        released = (data.ctrl[self._grip_act] > 0.8 * self._grip_open) & (d_tc > 0.03)
        placed = jp.maximum(info["placed"], (lifted * over * (cube[2] < TABLE_Z + 0.01) * released).astype(float))
        # Near the post: the TCP, hand origin or cube within 3 cm of the post's axis, below its top.
        pts = jp.stack([tcp, data.xpos[self._hand], cube])
        near = (jp.linalg.norm(pts[:, :2] - self._post, axis=1) < CLASSICAL["post_r"] + 0.03) & \
               (pts[:, 2] < CLASSICAL["post_h"] + 0.02)
        terms = {
            "reach": 1 - jp.tanh(10 * d_tc),
            "hold": held * reached,
            "lift": held * height,
            "transport": lifted * (1 - jp.tanh(5 * d_plate)),
            "place": placed,
            "post": near.any().astype(float),
        }
        return terms, {**info, "reached": reached, "lifted": lifted, "placed": placed}

    def _obs(self, data, info):
        tcp, R = self._tcp(data)
        cube = data.xpos[self._cube]
        Rc = data.xmat[self._cube].reshape(3, 3)
        return jp.concatenate([
            data.qpos[self._arm_q], data.qvel[self._arm_v], data.qpos[self._finger_q],
            tcp, R[:, :2].ravel(), cube, Rc[:, :2].ravel(), cube - tcp,
            self._plate - cube[:2], self._post - tcp[:2],
            data.ctrl[self._arm_act] - data.qpos[self._arm_q], jp.array([info["lifted"]]),
        ])
