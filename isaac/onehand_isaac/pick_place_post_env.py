"""Isaac Lab port of the classical benchmark task: Franka picks a 3 cm cube and places it on a plate behind a post.

Mirrors xbody/learned/mjx_pick.py (same scene numbers, action, observation and reward terms) so the MJX and Isaac
routes differ only in the simulator. Scene numbers come from results/learned/layouts_franka.json, which
scripts/export_layouts.py writes from the classical builder; evaluation uses its seeded cube positions.

Written against Isaac Lab 2.x (Isaac Sim 5.x). Not yet run: this container has no RTX GPU. Expect small API fixes
on first launch (see isaac/README.md).
"""
from __future__ import annotations

import json
import pathlib

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

ROOT = pathlib.Path(__file__).resolve().parents[2]
LAYOUT = json.loads((ROOT / "results/learned/layouts_franka.json").read_text())
TABLE_Z = LAYOUT["cube_edge"] / 2
REGION_LO = (0.18 + 0.30, -0.08)      # classical evaluation region, Franka layout offset applied
REGION_HI = (0.26 + 0.30, 0.06)


@configclass
class PickPlacePostEnvCfg(DirectRLEnvCfg):
    decimation = 10                    # 0.002 s physics, 0.02 s control: as in the MJX env
    episode_length_s = 8.0
    action_space = 8
    observation_space = 49            # same 49 numbers, same order, as the MJX env
    state_space = 0
    action_scale = 0.05
    cameras = False                    # True adds a wrist and a scene RGB camera (for VLA / world-model data)
    rew = dict(reach=1.0, hold=2.0, lift=4.0, transport=6.0, place=10.0, post=2.0, action_rate=0.01)

    sim: SimulationCfg = SimulationCfg(dt=0.002, render_interval=10)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(LAYOUT["cube_edge"],) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=LAYOUT["cube_mass"]),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=LAYOUT["cube_friction"],
                                                            dynamic_friction=LAYOUT["cube_friction"]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.0, TABLE_Z)),
    )
    post: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Post",
        spawn=sim_utils.CylinderCfg(
            radius=LAYOUT["post_r"], height=LAYOUT["post_h"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*LAYOUT["post_xy"], LAYOUT["post_h"] / 2)),
    )
    wrist_cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.065, 0.0, 0.03), rot=(0.0, 0.7071, 0.7071, 0.0), convention="ros"),
        data_types=["rgb"], width=128, height=128,
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955),
    )


class PickPlacePostEnv(DirectRLEnv):
    cfg: PickPlacePostEnvCfg

    def __init__(self, cfg: PickPlacePostEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        r = self.robot
        self.arm_ids = r.find_joints("panda_joint[1-7]")[0]
        self.finger_ids = r.find_joints("panda_finger_joint.*")[0]
        self.hand_idx = r.find_bodies("panda_hand")[0][0]
        self.lo = r.data.soft_joint_pos_limits[0, :, 0]
        self.hi = r.data.soft_joint_pos_limits[0, :, 1]
        self.targets = r.data.default_joint_pos.clone()
        dev = self.device
        self.plate = torch.tensor(LAYOUT["plate_xy"], device=dev)
        self.post_xy = torch.tensor(LAYOUT["post_xy"], device=dev)
        self.tcp_off = torch.tensor([0.0, 0.0, 0.1034], device=dev)
        n = self.num_envs
        self.flags = {k: torch.zeros(n, device=dev) for k in ("reached", "lifted", "placed")}
        self.last_act = torch.zeros(n, 8, device=dev)
        self.eval_layouts = None       # set by the evaluator: (num_envs, 2) cube positions from the classical seeds

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.cube = RigidObject(self.cfg.cube)
        self.post = RigidObject(self.cfg.post)
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cube"] = self.cube
        self.scene.rigid_objects["post"] = self.post
        if self.cfg.cameras:
            self.wrist_cam = TiledCamera(self.cfg.wrist_cam)
            self.scene.sensors["wrist_cam"] = self.wrist_cam
        sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    # ------------------------------------------------------------------ helpers
    def _tcp(self):
        from isaaclab.utils.math import quat_apply
        p = self.robot.data.body_pos_w[:, self.hand_idx] - self.scene.env_origins
        q = self.robot.data.body_quat_w[:, self.hand_idx]
        return p + quat_apply(q, self.tcp_off.expand(self.num_envs, 3)), q

    def _cube(self):
        return self.cube.data.root_pos_w - self.scene.env_origins, self.cube.data.root_quat_w

    # ------------------------------------------------------------------ RL hooks
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-1, 1)
        arm = self.targets[:, self.arm_ids] + self.actions[:, :7] * self.cfg.action_scale
        self.targets[:, self.arm_ids] = torch.clamp(arm, self.lo[self.arm_ids], self.hi[self.arm_ids])
        width = (self.actions[:, 7:8] + 1) / 2 * 0.04          # MJX: 0..255 tendon command = 0..0.04 m per finger
        self.targets[:, self.finger_ids] = width.expand(-1, len(self.finger_ids))

    def _apply_action(self):
        self.robot.set_joint_position_target(self.targets)

    def _get_observations(self) -> dict:
        from isaaclab.utils.math import matrix_from_quat
        tcp, qh = self._tcp()
        cube, qc = self._cube()
        Rh, Rc = matrix_from_quat(qh), matrix_from_quat(qc)
        jp, jv = self.robot.data.joint_pos, self.robot.data.joint_vel
        obs = torch.cat([
            jp[:, self.arm_ids], jv[:, self.arm_ids], jp[:, self.finger_ids],
            tcp, Rh[:, :, :2].reshape(-1, 6), cube, Rc[:, :, :2].reshape(-1, 6), cube - tcp,
            self.plate - cube[:, :2], self.post_xy - tcp[:, :2],
            self.targets[:, self.arm_ids] - jp[:, self.arm_ids], self.flags["lifted"][:, None],
        ], dim=-1)
        out = {"policy": obs}
        if self.cfg.cameras:
            out["wrist_rgb"] = self.wrist_cam.data.output["rgb"]
        return out

    def _get_rewards(self) -> torch.Tensor:
        w = self.cfg.rew
        tcp, _ = self._tcp()
        cube, _ = self._cube()
        d_tc = torch.linalg.norm(cube - tcp, dim=-1)
        fingers = self.robot.data.joint_pos[:, self.finger_ids]
        held = ((d_tc < 0.025) & (fingers > 0.009).all(-1) & (fingers < 0.021).all(-1)).float()
        self.flags["reached"] = torch.maximum(self.flags["reached"], (d_tc < 0.03).float())
        height = ((cube[:, 2] - TABLE_Z) / 0.05).clamp(0, 1)
        self.flags["lifted"] = torch.maximum(self.flags["lifted"], held * (height >= 1.0).float())
        d_plate = torch.linalg.norm(cube[:, :2] - self.plate, dim=-1)
        released = (self.targets[:, self.finger_ids].mean(-1) > 0.032) & (d_tc > 0.03)
        placed_now = self.flags["lifted"] * (d_plate < 0.8 * LAYOUT["plate_r"]).float() * \
            (cube[:, 2] < TABLE_Z + 0.01).float() * released.float()
        self.flags["placed"] = torch.maximum(self.flags["placed"], placed_now)
        hand = self.robot.data.body_pos_w[:, self.hand_idx] - self.scene.env_origins
        pts = torch.stack([tcp, hand, cube], 1)
        near = ((torch.linalg.norm(pts[..., :2] - self.post_xy, dim=-1) < LAYOUT["post_r"] + 0.03)
                & (pts[..., 2] < LAYOUT["post_h"] + 0.02)).any(-1).float()
        r = (w["reach"] * (1 - torch.tanh(10 * d_tc)) + w["hold"] * held * self.flags["reached"]
             + w["lift"] * held * height + w["transport"] * self.flags["lifted"] * (1 - torch.tanh(5 * d_plate))
             + w["place"] * self.flags["placed"] - w["post"] * near
             - w["action_rate"] * ((self.actions - self.last_act) ** 2).sum(-1))
        self.last_act = self.actions.clone()
        self.extras["log"] = {"success": self.flags["placed"].mean(), "post_near": near.mean()}
        return r

    def _get_dones(self):
        cube, _ = self._cube()
        out = (torch.linalg.norm(cube[:, :2] - self.plate, dim=-1) > 0.6) | (cube[:, 2] < -0.01)
        return out, self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        r = self.robot
        q = r.data.default_joint_pos[env_ids].clone()
        q[:, self.arm_ids] = torch.tensor(LAYOUT["home"], device=self.device)
        q[:, self.finger_ids] = 0.04
        r.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.targets[env_ids] = q
        if self.eval_layouts is not None:
            xy = self.eval_layouts[env_ids]
        else:
            lo, hi = torch.tensor(REGION_LO, device=self.device), torch.tensor(REGION_HI, device=self.device)
            xy = lo + (hi - lo) * torch.rand(n, 2, device=self.device)
        pose = torch.zeros(n, 7, device=self.device)
        pose[:, :2] = xy
        pose[:, 2] = TABLE_Z
        pose[:, 3] = 1.0
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.cube.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.cube.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)
        for f in self.flags.values():
            f[env_ids] = 0.0
        self.last_act[env_ids] = 0.0
