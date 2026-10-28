"""S1 classical pipeline and its one-layer variants (plan section "S1 classical variants"), on any MuJoCo body.

Default path: privileged object pose (layers 1-2, known geometry) -> top-down grasp waypoints -> IK with
position first and the approach axis held down (layers 3 and 7, hqp-axis) -> straight-line Cartesian path
(layer 4; the scene has no obstacles) -> trapezoidal timing (layer 5) -> joint-trajectory tracking (layer 6)
-> each body's own position servos (layers 8-9).

The default doubles as the Gate 0 scripted expert. Usage:
  python scripts/eval_s1.py <franka|so101> <episodes> [--ik hqp-axis|hqp-full|dls-full|newton-full]
      [--tracking joint|cartesian] [--timing trapezoid|quintic] [--kp-scale 1.0]
"""
import argparse
import json
import pathlib
import sys
import time

import mujoco
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody.bodies import SPECS, tcp_frame  # noqa: E402
from xbody.kinematics import Body, solve_ik  # noqa: E402
from xbody.scene import build, sample_layout  # noqa: E402
from xbody.stages import StageScorer, StepState  # noqa: E402
from xbody.stats import wilson  # noqa: E402

R_DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
CONTROL_HZ, V_MAX, T_MIN, EP_SECONDS = 50, 0.15, 0.4, 14.0
TRANSIT_Z = 0.09   # SO-101 keeps the approach within 4.1 deg of vertical here (Phase 0 IK check); 7.5 cm lift margin
IK_VARIANTS = {"hqp-axis": ("hqp", "axis"), "hqp-full": ("hqp", "full"), "dls-full": ("dls", "full"),
               "newton-full": ("newton", "full")}


def profile(s, kind):
    """Normalized progress in [0, 1] for normalized time s in [0, 1] (layer 5)."""
    s = np.clip(s, 0, 1)
    if kind == "quintic":                       # minimum-jerk, a smooth S-curve stand-in
        return 10 * s**3 - 15 * s**4 + 6 * s**5
    a = 0.25                                    # trapezoid: 25% accelerate, 50% cruise, 25% decelerate
    v = 1 / (1 - a)
    if s < a:
        return 0.5 * v / a * s * s
    if s > 1 - a:
        return 1 - 0.5 * v / a * (1 - s) ** 2
    return 0.5 * v * a + v * (s - a)


class Robot:
    def __init__(self, body_name, kp_scale):
        self.spec = SPECS[body_name]
        self.model = build(body_name, kp_scale=kp_scale)
        self.data = mujoco.MjData(self.model)
        bname, off, q = tcp_frame(self.model, self.spec)
        self.kin = Body(self.model, self.spec["arm"], bname, off, q)
        m = self.model
        act = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.arm_act = [i for i in range(m.nu) if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                        and m.actuator_trnid[i, 0] in self.kin.jids]
        self.arm_act.sort(key=lambda i: self.kin.jids.index(m.actuator_trnid[i, 0]))
        self.grip_act = act(self.spec["gripper_actuator"])
        self.cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.cube_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
        self.plate = m.site_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "plate")][:2].copy()
        jaw_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.spec["jaw_bodies"]]
        self.jaw_geoms = [set(np.where(m.geom_bodyid == j)[0]) for j in jaw_ids]
        self.substeps = max(1, int(round(1 / (CONTROL_HZ * m.opt.timestep))))

    def reset(self, cube_xy):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.kin.qadr] = self.spec["home"]
        d.qpos[self.cube_qadr:self.cube_qadr + 3] = [cube_xy[0], cube_xy[1], 0.015]
        d.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [1, 0, 0, 0]
        d.ctrl[self.arm_act] = self.spec["home"]
        d.ctrl[self.grip_act] = self.spec["gripper_open"]
        mujoco.mj_forward(m, d)

    def jaws_touch(self):
        hit = [False, False]
        for c in self.data.contact[: self.data.ncon]:
            pair = {c.geom1, c.geom2}
            if self.cube_geom in pair:
                other = (pair - {self.cube_geom}).pop() if len(pair) == 2 else -1
                for k in (0, 1):
                    hit[k] = hit[k] or other in self.jaw_geoms[k]
        return all(hit)

    def state(self, t):
        d = self.data
        tcp, _ = self.kin.fk(d.qpos[self.kin.qadr])
        return StepState(t=t, tcp=tcp, obj=d.xpos[self.cube_body].copy(),
                         obj_vel=float(np.linalg.norm(d.cvel[self.cube_body][3:])),
                         both_jaws_touch=self.jaws_touch(),
                         jaws_open=bool(d.ctrl[self.grip_act] == self.spec["gripper_open"]),
                         table_z=0.015, container_xy=self.plate, container_r=0.06)


def run_episode(robot, cube_xy, args, scorer_cfg):
    robot.reset(cube_xy)
    method, orient = IK_VARIANTS[args.ik]
    d = robot.data
    scorer = StageScorer(scorer_cfg)
    q = d.qpos[robot.kin.qadr].copy()
    tcp0, _ = robot.kin.fk(q)
    c, p = np.r_[cube_xy, 0.015], robot.plate
    R_t = R_DOWN
    if orient == "axis":
        # Grasp yaw (layer 3 endpoint selection): find the jaw yaw the arm reaches with the approach axis held down,
        # snap it to the nearest cube face (cube yaw is 0), then hold that full orientation for the episode.
        q_a, _ = solve_ik(robot.kin, c, R_DOWN, q0=q, method=method, orient="axis")
        _, R_a = robot.kin.fk(q_a)
        yaw = np.arctan2(R_a[1, 0], R_a[0, 0])
        yaw = np.round(yaw / (np.pi / 2)) * (np.pi / 2)
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R_t, orient = Rz @ R_DOWN, "full"
    # (target TCP, gripper command, dwell seconds)
    z = TRANSIT_Z
    plan = [(np.r_[cube_xy, z], "open", 0), (c, "open", 0.2), (c, "close", 0.6),
            (np.r_[cube_xy, z], "close", 0.6), (np.r_[p, z], "close", 0), (np.r_[p, 0.03], "close", 0.2),
            (np.r_[p, 0.03], "open", 0.5), (np.r_[p, z], "open", 0)]
    t, ik_fail, prev = 0.0, 0, tcp0
    q_prev = q
    for target, grip, dwell in plan:
        dist = np.linalg.norm(target - prev)
        T = max(T_MIN, dist / V_MAX) if dist > 1e-6 else 0.0
        if args.tracking == "joint" and T > 0:
            q_goal, info = solve_ik(robot.kin, target, R_t, q0=q_prev, method=method, orient=orient)
            ik_fail += info["pos_err_m"] > 0.005
        n = int(T * CONTROL_HZ) + int(dwell * CONTROL_HZ)
        for k in range(max(n, 1)):
            s = min(1.0, k / max(T * CONTROL_HZ, 1))
            u = profile(s, args.timing)
            if T > 0 and args.tracking == "joint":
                q_cmd = q_prev + u * (q_goal - q_prev)
            elif T > 0:
                q_cmd, info = solve_ik(robot.kin, prev + u * (target - prev), R_t, q0=d.ctrl[robot.arm_act].copy(),
                                       method=method, orient=orient, iters=30)
                ik_fail += info["pos_err_m"] > 0.005
            else:
                q_cmd = d.ctrl[robot.arm_act].copy()
            d.ctrl[robot.arm_act] = q_cmd
            d.ctrl[robot.grip_act] = robot.spec["gripper_open" if grip == "open" else "gripper_closed"]
            for _ in range(robot.substeps):
                mujoco.mj_step(robot.model, d)
            t += 1 / CONTROL_HZ
            scorer.update(robot.state(t))
            if t > EP_SECONDS:
                break
        q_prev = q_goal if (args.tracking == "joint" and T > 0) else d.ctrl[robot.arm_act].copy()
        prev = target
    for _ in range(int(0.6 * CONTROL_HZ)):      # settle so the place stage can see the cube at rest
        for _ in range(robot.substeps):
            mujoco.mj_step(robot.model, d)
        t += 1 / CONTROL_HZ
        scorer.update(robot.state(t))
    return scorer.stage, ik_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("body", choices=list(SPECS))
    ap.add_argument("episodes", type=int)
    ap.add_argument("--ik", default="hqp-axis", choices=list(IK_VARIANTS))
    ap.add_argument("--tracking", default="joint", choices=["joint", "cartesian"])
    ap.add_argument("--timing", default="trapezoid", choices=["trapezoid", "quintic"])
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--seed-base", type=int, default=20261028)
    args = ap.parse_args()
    proto = yaml.safe_load((ROOT / "protocol.yaml").read_text())
    robot = Robot(args.body, args.kp_scale)
    stages, fails, t0 = [], [], time.time()
    for ep in range(args.episodes):
        rng = np.random.default_rng(args.seed_base + ep)
        st, f = run_episode(robot, sample_layout(rng, args.body), args, proto["stages"])
        stages.append(st); fails.append(f)
    k, n = sum(s == 5 for s in stages), len(stages)
    tag = f"{args.body}_{args.ik}_{args.tracking}_{args.timing}_kp{args.kp_scale:g}"
    res = {"stack": "S1", "body": args.body, "ik": args.ik, "tracking": args.tracking, "timing": args.timing,
           "kp_scale": args.kp_scale, "episodes": n, "success": k, "rate": k / n, "wilson95": wilson(k, n),
           "stage_hist": {s: stages.count(s) for s in range(6)}, "mean_stage": float(np.mean(stages)),
           "ik_failures_per_episode": float(np.mean(fails)), "sec_per_episode": (time.time() - t0) / n,
           "stages": stages, "seed_base": args.seed_base,
           "stage_key": "0 none, 1 reach, 2 grasp, 3 lift, 4 transport, 5 place (protocol.yaml)"}
    out = ROOT / f"results/phase0/s1_{tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
