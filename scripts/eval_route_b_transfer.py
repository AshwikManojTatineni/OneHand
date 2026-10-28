"""Route B cross-body transfer: frozen Franka TD-MPC2 PickCube policy driving the SO100 (rung R6), no training.

Interfaces (plan section "Transfer interfaces"):
  K   : the Franka's end-effector position delta -> SO100 joint targets by differential IK (layer 7: dls or hqp-axis);
        the Franka gripper command -> SO100 jaw angle via a measured aperture table; task frames aligned by
        shifting SO100 scene coordinates onto the Franka's cube region (metric distances kept).
  K+P : as K, plus a virtual Franka (Menagerie model) whose TCP tracks the SO100 TCP supplies the joint block of
        the observation. Under K alone the joint block is frozen at the Franka rest pose.

Usage: python scripts/eval_route_b_transfer.py <K|K+P> <episodes> [--ik dls|hqp-axis] [--no-mpc] [--no-shift]
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: E402,F401
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import sapien  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody.kinematics import Body, solve_ik  # noqa: E402
from xbody.newt_agent import FrozenNewtAgent  # noqa: E402
from xbody.stats import wilson  # noqa: E402

ACTION_REPEAT, EP_LEN = 2, 25
FRANKA_BASE = np.array([-0.615, 0.0, 0.0])          # PickCube panda base (world)
FRANKA_REST = np.array([0.0, np.pi / 8, 0.0, -5 * np.pi / 8, 0.0, 3 * np.pi / 4, np.pi / 4])
SHIFT = np.array([0.46, 0.0, 0.0])                   # SO100 cube_spawn_center (-0.46, 0) -> panda (0, 0)
EE_DELTA_LIMIT = 0.1                                 # panda pd_ee_delta_pos bound (m) for a normalized action of 1
R_DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)


class SO100Kinematics:
    """Differential IK on the SO100 via ManiSkill's own Pinocchio model; TCP = midpoint of the two jaw tips."""

    def __init__(self, agent, ik="dls"):
        self.robot = agent.robot
        self.pm = self.robot.create_pinocchio_model()
        names = [l.name for l in self.robot.get_links()]
        self.i1, self.i2 = names.index("Fixed_Jaw_tip"), names.index("Moving_Jaw_tip")
        self.ijaw = names.index("Fixed_Jaw")
        self.lim = self.robot.get_qlimits().numpy()[0]
        self.ik = ik
        self.base = self.robot.pose.sp
        self.axis_local = None

    def tcp_world(self, q):
        self.pm.compute_forward_kinematics(q)
        p = (self.pm.get_link_pose(self.i1).p + self.pm.get_link_pose(self.i2).p) / 2
        return (self.base * sapien.Pose(p)).p

    def approach_world(self, q):
        self.pm.compute_forward_kinematics(q)
        R = (self.base * self.pm.get_link_pose(self.ijaw)).to_transformation_matrix()[:3, :3]
        return R @ self.axis_local

    def capture_axis(self, q):
        """Pick the Fixed_Jaw axis that points most nearly down at reset; hqp-axis holds it pointing down."""
        self.pm.compute_forward_kinematics(q)
        R = (self.base * self.pm.get_link_pose(self.ijaw)).to_transformation_matrix()[:3, :3]
        k = int(np.argmax(np.abs(R[2])))
        self.axis_local = np.eye(3)[k] * -np.sign(R[2, k])

    def solve(self, q, p_target, iters=15, damping=0.02, eps=1e-4):
        q = q.copy()
        arm = slice(0, 5)
        for _ in range(iters):
            p = self.tcp_world(q)
            e = p_target - p
            if np.linalg.norm(e) < 5e-4:
                break
            J = np.zeros((3, 5))
            for j in range(5):
                dq = q.copy(); dq[j] += eps
                J[:, j] = (self.tcp_world(dq) - p) / eps
            Jp = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(3))
            step = Jp @ e
            if self.ik == "hqp-axis":
                a = self.approach_world(q)
                ea = np.cross(a, np.array([0, 0, -1.0]))
                Ja = np.zeros((3, 5))
                for j in range(5):
                    dq = q.copy(); dq[j] += eps
                    Ja[:, j] = (np.cross(self.approach_world(dq), np.array([0, 0, -1.0])) - ea) / eps
                N = np.eye(5) - Jp @ J
                JaN = Ja @ N
                step += N @ (JaN.T @ np.linalg.solve(JaN @ JaN.T + 0.05 * np.eye(3), -ea - Ja @ step))
            n = np.linalg.norm(step)
            if n > 0.2:
                step *= 0.2 / n
            q[arm] = np.clip(q[arm] + step, self.lim[arm, 0], self.lim[arm, 1])
        return q


def calibrate_jaw(env, kin):
    """Measure jaw aperture (distance between tips) against gripper joint angle: a measurement, not training."""
    angles = np.linspace(kin.lim[5, 0], kin.lim[5, 1], 23)
    q0 = kin.robot.get_qpos().numpy()[0]
    ap = []
    for g in angles:
        q = q0.copy(); q[5] = g
        kin.pm.compute_forward_kinematics(q)
        ap.append(np.linalg.norm(kin.pm.get_link_pose(kin.i1).p - kin.pm.get_link_pose(kin.i2).p))
    ap = np.array(ap)
    order = np.argsort(ap)
    return ap[order], angles[order]


def franka_twin():
    m = mujoco.MjModel.from_xml_path(str(ROOT / "assets/mujoco_menagerie/franka_emika_panda/panda.xml"))
    return Body(m, [f"joint{i}" for i in range(1, 8)], "hand", np.array([0, 0, 0.1034]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("interface", choices=["K", "K+P"])
    ap.add_argument("episodes", type=int)
    ap.add_argument("--ik", default="dls", choices=["dls", "hqp-axis"])
    ap.add_argument("--no-mpc", action="store_true")
    ap.add_argument("--no-shift", action="store_true")
    ap.add_argument("--seed-base", type=int, default=20261028)
    ap.add_argument("--record", default=None, help="save policy-view observations and actions (.npz)")
    args = ap.parse_args()

    env = gym.make("PickCubeSO100-v1", obs_mode="state", control_mode="pd_joint_pos", num_envs=1, sim_backend="cpu")
    agent = FrozenNewtAgent(ROOT / "checkpoints/newt/ms-pick-cube.pt", "ms-pick-cube", 4, EP_LEN, mpc=not args.no_mpc)
    twin = franka_twin()
    shift = np.zeros(3) if args.no_shift else SHIFT
    results, t_start, rec = [], time.time(), []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed_base + ep)
        u = env.unwrapped
        kin = SO100Kinematics(u.agent, args.ik)
        q = u.agent.robot.get_qpos().numpy()[0]
        kin.capture_axis(q)
        ap_tab, ang_tab = calibrate_jaw(env, kin)
        q_twin, prev_twin = FRANKA_REST.copy(), FRANKA_REST.copy()
        stage, max_stage, grasp_steps, ok, ik_err = 0, 0, 0, False, []
        ep_obs, ep_act = [], []
        for t in range(EP_LEN):
            o = np.asarray(obs).reshape(-1)
            qpos, rest = o[:6], o[12:]
            rest = rest.copy()
            rest[1:4] += shift          # tcp position
            rest[8:11] += shift         # goal position
            rest[11:14] += shift        # object position
            aperture = np.linalg.norm(
                (lambda: (kin.pm.compute_forward_kinematics(qpos),
                          kin.pm.get_link_pose(kin.i1).p - kin.pm.get_link_pose(kin.i2).p)[1])())
            finger = np.clip(aperture / 2, 0.0, 0.04)
            if args.interface == "K+P":
                tcp_f = rest[1:4] - FRANKA_BASE
                q_twin, _ = solve_ik(twin, tcp_f, R_DOWN, q0=q_twin, method="hqp", iters=60)
            fq = np.concatenate([q_twin, [finger, finger]])
            fqd = np.concatenate([(q_twin - prev_twin) * 10.0, [0.0, 0.0]])   # 20 Hz control, action repeat 2
            prev_twin = q_twin.copy()
            f_obs = np.concatenate([fq, fqd, rest]).astype(np.float32)
            a = agent.act(f_obs, t0=(t == 0))
            ep_obs.append(f_obs); ep_act.append(a)
            dp = np.clip(a[:3], -1, 1) * EE_DELTA_LIMIT
            width = -0.01 + (np.clip(a[3], -1, 1) + 1) / 2 * 0.05              # panda finger target (m)
            g = np.interp(2 * max(width, 0.0), ap_tab, ang_tab)
            for _ in range(ACTION_REPEAT):
                q = u.agent.robot.get_qpos().numpy()[0]
                target = kin.tcp_world(q) + dp
                q_cmd = kin.solve(q, target)
                ik_err.append(float(np.linalg.norm(kin.tcp_world(q_cmd) - target)))
                q_cmd[5] = g
                obs, r, term, trunc, info = env.step(q_cmd[None].astype(np.float32))
                ok = ok or bool(np.asarray(info["success"]).item())
                o2 = np.asarray(obs).reshape(-1)
                # extra block layout: is_grasped 0, tcp_pose 1:8, goal 8:11, obj_pose 11:18,
                # tcp_to_obj 18:21, obj_to_goal 21:24
                tcp_to_obj = np.linalg.norm(o2[12 + 18:12 + 21])
                grasped = bool(o2[12] > 0.5)
                grasp_steps = grasp_steps + 1 if grasped else 0
                near_goal = np.linalg.norm(o2[12 + 21:12 + 24]) < 2 * u.goal_thresh
                stage = max(stage, 1 if tcp_to_obj < 0.03 else 0)
                stage = max(stage, 2 if (stage >= 1 and grasp_steps >= 2) else 0)
                stage = max(stage, 3 if (stage >= 2 and near_goal and grasped) else 0)
                stage = max(stage, 4 if ok else 0)
                if bool(np.asarray(term).item()) or bool(np.asarray(trunc).item()):
                    break
        rec.append((np.array(ep_obs), np.array(ep_act)))
        results.append({"ep": ep, "success": ok, "stage": stage, "ik_err_mm_median": 1000 * float(np.median(ik_err))})
        print(json.dumps(results[-1]), flush=True)
    k, n = sum(r["success"] for r in results), len(results)
    tag = f"{args.interface}_{args.ik}_{'prior' if args.no_mpc else 'mpc'}{'_noshift' if args.no_shift else ''}{'_rec' if args.record else ''}"
    summary = {"policy": "ms-pick-cube (Franka)", "target": "SO100 (PickCubeSO100-v1)", "interface": args.interface,
               "ik": args.ik, "mpc": not args.no_mpc, "shift": not args.no_shift, "episodes": n, "success": k,
               "rate": k / n, "wilson95": wilson(k, n),
               "stage_hist": {s: sum(r["stage"] == s for r in results) for s in range(5)},
               "mean_stage": float(np.mean([r["stage"] for r in results])),
               "sec_per_episode": (time.time() - t_start) / n,
               "stage_key": "0 none, 1 reach (<3 cm), 2 grasp (2 steps), 3 at goal while grasped, 4 env success"}
    if args.record:
        np.savez(args.record, obs=np.array([r[0] for r in rec], dtype=object), act=np.array([r[1] for r in rec], dtype=object))
    (ROOT / f"results/phase0/route_b_transfer_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
