"""Route B rung R3: frozen Franka TD-MPC2 PickCube policy on the xArm6 (6-DOF, Robotiq gripper), no training.

Both bodies use ManiSkill's pd_ee_delta_pos arm controller on the same PickCube config, so the arm side of K is
the stock controller IK. The gripper side differs: the Franka command is an absolute finger width (+1 = open);
the xArm6 command is a normalized knuckle delta (+ = close). K maps width -> knuckle target -> delta;
--naive-gripper passes the Franka value through unchanged (an ablation that drives the gripper backwards).

Usage: python scripts/eval_route_b_r3.py <K|K+P> <episodes> [--no-mpc] [--naive-gripper]
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
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_route_b_transfer import FRANKA_BASE, FRANKA_REST, R_DOWN, franka_twin  # noqa: E402
from xbody.kinematics import solve_ik  # noqa: E402
from xbody.newt_agent import FrozenNewtAgent  # noqa: E402
from xbody.stats import wilson  # noqa: E402

ACTION_REPEAT, EP_LEN = 2, 25
KNUCKLE_CLOSED, KNUCKLE_DELTA = 0.81, 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("interface", choices=["K", "K+P"])
    ap.add_argument("episodes", type=int)
    ap.add_argument("--no-mpc", action="store_true")
    ap.add_argument("--naive-gripper", action="store_true")
    ap.add_argument("--seed-base", type=int, default=20261028)
    ap.add_argument("--record", default=None, help="save policy-view observations and actions (.npz)")
    args = ap.parse_args()

    env = gym.make("PickCube-v1", robot_uids="xarm6_robotiq", obs_mode="state", control_mode="pd_ee_delta_pos",
                   num_envs=1, sim_backend="cpu")
    agent = FrozenNewtAgent(ROOT / "checkpoints/newt/ms-pick-cube.pt", "ms-pick-cube", 4, EP_LEN, mpc=not args.no_mpc)
    twin = franka_twin()
    results, t0, rec = [], time.time(), []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed_base + ep)
        u = env.unwrapped
        q_twin, prev = FRANKA_REST.copy(), FRANKA_REST.copy()
        stage, grasp_steps, ok = 0, 0, False
        ep_obs, ep_act = [], []
        for t in range(EP_LEN):
            o = np.asarray(obs).reshape(-1)
            qpos, rest = o[:12], o[24:]
            knuckle = float(qpos[6])
            finger = np.clip((1 - knuckle / KNUCKLE_CLOSED) * 0.04, 0.0, 0.04)
            if args.interface == "K+P":
                q_twin, _ = solve_ik(twin, rest[1:4] - FRANKA_BASE, R_DOWN, q0=q_twin, method="hqp", iters=60)
            fq = np.concatenate([q_twin, [finger, finger]])
            fqd = np.concatenate([(q_twin - prev) * 10.0, [0.0, 0.0]])
            prev = q_twin.copy()
            f_obs = np.concatenate([fq, fqd, rest]).astype(np.float32)
            a = agent.act(f_obs, t0=(t == 0))
            ep_obs.append(f_obs); ep_act.append(a)
            for _ in range(ACTION_REPEAT):
                o = np.asarray(obs).reshape(-1)
                if args.naive_gripper:
                    g = float(np.clip(a[3], -1, 1))
                else:
                    width = -0.01 + (np.clip(a[3], -1, 1) + 1) / 2 * 0.05
                    target = (1 - np.clip(width / 0.04, 0, 1)) * KNUCKLE_CLOSED
                    g = float(np.clip((target - o[6]) / KNUCKLE_DELTA, -1, 1))
                act = np.concatenate([np.clip(a[:3], -1, 1), [g]]).astype(np.float32)
                obs, r, term, trunc, info = env.step(act[None])
                ok = ok or bool(np.asarray(info["success"]).item())
                o2 = np.asarray(obs).reshape(-1)[24:]
                grasped = bool(o2[0] > 0.5)
                grasp_steps = grasp_steps + 1 if grasped else 0
                stage = max(stage, 1 if np.linalg.norm(o2[18:21]) < 0.03 else 0)
                stage = max(stage, 2 if (stage >= 1 and grasp_steps >= 2) else 0)
                stage = max(stage, 3 if (stage >= 2 and grasped and np.linalg.norm(o2[21:24]) < 2 * u.goal_thresh) else 0)
                stage = max(stage, 4 if ok else 0)
                if bool(np.asarray(term).item()) or bool(np.asarray(trunc).item()):
                    break
        rec.append((np.array(ep_obs), np.array(ep_act)))
        results.append({"ep": ep, "success": ok, "stage": stage})
        print(json.dumps(results[-1]), flush=True)
    k, n = sum(r["success"] for r in results), len(results)
    tag = f"{args.interface}_{'prior' if args.no_mpc else 'mpc'}{'_naivegrip' if args.naive_gripper else ''}{'_rec' if args.record else ''}"
    summary = {"policy": "ms-pick-cube (Franka)", "target": "xArm6 (PickCube-v1)", "rung": "R3",
               "interface": args.interface, "mpc": not args.no_mpc, "naive_gripper": args.naive_gripper,
               "episodes": n, "success": k, "rate": k / n, "wilson95": wilson(k, n),
               "stage_hist": {s: sum(r["stage"] == s for r in results) for s in range(5)},
               "mean_stage": float(np.mean([r["stage"] for r in results])),
               "sec_per_episode": (time.time() - t0) / n,
               "stage_key": "0 none, 1 reach (<3 cm), 2 grasp (2 steps), 3 at goal while grasped, 4 env success"}
    if args.record:
        np.savez(args.record, obs=np.array([r[0] for r in rec], dtype=object), act=np.array([r[1] for r in rec], dtype=object))
    (ROOT / f"results/phase0/route_b_r3_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
