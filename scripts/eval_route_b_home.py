"""Route B Gate 1: home success of each frozen Newt PickCube checkpoint on its own body (rung R0 / native).

Usage: python scripts/eval_route_b_home.py <task> <episodes> [--no-mpc]
"""
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
from xbody.newt_agent import FrozenNewtAgent  # noqa: E402
from xbody.stats import wilson  # noqa: E402

TASKS = {  # mirrors Newt's MANISKILL_TASKS
    "ms-pick-cube": ("PickCube-v1", "pd_ee_delta_pos", 4, {}),
    "ms-pick-cube-eepose": ("PickCube-v1", "pd_ee_delta_pose", 7, {}),
    "ms-pick-cube-so": ("PickCubeSO100-v1", "pd_joint_delta_pos", 6, {}),
    "ms-pick-cube-xarm6": ("PickCube-v1", "pd_ee_delta_pos", 4, {"robot_uids": "xarm6_robotiq"}),
}
ACTION_REPEAT, EP_LEN = 2, 25


def main(task, episodes, mpc=True, seed_base=20261028):
    env_id, cm, adim, kw = TASKS[task]
    env = gym.make(env_id, obs_mode="state", control_mode=cm, num_envs=1, sim_backend="cpu", **kw)
    agent = FrozenNewtAgent(ROOT / f"checkpoints/newt/{task}.pt", task, adim, EP_LEN, mpc=mpc)
    succ, t_start = [], time.time()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        ok = False
        for t in range(EP_LEN):
            a = agent.act(np.asarray(obs).reshape(-1), t0=(t == 0))
            for _ in range(ACTION_REPEAT):
                obs, r, term, trunc, info = env.step(a[None])
                ok = ok or bool(np.asarray(info["success"]).item())
                if bool(np.asarray(term).item()) or bool(np.asarray(trunc).item()):
                    break
        succ.append(ok)
        print(f"{task} ep {ep:3d} success={ok}", flush=True)
    k, n = int(sum(succ)), len(succ)
    lo, hi = wilson(k, n)
    res = {"task": task, "mpc": mpc, "episodes": n, "success": k, "rate": k / n, "wilson95": [lo, hi],
           "sec_per_episode": (time.time() - t_start) / n, "seed_base": seed_base}
    out = ROOT / f"results/phase0/route_b_home_{task}_{'mpc' if mpc else 'prior'}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), mpc="--no-mpc" not in sys.argv)
