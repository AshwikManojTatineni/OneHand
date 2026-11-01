"""Score an MJX-trained policy in the classical benchmark harness.

Physics: the CPU MuJoCo scene from xbody.scene.build_task (the cylinder post, Menagerie meshes), stepped with
mj_step. Layouts: the classical seeds (20261028 + episode, same region as scripts/eval_classical.py). Scoring:
xbody.stages.StageScorer with protocol.yaml thresholds, and the same post-contact metric. Observations are
computed by the training environment's own _obs on the live MuJoCo state, so they cannot drift from training.

Usage: python scripts/eval_learned_mjx.py checkpoints/learned/mjx_ppo_gpu_s0 [--episodes 20]
Writes results/learned/franka__mjx_ppo_<run>.json in the classical result format.
"""
import argparse
import json
import pathlib
import sys

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody.bodies import SPECS  # noqa: E402
from xbody.scene import build_task, sample_layout  # noqa: E402
from xbody.stages import StageScorer, StepState  # noqa: E402
from xbody.stats import wilson  # noqa: E402


def load_policy(run, env):
    from brax.io import model as brax_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    cfg = json.loads((run / "config.json").read_text())
    params = brax_model.load_params(str(run / "params.pkl"))
    obs_size = int(env.observation_size)
    nets = ppo_networks.make_ppo_networks(obs_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize,
                                          policy_hidden_layer_sizes=tuple(cfg["hidden"]))
    return jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    from mujoco import mjx

    from xbody.learned.mjx_pick import FrankaPickPlacePost, default_config
    env = FrankaPickPlacePost(default_config())          # impl="jax": used for ids and the observation function
    policy = load_policy(run, env)
    obs_fn = jax.jit(env._obs)
    reward_fn = jax.jit(env._reward)   # only to carry the episode flags (e.g. "lifted") exactly as in training
    proto = yaml.safe_load(open(ROOT / "protocol.yaml"))
    spec = SPECS["franka"]
    m = build_task("franka")
    m.opt.timestep = env.sim_dt
    d = mujoco.MjData(m)
    cube_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    post = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "post")
    rb = {b for b in range(m.nbody) if b not in (0, env._cube)}
    robot_geoms = set(np.where(np.isin(m.geom_bodyid, list(rb)))[0]) - {post}
    jaws = [set(np.where(m.geom_bodyid == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b))[0]) for b in spec["jaw_bodies"]]
    stages, post_hits = [], []
    for ep in range(a.episodes):
        seed = 20261028 + ep
        cube_xy = sample_layout(np.random.default_rng(seed), "franka", region_y=(-0.08, 0.06))
        mujoco.mj_resetData(m, d)
        d.qpos[:] = np.asarray(env._q0)
        d.qpos[env._cube_q:env._cube_q + 3] = [cube_xy[0], cube_xy[1], 0.015]
        d.qpos[env._cube_q + 3:env._cube_q + 7] = [1, 0, 0, 0]
        d.ctrl[:] = np.asarray(env._ctrl0)
        mujoco.mj_forward(m, d)
        scorer = StageScorer(proto["stages"])
        info = {"reached": jp.zeros(()), "lifted": jp.zeros(()), "placed": jp.zeros(()), "last_act": jp.zeros(8)}
        hit, t = False, 0.0
        rng = jax.random.PRNGKey(ep)
        for k in range(env._config.episode_length):
            mx = mjx.put_data(env.mj_model, d)
            obs = obs_fn(mx, info)
            act, _ = policy(obs, rng)
            act = np.clip(np.asarray(act), -1, 1)
            arm = np.clip(d.ctrl[env._arm_act] + act[:7] * env._config.action_scale,
                          m.actuator_ctrlrange[env._arm_act, 0], m.actuator_ctrlrange[env._arm_act, 1])
            d.ctrl[env._arm_act] = arm
            d.ctrl[env._grip_act] = (act[7] + 1) / 2 * spec["gripper_open"]
            for _ in range(env.n_substeps):
                mujoco.mj_step(m, d)
                t += m.opt.timestep
            touch = [False, False]
            for i in range(d.ncon):
                c = d.contact[i]
                pair = {c.geom1, c.geom2}
                if post in pair and (pair & robot_geoms or cube_geom in pair):
                    hit = True
                if cube_geom in pair:
                    other = (pair - {cube_geom}).pop() if len(pair) == 2 else -1
                    touch = [touch[j] or other in jaws[j] for j in (0, 1)]
            R = d.xmat[env._hand].reshape(3, 3)
            tcp = d.xpos[env._hand] + R @ np.asarray(env._tcp_off)
            cube = d.xpos[env._cube].copy()
            _, info = reward_fn(mjx.put_data(env.mj_model, d), info, jp.asarray(act))
            scorer.update(StepState(t=t, tcp=tcp, obj=cube, obj_vel=float(np.linalg.norm(d.cvel[env._cube][3:])),
                                    both_jaws_touch=all(touch), jaws_open=bool(d.ctrl[env._grip_act] >= 0.9 * spec["gripper_open"]),
                                    table_z=0.015, container_xy=np.asarray(env._plate), container_r=0.05))
        stages.append(scorer.stage)
        post_hits.append(hit)
        print(f"episode {ep:2d}: stage {scorer.stage}  post contact {hit}", flush=True)
    succ = int(sum(s == 5 for s in stages))
    lo, hi = wilson(succ, len(stages))
    out = {"body": "franka", "method": f"mjx_ppo:{run.name}", "episodes": len(stages), "success": succ,
           "rate": succ / len(stages), "ci": [lo, hi], "stages": stages, "mean_stage": float(np.mean(stages)),
           "post_contact_rate": float(np.mean(post_hits)), "not_applicable": ""}
    res = ROOT / "results/learned"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"franka__mjx_ppo_{run.name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "stages"}))


if __name__ == "__main__":
    main()
