"""Train or evaluate the Isaac Lab port of the benchmark task.

  Train:    ./isaaclab.sh -p isaac/scripts/run.py train --num_envs 4096 --headless
  Evaluate: ./isaaclab.sh -p isaac/scripts/run.py eval --checkpoint logs/rsl_rl/.../model_1499.pt --headless

Evaluation runs one environment per classical seed (results/learned/layouts_franka.json), scores each with the
protocol.yaml stage thresholds (reach, grasp held 0.5 s, lift 5 cm for 1 s, over the plate, released at rest 0.5 s)
using finger-cube contact sensors for "both jaws touch", and writes results/learned/franka__isaac_ppo_<run>.json
in the classical result format.
"""
import argparse
import json
import pathlib
import sys

from isaaclab.app import AppLauncher

ROOT = pathlib.Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["train", "eval"])
ap.add_argument("--num_envs", type=int, default=4096)
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--episodes", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, str(ROOT / "isaac"))
import onehand_isaac  # noqa: E402,F401
from onehand_isaac.agents.rsl_rl_ppo_cfg import PPORunnerCfg  # noqa: E402
from onehand_isaac.pick_place_post_env import LAYOUT, TABLE_Z, PickPlacePostEnvCfg  # noqa: E402


def make_env(num_envs, contacts=False):
    cfg = PickPlacePostEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = args.seed
    if contacts:
        cfg.robot.spawn.activate_contact_sensors = True
    env = gym.make("OneHand-Franka-PickPlacePost-v0", cfg=cfg)
    return env, cfg


def train():
    env, _ = make_env(args.num_envs)
    agent = PPORunnerCfg()
    agent.seed = args.seed
    log_dir = ROOT / "logs/rsl_rl" / agent.experiment_name / f"s{args.seed}"
    runner = OnPolicyRunner(RslRlVecEnvWrapper(env), agent.to_dict(), log_dir=str(log_dir), device="cuda:0")
    runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
    env.close()


def evaluate():
    proto = yaml.safe_load(open(ROOT / "protocol.yaml"))["stages"]
    n = args.episodes
    env, cfg = make_env(n, contacts=True)
    base = env.unwrapped
    sensors = []
    for side in ("leftfinger", "rightfinger"):
        s = ContactSensor(ContactSensorCfg(prim_path=f"/World/envs/env_.*/Robot/panda_{side}",
                                           filter_prim_paths_expr=["/World/envs/env_.*/Cube"]))
        base.scene.sensors[side] = s
        sensors.append(s)
    base.eval_layouts = torch.tensor(LAYOUT["cube_xy"][:n], device=base.device)
    wrapped = RslRlVecEnvWrapper(env)
    agent = PPORunnerCfg()
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=base.device)
    obs, _ = wrapped.reset()
    dev, dt = base.device, base.step_dt
    stage = torch.zeros(n, dtype=torch.long, device=dev)
    held_t, lift_t, rest_t = (torch.zeros(n, device=dev) for _ in range(3))
    post_hit = torch.zeros(n, dtype=torch.bool, device=dev)
    plate = torch.tensor(LAYOUT["plate_xy"], device=dev)
    for _ in range(int(cfg.episode_length_s / dt) - 1):
        with torch.inference_mode():
            obs, _, _, _ = wrapped.step(policy(obs))
        tcp, _ = base._tcp()
        cube, _ = base._cube()
        touch = torch.stack([(s.data.force_matrix_w.norm(dim=-1).sum(dim=(1, 2)) > 0.05) for s in sensors], -1).all(-1)
        jaws_open = base.targets[:, base.finger_ids].mean(-1) >= 0.9 * 0.04
        lifted = cube[:, 2] - TABLE_Z >= proto["3_lift"]["height_m"]
        over = torch.linalg.norm(cube[:, :2] - plate, dim=-1) <= LAYOUT["plate_r"]
        speed = torch.linalg.norm(base.cube.data.root_lin_vel_w, dim=-1)
        stage = torch.where((stage == 0) & (torch.linalg.norm(tcp - cube, dim=-1) <= proto["1_reach"]["gripper_to_object_m"]), 1, stage)
        held_t = torch.where(touch & (stage >= 1), held_t + dt, 0)
        stage = torch.where((stage == 1) & (held_t >= proto["2_grasp"]["held_s"]), 2, stage)
        lift_t = torch.where(lifted & touch & (stage >= 2), lift_t + dt, 0)
        stage = torch.where((stage == 2) & (lift_t >= proto["3_lift"]["hold_s"]), 3, stage)
        stage = torch.where((stage == 3) & over & lifted, 4, stage)
        rest_t = torch.where((stage == 4) & over & (speed < 0.01) & jaws_open & ~touch, rest_t + dt, 0)
        stage = torch.where((stage == 4) & (rest_t >= proto["5_place"]["at_rest_s"]), 5, stage)
        hand = base.robot.data.body_pos_w[:, base.hand_idx] - base.scene.env_origins
        near = torch.stack([tcp, hand, cube], 1)
        post_hit |= ((torch.linalg.norm(near[..., :2] - base.post_xy, dim=-1) < LAYOUT["post_r"] + 0.005)
                     & (near[..., 2] < LAYOUT["post_h"])).any(-1)
    st = stage.cpu().tolist()
    succ = sum(s == 5 for s in st)
    out = {"body": "franka", "method": f"isaac_ppo:{pathlib.Path(args.checkpoint).parent.name}", "episodes": n,
           "success": succ, "rate": succ / n, "stages": st, "mean_stage": sum(st) / n,
           "post_contact_rate": float(post_hit.float().mean()), "not_applicable": "",
           "note": "post contact is geometric (within 5 mm of the post) in Isaac; MuJoCo uses contacts"}
    res = ROOT / "results/learned"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"franka__isaac_ppo_{pathlib.Path(args.checkpoint).parent.name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "stages"}))
    env.close()


if __name__ == "__main__":
    train() if args.mode == "train" else evaluate()
    app.close()
