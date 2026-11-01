"""Train PPO on the classical benchmark task in MJX (xbody/learned/mjx_pick.py).

  GPU run (NVIDIA, MuJoCo Warp):   python scripts/train_mjx_ppo.py --preset gpu --seed 0
  CPU smoke test (pipeline only):  python scripts/train_mjx_ppo.py --preset smoke

Writes checkpoints/learned/mjx_ppo_<preset>_s<seed>/ (params.pkl, config.json, progress.jsonl). Evaluate with
scripts/eval_learned_mjx.py, which scores the policy in the classical CPU scene with the classical seeds.
"""
import argparse
import functools
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PRESETS = {
    # Settings follow MuJoCo Playground's PandaPickCube recipe, scaled for the longer, two-phase task.
    "gpu": dict(impl="warp", num_timesteps=150_000_000, num_evals=10, num_envs=2048, batch_size=512,
                num_minibatches=32, unroll_length=10, num_updates_per_batch=8, learning_rate=1e-3,
                entropy_cost=2e-2, discounting=0.97, reward_scaling=1.0, hidden=(256, 256, 256)),
    "smoke": dict(impl="jax", num_timesteps=4_000, num_evals=2, num_envs=8, batch_size=8,
                  num_minibatches=2, unroll_length=10, num_updates_per_batch=2, learning_rate=1e-3,
                  entropy_cost=2e-2, discounting=0.97, reward_scaling=1.0, hidden=(32, 32)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=None)
    a = ap.parse_args()
    p = dict(PRESETS[a.preset])
    if a.timesteps:
        p["num_timesteps"] = a.timesteps

    from brax.io import model as brax_model
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    from xbody.learned.mjx_pick import FrankaPickPlacePost, default_config

    cfg = default_config()
    cfg.impl = p["impl"]
    env = FrankaPickPlacePost(cfg)
    out = ROOT / f"checkpoints/learned/mjx_ppo_{a.preset}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"preset": a.preset, "seed": a.seed, **{k: list(v) if isinstance(v, tuple) else v for k, v in p.items()},
                                                 "env": cfg.to_dict()}, indent=2))
    log = open(out / "progress.jsonl", "w")
    t0 = time.time()

    def progress(step, metrics):
        rec = {"step": int(step), "time_s": time.time() - t0,
               **{k: float(v) for k, v in metrics.items() if k.startswith("eval/") and getattr(v, "ndim", 0) == 0}}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"step {step:>11,d}  reward {rec.get('eval/episode_reward', float('nan')):9.2f}  "
              f"success {rec.get('eval/episode_success', float('nan')):.3f}  ({rec['time_s']:.0f}s)", flush=True)

    make_inference_fn, params, _ = ppo.train(
        environment=env, wrap_env_fn=wrapper.wrap_for_brax_training, progress_fn=progress, seed=a.seed,
        num_timesteps=p["num_timesteps"], num_evals=p["num_evals"], num_envs=p["num_envs"],
        batch_size=p["batch_size"], num_minibatches=p["num_minibatches"], unroll_length=p["unroll_length"],
        num_updates_per_batch=p["num_updates_per_batch"], learning_rate=p["learning_rate"],
        entropy_cost=p["entropy_cost"], discounting=p["discounting"], reward_scaling=p["reward_scaling"],
        episode_length=cfg.episode_length, normalize_observations=True, action_repeat=1,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=p["hidden"]),
    )
    brax_model.save_params(str(out / "params.pkl"), params)
    print("saved", out / "params.pkl", f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
