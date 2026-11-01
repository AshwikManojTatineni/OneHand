# GPU training: MJX and Isaac Lab routes

Both routes train on the classical benchmark task (Franka, 3 cm cube, plate behind the post) and are scored on the
same 200 seeded layouts with the same stage thresholds as the classical methods.

| Route | Simulator | Scene source | Status |
| --- | --- | --- | --- |
| MJX | MuJoCo (MJX / MuJoCo Warp), same physics as the classical runs | `xbody.scene.build_task("franka")` | Environment and training pipeline smoke-tested on CPU |
| Isaac Lab | Isaac Sim (PhysX, RTX rendering) | Port, numbers from `results/learned/layouts_franka.json` | Written, not yet run (needs an RTX GPU) |

## Machine

- NVIDIA GPU with RT cores for Isaac Sim: L40S, A10G, L4, RTX 6000 Ada or RTX 4090. A100 / H100 work for MJX but
  NVIDIA lists them as unsupported for Isaac Sim.
- Ubuntu 22.04, a recent NVIDIA driver, Docker with the NVIDIA container toolkit.
- To let Claude drive the machine: install Claude Code there and run `claude remote-control` in the repo folder,
  or open the folder in the Claude Desktop app. The session then shows up in the Claude Code app.

## MJX route

```bash
git clone https://github.com/AshwikManojTatineni/OneHand && cd OneHand
uv venv .venv-mjx --python 3.13 && uv pip install --python .venv-mjx/bin/python playground "jax[cuda12]" mujoco-warp
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie assets/mujoco_menagerie
for s in 0 1 2; do .venv-mjx/bin/python scripts/train_mjx_ppo.py --preset gpu --seed $s; done
for s in 0 1 2; do .venv-mjx/bin/python scripts/eval_learned_mjx.py checkpoints/learned/mjx_ppo_gpu_s$s --episodes 200; done
```

Training uses `impl="warp"`, which keeps the classical cylinder post and every arm-link collision mesh. The CPU
smoke test (`--preset smoke`) swaps the post for a same-size capsule and keeps collisions on the hand and fingers
only, because the JAX backend lacks that collider and runs out of memory compiling the rest. Its numbers mean nothing.

## Isaac Lab route

```bash
docker run --gpus all -it --rm -v $PWD:/workspace/OneHand nvcr.io/nvidia/isaac-lab:<current 2.x tag> bash
cd /workspace/OneHand
/workspace/isaaclab/isaaclab.sh -p isaac/scripts/run.py train --num_envs 4096 --headless --seed 0
/workspace/isaaclab/isaaclab.sh -p isaac/scripts/run.py eval --checkpoint logs/rsl_rl/onehand_franka_pick_place_post/s0/model_1499.pt --headless
```

Expect a few API fixes on first launch: the port targets Isaac Lab 2.x and has not been run. The Franka asset is
Isaac Lab's own Panda USD (same robot as Menagerie; collision shapes differ), so the classical numbers and the
Isaac numbers differ by simulator as well as by method. `cameras=True` in the env config adds a wrist RGB camera
for VLA and world-model data.

## Results

Both evaluators write `results/learned/franka__<route>_<run>.json` in the classical result format, next to
`results/classical/`.
