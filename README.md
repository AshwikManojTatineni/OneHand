# xbody-bench: zero-training Franka → SO-101 cross-body benchmark

Code for the plan "Zero-Training Cross-Body Benchmark Plan". No network weight is ever updated: every policy is a
frozen public checkpoint, and every adapter is kinematics, calibration or test-time search.

## Layout

| Path | What it is |
| --- | --- |
| `protocol.yaml` | Frozen task, stage thresholds, seeds, gates and statistics (freezes at the end of Phase 0) |
| `registry/checkpoints.yaml` | Every checkpoint in the plan, by route; `scripts/check_registry.py` writes `checkpoints_status.yaml` |
| `xbody/kinematics.py` | Interface K: body-agnostic IK with conversion variants `newton`, `dls`, `hqp` (layers 3 and 7) and the 5-DOF approach-axis projection |
| `xbody/stages.py` | Staged-success scorer shared by every stack |
| `xbody/stats.py` | Wilson intervals, exact McNemar, Wilcoxon, Holm, retention |
| `xbody/newt_agent.py` | Frozen Newt / TD-MPC2 loader for CPU inference (route B, stack S5) |
| `scripts/test_k_interface.py` | Phase 0 IK check on the Menagerie Franka and SO-101 |
| `scripts/eval_route_b_home.py` | Route B Gate 1: each checkpoint on its own body |
| `scripts/eval_route_b_transfer.py` | Route B rung R6: the Franka policy on the SO100 under interfaces K and K+P |
| `scripts/run_route_b_grid.sh` | Route B screening grid (50 episodes per cell) |

## Setup (CPU machine)

```bash
uv venv -p 3.11 .venv-lite && . .venv-lite/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install mani_skill mujoco numpy scipy pyyaml tensordict omegaconf hydra-core termcolor
sudo apt-get install -y mesa-vulkan-drivers   # software Vulkan; ManiSkill needs it even for state-only runs
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
git clone --depth 1 https://github.com/nicklashansen/newt third_party/newt
# third_party/newt_cpu = copy of third_party/newt with two device strings set to "cpu"
#   (tdmpc2/tdmpc2.py and tdmpc2/common/scale.py)
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie assets/mujoco_menagerie
(cd assets/mujoco_menagerie && git sparse-checkout set franka_emika_panda robotstudio_so101)
```

Checkpoints go in `checkpoints/newt/` (hashes in `checkpoints/newt/SHA256SUMS`).

## What needs a GPU

Routes A (LIBERO-X VLAs and world-action models) and C (Isaac Sim, V-JEPA 2-AC, DROID models) need a CUDA GPU;
Isaac Sim needs an RTX card. Route B and all of interface K run on CPU.
