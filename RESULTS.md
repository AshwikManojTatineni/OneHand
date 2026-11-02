# OneHand: cross-body benchmark results (zero training, CPU)

Task: pick a 3 cm cube and place it on a plate behind an obstacle post. Robots: Franka (7 joints) and SO-101
(5 joints); route B also uses the xArm6 and SO100. Every comparison is paired on the same seeded layouts;
statistics are Wilson 95% intervals, exact McNemar, Wilcoxon and Holm correction. Detailed tables live in the
`SUMMARY.md` file of each `results/` folder.

## What was run

| Study | Size | Folder |
| --- | --- | --- |
| Classical decision tree, all 83 methods | 146 cells x 20 episodes (10 for MPC, NMPC, LQG), about 2,860 episodes | `results/classical/` |
| Iterative learning control | 2 robots x 20 layouts x 5 trials | `results/classical/*control-ilc.json` |
| Early classical variants | 7 variants x 2 robots x 50 | `results/phase0/` |
| Route B frozen TD-MPC2 policies | about 850 episodes on 3 bodies | `results/phase0/` |
| Teleop self-model pilots | 2 robots x 90 setups | `results/selfmodel_teleop*/` |
| Self-model and task success | 2 robots x (20 ideal + 5 arms x 20 x 2 models) | `results/selfmodel_task/` |
| Sensor-swap self-modeling | 2 robots x 40 recordings x 400 frames | `results/selfmodel_sensors*/` |

## Findings

### 1. Classical methods (S1)

- Defaults: Franka 15/20, SO-101 17/20.
- Planning, grasping and geometry decide transfer. Control laws barely matter: every competent law scores
  15 to 18/20 and fails on the same episodes.
- Clear body splits:

| Method | Franka | SO-101 | Why |
| --- | --- | --- | --- |
| Full-pose Newton IK | 17/20 | 0/20 | 6-D target on a 5-joint arm |
| CCD IK | 0/20 | 10/20 | Unstable on the redundant arm |
| Pseudoinverse / Jacobian transpose | 14 to 15/20 | 0/20 | SO-101 wrist singularity |
| Wrist-camera servoing (IBVS, 2.5-D) | 17/20 | 1 to 2/20 | SO-101 camera too close to its jaws |
| RRT / RRT* | 0/20 | 14 to 17/20 | Franka's long hand needs tight routes past the post |
| PRM* / LazyPRM | 18 to 19/20 | 17 to 18/20 | Robust on both |
| Voxel / OctoMap / TSDF maps | 4/20 | 2 to 16/20 | Discretisation inflates the post |
| P-only control | 5/20 | 18/20 | Heavy arm sags; servos hide it |

- ILC: tracking error falls 13% on the Franka and stays flat on the SO-101; success does not change.

### 2. Frozen RL policies (route B)

| Cell | Success | Mean stage (of 4) |
| --- | --- | --- |
| Franka home, position-delta policy | 50/50 | 4.00 |
| xArm6, K+P interface | 3/50 (11/50 with planning on) | 1.52 |
| SO100, any interface | 0/50 | 0.4 to 0.9 |
| Franka home, pose-delta policy | 49/50 | 3.92 |
| xArm6, pose-delta policy | 0/50 | 1.58 |
| SO100, pose-delta policy | 0/50 | 0.28 to 0.42 |

- On the SO100 the pose-action policy gets less far than the position policy (Wilcoxon p = 6e-6 and 0.001).
- Frame alignment matters (stage 0.86 vs 0.42, p = 1.6e-5); the virtual twin helps only with planning on the xArm6.
- The frozen world model's latent self-prediction error grows from 0.16 at home to 52 to 124 on the SO100.

### 3. Self-model from teleoperation

The arm is teleoperated through its model file, logging encoder readings and the measured pose of a frame on
the hand. A product-of-exponentials model is fitted to the pose changes, then refined on absolute poses. The
simulated arm carries sampled calibration errors. 400 poses, 1 mm / 0.3 deg noise, 40 reach targets per arm
(success within 5 mm and 2 deg).

| Robot | Model | Hand error | Reach |
| --- | --- | --- | --- |
| Franka | Model file | 11.4 mm | 100/400 |
| Franka | Self-model, changes only | 20 to 36 mm | 86/400 |
| Franka | Self-model, changes + absolute | 1.9 to 2.1 mm | 400/400 |
| SO-101 | Model file | 12.2 mm | 0/400 |
| SO-101 | Self-model, changes + absolute | 0.9 to 1.1 mm | 400/400 |

- Fitting changes alone counts sensor noise twice; the absolute refinement fixes it.
- Data needed: about 200 poses (joint teleop) to 400 (Cartesian). Below 100 the fit fails.
- Still 5 to 7 mm at 3 mm / 1 deg noise.
- One-step relative error does not separate good from bad models; use FK error and reach.

### 4. Self-model and task success

Classical default pipeline on mis-calibrated arms, planning with the model file or the calibrated one:

| Robot | Arm matches model | Model file | Self-model |
| --- | --- | --- | --- |
| Franka | 15/20 | 55/100 | 81/100 |
| SO-101 | 17/20 | 56/100 | 85/100 |

Paired McNemar p < 1e-4 on both robots. Calibration restores ideal-level success; one arm per robot went from
0/20 to 17/20.

### 5. Sensor-swap self-modeling

Rendered lab (ChArUco board, clutter, walls). The hand pose comes from a real perception pipeline: mono PnP,
stereo triangulation, or a wide-field lidar with ICP against a room map. Realism ladder: R0 ideal; R1 image
noise, lens and calibration error, lidar noise; R2 adds motion blur and 15 ms latency; R3 adds joint
compliance. Self-model reach out of 200 (the model file reaches 0/40 per arm throughout):

| Robot | Teleop | Sensor | R0 | R1 | R2 | R3 |
| --- | --- | --- | --- | --- | --- | --- |
| Franka | Cartesian | Mono | 200 | 200 | 184 | 183 |
| Franka | Cartesian | Stereo | 200 | 125 | 106 | 99 |
| Franka | Cartesian | Lidar | 199 | 200 | 197 | 196 |
| Franka | Joint | Mono | 199 | 185 | 108 | 104 |
| Franka | Joint | Stereo | 191 | 128 | 86 | 94 |
| SO-101 | Cartesian | Mono | 200 | 160 | 200 | 199 |
| SO-101 | Cartesian | Stereo | 200 | 200 | 197 | 199 |
| SO-101 | Cartesian | Lidar | 200 | 200 | 198 | 197 |
| SO-101 | Joint | Mono | 200 | 200 | 155 | 192 |
| SO-101 | Joint | Stereo | 80 | 120 | 79 | 75 |
| SO-101 | Joint | Lidar | 200 | 196 | 199 | 196 |

The Franka joint-teleop lidar row is in `results/selfmodel_sensors/SUMMARY.md`.

- Lidar is the most robust on both robots at every level (a Mid-360-class unit may be heavy for the SO-101).
- Mono degrades with blur and latency when joint teleop takes the board out of view.
- Narrow-baseline stereo is the weakest.

## Parameters varied

- Robots; classical layers (input, world model, IK, planner, timing, tracking, conversion, 17 control laws, actuator).
- Transfer interfaces (K, P, IK type, frame alignment, gripper map, planning, position vs pose actions).
- Self-model fit method, teleop style, 25 to 800 poses, 4 noise levels, calibration error sizes.
- Sensor (oracle, mono, stereo, lidar) and realism level R0 to R3.

## Not done here

- **GPU work:** route A (9 LIBERO VLA and world-model checkpoints), route C (4 Isaac / DROID checkpoints), and
  all training. MJX and Isaac Lab training routes are written and untested; see `isaac/README.md`.
- **Instant Policy:** ships as a closed compiled binary; running it needs explicit approval. Noncommercial license.
