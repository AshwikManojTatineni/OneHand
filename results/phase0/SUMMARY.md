# Phase 0 summary (CPU-only comparisons)

All policies frozen; every comparison is paired on the same seeds and layouts.

## Route B: frozen Franka TD-MPC2 PickCube policy

Stages: 0 none, 1 reach, 2 grasp, 3 at goal while grasped, 4 success.

| Cell | Success (Wilson 95%) | Mean stage | Retention | Stage retention |
| --- | --- | --- | --- | --- |
| R0 Franka home (prior) | 50/50 (93-100%) | 4.00 | 1.00 | 1.00 |
| R3 xArm6 K (prior) | 3/50 (2-16%) | 1.84 | 0.06 | 0.46 |
| R3 xArm6 K+P (prior) | 3/50 (2-16%) | 1.52 | 0.06 | 0.38 |
| R3 xArm6 K+P naive gripper (prior) | 1/50 (0-10%) | 0.84 | 0.02 | 0.21 |
| R6 SO100 K dls (prior) | 0/50 (0-7%) | 0.94 | 0.00 | 0.23 |
| R6 SO100 K+P dls (prior) | 0/50 (0-7%) | 0.86 | 0.00 | 0.21 |
| R6 SO100 K hqp-axis (prior) | 0/50 (0-7%) | 0.90 | 0.00 | 0.23 |
| R6 SO100 K+P hqp-axis (prior) | 0/50 (0-7%) | 0.78 | 0.00 | 0.20 |
| R6 SO100 K+P dls no frame shift (prior) | 0/50 (0-7%) | 0.42 | 0.00 | 0.10 |
| R6 SO100 K dls (planning on) | 0/20 (0-16%) | 0.00 | 0.00 | 0.00 |
| R6 SO100 K+P dls (planning on) | 0/20 (0-16%) | 0.30 | 0.00 | 0.07 |
| R3 xArm6 K (planning on) | 2/50 (1-13%) | 1.60 | 0.04 | 0.40 |
| R3 xArm6 K+P (planning on) | 11/50 (13-35%) | 1.86 | 0.22 | 0.47 |
| R3 xArm6 K+P naive gripper (planning on) | 0/50 (0-7%) | 0.88 | 0.00 | 0.22 |

Tests (exact McNemar on success and Wilcoxon on stage reached for paired rows; Fisher and Mann-Whitney for the unpaired row; Holm-adjusted stage p):

| Comparison | Paired episodes | Success | Mean stage | McNemar p | Wilcoxon p (Holm) |
| --- | --- | --- | --- | --- | --- |
| P (virtual twin), R6 | 50 | 0.00 vs 0.00 | 0.94 vs 0.86 | 1 | 0.102 (0.615) |
| IK conversion dls vs hqp-axis, R6 | 50 | 0.00 vs 0.00 | 0.86 vs 0.78 | 1 | 0.102 (0.615) |
| Frame alignment, R6 | 50 | 0.00 vs 0.00 | 0.86 vs 0.42 | 1 | 1.6e-05 (0.00016) |
| P (virtual twin), R3 | 50 | 0.06 vs 0.06 | 1.84 vs 1.52 | 1 | 0.153 (0.615) |
| Gripper map in K, R3 | 50 | 0.06 vs 0.02 | 1.52 vs 0.84 | 0.625 | 0.000419 (0.00335) |
| Rung R3 vs R6 (different envs: unpaired tests) | 50 | 0.06 vs 0.00 | 1.52 vs 0.86 | 0.242 | 2.83e-05 (0.000255) |
| Planning on vs off, R6 (first 20 seeds) | 20 | 0.00 vs 0.00 | 0.90 vs 0.30 | 1 | 0.00134 (0.00938) |
| Planning on vs off, R3 | 50 | 0.06 vs 0.04 | 1.84 vs 1.60 | 1 | 0.159 (0.615) |
| Planning on vs off, R3 with P | 50 | 0.06 vs 0.22 | 1.52 vs 1.86 | 0.0386 | 0.116 (0.615) |
| P (virtual twin), R3 with planning | 50 | 0.04 vs 0.22 | 1.60 vs 1.86 | 0.00391 | 0.185 (0.615) |

## S1 classical variants (MuJoCo, Menagerie Franka and SO-101)

Stages: 0 none, 1 reach, 2 grasp, 3 lift, 4 transport, 5 place. Retention = SO-101 / Franka.

| Variant | Franka | SO-101 | Retention | SO-101 vs default: McNemar p | SO-101 IK misses per episode |
| --- | --- | --- | --- | --- | --- |
| ik dls-full | 50/50 | 40/50 | 0.80 | 0.00195 | 5.1 |
| tracking cartesian | 50/50 | 47/50 | 0.94 | 0.25 | 0.0 |
| timing quintic | 50/50 | 50/50 | 1.00 | 1 | 0.0 |
| servo stiffness x0.017857 | 0/50 | 50/50 | n/a | 1 | 0.0 |
| default | 50/50 | 50/50 | 1.00 |  | 0.0 |
| ik hqp-full | 50/50 | 50/50 | 1.00 | 1 | 0.0 |
| ik newton-full | 50/50 | 0/50 | 0.00 | 1.78e-15 | 6.0 |

## Self-model readout vs transfer (route B, training-free)

Frozen Franka world model, one-step latent prediction on each body. Relative error divides by a no-change baseline; lower is a better self-model.

| Body / interface | Relative error | Self-prediction ratio | Mean stage reached |
| --- | --- | --- | --- |
| franka_home | 0.157 | 1.0 | 4.00 |
| xarm6_R3_KP | 0.865 | 22.7 | 1.52 |
| so100_R6_KP | 52.325 | 23.7 | 0.86 |
| so100_R6_K | 124.145 | 26.8 | 0.94 |
| so100_R6_KP_noshift | 1.760 | 10.9 | 0.42 |

Spearman correlation between relative error and mean stage: rho = -0.60 (p = 0.28, n = 5 cells; too few points for a firm claim).
