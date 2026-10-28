# Full classical benchmark (S1): every decision-tree method on the Franka and the SO-101

One-layer swaps from the default pipeline; same seeded layouts on both bodies. Success = the protocol's
stage 5 (cube resting on the plate, jaws open). Retention = SO-101 success / Franka success.

- Default on franka: 15/20 (53-89%), resolved config `{'input': 'pointcloud', 'world': 'primitives', 'endpoint': 'constrained', 'planner': 'rrtconnect', 'timing': 'toppra', 'tracking': 'joint', 'conversion': 'hqp', 'control': 'pdg', 'actuator': 'torque'}`
- Default on so101: 17/20 (64-95%), resolved config `{'input': 'pointcloud', 'world': 'primitives', 'endpoint': 'constrained', 'planner': 'rrtconnect', 'timing': 'toppra', 'tracking': 'joint', 'conversion': 'hqp', 'control': 'pd', 'actuator': 'cascade'}`

## 1 Input source

| Method | Franka | SO-101 | Retention | Mean stage F / S | Post contact F / S | vs default p (F / S) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| camera | 1/20 | 0/20 | 0.00 | 2.8 / 0.9 | 1.00 / 1.00 | 0.00012 / 1.5e-05 |  |
| known | 18/20 | 18/20 | 1.00 | 4.8 / 4.8 | 0.00 / 0.50 | 0.25 / 1 |  |

## 2 World model

| Method | Franka | SO-101 | Retention | Mean stage F / S | Post contact F / S | vs default p (F / S) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| esdf | 17/20 | 18/20 | 1.06 | 4.7 / 4.5 | 0.00 / 0.30 | 0.5 / 1 |  |
| mesh | 17/20 | 19/20 | 1.12 | 4.7 / 4.9 | 0.00 / 1.00 | 0.5 / 0.62 |  |
| octomap | 4/20 | 2/20 | 0.50 | 2.5 / 1.6 | 0.00 / 0.05 | 0.00098 / 6.1e-05 |  |
| pose | 15/20 | 20/20 | 1.33 | 4.5 / 5.0 | 0.00 / 0.50 | 1 / 0.25 |  |
| tsdf | 4/20 | 2/20 | 0.50 | 2.0 / 1.6 | 0.00 / 0.05 | 0.00098 / 6.1e-05 |  |
| voxel | 4/20 | 16/20 | 4.00 | 2.2 / 4.6 | 0.00 / 0.25 | 0.00098 / 1 |  |

## 3 Endpoint IK

| Method | Franka | SO-101 | Retention | Mean stage F / S | Post contact F / S | vs default p (F / S) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| analytic | n/a | 14/20 | - | 0.0 / 4.0 | 0.00 / 0.00 | 6.1e-05 / 0.38 | analytic IK: no closed form for this arm |
| ccd | 0/20 | 10/20 | - | 1.3 / 2.8 | 0.05 / 0.05 | 6.1e-05 / 0.016 |  |
| lm | 18/20 | 11/20 | 0.61 | 4.8 / 3.9 | 0.00 / 0.15 | 0.25 / 0.031 |  |
| newton | 17/20 | 0/20 | 0.00 | 4.7 / 1.1 | 0.00 / 0.00 | 0.62 / 1.5e-05 |  |
| taught | 4/20 | 2/20 | 0.50 | 1.2 / 0.8 | 0.00 / 1.00 | 0.00098 / 6.1e-05 |  |

## 4 Motion planning

| Method | Franka | SO-101 | Retention | Mean stage F / S | Post contact F / S | vs default p (F / S) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| astar | 0/20 | 0/20 | - | 0.0 / 0.0 | 0.00 / 0.00 | 6.1e-05 / 1.5e-05 |  |
| chomp | 15/20 | 18/20 | 1.20 | 4.5 / 4.5 | 0.25 / 0.40 | 1 / 1 |  |
| dampers | 0/20 | 0/20 | - | 3.0 / 3.0 | 0.00 / 0.00 | 6.1e-05 / 1.5e-05 |  |
| dijkstra | 0/20 | 0/20 | - | 0.0 / 0.0 | 0.00 / 0.00 | 6.1e-05 / 1.5e-05 |  |
| est | 16/20 | 7/20 | 0.44 | 4.2 / 3.0 | 0.00 / 0.15 | 1 / 0.002 |  |
| informedrrtstar | 0/20 | 12/20 | - | 3.0 / 3.9 | 0.00 / 0.15 | 6.1e-05 / 0.12 |  |
| kinorrt | 11/20 | 8/20 | 0.73 | 3.2 / 2.6 | 0.00 / 0.10 | 0.34 / 0.0039 |  |
| kpiece | 15/20 | 13/20 | 0.87 | 4.5 / 4.0 | 0.00 / 0.40 | 1 / 0.12 |  |
| lattice | 10/20 | 0/20 | 0.00 | 4.0 / 2.7 | 0.00 / 0.00 | 0.18 / 1.5e-05 |  |
| lazyprm | 19/20 | 18/20 | 0.95 | 4.9 / 4.5 | 0.00 / 0.40 | 0.22 / 1 |  |
| potential | 6/20 | 0/20 | 0.00 | 3.6 / 1.9 | 0.00 / 0.00 | 0.012 / 1.5e-05 |  |
| prm | 13/20 | 15/20 | 1.15 | 4.5 / 4.3 | 0.00 / 0.40 | 0.77 / 0.62 |  |
| prmstar | 18/20 | 17/20 | 0.94 | 4.8 / 4.5 | 0.00 / 0.40 | 0.45 / 1 |  |
| rrt | 0/20 | 17/20 | - | 3.0 / 4.4 | 0.00 / 0.40 | 6.1e-05 / 1 |  |
| rrtstar | 0/20 | 14/20 | - | 3.0 / 4.1 | 0.00 / 0.20 | 6.1e-05 / 0.38 |  |
| sst | 15/20 | 6/20 | 0.40 | 3.9 / 2.9 | 0.00 / 0.05 | 1 / 0.00098 |  |
| stomp | 17/20 | 18/20 | 1.06 | 4.7 / 4.5 | 0.00 / 0.40 | 0.69 / 1 |  |
| straight | 0/20 | 0/20 | - | 3.0 / 2.7 | 0.00 / 0.00 | 6.1e-05 / 1.5e-05 |  |
| trajopt | 17/20 | 14/20 | 0.82 | 4.7 / 4.1 | 0.00 / 0.35 | 0.69 / 0.38 |  |

## 5 Timing

| Method | Franka | SO-101 | Retention | Mean stage F / S | Post contact F / S | vs default p (F / S) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| iptp | 15/20 | 17/20 | 1.13 | 4.5 / 4.5 | 0.00 / 0.40 | 1 / 1 |  |
| ruckig | 16/20 | 9/20 | 0.56 | 4.7 / 3.2 | 0.00 / 0.40 | 1 / 0.0078 |  |
| scurve | 16/20 | 17/20 | 1.06 | 4.6 / 4.4 | 0.00 / 0.40 | 1 / 1 |  |
| topp | 16/20 | 17/20 | 1.06 | 4.6 / 4.5 | 0.00 / 0.40 | 1 / 1 |  |
| totg | 17/20 | 17/20 | 1.00 | 4.7 / 4.4 | 0.00 / 0.40 | 0.5 / 1 |  |
| trapezoid | 16/20 | 15/20 | 0.94 | 4.6 / 4.3 | 0.00 / 0.40 | 1 / 0.62 |  |

## 6 Tracking

| Method | Franka | SO-101 | Retention | Mean stage F / S | Post contact F / S | vs default p (F / S) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| admittance | - | 8/20 | - | - | - |  / 0.0039 |  |
| cart | - | 9/20 | - | - | - |  / 0.0078 |  |
| carttorque | 16/20 | 9/20 | 0.56 | 4.6 / 3.2 | 0.00 / 0.55 | 1 / 0.0078 | SO-101 torque: sim-only counterfactual |
| ibvs | 17/20 | 2/20 | 0.12 | 4.7 / 1.9 | 0.00 / 0.40 | 0.5 / 6.1e-05 |  |
| pbvs | 14/20 | 7/20 | 0.50 | 4.5 / 2.8 | 0.00 / 0.40 | 1 / 0.002 |  |
| vs25 | 17/20 | 1/20 | 0.06 | 4.7 / 1.2 | 0.00 / 0.40 | 0.5 / 3.1e-05 |  |

## 7 Conversion (Cartesian tracking)

| Method | Franka | SO-101 | Retention | Mean stage F / S | Post contact F / S | vs default p (F / S) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cbf | - | 11/20 | - | - | - |  / 0.031 |  |
| dls | 15/20 | 9/20 | 0.60 | 4.5 / 3.3 | 0.00 / 0.15 | 1 / 0.0078 |  |
| jinv | n/a | 1/20 | - | 0.0 / 1.7 | 0.00 / 0.45 | 6.1e-05 / 3.1e-05 | conversion jinv cannot run on this arm |
| jtrans | 14/20 | 0/20 | 0.00 | 4.4 / 1.1 | 0.10 / 0.40 | 1 / 1.5e-05 |  |
| pinv | 15/20 | 0/20 | 0.00 | 4.5 / 1.9 | 0.00 / 0.75 | 1 / 1.5e-05 |  |
| qp | - | 7/20 | - | - | - |  / 0.002 |  |

## 8 Control

| Method | Franka | SO-101 | Retention | Mean stage F / S | Post contact F / S | vs default p (F / S) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| backstep | 15/20 | 18/20 | 1.20 | 4.5 / 4.5 | 0.00 / 0.40 | 1 / 1 | SO-101 torque: sim-only counterfactual |
| cimp (task space) | 16/20 | 7/20 | 0.44 | 4.6 / 3.0 | 0.00 / 0.70 | 1 / 0.002 | SO-101 torque: sim-only counterfactual |
| ct | 15/20 | 18/20 | 1.20 | 4.5 / 4.5 | 0.00 / 0.40 | 1 / 1 | SO-101 torque: sim-only counterfactual |
| force (task space) | 0/20 | 0/20 | - | 0.4 / 1.8 | 0.55 / 0.20 | 6.1e-05 / 1.5e-05 | SO-101 torque: sim-only counterfactual |
| hinf | 15/20 | 18/20 | 1.20 | 4.5 / 4.5 | 0.00 / 0.40 | 1 / 1 | SO-101 torque: sim-only counterfactual |
| hybrid (task space) | 0/20 | 0/20 | - | 0.4 / 1.8 | 0.55 / 0.20 | 6.1e-05 / 1.5e-05 | SO-101 torque: sim-only counterfactual |
| jimp | 15/20 | 18/20 | 1.20 | 4.5 / 4.5 | 0.00 / 0.40 | 1 / 1 | SO-101 torque: sim-only counterfactual |
| lqg | 0/10 | 0/10 | - | 1.0 / 1.0 | 0.10 / 0.40 | 0.016 / 0.0039 | SO-101 torque: sim-only counterfactual |
| lqr | 15/20 | 18/20 | 1.20 | 4.5 / 4.5 | 0.00 / 0.40 | 1 / 1 | SO-101 torque: sim-only counterfactual |
| mpc | 0/10 | 9/10 | - | 0.7 / 4.5 | 0.90 / 0.30 | 0.016 / 1 | SO-101 torque: sim-only counterfactual |
| mrac | 16/20 | 18/20 | 1.12 | 4.6 / 4.5 | 0.05 / 0.40 | 1 / 1 | SO-101 torque: sim-only counterfactual |
| nmpc | 0/10 | 4/10 | - | 0.4 / 2.2 | 0.70 / 0.20 | 0.016 / 0.062 | SO-101 torque: sim-only counterfactual |
| p | 5/20 | 18/20 | 3.60 | 2.8 / 4.5 | 0.10 / 0.40 | 0.0063 / 1 |  |
| passivity | 15/20 | 18/20 | 1.20 | 4.5 / 4.5 | 0.00 / 0.40 | 1 / 1 | SO-101 torque: sim-only counterfactual |
| pd | 13/20 | 17/20 | 1.31 | 4.2 / 4.5 | 0.00 / 0.40 | 0.62 / 1 |  |
| pdg | 15/20 | 18/20 | 1.20 | 4.5 / 4.5 | 0.00 / 0.40 | 1 / 1 | SO-101 torque: sim-only counterfactual |
| pid | 15/20 | 16/20 | 1.07 | 4.5 / 4.3 | 0.05 / 0.40 | 1 / 1 |  |
| ptask (task space) | 0/20 | 0/20 | - | 0.0 / 1.4 | 0.00 / 0.30 | 6.1e-05 / 1.5e-05 | SO-101 torque: sim-only counterfactual |
| slotine | 17/20 | 18/20 | 1.06 | 4.7 / 4.5 | 0.00 / 0.40 | 0.5 / 1 | SO-101 torque: sim-only counterfactual |
| smc | 2/20 | 18/20 | 9.00 | 2.5 / 4.5 | 0.00 / 0.40 | 0.00024 / 1 | SO-101 torque: sim-only counterfactual |

## 9 Actuator

| Method | Franka | SO-101 | Retention | Mean stage F / S | Post contact F / S | vs default p (F / S) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cascade | 18/20 | 17/20 | 0.94 | 4.8 / 4.5 | 0.00 / 0.40 | 0.25 / 1 |  |
| torque | 15/20 | 18/20 | 1.20 | 4.5 / 4.5 | 0.00 / 0.40 | 1 / 1 | SO-101 torque: sim-only counterfactual |

