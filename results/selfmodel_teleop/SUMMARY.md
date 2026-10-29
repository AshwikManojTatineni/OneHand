# Teleop self-model pilot (Franka): kinematics from encoder and extrinsic changes

The arm is teleoperated through its nominal kinematic model, but differs from it by sampled calibration
errors (2 mm link offsets, 0.2 deg axis tilts, 0.5 deg encoder offsets, 1 mm tool offset). It logs encoder
readings and the measured pose of a frame on the hand. The self-models rebuild the kinematics from those logs
alone. 5 sampled arms per row. FK error is measured on 300 random configurations over the whole joint range.
Reach: IK on the model, run on the real arm, success = within 5 mm and 2 deg (40 targets per arm,
the same targets for every model; p = exact McNemar against the nominal model).

## Joint-space teleop (wiggle each joint, then random walk): 400 logged poses, sensor noise 1 mm / 0.3 deg

| Model | FK median (mm) | FK p95 (mm) | Rotation median (deg) | One-step relative error | Reach | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- |
| Nominal model file | 11.41 | 20.16 | 0.93 | 0.0005 | 50/200 (20-31%) |  |
| Differential only (blind) | 20.50 | 41.01 | 2.41 | 0.0005 | 57/200 (23-35%) | 0.49 |
| Differential + absolute (blind) | 1.87 | 3.94 | 0.28 | 0.0000 | 200/200 (98-100%) | 1.4e-45 |
| Calibrated nominal | 1.87 | 3.94 | 0.28 | 0.0000 | 200/200 (98-100%) | 1.4e-45 |
| Local Jacobian (model-free) | - | - | - | 0.1914 | - | |

## Cartesian teleop (gripper moved between waypoints, pointing down): 400 logged poses, sensor noise 1 mm / 0.3 deg

| Model | FK median (mm) | FK p95 (mm) | Rotation median (deg) | One-step relative error | Reach | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- |
| Nominal model file | 11.41 | 20.16 | 0.93 | 0.0008 | 50/200 (20-31%) |  |
| Differential only (blind) | 36.20 | 71.48 | 3.21 | 0.0026 | 29/200 (10-20%) | 0.0031 |
| Differential + absolute (blind) | 2.10 | 4.49 | 0.26 | 0.0003 | 200/200 (98-100%) | 1.4e-45 |
| Calibrated nominal | 2.10 | 4.49 | 0.26 | 0.0003 | 200/200 (98-100%) | 1.4e-45 |
| Local Jacobian (model-free) | - | - | - | 0.0363 | - | |

## How much teleop is needed (noise 1 mm / 0.3 deg)

FK median error in mm (median over arms), reach successes out of 200.

| Teleop | 25 | 50 | 100 | 200 | 400 | 800 |
| --- | --- | --- | --- | --- | --- | --- |
| joint: Nominal model file | 11.4 mm, 50/200 | 11.4 mm, 50/200 | 11.4 mm, 50/200 | 11.4 mm, 50/200 | 11.4 mm, 50/200 | 11.4 mm, 50/200 |
| joint: Differential only (blind) | 23772.8 mm, 0/200 | 9685.8 mm, 0/200 | 22.7 mm, 81/200 | 24.1 mm, 44/200 | 20.5 mm, 57/200 | 14.8 mm, 71/200 |
| joint: Differential + absolute (blind) | 9085.7 mm, 0/200 | 4153.4 mm, 0/200 | 8.1 mm, 194/200 | 4.2 mm, 200/200 | 1.9 mm, 200/200 | 0.8 mm, 200/200 |
| joint: Calibrated nominal | 9093.6 mm, 0/200 | 4153.4 mm, 0/200 | 8.1 mm, 194/200 | 4.2 mm, 200/200 | 1.9 mm, 200/200 | 0.8 mm, 200/200 |
| cartesian: Nominal model file | 11.4 mm, 50/200 | 11.4 mm, 50/200 | 11.4 mm, 50/200 | 11.4 mm, 50/200 | 11.4 mm, 50/200 | 11.4 mm, 50/200 |
| cartesian: Differential only (blind) | 33696.8 mm, 0/200 | 4420.6 mm, 0/200 | 141.0 mm, 2/200 | 109.5 mm, 5/200 | 36.2 mm, 29/200 | 18.0 mm, 108/200 |
| cartesian: Differential + absolute (blind) | 5755.1 mm, 0/200 | 939.9 mm, 0/200 | 40.4 mm, 53/200 | 7.7 mm, 178/200 | 2.1 mm, 200/200 | 1.3 mm, 200/200 |
| cartesian: Calibrated nominal | 5281.3 mm, 0/200 | 207.6 mm, 0/200 | 40.4 mm, 54/200 | 7.7 mm, 178/200 | 2.1 mm, 200/200 | 1.3 mm, 200/200 |

## Sensor noise (400 logged poses)

FK median error in mm (median over arms), reach successes out of 200.

| Teleop | none | low | mid | high |
| --- | --- | --- | --- | --- |
| joint: Nominal model file | 11.4 mm, 50/200 | 11.4 mm, 50/200 | 11.4 mm, 50/200 | 11.4 mm, 50/200 |
| joint: Differential only (blind) | 0.6 mm, 200/200 | 7.3 mm, 179/200 | 20.5 mm, 57/200 | 66.8 mm, 0/200 |
| joint: Differential + absolute (blind) | 0.1 mm, 200/200 | 0.9 mm, 200/200 | 1.9 mm, 200/200 | 5.4 mm, 200/200 |
| joint: Calibrated nominal | 0.1 mm, 200/200 | 0.9 mm, 200/200 | 1.9 mm, 200/200 | 5.4 mm, 200/200 |
| cartesian: Nominal model file | 11.4 mm, 50/200 | 11.4 mm, 50/200 | 11.4 mm, 50/200 | 11.4 mm, 50/200 |
| cartesian: Differential only (blind) | 1.7 mm, 199/200 | 12.2 mm, 158/200 | 36.2 mm, 29/200 | 144.0 mm, 1/200 |
| cartesian: Differential + absolute (blind) | 0.1 mm, 200/200 | 0.9 mm, 200/200 | 2.1 mm, 200/200 | 6.5 mm, 199/200 |
| cartesian: Calibrated nominal | 0.1 mm, 200/200 | 0.9 mm, 200/200 | 2.1 mm, 200/200 | 6.5 mm, 199/200 |

## Same data as the current self-model (20 recorded ManiSkill Franka pick-cube episodes)

Fit on 10 episodes, test on the other 10, both ways (480 test transitions). Relative error = mean squared
error of the predicted TCP change / mean squared TCP change, the current readout's normalisation.
The current readout (frozen TD-MPC2 latent) predicts from the commanded action; the differential
self-models predict from the measured joint change. They answer different questions: "what will
this command do" against "what does this joint change do to my hand".

| Sensor noise | Differential only | Differential + absolute | Local Jacobian | Current readout (latent) |
| --- | --- | --- | --- | --- |
| none | 4.4e-11 | 4.3e-11 | 0.02 | 0.157 |
| mid | 0.0002 | 1.8e-05 | 0.021 | 0.157 |
| high | 0.0017 | 0.00017 | 0.023 | 0.157 |

