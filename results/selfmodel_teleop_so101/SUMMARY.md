# Teleop self-model pilot (so101): kinematics from encoder and extrinsic changes

The arm is teleoperated through its nominal kinematic model, but differs from it by sampled calibration
errors (1 mm link offsets, 0.5 deg axis tilts, 1.5 deg servo zero offsets, 1 mm tool offset). It logs encoder
readings and the measured pose of a frame on the hand. The self-models rebuild the kinematics from those logs
alone. 5 sampled arms per row. FK error is measured on 300 random configurations over the whole joint range.
Reach: IK on the model, run on the real arm, success = within 5 mm and 2 deg (40 targets per arm,
the same targets for every model; p = exact McNemar against the nominal model).

## Joint-space teleop (wiggle each joint, then random walk): 400 logged poses, sensor noise 1 mm / 0.3 deg

| Model | FK median (mm) | FK p95 (mm) | Rotation median (deg) | One-step relative error | Reach | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- |
| Nominal model file | 12.23 | 15.00 | 2.67 | 0.0016 | 0/200 (0-2%) |  |
| Differential only (blind) | 6.50 | 13.09 | 1.31 | 0.0003 | 184/200 (87-95%) | 8.2e-56 |
| Differential + absolute (blind) | 1.05 | 2.13 | 0.21 | 0.0000 | 200/200 (98-100%) | 1.2e-60 |
| Calibrated nominal | 1.05 | 2.13 | 0.21 | 0.0000 | 200/200 (98-100%) | 1.2e-60 |
| Local Jacobian (model-free) | - | - | - | 0.1492 | - | |

## Cartesian teleop (gripper moved between waypoints, pointing down): 400 logged poses, sensor noise 1 mm / 0.3 deg

| Model | FK median (mm) | FK p95 (mm) | Rotation median (deg) | One-step relative error | Reach | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- |
| Nominal model file | 12.23 | 15.00 | 2.67 | 0.0087 | 0/200 (0-2%) |  |
| Differential only (blind) | 10.94 | 17.54 | 1.69 | 0.0020 | 135/200 (61-74%) | 4.6e-41 |
| Differential + absolute (blind) | 0.88 | 1.69 | 0.22 | 0.0004 | 200/200 (98-100%) | 1.2e-60 |
| Calibrated nominal | 0.88 | 1.69 | 0.22 | 0.0004 | 200/200 (98-100%) | 1.2e-60 |
| Local Jacobian (model-free) | - | - | - | 0.1601 | - | |

## How much teleop is needed (noise 1 mm / 0.3 deg)

FK median error in mm (median over arms), reach successes out of 200.

| Teleop | 25 | 50 | 100 | 200 | 400 | 800 |
| --- | --- | --- | --- | --- | --- | --- |
| joint: Nominal model file | 12.2 mm, 0/200 | 12.2 mm, 0/200 | 12.2 mm, 0/200 | 12.2 mm, 0/200 | 12.2 mm, 0/200 | 12.2 mm, 0/200 |
| joint: Differential only (blind) | 11036.1 mm, 0/200 | 15.1 mm, 114/200 | 8.9 mm, 167/200 | 9.4 mm, 176/200 | 6.5 mm, 184/200 | 5.7 mm, 171/200 |
| joint: Differential + absolute (blind) | 5836.0 mm, 0/200 | 5.3 mm, 180/200 | 3.6 mm, 200/200 | 2.4 mm, 200/200 | 1.1 mm, 200/200 | 0.4 mm, 200/200 |
| joint: Calibrated nominal | 5850.7 mm, 0/200 | 5.3 mm, 180/200 | 3.6 mm, 200/200 | 2.4 mm, 200/200 | 1.1 mm, 200/200 | 0.4 mm, 200/200 |
| cartesian: Nominal model file | 12.2 mm, 0/200 | 12.2 mm, 0/200 | 12.2 mm, 0/200 | 12.2 mm, 0/200 | 12.2 mm, 0/200 | 12.2 mm, 0/200 |
| cartesian: Differential only (blind) | 1783.4 mm, 0/200 | 115.1 mm, 3/200 | 59.5 mm, 27/200 | 9.5 mm, 119/200 | 10.9 mm, 135/200 | 4.6 mm, 191/200 |
| cartesian: Differential + absolute (blind) | 824.9 mm, 0/200 | 78.2 mm, 4/200 | 11.9 mm, 130/200 | 2.6 mm, 200/200 | 0.9 mm, 200/200 | 0.8 mm, 200/200 |
| cartesian: Calibrated nominal | 313.2 mm, 0/200 | 78.2 mm, 4/200 | 11.9 mm, 130/200 | 2.6 mm, 200/200 | 0.9 mm, 200/200 | 0.8 mm, 200/200 |

## Sensor noise (400 logged poses)

FK median error in mm (median over arms), reach successes out of 200.

| Teleop | none | low | mid | high |
| --- | --- | --- | --- | --- |
| joint: Nominal model file | 12.2 mm, 0/200 | 12.2 mm, 0/200 | 12.2 mm, 0/200 | 12.2 mm, 0/200 |
| joint: Differential only (blind) | 0.1 mm, 200/200 | 2.8 mm, 200/200 | 6.5 mm, 184/200 | 21.0 mm, 33/200 |
| joint: Differential + absolute (blind) | 0.0 mm, 200/200 | 0.5 mm, 200/200 | 1.1 mm, 200/200 | 3.3 mm, 200/200 |
| joint: Calibrated nominal | 0.0 mm, 200/200 | 0.5 mm, 200/200 | 1.1 mm, 200/200 | 3.3 mm, 200/200 |
| cartesian: Nominal model file | 12.2 mm, 0/200 | 12.2 mm, 0/200 | 12.2 mm, 0/200 | 12.2 mm, 0/200 |
| cartesian: Differential only (blind) | 0.3 mm, 200/200 | 4.6 mm, 200/200 | 10.9 mm, 135/200 | 36.2 mm, 18/200 |
| cartesian: Differential + absolute (blind) | 0.0 mm, 200/200 | 0.4 mm, 200/200 | 0.9 mm, 200/200 | 2.7 mm, 199/200 |
| cartesian: Calibrated nominal | 0.0 mm, 200/200 | 0.4 mm, 200/200 | 0.9 mm, 200/200 | 2.7 mm, 199/200 |

