# Sensor-swap self-modeling benchmark (Franka, rendered lab)

One arm with sampled calibration errors (links, joint axes, encoder offsets, base placement, sensor mount)
is teleoperated through its nominal model. Only the sensor that measures the hand changes. Each sensor's pose
comes from its own pipeline on rendered data: ChArUco PnP (mono), triangulated corners (stereo), scan-to-map
ICP (lidar). The self-model (differential fit, then absolute refinement, robust loss) never sees the model file.
FK error: the sensor frame over 300 random configurations of the whole joint range. Reach: IK on the model,
run on the real arm, success within 5 mm and 2 deg (40 targets per arm, the same targets for every model).
Camera sensors model the camera frame, lidar the lidar frame; each is compared with the nominal model of the
same frame.

## R0 (ideal renders), cartesian teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.55 | 0.9 / 1.5 | 200/200 (98-100%) | 88/200 | 26/200 | 8.4e-53 |
| Monocular camera | 87% | 0.58 | 0.6 / 1.0 | 200/200 (98-100%) | 200/200 | 26/200 | 8.4e-53 |
| Stereo camera | 78% | 0.60 | 0.7 / 1.0 | 200/200 (98-100%) | 183/200 | 26/200 | 8.4e-53 |
| Lidar | 100% | 0.06 | 0.2 / 0.3 | 200/200 (98-100%) | 200/200 | 26/200 | 8.4e-53 |

Lidar frames by outcome: ok 2000

## R0 (ideal renders), joint teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.48 | 0.8 / 1.7 | 200/200 (98-100%) | 190/200 | 26/200 | 8.4e-53 |
| Monocular camera | 36% | 0.50 | 0.8 / 1.3 | 200/200 (98-100%) | 200/200 | 26/200 | 8.4e-53 |
| Stereo camera | 22% | 0.50 | 0.9 / 2.2 | 80/80 (95-100%) | 80/80 | 26/200 | 6.9e-18 |
| Lidar | 98% | 0.10 | 0.1 / 0.2 | 200/200 (98-100%) | 200/200 | 26/200 | 8.4e-53 |

Lidar frames by outcome: degenerate 7, few_matches 35, ok 1956, poor_fit 2

## R1 (sensor noise and calibration error), cartesian teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.50 | 0.7 / 1.3 | 200/200 (98-100%) | 138/200 | 26/200 | 8.4e-53 |
| Monocular camera | 87% | nan | 1.0 / 1.6 | 160/160 (98-100%) | 160/160 | 26/200 | 4.6e-41 |
| Stereo camera | 78% | 1.08 | 2.6 / 4.3 | 200/200 (98-100%) | 193/200 | 26/200 | 8.4e-53 |
| Lidar | 100% | 1.49 | 1.7 / 3.6 | 200/200 (98-100%) | 177/200 | 26/200 | 8.4e-53 |

Lidar frames by outcome: ok 2000

## R1 (sensor noise and calibration error), joint teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.50 | 1.0 / 1.7 | 200/200 (98-100%) | 189/200 | 26/200 | 8.4e-53 |
| Monocular camera | 36% | 0.72 | 1.4 / 2.1 | 200/200 (98-100%) | 200/200 | 26/200 | 8.4e-53 |
| Stereo camera | 22% | 0.87 | 1.5 / 2.8 | 120/120 (97-100%) | 119/120 | 26/200 | 1.3e-29 |
| Lidar | 99% | 1.67 | 2.2 / 4.2 | 196/200 (95-99%) | 200/200 | 26/200 | 1.3e-51 |

Lidar frames by outcome: degenerate 15, ok 1985

## R2 (R1 + motion blur and 15 ms latency), cartesian teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.52 | 0.9 / 1.3 | 200/200 (98-100%) | 162/200 | 26/200 | 8.4e-53 |
| Monocular camera | 85% | 0.76 | 2.5 / 4.5 | 200/200 (98-100%) | 48/200 | 26/200 | 8.4e-53 |
| Stereo camera | 75% | 1.26 | 3.3 / 6.4 | 197/200 (96-99%) | 19/200 | 26/200 | 2.9e-50 |
| Lidar | 100% | 1.62 | 2.6 / 5.1 | 198/200 (96-100%) | 38/200 | 26/200 | 3.3e-52 |

Lidar frames by outcome: ok 2000

## R2 (R1 + motion blur and 15 ms latency), joint teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.56 | 0.8 / 1.5 | 200/200 (98-100%) | 197/200 | 26/200 | 8.4e-53 |
| Monocular camera | 27% | 1.58 | 4.9 / 10.8 | 155/160 (93-99%) | 4/160 | 26/200 | 4.9e-38 |
| Stereo camera | 14% | 1.46 | 3.6 / 7.6 | 79/80 (93-100%) | 2/80 | 26/200 | 1.4e-17 |
| Lidar | 99% | 2.67 | 4.2 / 6.9 | 199/200 (97-100%) | 7/200 | 26/200 | 1.7e-52 |

Lidar frames by outcome: degenerate 14, ok 1986

## R3 (R2 + joint compliance), cartesian teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.57 | 2.5 / 8.0 | 200/200 (98-100%) | 131/200 | 1/200 | 2.5e-60 |
| Monocular camera | 85% | 0.79 | 3.8 / 9.7 | 199/200 (97-100%) | 53/200 | 1/200 | 5e-60 |
| Stereo camera | 76% | 1.22 | 4.2 / 10.6 | 199/200 (97-100%) | 19/200 | 1/200 | 5e-60 |
| Lidar | 100% | 1.59 | 4.7 / 10.9 | 197/200 (96-99%) | 19/200 | 0/200 | 1e-59 |

Lidar frames by outcome: ok 2000

## R3 (R2 + joint compliance), joint teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.50 | 2.8 / 9.4 | 199/200 (97-100%) | 190/200 | 1/200 | 5e-60 |
| Monocular camera | 27% | 1.62 | 5.2 / 15.6 | 192/200 (92-98%) | 4/200 | 1/200 | 6.4e-58 |
| Stereo camera | 14% | 1.48 | 5.3 / 14.7 | 75/80 (86-97%) | 2/80 | 1/200 | 5.3e-23 |
| Lidar | 99% | 2.70 | 5.2 / 12.1 | 196/200 (95-99%) | 7/200 | 0/200 | 2e-59 |

Lidar frames by outcome: degenerate 12, ok 1988

## Realism ladder at a glance (self-model reach successes, all arms pooled)

| Teleop | Sensor | R0 | R1 | R2 | R3 |
| --- | --- | --- | --- | --- | --- |
| cartesian | Oracle (true pose + noise) | 200/200 | 200/200 | 200/200 | 200/200 |
| cartesian | Monocular camera | 200/200 | 160/200 | 200/200 | 199/200 |
| cartesian | Stereo camera | 200/200 | 200/200 | 197/200 | 199/200 |
| cartesian | Lidar | 200/200 | 200/200 | 198/200 | 197/200 |
| joint | Oracle (true pose + noise) | 200/200 | 200/200 | 200/200 | 199/200 |
| joint | Monocular camera | 200/200 | 200/200 | 155/200 | 192/200 |
| joint | Stereo camera | 80/200 | 120/200 | 79/200 | 75/200 |
| joint | Lidar | 200/200 | 196/200 | 199/200 | 196/200 |

