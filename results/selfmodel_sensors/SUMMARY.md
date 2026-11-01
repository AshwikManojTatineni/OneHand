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
| Oracle (true pose + noise) | 100% | 1.49 | 1.9 / 4.1 | 200/200 (98-100%) | 28/200 | 0/200 | 1.2e-60 |
| Monocular camera | 95% | 0.77 | 1.5 / 3.7 | 200/200 (98-100%) | 187/200 | 0/200 | 1.2e-60 |
| Stereo camera | 93% | 0.92 | 1.4 / 2.5 | 200/200 (98-100%) | 169/200 | 0/200 | 1.2e-60 |
| Lidar | 100% | 0.04 | 0.2 / 0.3 | 199/200 (97-100%) | 200/200 | 0/200 | 2.5e-60 |

Lidar frames by outcome: ok 2000

## R0 (ideal renders), joint teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.52 | 1.9 / 4.0 | 200/200 (98-100%) | 53/200 | 0/200 | 1.2e-60 |
| Monocular camera | 50% | 0.93 | 3.6 / 6.5 | 199/200 (97-100%) | 181/200 | 0/200 | 2.5e-60 |
| Stereo camera | 46% | 1.50 | 8.9 / 23.6 | 191/200 (92-98%) | 80/200 | 0/200 | 6.4e-58 |
| Lidar | 100% | 0.04 | 0.1 / 0.2 | 199/200 (97-100%) | 199/200 | 0/200 | 2.5e-60 |

Lidar frames by outcome: ok 2000

## R1 (sensor noise and calibration error), cartesian teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.54 | 2.1 / 4.0 | 200/200 (98-100%) | 29/200 | 0/200 | 1.2e-60 |
| Monocular camera | 95% | 0.98 | 4.0 / 7.9 | 200/200 (98-100%) | 192/200 | 0/200 | 1.2e-60 |
| Stereo camera | 93% | 3.72 | 14.4 / 23.0 | 125/200 (56-69%) | 95/200 | 0/200 | 4.7e-38 |
| Lidar | 100% | 1.05 | 1.2 / 2.8 | 200/200 (98-100%) | 183/200 | 0/200 | 1.2e-60 |

Lidar frames by outcome: ok 2000

## R1 (sensor noise and calibration error), joint teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.54 | 1.7 / 3.3 | 200/200 (98-100%) | 36/200 | 0/200 | 1.2e-60 |
| Monocular camera | 47% | 1.54 | 6.2 / 13.6 | 185/200 (88-95%) | 160/200 | 0/200 | 4.1e-56 |
| Stereo camera | 42% | 3.98 | 15.0 / 29.4 | 128/200 (57-70%) | 71/200 | 0/200 | 5.9e-39 |
| Lidar | 100% | 1.06 | 1.2 / 2.5 | 200/200 (98-100%) | 195/200 | 0/200 | 1.2e-60 |

Lidar frames by outcome: ok 2000

## R2 (R1 + motion blur and 15 ms latency), cartesian teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.54 | 2.1 / 4.7 | 200/200 (98-100%) | 28/200 | 0/200 | 1.2e-60 |
| Monocular camera | 95% | 1.48 | 8.5 / 15.9 | 184/200 (87-95%) | 114/200 | 0/200 | 8.2e-56 |
| Stereo camera | 93% | 3.70 | 18.7 / 30.5 | 106/200 (46-60%) | 70/200 | 0/200 | 2.5e-32 |
| Lidar | 100% | 1.49 | 5.4 / 10.1 | 197/200 (96-99%) | 114/200 | 0/200 | 1e-59 |

Lidar frames by outcome: ok 2000

## R2 (R1 + motion blur and 15 ms latency), joint teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.53 | 1.7 / 3.9 | 200/200 (98-100%) | 64/200 | 0/200 | 1.2e-60 |
| Monocular camera | 31% | 4.83 | 17.4 / 32.7 | 108/200 (47-61%) | 0/200 | 0/200 | 6.2e-33 |
| Stereo camera | 26% | 5.86 | 17.9 / 31.1 | 86/200 (36-50%) | 0/200 | 0/200 | 2.6e-26 |
| Lidar | 100% | 4.17 | 9.6 / 17.2 | 182/200 (86-94%) | 0/200 | 0/200 | 3.3e-55 |

Lidar frames by outcome: ok 2000

## R3 (R2 + joint compliance), cartesian teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.54 | 2.1 / 4.8 | 200/200 (98-100%) | 19/200 | 0/200 | 1.2e-60 |
| Monocular camera | 95% | 1.51 | 8.9 / 16.9 | 183/200 (87-95%) | 101/200 | 0/200 | 1.6e-55 |
| Stereo camera | 92% | 3.66 | 20.1 / 32.0 | 99/200 (43-56%) | 45/200 | 0/200 | 3.2e-30 |
| Lidar | 100% | 1.50 | 5.5 / 10.4 | 196/200 (95-99%) | 123/200 | 0/200 | 2e-59 |

Lidar frames by outcome: ok 2000

## R3 (R2 + joint compliance), joint teleop: 5 arms x 400 frames

| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (true pose + noise) | 100% | 1.54 | 2.5 / 5.3 | 200/200 (98-100%) | 56/200 | 0/200 | 1.2e-60 |
| Monocular camera | 30% | 4.83 | 17.2 / 29.3 | 104/200 (45-59%) | 0/200 | 0/200 | 9.9e-32 |
| Stereo camera | 26% | 6.01 | 17.9 / 31.6 | 94/200 (40-54%) | 0/200 | 0/200 | 1e-28 |
| Lidar | 100% | 4.23 | 10.6 / 17.6 | 182/200 (86-94%) | 0/200 | 0/200 | 3.3e-55 |

Lidar frames by outcome: ok 2000

## Realism ladder at a glance (self-model reach successes, all arms pooled)

| Teleop | Sensor | R0 | R1 | R2 | R3 |
| --- | --- | --- | --- | --- | --- |
| cartesian | Oracle (true pose + noise) | 200/200 | 200/200 | 200/200 | 200/200 |
| cartesian | Monocular camera | 200/200 | 200/200 | 184/200 | 183/200 |
| cartesian | Stereo camera | 200/200 | 125/200 | 106/200 | 99/200 |
| cartesian | Lidar | 199/200 | 200/200 | 197/200 | 196/200 |
| joint | Oracle (true pose + noise) | 200/200 | 200/200 | 200/200 | 200/200 |
| joint | Monocular camera | 199/200 | 185/200 | 108/200 | 104/200 |
| joint | Stereo camera | 191/200 | 128/200 | 86/200 | 94/200 |
| joint | Lidar | 199/200 | 200/200 | 182/200 | 182/200 |

