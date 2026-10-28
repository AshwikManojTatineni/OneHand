#!/usr/bin/env bash
# S1 one-layer variant grid on both MuJoCo bodies, 50 episodes per cell, same seeds for every cell.
set -u
cd "$(dirname "$0")/.."
. .venv-lite/bin/activate
N=${1:-50}
for body in franka so101; do
  python scripts/eval_s1.py $body $N                                  # default path (Gate 0 expert)
  for ik in hqp-full dls-full newton-full; do python scripts/eval_s1.py $body $N --ik $ik; done
  python scripts/eval_s1.py $body $N --tracking cartesian
  python scripts/eval_s1.py $body $N --timing quintic
  python scripts/eval_s1.py $body $N --kp-scale 0.017857               # about 56x softer servos
done > results/phase0/s1_grid.out 2>&1
echo S1_GRID_DONE
