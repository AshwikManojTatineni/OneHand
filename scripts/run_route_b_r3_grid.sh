#!/usr/bin/env bash
# Route B rung R3 (xArm6) screening grid plus xArm6 home baselines, 50 episodes per cell.
set -u
cd "$(dirname "$0")/.."
. .venv-lite/bin/activate
export OMP_NUM_THREADS=1
N=${1:-50}
python scripts/eval_route_b_home.py ms-pick-cube-xarm6 $N --no-mpc > results/phase0/log_xarm6_prior.txt 2>&1
for mode in --no-mpc ""; do
  for iface in K K+P; do
    python scripts/eval_route_b_r3.py $iface $N $mode > "results/phase0/log_r3_${iface}${mode}.txt" 2>&1
  done
  python scripts/eval_route_b_r3.py K+P $N --naive-gripper $mode > "results/phase0/log_r3_K+P_naivegrip${mode}.txt" 2>&1
done
echo R3_GRID_DONE
