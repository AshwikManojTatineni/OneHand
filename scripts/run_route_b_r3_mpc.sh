#!/usr/bin/env bash
# Rerun the R3 (xArm6) planning-on cells that a container restart interrupted.
set -u
cd "$(dirname "$0")/.."
. .venv-lite/bin/activate
export OMP_NUM_THREADS=1
for args in "K" "K+P" "K+P --naive-gripper"; do
  tag=$(echo "$args" | tr ' ' '_')
  python scripts/eval_route_b_r3.py $args 50 > "results/phase0/log_r3_${tag}_mpc.txt" 2>&1
done
echo R3_MPC_DONE
