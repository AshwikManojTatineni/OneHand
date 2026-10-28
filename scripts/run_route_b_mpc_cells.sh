#!/usr/bin/env bash
# Route B planning-on (MPC) cells at R6, 20 episodes each (MPPI is slow on one CPU thread).
set -u
cd "$(dirname "$0")/.."
. .venv-lite/bin/activate
export OMP_NUM_THREADS=1
for iface in K K+P; do
  python scripts/eval_route_b_transfer.py $iface 20 --ik dls > "results/phase0/log_transfer_${iface}_dls_mpc20.txt" 2>&1
done
echo MPC_CELLS_DONE
