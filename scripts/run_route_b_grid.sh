#!/usr/bin/env bash
# Route B Phase 3-style screening grid: 50 episodes per cell, same seeds for every cell.
set -u
cd "$(dirname "$0")/.."
. .venv-lite/bin/activate
export OMP_NUM_THREADS=1
N=${1:-50}
for mode in --no-mpc ""; do
  for iface in K K+P; do
    for ik in dls hqp-axis; do
      python scripts/eval_route_b_transfer.py $iface $N --ik $ik $mode > "results/phase0/log_transfer_${iface}_${ik}${mode}.txt" 2>&1
    done
  done
  python scripts/eval_route_b_transfer.py K+P $N --ik dls --no-shift $mode > "results/phase0/log_transfer_K+P_dls_noshift${mode}.txt" 2>&1
done
echo GRID_DONE
