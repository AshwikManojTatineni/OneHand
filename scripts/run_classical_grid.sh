#!/usr/bin/env bash
# Full classical (S1) grid: every decision-tree method as a one-layer swap from the default, on both bodies.
# 20 episodes per cell (10 for the slow MPC / NMPC / LQG cells), 4 cells in parallel. Results: results/classical/.
cd "$(dirname "$0")/.."
N=${N:-20}; NSLOW=${NSLOW:-10}
cells=( "" input=known input=camera
  world=octomap world=voxel world=esdf world=tsdf world=mesh world=pose
  endpoint=analytic endpoint=newton endpoint=lm endpoint=ccd endpoint=taught
  planner=prm planner=prmstar planner=lazyprm planner=rrt planner=rrtstar planner=informedrrtstar planner=est
  planner=kpiece planner=kinorrt planner=sst planner=astar planner=dijkstra planner=lattice planner=chomp
  planner=stomp planner=trajopt planner=potential planner=dampers planner=straight
  timing=topp timing=totg timing=iptp timing=trapezoid timing=scurve timing=ruckig
  tracking=cart tracking=carttorque tracking=admittance tracking=ibvs tracking=pbvs tracking=vs25
  "tracking=cart conversion=dls" "tracking=cart conversion=pinv" "tracking=cart conversion=jinv"
  "tracking=cart conversion=jtrans" "tracking=cart conversion=qp" "tracking=cart conversion=cbf"
  control=p control=pd control=pid control=pdg control=ct control=lqr control=smc control=hinf control=backstep
  control=slotine control=mrac control=jimp control=passivity
  "tracking=carttorque control=cimp" "tracking=carttorque control=force" "tracking=carttorque control=hybrid"
  "tracking=carttorque control=ptask" actuator=cascade actuator=torque )
slow=( control=lqg control=mpc control=nmpc )
jobs=()
for b in franka so101; do
  for c in "${cells[@]}"; do jobs+=("$b $N $c"); done
  for c in "${slow[@]}"; do jobs+=("$b $NSLOW $c"); done
done
# resume: skip cells whose result file already exists (same naming as eval_classical.py)
todo=()
for j in "${jobs[@]}"; do
  set -- $j; b=$1; shift 2
  tag=$(printf '%s\n' "$@" | sed 's/=/-/' | paste -sd_ -); tag=${tag:-default}
  [ -f "results/classical/${b}__${tag}.json" ] || todo+=("$j")
done
echo "cells to run: ${#todo[@]} of ${#jobs[@]}"
printf '%s\n' "${todo[@]}" | xargs -P 4 -I{} sh -c '.venv-lite/bin/python scripts/eval_classical.py {} > /dev/null 2>&1; echo "done: {}"'
echo CLASSICAL_GRID_DONE
