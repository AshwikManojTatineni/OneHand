#!/usr/bin/env bash
# One episode of each non-default method family on both bodies, to catch crashes before the full grid.
cd "$(dirname "$0")/.."
for b in franka so101; do
  for v in input=known input=camera world=octomap world=voxel world=esdf world=tsdf world=mesh world=pose \
           endpoint=analytic endpoint=newton endpoint=lm endpoint=ccd endpoint=taught \
           planner=chomp planner=sst timing=topp timing=totg timing=iptp timing=trapezoid timing=scurve timing=ruckig \
           tracking=cart tracking=carttorque tracking=admittance tracking=ibvs tracking=pbvs tracking=vs25 \
           "tracking=cart conversion=jinv" "tracking=cart conversion=qp" "tracking=cart conversion=cbf" \
           control=p control=pid control=ct control=lqr control=lqg control=mpc control=nmpc control=smc control=hinf \
           control=backstep control=slotine control=mrac control=jimp control=passivity \
           "tracking=carttorque control=cimp" "tracking=carttorque control=force" "tracking=carttorque control=hybrid" \
           "tracking=carttorque control=ptask" actuator=cascade actuator=torque; do
    out=$(timeout 600 .venv-lite/bin/python scripts/eval_classical.py $b 1 $v 2>&1 | grep -v "nanobind\|leaked\| - \|See https" | tail -1)
    echo "$b | $v | $(echo "$out" | python3 -c "import json,sys
try:
    d=json.loads(sys.stdin.read()); print('stage',d['stages'] if 'stages' in d else d['mean_stage'],'NA:',d['not_applicable'],'%.0fs'%d['sec_per_episode'])
except Exception as e: print('ERROR')")"
  done
done
