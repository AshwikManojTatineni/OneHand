#!/bin/sh
# Re-run classical cells that need no OMPL and save their JSON next to this script; the committed results are restored afterwards.
cd "$(dirname "$0")/../../.." || exit 1
export MUJOCO_GL=glfw
for body in franka so101; do
  for p in lazyprm astar potential; do
    .venv/Scripts/python.exe scripts/eval_classical.py $body 20 planner=$p > docs/analysis/repro/log_${body}_$p.txt 2>&1
    cp results/classical/${body}__planner-$p.json docs/analysis/repro/${body}__planner-$p.json
  done
done
git checkout -- results/classical
echo done
