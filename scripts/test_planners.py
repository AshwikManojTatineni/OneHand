"""Phase 0 check of layer 4: every planner on the transit above the cube -> above the plate (past the post),
both bodies, mesh world model. Reports success, time, path length and collision checks."""
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody.bodies import SPECS, robot_body_ids, tcp_frame  # noqa: E402
from xbody.classical.ik import EndpointSolver  # noqa: E402
from xbody.classical.planning import PLANNERS, Space, path_length  # noqa: E402
from xbody.classical.worldmodel import RobotPoints, WorldModel  # noqa: E402
from xbody.kinematics import Body  # noqa: E402
from xbody.scene import CLASSICAL, build_task  # noqa: E402

R_DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])


def main(only=None):
    for body in ["franka", "so101"]:
        cfg = SPECS[body]
        m = build_task(body); d = mujoco.MjData(m)
        bn, off, q = tcp_frame(m, cfg); kin = Body(m, cfg["arm"], bn, off, q)
        rb = robot_body_ids(m, cfg)
        rp = RobotPoints(m, rb)
        wm = WorldModel("mesh", model=m, data=d, post_geom=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "post"),
                        robot_points=rp)
        o = np.r_[cfg["layout_offset"], 0]
        ik = EndpointSolver(kin, "constrained", body)
        base = d.qpos.copy()
        best = None
        for yaw in (0, 90):     # endpoint selection: cube-symmetric yaw with the most clearance at the plate
            t = np.radians(yaw)
            Rt = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]) @ R_DOWN
            qa, _ = ik(np.r_[0.22, 0.0, 0.09] + o, Rt, np.array(cfg["home"], float))
            qb, _ = ik(np.r_[*CLASSICAL["plate_xy"], 0.09] + o, Rt, qa)
            x = base.copy(); x[kin.qadr] = qb
            c = wm.clearance_q(x)
            if best is None or c > best[0]:
                best = (c, qa, qb, yaw)
        _, qa, qb, yaw = best

        def clear(qarm):
            x = base.copy(); x[kin.qadr] = qarm
            return wm.clearance_q(x)

        lo, hi = kin.lo, kin.hi
        sp = Space(lo, hi, clear)
        print(f"\n{body}: yaw {yaw}, straight transit clear: {sp.motion_ok(qa, qb)}")
        for name, fn in PLANNERS.items():
            if only and name not in only:
                continue
            sp = Space(lo, hi, clear, budget_s=3.0, seed=1)
            t0 = time.time()
            try:
                path = fn(sp, qa, qb)
            except Exception as e:
                path = None; print("   error", name, repr(e)[:120])
            dt = time.time() - t0
            ok = path is not None and all(sp.motion_ok(a, b) for a, b in zip(path[:-1], path[1:]))
            print(f"  {name:16s} {'ok ' if ok else 'FAIL'} {dt:5.2f}s  length {path_length(path) if path else float('nan'):5.2f} rad  "
                  f"checks {sp.checks}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
