"""Phase 0 check of layers 1-2: cube pose error and each world representation's clearance at a free and a
colliding configuration, on both bodies."""
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
from xbody.classical.perception import Perceiver, estimate_cube_pose, obstacle_points  # noqa: E402
from xbody.classical.worldmodel import RobotPoints, WorldModel  # noqa: E402
from xbody.kinematics import Body, solve_ik  # noqa: E402
from xbody.scene import CLASSICAL, build_task  # noqa: E402


def setup(body):
    cfg = SPECS[body]
    m = build_task(body)
    d = mujoco.MjData(m)
    bname, off, q = tcp_frame(m, cfg)
    kin = Body(m, cfg["arm"], bname, off, q)
    d.qpos[kin.qadr] = cfg["home"]
    mujoco.mj_forward(m, d)
    return cfg, m, d, kin


def main():
    for body in ["franka", "so101"]:
        cfg, m, d, kin = setup(body)
        rb = robot_body_ids(m, cfg)
        per = Perceiver(m, d, rb)
        t0 = time.time()
        frames = per.frames()
        P, red = per.point_cloud(frames)
        cube_est = estimate_cube_pose(P, red)
        cube_true = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")]
        obs = obstacle_points(P, red, cube_est)
        print(f"\n{body}: {len(P)} points, perception {time.time() - t0:.2f}s, cube error "
              f"{1000 * np.linalg.norm(cube_est - cube_true):.1f} mm")
        off = np.array(cfg["layout_offset"])
        lo = [off[0] - 0.05, -0.25, 0.0]; hi = [off[0] + 0.45, 0.40, 0.35]
        ext = []
        for c, rgb, depth, seg, Pw in frames:
            cid = per.cam_ids[c]
            R = d.cam_xmat[cid].reshape(3, 3); t = d.cam_xpos[cid].copy()
            f = 0.5 * per.H / np.tan(np.radians(m.cam_fovy[cid]) / 2)
            keep = (Pw[:, 2] > 0.008) & ~np.isin(seg.reshape(-1), list(per.robot_geoms)) & (depth.reshape(-1) < 2)
            ext.append((c, rgb, np.where(np.isin(seg, list(per.robot_geoms)), np.inf, depth), seg, Pw[keep], t))
        rp = RobotPoints(m, rb)
        post = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "post")
        # colliding configuration: TCP pushed into the post
        ox, oy = CLASSICAL["post_xy"]
        q_hit, _ = solve_ik(kin, np.array([ox + off[0], oy + off[1], 0.06]), None, q0=np.array(cfg["home"], float))
        q_free = d.qpos.copy()
        q_col = d.qpos.copy(); q_col[kin.qadr] = q_hit
        for kind in ["octomap", "voxel", "esdf", "tsdf", "mesh", "primitives", "pose"]:
            t1 = time.time()
            frames_k = ext if kind == "octomap" else [
                (c, rgb, dep, seg, Pw, (d.cam_xmat[per.cam_ids[c]].reshape(3, 3), d.cam_xpos[per.cam_ids[c]].copy(),
                                        0.5 * per.H / np.tan(np.radians(m.cam_fovy[per.cam_ids[c]]) / 2), per.H, per.W))
                for c, rgb, dep, seg, Pw, _ in ext]
            wm = WorldModel(kind, obstacle_pts=obs, frames=frames_k, model=m, data=d, post_geom=post,
                            bounds=(lo, hi), robot_points=rp)
            build = time.time() - t1
            t2 = time.time()
            cf, cc = wm.clearance_q(q_free), wm.clearance_q(q_col)
            print(f"  {kind:10s} build {build:5.2f}s  query {1000 * (time.time() - t2) / 2:6.1f} ms  "
                  f"home {100 * cf:6.1f} cm  into-post {100 * cc:6.1f} cm  {'OK' if cf > 0 and cc < 0.005 else 'CHECK'}")
        per.close()


if __name__ == "__main__":
    main()
