"""Phase 0 check of interface K: top-down grasp targets over the task region, on the Franka and the SO-101.

Reports reach rate and orientation residual per IK method (decision-tree layers 3 and 7).
"""
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody.kinematics import Body, solve_ik  # noqa: E402

MEN = ROOT / "assets/mujoco_menagerie"


def so101():
    m = mujoco.MjModel.from_xml_path(str(MEN / "robotstudio_so101/so101.xml"))
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[sid])
    # Canonical TCP convention (shared by all bodies): z = approach direction. The SO-101 site approaches
    # along its x axis, so compose the site frame with a +90 deg turn about y.
    q_canon = np.zeros(4)
    mujoco.mju_mulQuat(q_canon, m.site_quat[sid], np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]))
    return Body(m, ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                body, m.site_pos[sid].copy(), q_canon)


def franka():
    m = mujoco.MjModel.from_xml_path(str(MEN / "franka_emika_panda/panda.xml"))
    return Body(m, [f"joint{i}" for i in range(1, 8)], "hand", np.array([0, 0, 0.1034]))


def run(name, body, q0, targets, R_down):
    print(f"\n{name}: {body.n} arm joints")
    for method, orient in (("newton", "full"), ("dls", "full"), ("hqp", "full"), ("hqp", "axis")):
        pe, re, ok = [], [], 0
        for p in targets:
            q, info = solve_ik(body, p, R_down, q0=q0, method=method, orient=orient)
            pe.append(info["pos_err_m"]); re.append(info["rot_err_rad"]); ok += info["pos_err_m"] < 0.005
        print(f"  {method:6s} {orient:4s} reach<5mm {ok:2d}/{len(targets)}  median pos err "
              f"{1000*np.median(pe):6.1f} mm  median rot residual {np.degrees(np.median(re)):5.1f} deg")


def main():
    rng = np.random.default_rng(20261028)
    # Top-down approach: TCP z axis pointing down.
    so = so101()
    p_home, R_home = so.fk(np.zeros(so.n))
    print("SO-101 TCP at zero pose:", np.round(p_home, 3))
    # Task region in the SO-101 base frame (protocol.yaml), cube centre height 0.015 m above the base plane.
    tgt_so = np.column_stack([rng.uniform(0.18, 0.26, 40), rng.uniform(-0.08, 0.08, 40), np.full(40, 0.015)])
    # Find the TCP orientation that points the jaws down: try both z-axis conventions, keep the reachable one.
    R_down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    q0_so = np.array([0.0, -0.5, 0.8, 0.8, 0.0])
    run("SO-101", so, q0_so, tgt_so, R_down)

    fr = franka()
    tgt_fr = tgt_so + np.array([0.3, 0.0, 0.0])   # same layout, shifted into the Franka's workspace
    q0_fr = np.array([0, -0.4, 0, -2.4, 0, 2.0, 0.785])
    run("Franka", fr, q0_fr, tgt_fr, R_down)


if __name__ == "__main__":
    main()
