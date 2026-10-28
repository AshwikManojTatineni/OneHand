"""Body definitions shared by every MuJoCo stack: what interface K and S1 need to know about each arm.

Canonical TCP convention for all bodies: z axis = approach direction.
"""
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MENAGERIE = ROOT / "assets/mujoco_menagerie"

SPECS = {
    "franka": dict(
        xml="franka_emika_panda/panda.xml",
        arm=[f"joint{i}" for i in range(1, 8)],
        tcp_body="hand", tcp_offset=[0, 0, 0.1034], tcp_quat=None,
        home=[0, -0.4, 0, -2.4, 0, 2.0, 0.785],
        gripper_actuator="actuator8", gripper_open=255.0, gripper_closed=0.0,
        jaw_bodies=["left_finger", "right_finger"],
        layout_offset=[0.30, 0.0],   # task region sits inside the Franka's minimum reach otherwise (Phase 0)
        torque_capable=True,
    ),
    "so101": dict(
        xml="robotstudio_so101/so101.xml",
        arm=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        tcp_site="gripperframe",
        home=[0.0, -0.5, 0.8, 0.8, 0.0],
        gripper_actuator="gripper", gripper_open=1.2, gripper_closed=-0.17,
        jaw_bodies=["gripper", "moving_jaw_so101_v1"],
        layout_offset=[0.0, 0.0],
        torque_capable=False,
    ),
}


def tcp_frame(model, spec):
    """(body name, position offset, quaternion offset) of the canonical TCP."""
    if "tcp_site" in spec:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec["tcp_site"])
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.site_bodyid[sid])
        q = np.zeros(4)
        # The SO-101 site approaches along its x axis: turn +90 deg about y so z is the approach.
        mujoco.mju_mulQuat(q, model.site_quat[sid], np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]))
        return body, model.site_pos[sid].copy(), q
    return spec["tcp_body"], np.array(spec["tcp_offset"], float), spec["tcp_quat"]
