"""MuJoCo task scene for the single benchmark task (protocol.yaml): one robot, a floor, a cube and a plate zone.

Object placement is given in the robot base frame: every body gets the same layout shape, shifted by a
body-specific offset when the shared region lies outside its workspace (the Franka cannot reach it).
"""
import mujoco
import numpy as np

from xbody.bodies import MENAGERIE, SPECS


def build(body_name, cube_edge=0.03, cube_mass=0.03, plate_xy=(0.20, 0.14), plate_r=0.06, kp_scale=1.0):
    spec_cfg = SPECS[body_name]
    off = np.array(spec_cfg["layout_offset"], float)
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / spec_cfg["xml"]))
    wb = spec.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.05], rgba=[0.8, 0.8, 0.8, 1])
    cube = wb.add_body(name="cube", pos=[0.22, 0.0, cube_edge / 2])
    cube.add_freejoint(name="cube_free")
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[cube_edge / 2] * 3, mass=cube_mass,
                  friction=[1.5, 0.01, 0.001], rgba=[0.9, 0.1, 0.1, 1])
    wb.add_site(name="plate", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[plate_xy[0] + off[0], plate_xy[1] + off[1], 0.001],
                size=[plate_r, 0.001, 0], rgba=[0.2, 0.6, 0.2, 0.4])
    if kp_scale != 1.0:   # actuation-gap condition: rescale position-servo stiffness (and damping with it)
        for act in spec.actuators:
            act.gainprm[0] *= kp_scale
            act.biasprm[1] *= kp_scale
            act.biasprm[2] *= np.sqrt(kp_scale)
    model = spec.compile()
    return model


CLASSICAL = dict(plate_xy=(0.20, 0.21), post_xy=(0.21, 0.135), post_r=0.015, post_h=0.12)


def build_task(body_name, obstacle=True, torque=False, kp_scale=1.0, cube_edge=0.03, cube_mass=0.03, plate_r=0.05):
    """Scene for the full classical benchmark: the plate sits behind an obstacle post, so a straight transit
    collides and the planning layer matters. Adds a scene camera ("scene") and makes sure there is a wrist camera
    on the hand ("wrist_cam"). torque=True turns the arm's position servos into torque motors
    (layer 9); the gripper stays position-controlled.
    """
    from xbody.bodies import tcp_frame
    cfg = SPECS[body_name]
    off = np.array(cfg["layout_offset"], float)
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / cfg["xml"]))
    wb = spec.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.05], rgba=[0.8, 0.8, 0.8, 1])
    wb.add_light(pos=[0.3 + off[0], 0, 1.2], dir=[0, 0, -1])
    cube = wb.add_body(name="cube", pos=[0.22 + off[0], off[1], cube_edge / 2])
    cube.add_freejoint(name="cube_free")
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[cube_edge / 2] * 3, mass=cube_mass,
                  friction=[1.5, 0.01, 0.001], rgba=[0.9, 0.1, 0.1, 1])
    px, py = CLASSICAL["plate_xy"]
    wb.add_site(name="plate", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[px + off[0], py + off[1], 0.001],
                size=[plate_r, 0.001, 0], rgba=[0.2, 0.6, 0.2, 0.4])
    if obstacle:
        ox, oy = CLASSICAL["post_xy"]
        wb.add_geom(name="post", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[ox + off[0], oy + off[1], CLASSICAL["post_h"] / 2],
                    size=[CLASSICAL["post_r"], CLASSICAL["post_h"] / 2, 0], rgba=[0.2, 0.3, 0.9, 1])
    cx = 0.22 + off[0]
    wb.add_camera(name="scene", pos=[cx + 0.55, off[1] + 0.06, 0.75], xyaxes=[0, 1, 0, -0.8, 0, 0.6], fovy=45)
    if torque:
        for act in spec.actuators:
            if act.name == cfg["gripper_actuator"]:
                continue
            fr = act.forcerange.copy()
            act.gainprm[:3] = [1, 0, 0]
            act.biasprm[:3] = [0, 0, 0]
            act.ctrlrange = fr
            act.ctrllimited = 1
    if kp_scale != 1.0:
        for act in spec.actuators:
            act.gainprm[0] *= kp_scale
            act.biasprm[1] *= kp_scale
            act.biasprm[2] *= np.sqrt(kp_scale)
    if "wrist_cam" not in [c.name for c in spec.cameras]:
        # Franka: no camera in the model, so mount one on the side of the hand aimed at the point between the
        # fingers (the SO-101 model already carries "wrist_cam").
        pos = np.array([0.065, 0.0, 0.03])
        look = np.array([0.0, 0.0, 0.1034]) - pos
        zc = -look / np.linalg.norm(look)                  # MuJoCo cameras look along -z
        xc = np.cross([0, 1.0, 0], zc); xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        spec.body(cfg["tcp_body"]).add_camera(name="wrist_cam", pos=pos, xyaxes=np.r_[xc, yc], fovy=75)
    return spec.compile()


def sample_layout(rng, body_name, region_x=(0.18, 0.26), region_y=(-0.08, 0.08)):
    """Same draws for every body (same seed), shifted by the body's layout offset."""
    return np.array([rng.uniform(*region_x), rng.uniform(*region_y)]) + np.array(SPECS[body_name]["layout_offset"])
