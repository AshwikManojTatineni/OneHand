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


def sample_layout(rng, body_name, region_x=(0.18, 0.26), region_y=(-0.08, 0.08)):
    """Same draws for every body (same seed), shifted by the body's layout offset."""
    return np.array([rng.uniform(*region_x), rng.uniform(*region_y)]) + np.array(SPECS[body_name]["layout_offset"])
