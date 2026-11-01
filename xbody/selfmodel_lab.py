"""A more realistic self-modeling lab: the Franka with a sensor head, in a room it can see.

What changes against the first pilot (which handed the self-model a true pose plus noise):
- The hand pose is no longer given. It comes from a real perception pipeline on rendered sensor data:
  monocular (ChArUco board, PnP), stereo (triangulated board corners), or lidar (scan-to-map ICP on the
  room's geometry). A sensor can fail on a frame (board out of view, too few points, degenerate geometry).
- The arm is built from a perturbed copy of the model: link offsets, joint-axis tilts, encoder offsets, base
  placement and sensor-mount errors. The nominal model knows none of them.
- Optional realism on top (levels in REALISM): image noise, lens distortion with imperfect calibration,
  exposure changes, lidar range noise and dropouts, motion blur, sensor latency, joint compliance under gravity.

World frame = the board / room-map frame. Everything a sensor measures is in that frame.
"""
import pathlib

import cv2
import mujoco
import numpy as np

from xbody.bodies import MENAGERIE, SPECS

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/generated"
SPEC = SPECS["franka"]
LINKS = [f"link{i}" for i in range(1, 8)] + ["hand"]

W, H, FOVY = 640, 480, 60.0
BASELINE = 0.06
BOARD = dict(squares=(10, 7), square=0.04, marker=0.03, center=(0.50, 0.0))
# Livox Mid-360-like: 360 deg around its axis, -7 to +52 deg elevation, 2 cm range noise (datasheet scale).
LIDAR = dict(az_step=2.0, el_min=-7.0, el_max=52.0, el_step=2.0, rmin=0.1, rmax=4.0)

# Sensor head on the hand (hand frame: z = approach). Cameras look along +z_hand; MuJoCo cameras look along -z.
CAM_POS = np.array([0.065, -0.03, 0.035])
LIDAR_POS = np.array([-0.065, 0.0, 0.035])
CAM_XYAXES = [0, 1, 0, 1, 0, 0]

REALISM = {
    # image: (noise std in grey levels, blur sigma px, distortion k1, calibration errors on/off)
    # lidar: (range std m, dropout fraction)
    "R0": dict(img_noise=0.0, defocus=0.0, k1=0.0, calib_err=False, exposure=0.0, lidar_std=0.0, dropout=0.0,
               blur_steps=1, latency=0.0, compliance=False),
    "R1": dict(img_noise=3.0, defocus=0.6, k1=-0.12, calib_err=True, exposure=0.15, lidar_std=0.02, dropout=0.03,
               blur_steps=1, latency=0.0, compliance=False),
    "R2": dict(img_noise=3.0, defocus=0.6, k1=-0.12, calib_err=True, exposure=0.15, lidar_std=0.02, dropout=0.03,
               blur_steps=4, latency=0.015, compliance=False),
    "R3": dict(img_noise=3.0, defocus=0.6, k1=-0.12, calib_err=True, exposure=0.15, lidar_std=0.02, dropout=0.03,
               blur_steps=4, latency=0.015, compliance=True),
}
JOINT_STIFFNESS = np.array([2.0e4, 2.0e4, 1.5e4, 1.5e4, 8e3, 8e3, 8e3])   # Nm/rad, harmonic-drive scale


def board_image():
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / "charuco_10x7.png"
    if not path.exists():
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        b = cv2.aruco.CharucoBoard(BOARD["squares"], BOARD["square"], BOARD["marker"], d)
        img = b.generateImage((2000, 1400), marginSize=0)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    return path


def charuco():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    return cv2.aruco.CharucoBoard(BOARD["squares"], BOARD["square"], BOARD["marker"], d)


def _quat_small(rng, deg):
    ax = rng.normal(size=3)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, ax / np.linalg.norm(ax), np.deg2rad(deg) * rng.normal())
    return q


def _mulq(a, b):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.asarray(a, float), np.asarray(b, float))
    return q / np.linalg.norm(q)


ROOM_BOXES = [   # (centre, half sizes, yaw deg): clutter around the board and the room's walls
    ((0.78, 0.30, 0.06), (0.06, 0.05, 0.06), 20), ((0.80, -0.28, 0.09), (0.05, 0.08, 0.09), -15),
    ((0.25, 0.42, 0.04), (0.08, 0.04, 0.04), 35), ((0.30, -0.45, 0.12), (0.04, 0.04, 0.12), 0),
    ((0.62, 0.45, 0.03), (0.10, 0.03, 0.03), -40), ((0.95, 0.0, 0.15), (0.04, 0.15, 0.15), 10),
    # tabletop objects ringing the board (board footprint x 0.30-0.70, y -0.14-0.14 stays clear)
    ((0.50, 0.22, 0.04), (0.05, 0.03, 0.04), 10), ((0.38, -0.22, 0.05), (0.03, 0.03, 0.05), 0),
    ((0.64, -0.21, 0.025), (0.06, 0.04, 0.025), 30), ((0.78, 0.05, 0.07), (0.03, 0.06, 0.07), -20),
    ((0.22, 0.12, 0.03), (0.04, 0.04, 0.03), 45), ((0.24, -0.10, 0.06), (0.025, 0.05, 0.06), 15),
]


def build(seed=None, link_mm=2.0, axis_deg=0.2, base_mm=3.0, base_deg=0.3, mount_mm=1.0, mount_deg=0.3):
    """Compile the lab. seed=None: the nominal robot (what the CAD says). seed=k: the k-th 'real' arm."""
    rng = np.random.default_rng(seed) if seed is not None else None
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / SPEC["xml"]))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    wb = spec.worldbody
    if rng is not None:
        for name in LINKS:
            b = spec.body(name)
            b.pos = np.asarray(b.pos) + rng.normal(0, link_mm / 1000, 3)
            b.quat = _mulq(b.quat, _quat_small(rng, axis_deg))
        base = spec.body("link0")
        base.pos = np.asarray(base.pos) + rng.normal(0, base_mm / 1000, 3)
        base.quat = _mulq(base.quat, _quat_small(rng, base_deg))
    # Sensor head (mount errors only on the real arm).
    hand = spec.body("hand")
    dm = rng.normal(0, mount_mm / 1000, 3) if rng is not None else np.zeros(3)
    head = hand.add_body(name="sensor_head", pos=dm, quat=_quat_small(rng, mount_deg) if rng is not None else [1, 0, 0, 0])
    head.add_camera(name="cam_l", pos=CAM_POS, xyaxes=CAM_XYAXES, fovy=FOVY)
    head.add_camera(name="cam_r", pos=CAM_POS + np.array([0, BASELINE, 0]), xyaxes=CAM_XYAXES, fovy=FOVY)
    head.add_site(name="lidar", pos=LIDAR_POS, quat=[1, 0, 0, 0])
    # Room: floor/table, walls, clutter, the board, lights.
    spec.add_texture(name="board_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(board_image()))
    spec.add_material(name="board_mat", textures=["", "board_tex"], texrepeat=[1, 1], texuniform=False)
    spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=[0.55, 0.5, 0.45], rgb2=[0.5, 0.45, 0.4], width=64, height=64)
    spec.add_material(name="floor_mat", rgba=[0.62, 0.58, 0.52, 1])
    spec.add_material(name="wall_mat", rgba=[0.80, 0.80, 0.78, 1])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], material="floor_mat")
    bw, bh = BOARD["squares"][0] * BOARD["square"], BOARD["squares"][1] * BOARD["square"]
    wb.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[bw / 2, bh / 2, 0.01],
                pos=[BOARD["center"][0], BOARD["center"][1], 0.0005], material="board_mat", contype=0, conaffinity=0)
    for name, pos, size in (("wall_back", [1.25, 0, 0.75], [0.02, 1.4, 0.75]),
                            ("wall_left", [0.2, 0.9, 0.75], [1.1, 0.02, 0.75]),
                            ("wall_right", [0.2, -0.9, 0.75], [1.1, 0.02, 0.75]),
                            ("wall_rear", [-0.85, 0, 0.75], [0.02, 1.4, 0.75])):
        wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, material="wall_mat")
    for i, (c, s, yaw) in enumerate(ROOM_BOXES):
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), np.deg2rad(yaw))
        wb.add_geom(name=f"clutter{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=c, size=s, quat=q,
                    rgba=[0.3 + 0.1 * (i % 3), 0.35, 0.55 - 0.08 * i, 1])
    wb.add_light(name="key", pos=[0.4, 0.3, 2.0], dir=[0, -0.1, -1], castshadow=1,
                 diffuse=[0.7, 0.7, 0.7])
    wb.add_light(name="fill", pos=[-0.5, -0.5, 1.5], dir=[0.5, 0.5, -1], castshadow=0, diffuse=[0.3, 0.3, 0.3])
    m = spec.compile()
    enc = np.deg2rad(rng.normal(0, 0.5, 7)) if rng is not None else np.zeros(7)
    return m, enc


def env_geom_ids(m):
    """Static room geoms (everything not on the robot)."""
    from xbody.bodies import robot_body_ids
    rb = robot_body_ids(m, SPEC)
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] not in rb]
