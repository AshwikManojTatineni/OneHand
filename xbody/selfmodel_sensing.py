"""Sensor pipelines for the self-modeling lab: from rendered data to a measured sensor pose in the world frame.

  mono    left camera image -> ChArUco corners -> PnP (planar IPPE, then LM refinement)
  stereo  left + right images -> corners matched by id -> triangulated with the calibrated rig -> Kabsch onto
          the board model
  lidar   wrist lidar ranges (mj_multiRay against the room only) -> point-to-plane ICP onto a map of the room,
          tracked from the previous frame's estimate

Every pipeline sees only what a real one would: the image or the ranges, its own (imperfect) calibration and the
board or room model. It returns None when it cannot produce a pose it trusts.
"""
import cv2
import mujoco
import numpy as np
from scipy.spatial import cKDTree

from xbody import selfmodel_diff as sd
from xbody import selfmodel_lab as L

CV_FLIP = np.diag([1.0, -1.0, -1.0])   # MuJoCo camera frame (y up, looks -z) -> OpenCV (y down, looks +z)


def intrinsics():
    f = (L.H / 2) / np.tan(np.deg2rad(L.FOVY) / 2)
    return np.array([[f, 0, L.W / 2], [0, f, L.H / 2], [0, 0, 1.0]])


class Rig:
    """True camera/lidar parameters and the calibrated estimates a real system would carry."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.K = intrinsics()
        self.D = np.array([cfg["k1"], 0.25 * -cfg["k1"], 0, 0, 0.0])
        if cfg["calib_err"]:
            self.K_est = self.K.copy()
            self.K_est[0, 0] *= 1 + rng.normal(0, 0.002)
            self.K_est[1, 1] = self.K_est[0, 0] * (1 + rng.normal(0, 0.0005))
            self.K_est[:2, 2] += rng.normal(0, 0.5, 2)
            self.D_est = self.D * (1 + rng.normal(0, 0.02, 5))
            self.R_lr_est = sd.so3_exp(rng.normal(0, np.deg2rad(0.03), 3))
            self.t_lr_est = np.array([-L.BASELINE, 0, 0]) + rng.normal(0, 0.0002, 3)
        else:
            self.K_est, self.D_est = self.K.copy(), self.D.copy()
            self.R_lr_est, self.t_lr_est = np.eye(3), np.array([-L.BASELINE, 0, 0])
        # Distortion map: for each pixel of the distorted image, where it comes from in the ideal render.
        u, v = np.meshgrid(np.arange(L.W, dtype=np.float32), np.arange(L.H, dtype=np.float32))
        pts = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
        if np.any(self.D):
            und = cv2.undistortPoints(pts, self.K, self.D, P=self.K).reshape(L.H, L.W, 2)
            self.mapx, self.mapy = und[..., 0].astype(np.float32), und[..., 1].astype(np.float32)
        else:
            self.mapx = None


def degrade(img, rig, rng):
    """Lens distortion, exposure change, defocus and sensor noise on an ideal RGB render; returns grey."""
    c = rig.cfg
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if rig.mapx is not None:
        g = cv2.remap(g, rig.mapx, rig.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if c["exposure"]:
        g = g * (1 + rng.uniform(-c["exposure"], c["exposure"]))
    if c["defocus"]:
        g = cv2.GaussianBlur(g, (0, 0), c["defocus"])
    if c["img_noise"]:
        g = g + rng.normal(0, c["img_noise"], g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


class BoardDetector:
    def __init__(self):
        self.board = L.charuco()
        self.det = cv2.aruco.CharucoDetector(self.board)
        self.obj_all = self.board.getChessboardCorners()   # board frame, metres

    def corners(self, grey):
        cc, ids, _, _ = self.det.detectBoard(grey)
        if ids is None or len(ids) < 6:
            return None, None
        return cc.reshape(-1, 2), ids.ravel()


def mono_pose(grey, rig, det, min_corners=8, max_rms=1.5, found=None):
    """T_board_cam (OpenCV camera frame) from one image, or None. found: corners already detected."""
    img, ids = found if found is not None else det.corners(grey)
    if img is None or len(ids) < min_corners:
        return None
    obj = det.obj_all[ids].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img.astype(np.float64), rig.K_est, rig.D_est, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img.astype(np.float64), rig.K_est, rig.D_est, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, rig.K_est, rig.D_est)
    if np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(1).mean()) > max_rms:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return sd.inv(sd.pose(tvec.ravel(), R))   # PnP gives board in camera; invert to camera in board


def stereo_pose(grey_l, grey_r, rig, det, min_common=8, found_l=None):
    il, idl = found_l if found_l is not None else det.corners(grey_l)
    ir, idr = det.corners(grey_r)
    if il is None or ir is None:
        return None
    common, a, b = np.intersect1d(idl, idr, return_indices=True)
    if len(common) < min_common:
        return None
    nl = cv2.undistortPoints(il[a].reshape(-1, 1, 2).astype(np.float64), rig.K_est, rig.D_est).reshape(-1, 2)
    nr = cv2.undistortPoints(ir[b].reshape(-1, 1, 2).astype(np.float64), rig.K_est, rig.D_est).reshape(-1, 2)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([rig.R_lr_est, rig.t_lr_est[:, None]])
    X = cv2.triangulatePoints(P1, P2, nl.T, nr.T)
    X = (X[:3] / X[3]).T                                   # left camera frame
    if np.any(X[:, 2] <= 0):
        return None
    obj = det.obj_all[common].astype(np.float64)
    R, t = kabsch(obj, X)                                  # X = R obj + t: board in camera
    if np.sqrt((((obj @ R.T + t) - X) ** 2).sum(1).mean()) > 0.01:
        return None
    return sd.inv(sd.pose(t, R))


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    S = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    return R, cb - R @ ca


# ---------- lidar ----------
def lidar_dirs():
    """Ray directions in the lidar frame: a full turn about its z axis (mounted along the approach axis, so it
    points down when the gripper does), elevation measured from the plane normal to z, positive toward +z."""
    c = L.LIDAR
    a = np.deg2rad(np.arange(0, 360, c["az_step"]))
    e = np.deg2rad(np.arange(c["el_min"], c["el_max"] + 1e-9, c["el_step"]))
    aa, ee = np.meshgrid(a, e)
    return np.stack([np.cos(ee) * np.cos(aa), np.cos(ee) * np.sin(aa), np.sin(ee)], -1).reshape(-1, 3)


def lidar_scan(m, d, site_id, dirs, rig, rng):
    """Points in the lidar frame. Rays hit the room only (geom group 0): the robot is filtered as a real driver
    would with a self-mask."""
    c = rig.cfg
    R = d.site_xmat[site_id].reshape(3, 3)
    p = d.site_xpos[site_id].copy()
    vec = (dirs @ R.T).ravel()
    n = len(dirs)
    geomid = np.zeros(n, np.int32)
    dist = np.zeros(n)
    mujoco.mj_multiRay(m, d, p, vec, np.array([1, 0, 0, 0, 0, 0], np.uint8), True, -1, geomid, dist, None, n,
                       L.LIDAR["rmax"])
    ok = (geomid >= 0) & (dist > L.LIDAR["rmin"])
    if c["dropout"]:
        ok &= rng.random(n) > c["dropout"]
    r = dist[ok] + (rng.normal(0, c["lidar_std"], ok.sum()) if c["lidar_std"] else 0)
    return dirs[ok] * r[:, None]


def room_map(m, spacing=0.01):
    """Points and normals on the room's visible surfaces (floor, inner wall faces, clutter boxes)."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    P, N = [], []

    def grid(c, u, v, nu, nv, n):
        a = np.arange(-nu, nu + 1e-9, spacing)
        b = np.arange(-nv, nv + 1e-9, spacing)
        aa, bb = np.meshgrid(a, b)
        pts = c + aa.ravel()[:, None] * u + bb.ravel()[:, None] * v
        P.append(pts)
        N.append(np.repeat(n[None], len(pts), 0))

    grid(np.array([0.2, 0, 0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 1.03, 0.88, np.array([0, 0, 1.0]))
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not (name.startswith("wall") or name.startswith("clutter")):
            continue
        c, R, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        for ax in range(3):
            for sign in (-1, 1):
                n = R[:, ax] * sign
                if n[2] < -0.5:
                    continue                      # bottom face, on the floor
                fc = c + n * s[ax]
                if name.startswith("wall") and np.dot(n, -fc * np.array([1, 1, 0])) <= 0:
                    continue                      # outer face of a wall
                u, v = [R[:, k] for k in range(3) if k != ax]
                su, sv = [s[k] for k in range(3) if k != ax]
                grid(fc, u, v, su, sv, n)
    return np.concatenate(P), np.concatenate(N)


class LidarLocalizer:
    """Point-to-plane ICP of each scan onto the room map, started from the previous estimate."""

    def __init__(self, map_pts, map_nrm):
        self.P, self.N = map_pts, map_nrm
        self.tree = cKDTree(map_pts)

    def localize(self, scan, T0, iters=30, min_pts=300, max_pts=3000, seed=0):
        if len(scan) < min_pts:
            return None, "few_points"
        if len(scan) > max_pts:
            scan = scan[np.random.default_rng(seed).choice(len(scan), max_pts, replace=False)]
        T = T0.copy()
        for it in range(iters):
            thr = 0.10 if it < 10 else 0.03
            X = scan @ T[:3, :3].T + T[:3, 3]
            dd, idx = self.tree.query(X, distance_upper_bound=thr)
            ok = np.isfinite(dd)
            if ok.sum() < min_pts:
                return None, "few_matches"
            x, q, n = X[ok], self.P[idx[ok]], self.N[idx[ok]]
            A = np.hstack([np.cross(x, n), n])           # d/d(omega, t) of n.(x - q) for a world-frame twist
            b = -np.einsum("ij,ij->i", n, x - q)
            w = np.minimum(1.0, 0.02 / np.maximum(np.abs(b), 1e-9))   # Huber: points off the map count less
            H = A.T @ (A * w[:, None])
            xi = np.linalg.solve(H + 1e-9 * np.eye(6), A.T @ (b * w))
            dR = sd.so3_exp(xi[:3])
            T = sd.pose(dR @ T[:3, 3] + xi[3:], dR @ T[:3, :3])
            if np.linalg.norm(xi) < 1e-6:
                break
        # Degeneracy: can the geometry pin all six directions? Rotation columns are about the sensor (lever arms of
        # metres would hide a free rotation), so recentre before the eigenvalue test.
        Ac = np.hstack([np.cross(x - T[:3, 3], n), n]) * np.sqrt(w)[:, None]
        ev = np.linalg.eigvalsh(Ac.T @ Ac / ok.sum())
        if ev[0] < 2e-3 * ev[-1]:
            return None, "degenerate"
        # Slip check: after convergence most of the scan must lie on the map.
        X = scan @ T[:3, :3].T + T[:3, 3]
        dd, idx = self.tree.query(X, distance_upper_bound=0.05)
        ok = np.isfinite(dd)
        r = np.abs(np.einsum("ij,ij->i", self.N[idx[ok]], X[ok] - self.P[idx[ok]]))
        if ok.mean() < 0.85 or np.median(r) > 0.03:
            return None, "poor_fit"
        return T, "ok"
