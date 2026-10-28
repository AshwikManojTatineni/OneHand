"""Decision-tree layer 1 (input source): rendered depth point clouds, known geometry, or camera image features.

Everything here is classical: depth back-projection, robot self-filtering from the robot's own model, colour
thresholding for the red cube, and a top-face fit for its pose. No learned component.
"""
import mujoco
import numpy as np

CUBE_EDGE = 0.03


def red_mask(col):
    """Hue test for the red cube, robust to its shadowed faces (dark but still red-dominant)."""
    r, g, b = col[:, 0], col[:, 1], col[:, 2]
    return (r > 0.12) & (r > 1.8 * g) & (r > 1.8 * b)


class Perceiver:
    def __init__(self, model, data, robot_body_ids, H=120, W=160, cams=("scene", "wrist_cam")):
        self.m, self.d, self.H, self.W, self.cams = model, data, H, W, cams
        self.r = mujoco.Renderer(model, H, W)
        self.robot_geoms = set(np.where(np.isin(model.geom_bodyid, list(robot_body_ids)))[0])
        self.cam_ids = {c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in cams}

    def close(self):
        self.r.close()

    def _render(self, cam):
        r = self.r
        r.disable_depth_rendering(); r.disable_segmentation_rendering()
        r.update_scene(self.d, camera=cam); rgb = r.render().copy()
        r.enable_depth_rendering(); r.update_scene(self.d, camera=cam); depth = r.render().copy()
        r.disable_depth_rendering(); r.enable_segmentation_rendering()
        r.update_scene(self.d, camera=cam); seg = r.render()[..., 0].copy()
        r.disable_segmentation_rendering()
        return rgb, depth, seg

    def _backproject(self, cam, depth):
        cid = self.cam_ids[cam]
        fovy = np.radians(self.m.cam_fovy[cid])
        f = 0.5 * self.H / np.tan(fovy / 2)
        u, v = np.meshgrid(np.arange(self.W), np.arange(self.H))
        x = (u - self.W / 2 + 0.5) / f * depth
        y = -(v - self.H / 2 + 0.5) / f * depth
        pc = np.stack([x, y, -depth], -1).reshape(-1, 3)          # camera frame (looks along -z)
        R = self.d.cam_xmat[cid].reshape(3, 3)
        return pc @ R.T + self.d.cam_xpos[cid]

    def frames(self):
        """[(cam, rgb, depth, seg, world points)] for each camera."""
        out = []
        for c in self.cams:
            rgb, depth, seg = self._render(c)
            out.append((c, rgb, depth, seg, self._backproject(c, depth)))
        return out

    def point_cloud(self, frames=None):
        """Scene points with the robot removed (self-filter from the robot's own geometry), plus a cube mask."""
        frames = frames or self.frames()
        pts, red = [], []
        for c, rgb, depth, seg, P in frames:
            valid = (depth.reshape(-1) < 2.0) & ~np.isin(seg.reshape(-1), list(self.robot_geoms))
            col = rgb.reshape(-1, 3).astype(float) / 255
            is_red = red_mask(col)
            pts.append(P[valid]); red.append(is_red[valid])
        return np.concatenate(pts), np.concatenate(red)

    def image_features(self, cam="wrist_cam"):
        """Cube centroid (pixels), area (pixels) and depth at the centroid, from the wrist camera (VS branch)."""
        rgb, depth, seg = self._render(cam)
        col = rgb.astype(float) / 255
        mask = red_mask(col.reshape(-1, 3)).reshape(col.shape[:2])
        if mask.sum() < 4:
            return None
        v, u = np.nonzero(mask)
        return {"u": u.mean(), "v": v.mean(), "area": float(mask.sum()), "depth": float(np.median(depth[mask])),
                "W": self.W, "H": self.H, "f": 0.5 * self.H / np.tan(np.radians(self.m.cam_fovy[self.cam_ids[cam]]) / 2)}


def clusters(P, tol=0.015, min_pts=5):
    """Euclidean cluster extraction: connected components of the radius-tol neighbour graph."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    if len(P) == 0:
        return []
    pairs = cKDTree(P).query_pairs(tol, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    n, lab = connected_components(g, directed=False)
    out = [np.where(lab == k)[0] for k in range(n)]
    return sorted([c for c in out if len(c) >= min_pts], key=len, reverse=True)


def estimate_cube_pose(points, is_red):
    """Cube pose from its red points: known-model (3 cm box) least-squares fit, started at its top-face centre."""
    P = points[is_red & (points[:, 2] > 0.006)]
    if len(P) < 5:
        return None
    P = P[np.random.default_rng(0).choice(len(P), min(len(P), 4000), replace=False)]
    cl = clusters(P, tol=0.01)
    if not cl:
        return None
    # views of one cube can form separate clusters (one camera sees a side, another the top): merge every red
    # cluster whose centroid is within a cube size of the largest one
    c0 = P[cl[0]].mean(0)
    P = P[np.concatenate([c for c in cl if np.linalg.norm(P[c].mean(0) - c0) < 2 * CUBE_EDGE])]
    top = P[P[:, 2] > P[:, 2].max() - 0.002]           # top face only: side faces bias a centroid
    xy0 = 0.5 * (np.percentile(top[:, :2], 5, 0) + np.percentile(top[:, :2], 95, 0))  # robust box centre
    z = max(top[:, 2].mean() - CUBE_EDGE / 2, CUBE_EDGE / 2)
    if top[:, 2].mean() < CUBE_EDGE * 0.8:              # top face not seen: centre is a side-face guess
        z = CUBE_EDGE / 2
    # model fit: centre (and yaw) that put all visible points on the surface of the known 3 cm box
    from scipy.optimize import least_squares

    def resid(x):
        c, yaw = np.r_[x[:2], z], x[2]
        R = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        L = (P - c) @ R.T
        h = CUBE_EDGE / 2
        out = np.abs(L) - h                               # signed distance to the box surface (inside < 0)
        inside = np.all(out < 0, axis=1)
        dist = np.where(inside, out.max(1), np.linalg.norm(np.maximum(out, 0), axis=1))
        return dist
    r = least_squares(resid, np.r_[xy0, 0.0], loss="soft_l1", f_scale=0.002, bounds=([-np.inf, -np.inf, -np.pi / 4], [np.inf, np.inf, np.pi / 4]))
    return np.r_[r.x[:2], z]


def obstacle_points(points, is_red, cube_xyz=None, floor_z=0.008, keep_radius=0.04):
    """Non-floor, non-cube scene points, clustered; clusters next to the target cube are dropped."""
    P = points[(points[:, 2] > floor_z) & ~is_red]
    if len(P) == 0:
        return P
    P = P[np.random.default_rng(0).choice(len(P), min(len(P), 6000), replace=False)]
    keep = []
    for c in clusters(P, tol=0.015, min_pts=8):
        centre = P[c, :2].mean(0)
        if cube_xyz is not None and np.linalg.norm(centre - cube_xyz[:2]) < keep_radius:
            continue
        keep.append(c)
    return P[np.concatenate(keep)] if keep else P[:0]
