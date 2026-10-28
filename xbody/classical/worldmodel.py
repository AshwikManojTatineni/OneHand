"""Decision-tree layer 2 (world / target representation): each representation answers one question for the
planners: how far is the robot from the obstacles at configuration q.

  octomap    probabilistic occupancy (log-odds, ray-cast free space) at 1 cm
  voxel      binary occupancy grid at 1 cm
  esdf       Euclidean signed distance field from the voxel grid (also gives gradients for CHOMP)
  tsdf       depth frames fused into a TSDF, surface extracted, then an ESDF
  mesh       exact link-to-scene distances from the known CAD geometry (MuJoCo geom distance)
  primitives obstacle fitted from points as a vertical cylinder
  pose       known obstacle model placed at its estimated pose (centroid of its points)

The robot side is the same for every representation: surface points sampled from the robot's own collision
geometry, i.e. its kinematic self-model.
"""
import mujoco
import numpy as np
from scipy import ndimage

RES = 0.01
POST_R, POST_H = 0.015, 0.12          # CAD of the obstacle post, used by "pose"


class RobotPoints:
    """Surface sample points of the robot's collision geometry, posed by forward kinematics."""

    def __init__(self, model, robot_body_ids, per_geom=40, seed=0):
        rng = np.random.default_rng(seed)
        self.m = model
        self.d = mujoco.MjData(model)
        self.geoms, self.local = [], []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] not in robot_body_ids:
                continue
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            t = model.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
                pts = v[rng.choice(len(v), min(per_geom, len(v)), replace=False)]
            else:
                s = model.geom_size[g]
                dirs = rng.normal(size=(per_geom, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                    pts = dirs * s[0]
                elif t == mujoco.mjtGeom.mjGEOM_CAPSULE or t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    pts = np.c_[dirs[:, :2] / np.linalg.norm(dirs[:, :2], axis=1, keepdims=True) * s[0],
                                rng.uniform(-s[1], s[1], per_geom)]
                elif t == mujoco.mjtGeom.mjGEOM_BOX:
                    pts = rng.uniform(-1, 1, (per_geom, 3)) * s
                else:
                    continue
            self.geoms.append(g); self.local.append(pts)
        self.attached = None                       # (body id, points in that body's frame), e.g. a held cube

    def world(self, qpos_full):
        d = self.d
        d.qpos[:] = qpos_full
        mujoco.mj_kinematics(self.m, d)
        out = [d.geom_xpos[g] + p @ d.geom_xmat[g].reshape(3, 3).T for g, p in zip(self.geoms, self.local)]
        if self.attached is not None:
            b, p = self.attached
            out.append(d.xpos[b] + p @ d.xmat[b].reshape(3, 3).T)
        return np.concatenate(out)


class Grid:
    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, float); self.hi = np.asarray(hi, float)
        self.shape = np.ceil((self.hi - self.lo) / RES).astype(int)

    def idx(self, P):
        i = np.floor((P - self.lo) / RES).astype(int)
        inside = np.all((i >= 0) & (i < self.shape), axis=1)
        return i, inside


class WorldModel:
    """clearance(points) -> per-point distance to the nearest obstacle (m); negative means inside."""

    def __init__(self, kind, obstacle_pts=None, frames=None, model=None, data=None, post_geom=None, bounds=None,
                 robot_points=None):
        self.kind = kind
        self.robot_points = robot_points
        self.grid = Grid(*bounds) if bounds is not None else None
        if kind in ("voxel", "esdf"):
            occ = self._occupancy(obstacle_pts)
            self.occ = occ
            if kind == "esdf":
                self.dist = ndimage.distance_transform_edt(~occ) * RES
        elif kind == "octomap":
            self.occ = self._octomap(frames)
        elif kind == "tsdf":
            occ = self._tsdf(frames)
            self.occ = occ
            self.dist = ndimage.distance_transform_edt(~occ) * RES
        elif kind == "mesh":
            self.m, self.d, self.post = model, data, post_geom
            self.dm = mujoco.MjData(model)
        elif kind in ("primitives", "pose"):
            P = obstacle_pts
            c = P[:, :2].mean(0) if len(P) else np.array([9.0, 9.0])
            if kind == "pose":
                self.cyl = (c, POST_R, POST_H)
            else:
                r = float(np.percentile(np.linalg.norm(P[:, :2] - c, axis=1), 95)) if len(P) else 0.0
                self.cyl = (c, r, float(P[:, 2].max()) if len(P) else 0.0)
        else:
            raise ValueError(kind)
        self.prepare()

    # ---- builders
    def _occupancy(self, P):
        occ = np.zeros(self.grid.shape, bool)
        i, inside = self.grid.idx(P)
        occ[tuple(i[inside].T)] = True
        return occ

    def _octomap(self, frames, hit=0.85, miss=-0.4):
        L = np.zeros(self.grid.shape, np.float32)
        for c, rgb, depth, seg, P, origin in frames:
            sel = np.random.default_rng(0).choice(len(P), min(3000, len(P)), replace=False)
            for p in P[sel]:
                ray = p - origin
                n = int(np.linalg.norm(ray) / RES)
                if n == 0:
                    continue
                ts = np.linspace(0, 1, n, endpoint=False)[:, None]
                i, inside = self.grid.idx(origin + ts * ray)
                L[tuple(i[inside].T)] += miss          # free space along the ray
                j, ins = self.grid.idx(p[None])
                if ins[0]:
                    L[tuple(j[0])] += hit - miss       # endpoint occupied (undo its miss)
        return L > 0.5

    def _tsdf(self, frames, trunc=0.03):
        g = self.grid
        idx = np.indices(g.shape).reshape(3, -1).T
        centers = g.lo + (idx + 0.5) * RES
        tsdf = np.ones(len(centers), np.float32); w = np.zeros(len(centers), np.float32)
        for c, rgb, depth, seg, P, cam in frames:
            R, t, f, H, W = cam
            pc = (centers - t) @ R                    # camera frame, looks along -z
            z = -pc[:, 2]
            ok = z > 0.05
            u = np.round(pc[:, 0] / np.maximum(z, 1e-6) * f + W / 2 - 0.5).astype(int)
            v = np.round(-pc[:, 1] / np.maximum(z, 1e-6) * f + H / 2 - 0.5).astype(int)
            ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
            dmeas = np.full(len(centers), np.inf)
            dmeas[ok] = depth[v[ok], u[ok]]
            sdf = dmeas - z
            upd = ok & (sdf > -trunc) & np.isfinite(dmeas)
            val = np.clip(sdf[upd] / trunc, -1, 1)
            tsdf[upd] = (tsdf[upd] * w[upd] + val) / (w[upd] + 1); w[upd] += 1
        surf = (np.abs(tsdf) < RES / trunc * 1.2) & (w > 0)
        occ = surf.reshape(g.shape)
        occ[:, :, : int(0.008 / RES) + 1] = False     # drop the floor, as for the other representations
        return occ

    # ---- queries
    def clearance(self, P):
        if self.kind in ("voxel", "octomap"):
            i, inside = self.grid.idx(P)
            occ = np.zeros(len(P), bool)
            occ[inside] = self._dil[tuple(i[inside].T)]
            # binary maps carry no distance: occupancy is dilated by the planning margin instead, and a free cell
            # reports "clear" (1 m) so the planner's margin test passes
            return np.where(occ, -RES, 1.0)
        if self.kind in ("esdf", "tsdf"):
            i, inside = self.grid.idx(P)
            out = np.full(len(P), 1.0)
            out[inside] = self.dist[tuple(i[inside].T)] - RES / 2
            return out
        if self.kind in ("primitives", "pose"):
            c, r, h = self.cyl
            radial = np.linalg.norm(P[:, :2] - c, axis=1) - r
            above = P[:, 2] - h
            return np.where(above > 0, np.hypot(np.maximum(radial, 0), above), radial)
        raise RuntimeError("mesh uses clearance_q")

    def prepare(self):
        if self.kind in ("voxel", "octomap"):
            self._dil = ndimage.binary_dilation(self.occ, iterations=3)   # 3 cells = the 2.5 cm planning margin
        if self.kind in ("esdf", "tsdf"):
            self._grad = np.gradient(self.dist, RES)
        return self

    def clearance_q(self, qpos_full):
        """Minimum robot-obstacle distance at a full qpos."""
        if self.kind == "mesh":
            d = self.dm
            d.qpos[:] = qpos_full
            mujoco.mj_kinematics(self.m, d)
            fromto = np.zeros(6)
            best = 1.0
            for g in self.robot_points.geoms:
                best = min(best, mujoco.mj_geomDistance(self.m, d, g, self.post, 0.2, fromto))
            if self.robot_points.attached is not None:
                best = min(best, mujoco.mj_geomDistance(self.m, d, self.cube_geom, self.post, 0.2, fromto))
            return best
        return float(self.clearance(self.robot_points.world(qpos_full)).min())

    def gradient(self, P):
        """Distance-field gradient at points (ESDF/TSDF only), for CHOMP."""
        gx, gy, gz = self._grad
        i, inside = self.grid.idx(P)
        G = np.zeros_like(P)
        ii = tuple(i[inside].T)
        G[inside] = np.c_[gx[ii], gy[ii], gz[ii]]
        return G
