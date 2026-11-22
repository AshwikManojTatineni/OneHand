"""Visual self-perception for the body-scheme self-model: fiducial markers on the robot's body parts, seen by fixed
cameras (Sturm et al. track ARToolkit markers on each link with one monocular camera).

Two perception modes return the same thing, one 4x4 pose per marker per frame (NaN when it is not seen):
  camera   ArUco markers rendered on each part, detected in the images of fixed calibrated cameras, pose by planar
           PnP (IPPE square); the camera with the most frontal view wins
  oracle   the marker's true pose plus camera-like noise (depth noisier than the image plane, as Sturm reports) and
           dropout when the marker faces away from every camera; fast, for long sweeps
"""
import pathlib

import cv2
import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/generated"
DICT = cv2.aruco.DICT_4X4_50
BACK = 16                                # id offset of each marker's back face
MAX_RMS_PX = 1.0                         # reprojection error accepted for a marker pose
AMBIGUITY = 3.0                          # the second pose solution must reproject this much worse
FUSE_MM, FUSE_DEG = 6.0, 4.0             # views of one marker further than this from the consensus are dropped
CV_FLIP = np.diag([1.0, -1.0, -1.0])    # MuJoCo camera frame (looks -z, y up) -> OpenCV (looks +z, y down)

PARTS = {   # body parts that carry a marker, base first; size of the marker's black square (m)
    "franka": (["link0", "link2", "link3", "link4", "link5", "link6", "link7", "hand"], 0.06),
    "so101": (["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper"], 0.03),
}


def marker_png(k):
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"aruco4x4_{k}_q60.png"
    if not path.exists():
        img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICT), k, 240, borderBits=1)
        img = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)   # wide white quiet zone
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    return path


def _look_quat(direction):
    """Quaternion of a frame whose +z points along `direction` (the marker's face normal)."""
    z = direction / np.linalg.norm(direction)
    x = np.cross([0, 0, 1.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.c_[x, y, z].ravel())
    return q


def add_markers(spec, body_name, home, arm_joints):
    """Add a marker on a short bracket to every part, facing the cameras at the home pose, and three fixed cameras.
    Returns the marker names and the camera names."""
    parts, size = PARTS[body_name]
    m0 = spec.copy().compile()
    d0 = mujoco.MjData(m0)
    for j, q in zip(arm_joints, home):
        d0.qpos[m0.jnt_qposadr[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, j)]] = q
    mujoco.mj_forward(m0, d0)
    bids = [mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, p) for p in parts]
    centre = np.mean([d0.xpos[b] for b in bids], 0)
    span = max(np.linalg.norm(d0.xpos[b] - centre) for b in bids) + 0.15
    cams = [centre + span * 1.7 * np.array([np.cos(a), np.sin(a), 0.55]) for a in np.deg2rad([-35, 25, 85, 145, 205, 265])]
    eye = np.mean(cams, 0)
    names = []
    for k, (p, b) in enumerate(zip(parts, bids)):
        tex = f"mk_tex{k}"
        spec.add_texture(name=tex, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(marker_png(k)))
        spec.add_material(name=f"mk_mat{k}", textures=["", tex], texuniform=False, texrepeat=[1, 1])
        R = d0.xmat[b].reshape(3, 3)
        # Mount on the part's own geometry (link frames of neighbouring parts often share an origin): start at the
        # centroid of its geoms, face outward from the arm and toward the cameras, and stand clear of its surface.
        geoms = [g for g in range(m0.ngeom) if m0.geom_bodyid[g] == b and m0.geom_rbound[g] > 0]
        cg = np.mean([d0.geom_xpos[g] for g in geoms], 0) if geoms else d0.xpos[b].copy()
        away = cg - centre
        away = away / np.linalg.norm(away) if np.linalg.norm(away) > 1e-6 else np.zeros(3)
        to_eye = (eye - cg) / np.linalg.norm(eye - cg)
        out_w = away + 1.5 * to_eye
        out_w /= np.linalg.norm(out_w)
        reach = max([float((d0.geom_xpos[g] - cg) @ out_w) + 0.9 * m0.geom_rbound[g] for g in geoms] or [0.03])
        local_dir = R.T @ out_w
        pos = R.T @ (cg - d0.xpos[b]) + local_dir * (reach + 0.006)
        mk = spec.body(p).add_body(name=f"marker{k}", pos=pos, quat=_look_quat(local_dir))
        full = size * 360 / 240                          # texture includes the white quiet zone
        # Double-sided: a box face shows its texture mirrored from behind, so the back carries its own id (k + 16)
        # on a second face turned 180 degrees about x. Either side identifies the same part.
        tex_b = f"mk_tex{k + BACK}"
        spec.add_texture(name=tex_b, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(marker_png(k + BACK)))
        spec.add_material(name=f"mk_mat{k + BACK}", textures=["", tex_b], texuniform=False, texrepeat=[1, 1])
        mk.add_geom(name=f"marker{k}_face", type=mujoco.mjtGeom.mjGEOM_BOX, size=[full / 2, full / 2, 0.0008],
                    pos=[0, 0, 0.0009], material=f"mk_mat{k}", contype=0, conaffinity=0, group=3, mass=0)
        mk.add_geom(name=f"marker{k}_back", type=mujoco.mjtGeom.mjGEOM_BOX, size=[full / 2, full / 2, 0.0008],
                    pos=[0, 0, -0.0009], quat=[0, 1, 0, 0], material=f"mk_mat{k + BACK}", contype=0, conaffinity=0,
                    group=3, mass=0)
        names.append(f"marker{k}")
    cam_names = []
    for c, pos in enumerate(cams):
        fwd = centre - pos
        fwd /= np.linalg.norm(fwd)
        xr = np.cross(fwd, [0, 0, 1.0]); xr /= np.linalg.norm(xr)
        up = np.cross(xr, fwd)
        spec.worldbody.add_camera(name=f"self_cam{c}", pos=pos, xyaxes=np.r_[xr, up], fovy=45)
        cam_names.append(f"self_cam{c}")
    return names, cam_names, size


class Perception:
    """Observed marker-body poses, one 4x4 per part per frame (NaN when unseen). Both faces of a marker map to the
    same marker-body frame, so the self-model sees one consistent frame per part."""

    def __init__(self, model, data, marker_names, cam_names, size, mode="oracle", W=1280, H=960, rng=None,
                 noise=(0.001, 0.003, 0.5), dropout=0.03):
        self.m, self.d, self.mode = model, data, mode
        self.mk = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in marker_names]
        self.cams = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in cam_names]
        self.size, self.W, self.H = size, W, H
        self.rng = rng or np.random.default_rng(0)
        self.noise, self.dropout = noise, dropout
        # each face's pose in its marker body, G, and the texture's frame on a face, F0 (calibrated once)
        self.G = {}
        for k, n in enumerate(marker_names):
            for side, offset in (("face", 0), ("back", BACK)):
                g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{n}_{side}")
                T = np.eye(4)
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, model.geom_quat[g])
                T[:3, :3], T[:3, 3] = R.reshape(3, 3), model.geom_pos[g]
                self.G[k + offset] = (k, T)
        self.F0 = np.eye(4)
        if mode == "camera":
            model.vis.global_.offwidth = max(W, model.vis.global_.offwidth)
            model.vis.global_.offheight = max(H, model.vis.global_.offheight)
            self.r = mujoco.Renderer(model, H, W)
            self.opt = mujoco.MjvOption()
            self.opt.geomgroup[:] = 1
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)
            f = 0.5 * H / np.tan(np.deg2rad(45) / 2)
            self.K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
            h = size / 2
            self.obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)

    def true_poses(self):
        out = []
        for b in self.mk:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = self.d.xmat[b].reshape(3, 3), self.d.xpos[b]
            out.append(T)
        return out

    def _cam_T(self, c):
        T = np.eye(4)
        T[:3, :3] = self.d.cam_xmat[c].reshape(3, 3) @ CV_FLIP
        T[:3, 3] = self.d.cam_xpos[c]
        return T

    def observe(self):
        """One frame: a 4x4 world pose per marker body, NaN where no camera sees either face."""
        return self._oracle() if self.mode == "oracle" else self._camera()

    def _oracle(self):
        sxy, sz, srot = self.noise
        out = []
        views = [self._cam_T(c) for c in self.cams]
        for T in self.true_poses():
            # double-sided marker: seen when either face turns toward a camera, at more than ~75 degrees off-axis
            facing = [abs(float(V[:3, 2] @ T[:3, 2])) for V in views]
            if max(facing) < 0.25 or self.rng.random() < self.dropout:
                out.append(np.full((4, 4), np.nan)); continue
            Rc = views[int(np.argmax(facing))][:3, :3]
            e = Rc @ np.array([self.rng.normal(0, sxy), self.rng.normal(0, sxy), self.rng.normal(0, sz)])
            w = self.rng.normal(0, np.deg2rad(srot), 3)
            K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            U, _, Vt = np.linalg.svd(np.eye(3) + K + K @ K / 2)
            Tn = T.copy()
            Tn[:3, :3] = (U @ Vt) @ T[:3, :3]
            Tn[:3, 3] += e
            out.append(Tn)
        return out

    def _detections(self):
        """(marker index, face id, world pose of the PnP frame, image area) for every decoded marker face."""
        found = []
        for c in self.cams:
            self.r.update_scene(self.d, camera=c, scene_option=self.opt)
            grey = cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2GRAY)
            corners, ids, _ = self.det.detectMarkers(grey)
            if ids is None:
                continue
            Tc = self._cam_T(c)
            for cc, fid in zip(corners, ids.ravel()):
                if int(fid) not in self.G:
                    continue
                # A planar square has two pose solutions that mirror its tilt; keep the detection only when the best
                # one reprojects well and clearly beats the other (the flipped one corrupts every relative pose)
                n, rvecs, tvecs, errs = cv2.solvePnPGeneric(self.obj, cc.reshape(4, 2).astype(np.float64), self.K,
                                                            None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if n == 0:
                    continue
                errs = np.ravel(errs)
                order = np.argsort(errs)
                if errs[order[0]] > MAX_RMS_PX or (n > 1 and errs[order[1]] < AMBIGUITY * max(errs[order[0]], 1e-3)):
                    continue
                rvec, tvec = rvecs[order[0]], tvecs[order[0]]
                Rm, _ = cv2.Rodrigues(rvec)
                Tm = np.eye(4)
                Tm[:3, :3], Tm[:3, 3] = Rm, tvec.ravel()
                found.append((self.G[int(fid)][0], int(fid), Tc @ Tm, cv2.contourArea(cc.reshape(4, 2))))
        return found

    def _camera(self):
        """Fuse every camera's view of each marker (both faces map to the same body frame): drop views more than
        FUSE_MM from the median position, then average the rest (positions mean, rotations chordal mean). A marker
        seen by one camera only is kept as is."""
        views = [[] for _ in self.mk]
        for k, fid, Tw, area in self._detections():
            views[k].append(Tw @ np.linalg.inv(self.G[fid][1] @ self.F0))
        out = []
        for vs in views:
            if not vs:
                out.append(np.full((4, 4), np.nan)); continue
            P = np.array([v[:3, 3] for v in vs])
            med = np.median(P, 0)
            ok = np.linalg.norm(P - med, axis=1) <= FUSE_MM / 1000
            if len(vs) >= 3 and ok.sum() == 0:
                out.append(np.full((4, 4), np.nan)); continue
            sel = [v for v, o in zip(vs, ok) if o] or vs
            if len(sel) > 1:     # rotation outliers too: keep views within FUSE_DEG of the chordal mean
                U, _, Vt = np.linalg.svd(np.mean([v[:3, :3] for v in sel], 0))
                Rm = U @ Vt
                ang = [np.degrees(np.arccos(np.clip((np.trace(Rm.T @ v[:3, :3]) - 1) / 2, -1, 1))) for v in sel]
                sel = [v for v, a in zip(sel, ang) if a <= FUSE_DEG] or sel
            T = np.eye(4)
            T[:3, 3] = np.mean([v[:3, 3] for v in sel], 0)
            U, _, Vt = np.linalg.svd(np.mean([v[:3, :3] for v in sel], 0))
            T[:3, :3] = U @ Vt
            out.append(T)
        return out

    def calibrate_frame(self):
        """Camera mode: F0, the frame PnP reports relative to the textured face (fixed by the texture mapping),
        averaged over every face seen in this view."""
        if self.mode != "camera":
            return
        true = self.true_poses()
        Fs = [np.linalg.inv(true[k] @ self.G[fid][1]) @ Tw for k, fid, Tw, _ in self._detections()]
        if Fs:
            U, _, Vt = np.linalg.svd(np.mean([F[:3, :3] for F in Fs], 0))
            self.F0 = np.eye(4)
            self.F0[:3, :3] = U @ Vt

    def close(self):
        if self.mode == "camera":
            self.r.close()
