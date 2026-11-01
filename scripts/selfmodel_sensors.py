"""Sensor-swap self-modeling benchmark on one robot (Franka): monocular vs stereo vs lidar.

Same arm, same teleop, same joint trajectory; only the sensor that measures the hand changes. Each sensor's
pose comes out of its own perception pipeline on rendered data (xbody/selfmodel_sensing.py) in a lab with a
calibration board, clutter and walls (xbody/selfmodel_lab.py). The arm differs from its model file by sampled
calibration errors, including where the base and the sensor head sit.

Sensors: oracle (true camera pose + 1 mm / 0.3 deg noise, the first pilot's sensor), mono, stereo, lidar.
Realism ladder (selfmodel_lab.REALISM): R0 ideal renders, R1 + image noise, lens distortion with imperfect
calibration, exposure changes, lidar range noise and dropouts, R2 + motion blur and 15 ms sensor latency,
R3 + joint compliance under gravity (encoders read the motor side).

Per sensor: valid-frame rate, pose error of the sensor itself, then the self-model (differential fit + absolute
refinement, robust loss) against the nominal model: FK error of the sensor frame over the joint range and
open-loop reach (IK on the model, run on the real arm, success within 5 mm and 2 deg).

Usage: MUJOCO_GL=osmesa python scripts/selfmodel_sensors.py [--quick]
"""
import json
import pathlib
import sys
import time
from multiprocessing import Pool

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from selfmodel_teleop import teleop_commands  # noqa: E402
from xbody import selfmodel_diff as sd  # noqa: E402
from xbody import selfmodel_lab as L  # noqa: E402
from xbody import selfmodel_sensing as S  # noqa: E402
from xbody.kinematics import Body  # noqa: E402
from xbody.stats import wilson  # noqa: E402

OUT = ROOT / "results/selfmodel_sensors"
SENSORS = ["oracle", "mono", "stereo", "lidar"]
FRAME = {"oracle": "cam", "mono": "cam", "stereo": "cam", "lidar": "lidar"}
DT, EXPOSURE = 0.1, 0.008      # teleop log period, camera exposure (s)
ENC_NOISE = 1e-4
REACH_TOL = (0.005, 2.0)
T_WORLD_BOARD = sd.pose(np.array([L.BOARD["center"][0] - L.BOARD["squares"][0] * L.BOARD["square"] / 2,
                                  L.BOARD["center"][1] + L.BOARD["squares"][1] * L.BOARD["square"] / 2, 0.0]),
                        np.diag([1.0, -1.0, -1.0]))   # board corner origin, x along +x, y along -y (measured once)


class Arm:
    """One compiled lab (nominal or real) with frame lookups."""

    def __init__(self, seed, compliance=False):
        self.m, self.enc = L.build(seed)
        self.d = mujoco.MjData(self.m)
        m = self.m
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in L.SPEC["arm"]]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jids])
        self.lo, self.hi = m.jnt_range[jids, 0].copy(), m.jnt_range[jids, 1].copy()
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_l")
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        self.compliance = compliance

    def link_angles(self, q_motor):
        """Joint compliance: the links sag under gravity away from what the motor-side encoders read."""
        if not self.compliance:
            return q_motor
        d = self.d
        d.qpos[self.qadr] = q_motor
        d.qvel[:] = 0
        mujoco.mj_forward(self.m, d)
        return q_motor - d.qfrc_bias[self.dadr] / L.JOINT_STIFFNESS

    def set(self, q_link):
        self.d.qpos[self.qadr] = q_link
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)

    def frame(self, which):
        d = self.d
        if which == "cam":
            return sd.pose(d.cam_xpos[self.cam].copy(), d.cam_xmat[self.cam].reshape(3, 3) @ S.CV_FLIP)
        return sd.pose(d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy())

    def fk_joint(self, q_joint, which):
        """Pose of a frame when the joints read q_joint (encoder coordinates, true robot physics)."""
        self.set(self.link_angles(q_joint - self.enc))
        return self.frame(which)


def record(seed, mode, level, steps, rng_seed):
    """Teleop the real arm and log encoders plus every sensor's measurement for each frame."""
    cfg = L.REALISM[level]
    rng = np.random.default_rng(rng_seed)
    nom = Arm(None)
    real = Arm(seed, compliance=cfg["compliance"])
    drive = Body(nom.m, L.SPEC["arm"], L.SPEC["tcp_body"], L.SPEC["tcp_offset"])
    cmd = teleop_commands(mode, steps, rng, drive)
    q_motor = cmd - real.enc
    rig = S.Rig(cfg, rng)
    det = S.BoardDetector()
    rend = mujoco.Renderer(real.m, L.H, L.W)
    dirs = S.lidar_dirs()
    loc = S.LidarLocalizer(*S.room_map(real.m))
    log = {"Q": cmd + rng.normal(0, ENC_NOISE, cmd.shape), "true": {"cam": [], "lidar": []},
           "meas": {s: [] for s in SENSORS}, "lidar_status": []}
    T_lidar_prev = None
    for k in range(steps):
        qk = real.link_angles(q_motor[k])
        real.set(qk)
        log["true"]["cam"].append(real.frame("cam"))
        log["true"]["lidar"].append(real.frame("lidar"))
        G = log["true"]["cam"][-1]
        log["meas"]["oracle"].append(sd.pose(G[:3, 3] + rng.normal(0, 0.001, 3),
                                             sd.so3_exp(rng.normal(0, np.deg2rad(0.3), 3)) @ G[:3, :3]))
        # What the sensors saw: captured `latency` before the encoder stamp, blurred over the exposure.
        prev = q_motor[k - 1] if k else q_motor[k]
        subs = []
        for j in range(cfg["blur_steps"]):
            back = cfg["latency"] + (EXPOSURE * j / max(cfg["blur_steps"] - 1, 1) if cfg["blur_steps"] > 1 else 0)
            subs.append(real.link_angles(q_motor[k] + (prev - q_motor[k]) * min(back / DT, 1.0)))
        imgs = {"cam_l": 0.0, "cam_r": 0.0}
        for qs in subs:
            real.set(qs)
            for cam in imgs:
                rend.update_scene(real.d, camera=cam)
                imgs[cam] = imgs[cam] + rend.render().astype(np.float32) / len(subs)
        gl = S.degrade(imgs["cam_l"].astype(np.uint8), rig, rng)
        gr = S.degrade(imgs["cam_r"].astype(np.uint8), rig, rng)
        found = det.corners(gl)
        Tm = S.mono_pose(gl, rig, det, found=found)
        Ts = S.stereo_pose(gl, gr, rig, det, found_l=found)
        log["meas"]["mono"].append(None if Tm is None else T_WORLD_BOARD @ Tm)
        log["meas"]["stereo"].append(None if Ts is None else T_WORLD_BOARD @ Ts)
        real.set(subs[0])
        scan = S.lidar_scan(real.m, real.d, real.site, dirs, rig, rng)
        if T_lidar_prev is None:   # stand-in for a global registration step at start-up: 2 cm / 2 deg off
            Gl = log["true"]["lidar"][-1]
            T_lidar_prev = sd.pose(Gl[:3, 3] + rng.normal(0, 0.02, 3), sd.so3_exp(rng.normal(0, 0.035, 3)) @ Gl[:3, :3])
        Tl, st = loc.localize(scan, T_lidar_prev)
        log["lidar_status"].append(st)
        log["meas"]["lidar"].append(Tl)
        if Tl is not None:
            T_lidar_prev = Tl
    rend.close()
    return log, nom, real


def pose_err(A, B):
    return np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000, np.rad2deg(np.linalg.norm(sd.so3_log(A[:3, :3].T @ B[:3, :3])))


def fk_errors(fk, real, which, n=300, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = real.lo + 0.1 * (real.hi - real.lo), real.hi - 0.1 * (real.hi - real.lo)
    e = np.array([pose_err(fk(q), real.fk_joint(q, which)) for q in rng.uniform(lo, hi, (n, 7)) + real.enc])
    return {"pos_mm_median": float(np.median(e[:, 0])), "pos_mm_p95": float(np.percentile(e[:, 0], 95)),
            "rot_deg_median": float(np.median(e[:, 1]))}


def reach(fk, real, which, n=40, seed=1):
    rng = np.random.default_rng(seed)
    home = np.array(L.SPEC["home"])
    hits, errs = [], []
    while len(hits) < n:
        qt = np.clip(home + rng.normal(0, 0.5, 7), real.lo + 0.05, real.hi - 0.05)
        Tt = real.fk_joint(qt + real.enc, which)
        tcp_z = Tt[2, 3]
        if not (0.2 < Tt[0, 3] < 0.75 and abs(Tt[1, 3]) < 0.35 and 0.15 < tcp_z < 0.65):
            continue
        q_cmd = sd.ik(fk, Tt, home, real.lo, real.hi)
        ep, er = pose_err(real.fk_joint(q_cmd, which), Tt)
        errs.append((ep, er))
        hits.append(bool(ep < REACH_TOL[0] and er < REACH_TOL[1]))
    e = np.array(errs)
    return {"success": int(sum(hits)), "n": n, "hits": hits, "pos_mm_median": float(np.median(e[:, 0]))}


def run_job(args):
    seed, mode, level, steps = args
    t0 = time.time()
    log, nom, real = record(seed, mode, level, steps, rng_seed=1000 + seed)
    Q = log["Q"]
    out = {"seed": seed, "mode": mode, "level": level, "steps": steps, "record_s": time.time() - t0,
           "lidar_status": {s: log["lidar_status"].count(s) for s in set(log["lidar_status"])}}
    fk_nom = {w: (lambda q, w=w: nom.fk_joint(q, w)) for w in ("cam", "lidar")}
    for w in ("cam", "lidar"):
        out[f"nominal_{w}"] = {"fk": fk_errors(fk_nom[w], real, w), "reach": reach(fk_nom[w], real, w)}
    for s in SENSORS:
        w = FRAME[s]
        meas = log["meas"][s]
        ok = np.array([T is not None for T in meas])
        r = {"valid": int(ok.sum()), "frames": len(meas)}
        if ok.sum():
            e = np.array([pose_err(T, G) for T, G in zip(meas, log["true"][w]) if T is not None])
            r["sensor_pos_mm_median"], r["sensor_pos_mm_p95"] = float(np.median(e[:, 0])), float(np.percentile(e[:, 0], 95))
            r["sensor_rot_deg_median"] = float(np.median(e[:, 1]))
        if ok.sum() >= 60:
            idx = np.where(ok)[0]
            Qv, Tv = Q[idx], np.array([meas[i] for i in idx])
            pairs = np.diff(idx) == 1
            t1 = time.time()
            for name, kw in (("diff_only", {}), ("diff_abs", {"absolute": True})):
                try:
                    model = sd.fit_poe(Qv, Tv, "data", valid=pairs, robust=True, **kw)
                    r[name] = {"fk": fk_errors(model.fk, real, w), "reach": reach(model.fk, real, w)}
                except Exception as ex:   # e.g. too few consecutive pairs for the initial Jacobian
                    r[name] = {"error": repr(ex)}
            r["fit_s"] = time.time() - t1
        out[s] = r
    out["total_s"] = time.time() - t0
    return out


def main(quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1] if quick else [0, 1, 2, 3, 4]
    levels = ["R1"] if quick else ["R1", "R0", "R2", "R3"]
    modes = ["cartesian"] if quick else ["cartesian", "joint"]
    steps = 150 if quick else 400
    jobs = [(s, mo, lv, steps) for lv in levels for mo in modes for s in seeds]
    path = OUT / ("cells_quick.json" if quick else "cells.json")
    done = json.loads(path.read_text()) if path.exists() else []
    have = {(c["seed"], c["mode"], c["level"], c["steps"]) for c in done}
    jobs = [j for j in jobs if j not in have]
    with Pool(4) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            done.append(r)
            path.write_text(json.dumps(done, indent=1))
            line = " ".join(f"{s}: {r[s]['valid']}/{r[s]['frames']} "
                            + (f"{r[s]['diff_abs']['fk']['pos_mm_median']:.1f}mm {r[s]['diff_abs']['reach']['success']}/40"
                               if "fk" in r[s].get("diff_abs", {}) else "-") for s in SENSORS)
            print(f"{r['level']} {r['mode']:9s} seed={r['seed']} nominal {r['nominal_cam']['fk']['pos_mm_median']:.1f}mm "
                  f"{r['nominal_cam']['reach']['success']}/40 | {line} ({r['total_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main("--quick" in sys.argv)
