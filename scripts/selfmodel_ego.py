"""Ego-centric Bayesian self-model benchmark: the arm learns its own body from one sensor it carries.

Sensors (xbody/selfmodel_sensing.py, the sensor-swap lab of xbody/selfmodel_lab.py): a wrist mono camera (ChArUco
board, PnP), a wrist stereo pair (triangulated board corners), a 3D lidar on the hand (scan-to-map ICP against the
room), and an oracle (true sensor pose + 1 mm / 0.3 deg) as the noise floor. Realism levels R0, R1, R3 of
selfmodel_lab.REALISM (R2 adds motion blur and latency, which move-and-settle capture removes, so it equals R1).
The arm is a perturbed copy of its model file (link, axis, encoder, base and sensor-mount errors); the self-model
never sees the model file.

Per cell (arm, sensor, level, babbling, seed):
  1 learn     random babbling (a random walk with jumps) or active babbling (estimation-exploration: after a random
              start, each batch of poses goes where the posterior is most uncertain and competing structures
              disagree, among poses where the sensor is predicted to see its board); structure by evidence
              (xbody/egoscheme.py), revolute/prismatic choice, optional GP residual
  2 evaluate  structure recovered (order of the action signals along the chain, joint types), forward error of the
              sensor pose on held-out poses, open-loop reach on the real arm (5 mm and 2 deg), against the nominal
              model file and the existing product-of-exponentials fit on the same data
  3 deploy    a session with a disturbance on the encoders (none, constant offset, linear drift); the disturbance
              node is inferred on the first half with the body model fixed and the correction scored on the second

Usage (one arm per process, as the lab reads SELFMODEL_BODY at import):
  python scripts/selfmodel_ego.py --body so101 [--preset full|core|smoke] [--sensors oracle,mono,stereo,lidar]
      [--levels R0,R1,R3] [--babbling random,active] [--seeds 5] [--workers 4] [--quick]
Writes results/selfmodel_ego/cells/<body>_<sensor>_<level>_<babbling>_<seed>.json (resumable) and a summary.
"""
import argparse
import json
import os
import pathlib
import sys
import time


def _body_from_argv():
    for i, a in enumerate(sys.argv):
        if a == "--body" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.environ.get("SELFMODEL_BODY", "franka")


os.environ["SELFMODEL_BODY"] = _body_from_argv()
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "win32" else "egl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import selfmodel_sensors as SS  # noqa: E402  (Arm, REACH, T_WORLD_BOARD, ENC_NOISE)
from xbody import egoscheme as E  # noqa: E402
from xbody import selfmodel_diff as sd  # noqa: E402
from xbody import selfmodel_lab as L  # noqa: E402
from xbody import selfmodel_sensing as S  # noqa: E402

BODY = L.BODY
N = len(L.SPEC["arm"])
OUT = ROOT / "results/selfmodel_ego"
FULL = dict(n_learn=400, n_start=80, batch=20, n_test=100, n_reach=30, n_deploy=200)
QUICK = dict(n_learn=160, n_start=60, batch=20, n_test=40, n_reach=10, n_deploy=80)
DISTURB = dict(constant=np.deg2rad(3.0), linear=np.deg2rad(6.0))
FRAME = {"oracle": "cam", "mono": "cam", "stereo": "cam", "lidar": "lidar"}


class SensorRig:
    """The real arm and one on-body sensor. measure(cmd) moves the arm so its encoders read `cmd`, waits for it to
    settle, and returns the sensor's own estimate of where it is (None when the pipeline gives no pose)."""

    def __init__(self, seed, level, sensor):
        self.cfg = dict(L.REALISM[level], blur_steps=1, latency=0.0)     # move-and-settle capture
        self.sensor, self.level = sensor, level
        self.rng = np.random.default_rng(3000 + seed)
        self.nom = SS.Arm(None)
        self.real = SS.Arm(seed, compliance=self.cfg["compliance"])
        self.rig = S.Rig(self.cfg, self.rng)
        self.lo, self.hi = self.real.lo, self.real.hi
        self.which = FRAME[sensor]
        if sensor in ("mono", "stereo"):
            self.det = S.BoardDetector()
            self.rend = mujoco.Renderer(self.real.m, L.H, L.W)
        if sensor == "lidar":
            self.dirs = S.lidar_dirs()
            self.loc = S.LidarLocalizer(*S.room_map(self.real.m))
            self.prev = None
        self.model = None           # the robot's current self-model, used as the lidar tracker's motion prior

    def true_pose(self, cmd):
        return self.real.fk_joint(np.asarray(cmd, float), self.which)

    def measure(self, cmd):
        qk = self.real.link_angles(np.asarray(cmd, float) - self.real.enc)
        self.real.set(qk)
        G = self.real.frame(self.which)
        if self.sensor == "oracle":
            return sd.pose(G[:3, 3] + self.rng.normal(0, 0.001, 3), sd.so3_exp(self.rng.normal(0, np.deg2rad(0.3), 3)) @ G[:3, :3])
        if self.sensor in ("mono", "stereo"):
            imgs = {}
            for cam in (("cam_l",) if self.sensor == "mono" else ("cam_l", "cam_r")):
                self.rend.update_scene(self.real.d, camera=cam)
                imgs[cam] = S.degrade(self.rend.render(), self.rig, self.rng)
            found = self.det.corners(imgs["cam_l"])
            T = (S.mono_pose(imgs["cam_l"], self.rig, self.det, found=found) if self.sensor == "mono"
                 else S.stereo_pose(imgs["cam_l"], imgs["cam_r"], self.rig, self.det, found_l=found))
            return None if T is None else SS.T_WORLD_BOARD @ T
        scan = S.lidar_scan(self.real.m, self.real.d, self.real.site, self.dirs, self.rig, self.rng)
        if self.model is not None:              # the self-model predicts where its sensor is
            prior = self.model.fk1(np.asarray(cmd, float))
        elif self.prev is not None:
            prior = self.prev
        else:                                   # start-up global registration stand-in: 2 cm / 2 deg off
            prior = sd.pose(G[:3, 3] + self.rng.normal(0, 0.02, 3), sd.so3_exp(self.rng.normal(0, 0.035, 3)) @ G[:3, :3])
        T, _ = self.loc.localize(scan, prior)
        if T is not None:
            self.prev = T
        return T

    def board_visible(self, X):
        """For camera sensors: is the board centre inside the predicted camera's view (with margin)?"""
        if self.sensor not in ("mono", "stereo"):
            return np.ones(len(X), bool)
        c = np.array([L.BOARD["center"][0], L.BOARD["center"][1], 0.0, 1.0])
        p = np.einsum("nij,j->ni", sd.inv_batch(X), c)[:, :3]
        half = np.tan(np.deg2rad(L.FOVY) / 2) * 0.7
        return (p[:, 2] > 0.12) & (np.abs(p[:, 0] / p[:, 2]) < half * L.W / L.H) & (np.abs(p[:, 1] / p[:, 2]) < half)

    def close(self):
        if self.sensor in ("mono", "stereo"):
            self.rend.close()


def babble(rig, n_valid, rng, step=0.06, jump=0.1, max_attempts=4, disturb=None, start=None):
    """Sensor-guided motor babbling: random steps from the last pose where the sensor gave a pose (when it gives
    none, the next step starts again from there), and with probability `jump` a uniformly random pose to explore;
    a jump the sensor cannot see costs one attempt. Stops after n_valid valid frames or max_attempts * n_valid
    tries. Uses only the sensor's own success signal, never the model file.
    Session clock: tau = attempt index / (n_valid - 1), so an encoder disturbance d(tau) and its inference share one
    time axis. Returns commands, encoder readings, sensor poses, jump flags and tau."""
    lo, hi = rig.lo + 0.15 * (rig.hi - rig.lo), rig.hi - 0.15 * (rig.hi - rig.lo)
    anchor = np.clip(np.array(L.SPEC["home"], float) if start is None else start, lo, hi)
    cmds, A, Y, jumps, taus = [], [], [], [], []
    for k in range(max_attempts * n_valid):
        j = rng.random() < jump
        q = rng.uniform(lo, hi) if j else np.clip(anchor + rng.normal(0, step, N), lo, hi)
        y = rig.measure(q)
        tau = k / max(n_valid - 1, 1)
        d = disturb(tau) if disturb is not None else 0.0
        cmds.append(q); Y.append(y); jumps.append(j); taus.append(tau)
        A.append(q + d + rng.normal(0, SS.ENC_NOISE, N))
        if y is not None:
            anchor = q
            if sum(v is not None for v in Y) >= n_valid:
                break
    return np.array(cmds), np.array(A), Y, np.array(jumps), np.array(taus)


def collect(rig, cmds, rng):
    """Encoder readings (cmd + noise) and sensor poses (None where the pipeline failed) at chosen poses."""
    Y = [rig.measure(c) for c in cmds]
    return np.asarray(cmds) + rng.normal(0, SS.ENC_NOISE, np.shape(cmds)), Y


def noise_inflation(chain, Q, T):
    """Errors of a real sensor are correlated along a babbling walk (an ICP registration bias, a board seen from one
    side), which makes a Gaussian evidence overconfident. Scale the variance by (1 + rho) / (1 - rho), rho the
    lag-1 autocorrelation of the residuals in walk order (the effective-sample-size correction for AR(1) errors)."""
    r = E.pose_residuals(chain.fk(Q), T)
    r = r - r.mean(0)
    rho = np.mean([np.corrcoef(r[:-1, i], r[1:, i])[0, 1] for i in range(6) if np.std(r[:, i]) > 0])
    rho = float(np.clip(np.nan_to_num(rho), 0.0, 0.95))
    return (1 + rho) / (1 - rho), rho


def learn(A, Y, jumps, log=None):
    ok = np.array([y is not None for y in Y])
    idx = np.where(ok)[0]
    Q, T = A[idx], np.array([Y[i] for i in idx])
    valid = (np.diff(idx) == 1) & ~jumps[idx[1:]]
    best, ranked, inliers = E.structure_search(Q, T, valid, log=log)
    best = E.choose_types(best, Q, T, inliers, log=log)
    return best, ranked, (Q, T, valid, inliers)


def run_cell(sensor, level, babbling, seed, cfg, log=print):
    t0 = time.time()
    rng = np.random.default_rng(100 + seed)
    rig = SensorRig(seed, level, sensor)
    times = {}
    # ---------------------------------------------------------------- 1 learn
    t = time.time()
    # Lidar localization is tracking: each scan is registered from the previous fix, so until the robot has a self-
    # model to predict where its lidar is, it must babble continuously (a jump makes ICP lock onto the wrong place).
    first_jump = 0.0 if sensor == "lidar" else 0.1
    if babbling == "random":
        cmds, A, Y, jumps, _ = babble(rig, cfg["n_learn"], rng, jump=first_jump)
    else:
        cmds, A, Y, jumps, _ = babble(rig, cfg["n_start"], rng, jump=first_jump)
        lo, hi = rig.lo + 0.15 * (rig.hi - rig.lo), rig.hi - 0.15 * (rig.hi - rig.lo)
        models = None
        while sum(y is not None for y in Y) < cfg["n_learn"]:
            ok = np.array([y is not None for y in Y])
            idx = np.where(ok)[0]
            Q, T = A[idx], np.array([Y[i] for i in idx])
            if models is None:
                best, ranked, _ = learn(A, Y, jumps)
                models = ranked[:3]
            else:
                models = E.refit(models, Q, T)
            rig.model = models[0]
            # candidates near poses that worked (the sensor must still see its target), and anywhere
            seeds = Q[rng.integers(len(Q), size=200)]
            cand = np.clip(np.r_[seeds + rng.normal(0, 0.25, (200, N)), rng.uniform(lo, hi, (100, N))], lo, hi)
            pick = E.choose_next(models, cand, k=cfg["batch"], visible=rig.board_visible)
            An, Yn = collect(rig, cand[pick], rng)
            A, Y = np.r_[A, An], Y + Yn
            jumps = np.r_[jumps, np.ones(len(pick), bool)]
            if len(A) > 4 * cfg["n_learn"]:
                break
    times["collect_s"] = time.time() - t
    n_ok = sum(y is not None for y in Y)
    if n_ok < 40:
        rig.close()
        return dict(body=BODY, sensor=sensor, level=level, babbling=babbling, seed=seed, cfg=cfg, failed=True,
                    valid_frames=int(n_ok), frames=len(Y), reason="sensor gave too few poses to learn from")
    t = time.time()
    best, ranked, (Q, T, valid, inliers) = learn(A, Y, jumps)
    vmask = np.arange(len(Q)) % 5 == 0                      # held-out frames decide the residual GP
    best = E.fit_residual(best, Q[~vmask], T[~vmask], inliers[~vmask], Q[vmask], T[vmask], seed=seed)
    rig.model = best
    times["learn_s"] = time.time() - t
    out = dict(body=BODY, sensor=sensor, level=level, babbling=babbling, seed=seed, cfg=cfg,
               valid_frames=int(len(Q)), frames=int(len(A)),
               structure=dict(order=list(best.order), types="".join(best.types),
                              order_correct=bool(list(best.order) == list(range(N))),
                              types_correct=bool(set(best.types) == {"R"}),
                              runner_up=[dict(order=list(c.order), log_ev=c.log_evidence) for c in ranked[1:3]],
                              log_evidence=best.log_evidence, sigma_mm=best.sigma * 1000),
               fit=best.info)
    poe = None
    try:
        poe = sd.fit_poe(Q, T, "data", valid=valid, robust=True, absolute=True)
    except Exception as ex:
        out["poe_error"] = repr(ex)
    # ---------------------------------------------------------------- 2 evaluate against the true sensor pose
    t = time.time()
    lo, hi = rig.lo + 0.1 * (rig.hi - rig.lo), rig.hi - 0.1 * (rig.hi - rig.lo)
    test = rng.uniform(lo, hi, (cfg["n_test"], N))
    truth = np.array([rig.true_pose(q) for q in test])
    models = dict(bayes=best.fk1, nominal=lambda q: rig.nom.fk_joint(q, rig.which))
    if poe:
        models["poe"] = poe.fk
    out["forward_mm"] = {}
    for k, f in models.items():
        e = [SS.pose_err(f(q), G)[0] for q, G in zip(test, truth)]
        out["forward_mm"][k] = dict(median=float(np.median(e)), p95=float(np.percentile(e, 95)))
    reach = {k: [] for k in models}
    home = np.array(L.SPEC["home"], float)
    while len(reach["bayes"]) < cfg["n_reach"]:
        qt = np.clip(home + rng.normal(0, SS.REACH["sigma"], N), rig.lo + 0.05, rig.hi - 0.05)
        Tt = rig.true_pose(qt)
        if not all(a < Tt[i, 3] < b for i, (a, b) in enumerate(SS.REACH["box"])):
            continue
        for k, f in models.items():
            qc = sd.ik(f, Tt, home, rig.lo, rig.hi)
            reach[k].append(SS.pose_err(rig.true_pose(qc), Tt))
    out["reach"] = {k: dict(success=int(sum(e[0] < 5 and e[1] < 2 for e in v)), n=len(v),
                            median_mm=float(np.median([e[0] for e in v]))) for k, v in reach.items()}
    times["evaluate_s"] = time.time() - t
    # ---------------------------------------------------------------- 3 deploy with an encoder disturbance
    out["deploy"] = {}
    t = time.time()
    infl, rho = noise_inflation(best, Q, T)
    sigma_eff = best.sigma * np.sqrt(infl)
    out["noise"] = dict(sigma_mm=best.sigma * 1000, residual_autocorr=rho, sigma_eff_mm=sigma_eff * 1000)
    for kind in ("none", "constant", "linear"):
        drng = np.random.default_rng(900 + seed)
        c = drng.normal(0, DISTURB["constant"], N) if kind == "constant" else np.zeros(N)
        v = drng.normal(0, DISTURB["linear"], N) if kind == "linear" else np.zeros(N)
        cmds_d, Ad, Yd, _, tau = babble(rig, cfg["n_deploy"], drng, jump=0.3, disturb=lambda s, c=c, v=v: c + v * s)
        half = len(Ad) // 2
        sel, fits = E.select_disturbance(best, Ad[:half], tau[:half], Yd[:half], sigma=sigma_eff)
        truth_d = np.array([rig.true_pose(q) for q in cmds_d[half:]])
        raw = best.fk(Ad[half:])
        cor = best.fk(Ad[half:] - sel["c"] - np.outer(tau[half:], sel["v"]))
        out["deploy"][kind] = dict(
            selected=sel["kind"], log_evidence={k: f["log_evidence"] for k, f in fits.items()},
            c_true_deg=np.rad2deg(c).tolist(), c_est_deg=np.rad2deg(sel["c"]).tolist(),
            v_true_deg=np.rad2deg(v).tolist(), v_est_deg=np.rad2deg(sel["v"]).tolist(),
            posterior_sd_deg=np.rad2deg(sel["sd"]).tolist(),
            error_mm_uncorrected=float(np.median([SS.pose_err(P_, G)[0] for P_, G in zip(raw, truth_d)])),
            error_mm_corrected=float(np.median([SS.pose_err(P_, G)[0] for P_, G in zip(cor, truth_d)])))
    times["deploy_s"] = time.time() - t
    rig.close()
    out["times"], out["seconds"] = times, time.time() - t0
    s = out["structure"]
    log(f"{BODY} {sensor} {level} {babbling} seed {seed}: order {''.join(map(str, s['order']))} "
        f"{'ok' if s['order_correct'] else 'WRONG'} types {s['types']} | valid {out['valid_frames']}/{out['frames']} | "
        f"fwd bayes {out['forward_mm']['bayes']['median']:.2f} poe {out['forward_mm'].get('poe', {}).get('median', float('nan')):.2f} "
        f"nominal {out['forward_mm']['nominal']['median']:.2f} mm | reach {out['reach']['bayes']['success']}/{out['reach']['bayes']['n']} | "
        + " ".join(f"{k}->{d['selected']} {d['error_mm_uncorrected']:.1f}->{d['error_mm_corrected']:.1f}mm" for k, d in out["deploy"].items())
        + f" ({out['seconds']:.0f}s)", flush=True)
    return out


def _job(args):
    sensor, level, babbling, seed, cfg, force = args
    path = OUT / "cells" / f"{BODY}_{sensor}_{level}_{babbling}_{seed}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    r = run_cell(sensor, level, babbling, seed, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=1, default=float))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=BODY)
    ap.add_argument("--sensors", default="oracle,mono,stereo,lidar")
    ap.add_argument("--levels", default="R0,R1,R3")
    ap.add_argument("--babbling", default="random,active")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preset", choices=["full", "core", "smoke"], default=None,
                    help="full: R0,R1,R3 x 5 seeds; core: R1,R3 x 3 seeds; smoke: R1 x 1 seed (all sensors, both babbling)")
    a = ap.parse_args()
    if a.preset:
        a.levels, a.seeds = {"full": ("R0,R1,R3", 5), "core": ("R1,R3", 3), "smoke": ("R1", 1)}[a.preset]
    cfg = dict(QUICK if a.quick else FULL)
    jobs = [(s, lv, b, seed, cfg, a.force) for s in a.sensors.split(",") for lv in a.levels.split(",")
            for b in a.babbling.split(",") for seed in range(a.seeds)]
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        cells = pool.map(_job, jobs)
    summary = {}
    for c in cells:
        key = f"{c['sensor']}_{c['level']}_{c['babbling']}"
        if c.get("failed"):
            summary.setdefault("failed", []).append(dict(cell=key, seed=c["seed"], valid=c["valid_frames"], frames=c["frames"]))
            continue
        s = summary.setdefault(key, dict(cells=0, order_correct=0, types_correct=0, fwd_bayes=[], fwd_poe=[], fwd_nominal=[],
                                           reach_bayes=0, reach_n=0, deploy={k: dict(right=0, unc=[], cor=[]) for k in ("none", "constant", "linear")}))
        s["cells"] += 1
        s["order_correct"] += c["structure"]["order_correct"]
        s["types_correct"] += c["structure"]["types_correct"]
        for k in ("bayes", "poe", "nominal"):
            if k in c["forward_mm"]:
                s[f"fwd_{k}"].append(c["forward_mm"][k]["median"])
        s["reach_bayes"] += c["reach"]["bayes"]["success"]
        s["reach_n"] += c["reach"]["bayes"]["n"]
        for k, d in c["deploy"].items():
            s["deploy"][k]["right"] += d["selected"] == k
            s["deploy"][k]["unc"].append(d["error_mm_uncorrected"])
            s["deploy"][k]["cor"].append(d["error_mm_corrected"])
    for key, s in summary.items():
        if key == "failed":
            continue
        for k in ("fwd_bayes", "fwd_poe", "fwd_nominal"):
            s[k] = float(np.median(s[k])) if s[k] else None
        for d in s["deploy"].values():
            d["unc"], d["cor"] = float(np.median(d["unc"])), float(np.median(d["cor"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"summary_{BODY}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
