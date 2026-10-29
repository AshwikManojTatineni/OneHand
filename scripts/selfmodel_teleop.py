"""Single-robot pilot (Franka): self-model from teleoperation and differential sensor changes.

A teleoperator drives the arm through its kinematic model (the nominal Menagerie Panda). The real arm differs
from that model by typical calibration errors (link offsets, joint-axis tilts, encoder zero offsets). It logs
encoder readings and the measured pose of a frame on the hand (wrist-camera extrinsics), with sensor noise.
A second model, which never sees the model file, rebuilds the kinematics from those logs (xbody/selfmodel_diff.py).

Compared:
  nominal     the kinematic model file itself (what every other method in the benchmark assumes)
  diff_only   differential PoE fit only (screws from pose changes; M averaged), initialised from the data alone
  diff_abs    the same, then refined on the absolute logged poses (still no model file)
  calib       nominal model as the starting point, refined on the same logs (classical calibration)
  local_jac   model-free online Jacobian (recursive least squares), one-step prediction only
  newt        the current self-model readout (frozen TD-MPC2 latent), on the same recorded ManiSkill episodes

Metrics: one-step relative error (squared error of the predicted change / squared size of the change, the
same normalisation as the current readout), absolute FK error on random configurations, and an open-loop
reach test: solve IK on the model, run the joints on the real arm, measure where the hand really ends up.

Usage: python scripts/selfmodel_teleop.py [--quick]
"""
import json
import pathlib
import sys
import time

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody import selfmodel_diff as sd  # noqa: E402
from xbody.bodies import MENAGERIE, SPECS  # noqa: E402
from xbody.kinematics import Body, solve_ik  # noqa: E402
from xbody.stats import wilson  # noqa: E402

OUT = ROOT / "results/selfmodel_teleop"
SPEC = SPECS["franka"]
LINKS = [f"link{i}" for i in range(1, 8)] + ["hand"]
NOISE = {"none": (0.0, 0.0), "low": (0.0005, 0.1), "mid": (0.001, 0.3), "high": (0.003, 1.0)}   # (m, deg)
ENC_NOISE = 1e-4   # rad, ~ a 16-bit absolute encoder
REACH_TOL = (0.005, 2.0)   # m, deg


def nominal_body():
    m = mujoco.MjModel.from_xml_path(str(MENAGERIE / SPEC["xml"]))
    return Body(m, SPEC["arm"], SPEC["tcp_body"], SPEC["tcp_offset"])


def real_robot(seed, link_mm=2.0, axis_deg=0.2, enc_deg=0.5, tcp_mm=1.0):
    """The physical arm: the same model with sampled calibration errors, plus encoder zero offsets."""
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_path(str(MENAGERIE / SPEC["xml"]))
    for name in LINKS:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        m.body_pos[b] += rng.normal(0, link_mm / 1000, 3)
        ax = rng.normal(size=3)
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, ax / np.linalg.norm(ax), np.deg2rad(axis_deg) * rng.normal())
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, m.body_quat[b].copy(), dq)
        m.body_quat[b] = q / np.linalg.norm(q)
    tcp = np.array(SPEC["tcp_offset"]) + rng.normal(0, tcp_mm / 1000, 3)
    body = Body(m, SPEC["arm"], SPEC["tcp_body"], tcp)
    body.enc_offset = np.deg2rad(rng.normal(0, enc_deg, 7))
    return body


def T_of(body, q):
    p, R = body.fk(q)
    return sd.pose(p, R)


def measure(rng, T, noise):
    sp, sr = NOISE[noise]
    R = sd.so3_exp(rng.normal(0, np.deg2rad(sr), 3)) @ T[:3, :3]
    return sd.pose(T[:3, 3] + rng.normal(0, sp, 3), R)


def teleop(mode, steps, seed, nom, real, noise):
    """Returns encoder log Q (encoder coordinates), true joint angles, true and measured hand poses."""
    rng = np.random.default_rng(seed)
    lo, hi = nom.lo + 0.15 * (nom.hi - nom.lo), nom.hi - 0.15 * (nom.hi - nom.lo)
    home = np.array(SPEC["home"])
    cmd = [home.copy()]
    if mode == "joint":
        # Joint-space teleop: wiggle each joint about home, then a random walk through the workspace.
        for i in range(7):
            for a in np.sin(np.linspace(0, 2 * np.pi, 13)[1:]) * 0.3:
                c = home.copy(); c[i] += a; cmd.append(c)
        q = home.copy()
        while len(cmd) < steps:
            q = np.clip(q + rng.normal(0, 0.04, 7), lo, hi); cmd.append(q.copy())
    else:
        # Cartesian teleop: the operator moves the gripper between waypoints, pointing roughly down; the
        # driver turns each 1 cm / 2 deg increment into joints with the nominal model's IK.
        q = home.copy()
        p, R = nom.fk(q)
        while len(cmd) < steps:
            pt = np.array([rng.uniform(0.3, 0.6), rng.uniform(-0.25, 0.25), rng.uniform(0.15, 0.5)])
            Rt = sd.so3_exp(np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), 0])) @ \
                np.diag([1, -1, -1]) @ sd.so3_exp(np.array([0, 0, rng.uniform(-1.5, 1.5)]))
            dr = sd.so3_log(Rt @ R.T)
            k = int(max(np.linalg.norm(pt - p) / 0.01, np.linalg.norm(dr) / np.deg2rad(2), 1))
            for s in range(1, k + 1):
                pi, Ri = p + (pt - p) * s / k, sd.so3_exp(dr * s / k) @ R
                q, _ = solve_ik(nom, pi, Ri, q0=q, method="dls", iters=50)
                cmd.append(q.copy())
                if len(cmd) >= steps:
                    break
            p, R = nom.fk(q)
    cmd = np.array(cmd[:steps])
    q_true = cmd - real.enc_offset              # the arm reaches what its encoders read as cmd
    Q = cmd + rng.normal(0, ENC_NOISE, cmd.shape)
    T_true = np.array([T_of(real, q) for q in q_true])
    T_meas = np.array([measure(rng, T, noise) for T in T_true])
    return Q, q_true, T_true, T_meas


def nominal_J(nom, q):
    """Nominal spatial Jacobian at q (rotation rows scaled), for the calibration initialisation."""
    p, R, jp, jr = nom.jacobian(q)
    # mj_jac gives the TCP point velocity; the spatial twist's v is the velocity of the world origin point.
    return np.vstack([jr * sd.L_ROT, jp - np.cross(jr.T, p).T])


class NominalModel:
    def __init__(self, nom):
        self.nom = nom

    def fk(self, q):
        return T_of(self.nom, q)


def one_step(fk_model, Q, T_true):
    """Relative one-step error of predicted TCP position and rotation changes (same normalisation as the
    current readout: mean squared error / mean squared change)."""
    P = np.array([fk_model(q) for q in Q])
    dp_pred, dp = np.diff(P[:, :3, 3], axis=0), np.diff(T_true[:, :3, 3], axis=0)
    dr_pred = sd.rotvec_batch(P[1:, :3, :3] @ np.transpose(P[:-1, :3, :3], (0, 2, 1)))
    dr = sd.rotvec_batch(T_true[1:, :3, :3] @ np.transpose(T_true[:-1, :3, :3], (0, 2, 1)))
    return {"rel_pos": float(((dp_pred - dp) ** 2).sum(1).mean() / (dp ** 2).sum(1).mean()),
            "rel_rot": float(((dr_pred - dr) ** 2).sum(1).mean() / (dr ** 2).sum(1).mean())}


def one_step_local(lj, Q, T_true, T_meas):
    """The model-free baseline predicts each step, then updates on the measured (noisy) change."""
    ep, er, sp, sr = [], [], [], []
    for k in range(len(Q) - 1):
        dq = Q[k + 1] - Q[k]
        pred = lj.predict(dq)
        tru = sd.delta_x(T_true[k], T_true[k + 1])
        ep.append(((pred[:3] - tru[:3]) ** 2).sum()); sp.append((tru[:3] ** 2).sum())
        er.append(((pred[3:] - tru[3:]) ** 2).sum() / sd.L_ROT ** 2); sr.append((tru[3:] ** 2).sum() / sd.L_ROT ** 2)
        lj.update(dq, sd.delta_x(T_meas[k], T_meas[k + 1]))
    return {"rel_pos": float(np.mean(ep) / np.mean(sp)), "rel_rot": float(np.mean(er) / np.mean(sr))}


def fk_errors(fk_model, real, rng, n=300):
    lo, hi = real.lo + 0.1 * (real.hi - real.lo), real.hi - 0.1 * (real.hi - real.lo)
    ep, er = [], []
    for _ in range(n):
        qt = rng.uniform(lo, hi)
        T, P = T_of(real, qt), fk_model(qt + real.enc_offset)
        ep.append(np.linalg.norm(P[:3, 3] - T[:3, 3]) * 1000)
        er.append(np.rad2deg(np.linalg.norm(sd.so3_log(T[:3, :3].T @ P[:3, :3]))))
    return {"pos_mm_median": float(np.median(ep)), "pos_mm_p95": float(np.percentile(ep, 95)),
            "rot_deg_median": float(np.median(er)), "rot_deg_p95": float(np.percentile(er, 95))}


def reach(fk_model, real, rng, n=40):
    """Open-loop reach: IK on the model, run on the real arm. Targets are poses the real arm can reach,
    inside the Cartesian teleop workspace."""
    lo, hi = real.lo, real.hi
    home = np.array(SPEC["home"])
    errs, hits = [], []
    while len(errs) < n:
        qt = np.clip(home + rng.normal(0, 0.5, 7), lo + 0.05, hi - 0.05)
        Tt = T_of(real, qt)
        if not (0.25 < Tt[0, 3] < 0.65 and abs(Tt[1, 3]) < 0.3 and 0.1 < Tt[2, 3] < 0.55):
            continue
        q_cmd = sd.ik(fk_model, Tt, home, lo, hi)
        Ta = T_of(real, q_cmd - real.enc_offset)
        e_p = np.linalg.norm(Ta[:3, 3] - Tt[:3, 3])
        e_r = np.rad2deg(np.linalg.norm(sd.so3_log(Tt[:3, :3].T @ Ta[:3, :3])))
        errs.append((e_p * 1000, e_r))
        hits.append(bool(e_p < REACH_TOL[0] and e_r < REACH_TOL[1]))
    e, ok = np.array(errs), int(sum(hits))
    lo_, hi_ = wilson(ok, n)
    return {"success": ok, "n": n, "ci": [lo_, hi_], "hits": hits, "pos_mm_median": float(np.median(e[:, 0])),
            "rot_deg_median": float(np.median(e[:, 1]))}


def run_cell(seed, mode, steps, noise, nom, evaluate_reach=True):
    real = real_robot(seed)
    Q, qt, Tt, Tm = teleop(mode, steps, 1000 + seed, nom, real, noise)
    Qh, _, Tth, Tmh = teleop(mode, 200, 5000 + seed, nom, real, noise)   # held-out teleop log
    out = {"seed": seed, "mode": mode, "steps": steps, "noise": noise}
    models = {"nominal": NominalModel(nom).fk}
    t0 = time.time()
    diff = sd.fit_poe(Q, Tm, "data")
    models["diff_only"] = diff.fk
    out["fit_s_diff_only"] = time.time() - t0
    t0 = time.time()
    models["diff_abs"] = sd.fit_poe(Q, Tm, "data", absolute=True).fk
    out["fit_s_diff_abs"] = time.time() - t0
    models["calib"] = sd.fit_poe(Q, Tm, "nominal", nominal_J=nominal_J(nom, Q[0]), absolute=True).fk
    for name, fk in models.items():
        # Same test configurations and reach targets for every model, so comparisons are paired.
        r = {"one_step": one_step(fk, Qh, Tth), "fk": fk_errors(fk, real, np.random.default_rng(9000 + seed))}
        if evaluate_reach:
            r["reach"] = reach(fk, real, np.random.default_rng(9500 + seed))
        out[name] = r
    lj = sd.LocalJacobian(7)
    for k in range(len(Q) - 1):
        lj.update(Q[k + 1] - Q[k], sd.delta_x(Tm[k], Tm[k + 1]))
    out["local_jac"] = {"one_step": one_step_local(lj, Qh, Tth, Tmh)}
    return out


def maniskill_same_data(noise="none", seed=0):
    """The current readout's own data: 20 recorded ManiSkill Franka pick-cube episodes. Fit on 10, test on the
    other 10 (both folds), against the true logged poses. ManiSkill state obs: qpos 0:9, tcp pose 19:26 (wxyz)."""
    d = np.load(ROOT / "results/phase0/traj/franka_home.npz", allow_pickle=True)
    eps = [np.asarray(o, float) for o in d["obs"]]

    def to_T(o):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, o[22:26] / np.linalg.norm(o[22:26]))
        return sd.pose(o[19:22], R.reshape(3, 3))

    def stack(idx):
        Q = np.concatenate([eps[i][:, :7] for i in idx])
        T = np.array([to_T(o) for i in idx for o in eps[i]])
        valid = np.concatenate([np.r_[np.ones(len(eps[i]) - 1, bool), False] for i in idx])[:-1]
        return Q, T, valid

    rng = np.random.default_rng(seed)
    res = {"noise": noise}
    for name, (tr, te) in {"fold_a": (range(10), range(10, 20)), "fold_b": (range(10, 20), range(10))}.items():
        Q, T, v = stack(tr)
        Q = Q + rng.normal(0, ENC_NOISE, Q.shape) * (noise != "none")
        Tm = np.array([measure(rng, t, noise) for t in T])
        fits = {"diff_only": sd.fit_poe(Q, Tm, "data", valid=v), "diff_abs": sd.fit_poe(Q, Tm, "data", valid=v, absolute=True)}
        lj = sd.LocalJacobian(7)
        for k in np.where(v)[0]:
            lj.update(Q[k + 1] - Q[k], sd.delta_x(Tm[k], Tm[k + 1]))
        ps = {k: [] for k in fits}
        pl, sq = [], []
        for i in te:
            Qe, Te = eps[i][:, :7], np.array([to_T(o) for o in eps[i]])   # targets: the true logged poses
            dp = np.diff(Te[:, :3, 3], axis=0)
            for k, m in fits.items():
                P = np.array([m.fk(q) for q in Qe])
                ps[k].append(((np.diff(P[:, :3, 3], axis=0) - dp) ** 2).sum(1))
            Tme = np.array([measure(rng, t, noise) for t in Te])
            for k in range(len(Qe) - 1):
                pl.append(((lj.predict(Qe[k + 1] - Qe[k])[:3] - dp[k]) ** 2).sum())
                lj.update(Qe[k + 1] - Qe[k], sd.delta_x(Tme[k], Tme[k + 1]))
            sq.append((dp ** 2).sum(1))
        sq = np.concatenate(sq)
        res[name] = {"transitions": int(len(sq)), "mean_step_mm": float(np.sqrt(sq).mean() * 1000),
                     "local_jac_rel_pos": float(np.mean(pl) / sq.mean())}
        for k in fits:
            e = np.concatenate(ps[k])
            res[name][f"{k}_rel_pos"] = float(e.mean() / sq.mean())
            res[name][f"{k}_step_err_mm_median"] = float(np.median(np.sqrt(e)) * 1000)
    newt = json.loads((ROOT / "results/phase0/selfmodel_readout_route_b.json").read_text())["franka_home"]
    res["newt_latent_rel"] = newt["relative_error"]
    return res


def _cell(args):
    return run_cell(*args, nominal_body())


def main(quick=False):
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1] if quick else [0, 1, 2, 3, 4]
    plan = [(m, 400, "mid") for m in ("joint", "cartesian")]
    plan += [(m, n, "mid") for m in ("joint", "cartesian") for n in (25, 50, 100, 200, 800)]
    plan += [(m, 400, z) for m in ("joint", "cartesian") for z in ("none", "low", "high")]
    if quick:
        plan = plan[:2]
    jobs = [(s, mode, steps, noise) for mode, steps, noise in plan for s in seeds]
    cells = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(_cell, jobs):
            cells.append(r)
            print(f"{r['mode']:9s} N={r['steps']:4d} noise={r['noise']:5s} seed={r['seed']} "
                  + " ".join(f"{k}: fk {r[k]['fk']['pos_mm_median']:.2f}mm reach {r[k]['reach']['success']}/40"
                             for k in ("nominal", "diff_only", "diff_abs", "calib"))
                  + f" lj {r['local_jac']['one_step']['rel_pos']:.3f}", flush=True)
            (OUT / "cells.json").write_text(json.dumps(cells, indent=1))
    ms = [maniskill_same_data(z) for z in ("none", "mid", "high")]
    print("maniskill", json.dumps(ms), flush=True)
    (OUT / "maniskill_same_data.json").write_text(json.dumps(ms, indent=2))


if __name__ == "__main__":
    main("--quick" in sys.argv)
