"""Bayesian-network self-model benchmark (Sturm-style body scheme from visual self-perception) with disturbances.

Per arm (Franka, SO-101), per seeded "real" arm (calibration errors from xbody/calibration.ERRORS), per perception
mode (camera: ArUco markers on every part, detected in rendered images; oracle: true marker pose + camera noise):

  1 learn     motor babbling (a random walk through joint space), encoders + observed marker poses; learn the
              body scheme with no model file: local GP models for every pair of parts, structure by minimum
              spanning tree (xbody/bodyscheme.py)
  2 evaluate  structure recovered (does the tree match the real chain, and does each edge depend on the right
              joints?), forward error of the end part on held-out poses, open-loop reach on the real arm, against
              the nominal model file and the product-of-exponentials fit (xbody/selfmodel_diff.py) on the same data
  3 deploy    a new session in which a disturbance acts on the joint encoders: none, constant (a slipped zero) or
              linear (a reading that drifts over the session, as wear or heating would). The learned network is
              kept fixed; the disturbance node is inferred from the first frames (MAP, Laplace posterior, model
              selection by evidence) and the corrected model is scored on the rest of the session.

Usage:
  MUJOCO_GL=glfw python scripts/selfmodel_bayes.py [--bodies franka,so101] [--seeds 5] [--perception camera,oracle]
      [--workers 4] [--quick] [--estimate]
--estimate times one small cell per arm and perception and prints the projected time for the full grid.
Writes results/selfmodel_bayes/cells/<body>_<perception>_<seed>.json (resumable) and summary.json.
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "win32" else "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody import bodyscheme as B  # noqa: E402
from xbody import markers as MK  # noqa: E402
from xbody import selfmodel_diff as sd  # noqa: E402
from xbody.bodies import MENAGERIE, SPECS  # noqa: E402
from xbody.calibration import ERRORS, perturb  # noqa: E402

OUT = ROOT / "results/selfmodel_bayes"
FULL = dict(n_learn=400, n_test=100, n_reach=30, n_deploy=200, n_infer=100)
QUICK = dict(n_learn=150, n_test=40, n_reach=8, n_deploy=80, n_infer=40)
DISTURB = dict(constant=np.deg2rad(3.0), linear=np.deg2rad(6.0))   # sd of c (constant) and of v (drift per session)
REACH_TOL = (0.005, 2.0)


class Lab:
    """One arm with markers and cameras: a nominal copy and a perturbed 'real' copy share the same marker layout."""

    def __init__(self, body, seed, mode):
        spec_cfg = SPECS[body]
        spec = mujoco.MjSpec.from_file(str(MENAGERIE / spec_cfg["xml"]))
        spec.worldbody.add_light(pos=[0, 0, 2.0], dir=[0, 0, -1])
        spec.worldbody.add_light(pos=[1.0, 1.0, 1.5], dir=[-1, -1, -1], castshadow=False)
        self.markers, self.cams, self.size = MK.add_markers(spec, body, spec_cfg["home"], spec_cfg["arm"])
        self.nom = spec.compile()
        self.real = spec.compile()
        perturb(self.real, spec_cfg, seed, ERRORS[body])
        self.body, self.spec = body, spec_cfg
        jid = [mujoco.mj_name2id(self.nom, mujoco.mjtObj.mjOBJ_JOINT, j) for j in spec_cfg["arm"]]
        self.qadr = np.array([self.nom.jnt_qposadr[j] for j in jid])
        self.lo, self.hi = self.nom.jnt_range[jid, 0].copy(), self.nom.jnt_range[jid, 1].copy()
        self.dn, self.dr = mujoco.MjData(self.nom), mujoco.MjData(self.real)
        self.per = MK.Perception(self.real, self.dr, self.markers, self.cams, self.size, mode=mode,
                                 rng=np.random.default_rng(7000 + seed))
        self.set_real(np.array(spec_cfg["home"], float))
        self.per.calibrate_frame()
        self.n_parts = len(self.markers)
        self.true_edges = self._true_chain()

    def _true_chain(self):
        """Joint indices between consecutive marker parts in the real kinematic chain."""
        parts = MK.PARTS[self.body][0]
        m = self.nom
        bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, p) for p in parts]
        joint_body = [int(m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in self.spec["arm"]]

        def ancestors(b):
            out = []
            while b != 0:
                out.append(b)
                b = int(m.body_parentid[b])
            return out
        edges = {}
        for k in range(1, len(parts)):
            anc_k, anc_p = set(ancestors(bid[k])), set(ancestors(bid[k - 1]))
            edges[(k - 1, k)] = tuple(sorted(i for i, jb in enumerate(joint_body) if jb in anc_k and jb not in anc_p))
        return edges

    def set_real(self, q_true):
        self.dr.qpos[self.qadr] = q_true
        mujoco.mj_forward(self.real, self.dr)

    def end_pose_nominal(self, q):
        self.dn.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.nom, self.dn)
        b = mujoco.mj_name2id(self.nom, mujoco.mjtObj.mjOBJ_BODY, self.markers[-1])
        return sd.pose(self.dn.xpos[b].copy(), self.dn.xmat[b].reshape(3, 3).copy())

    def babble(self, n, rng, step=0.08, disturb=None, jump=0.1):
        """Motor babbling as a random walk; encoders read q + d(tau) + noise. Returns A, Y (per part), tau, q."""
        inset = 0.15 * (self.hi - self.lo)
        lo, hi = self.lo + inset, self.hi - inset
        q = np.clip(np.array(self.spec["home"], float), lo, hi)
        A, Q, Ys, tau, jumps = [], [], [], [], []
        for t in range(n):
            q = np.clip(q + rng.normal(0, step, len(q)), lo, hi)
            jumps.append(rng.random() < jump)
            if jumps[-1]:
                q = rng.uniform(lo, hi)                         # occasional jump to a new region
            self.set_real(q)
            Ys.append(self.per.observe())
            s = t / max(n - 1, 1)
            d = disturb(s) if disturb else 0.0
            A.append(q + d + rng.normal(0, 1e-4, len(q)))
            Q.append(q.copy()); tau.append(s)
        Y = [np.array([f[k] for f in Ys]) for k in range(self.n_parts)]
        self.last_jumps = np.array(jumps)
        return np.array(A), Y, np.array(tau), np.array(Q)

    def close(self):
        self.per.close()


def pose_err(A, Bm):
    return np.linalg.norm(A[:3, 3] - Bm[:3, 3]) * 1000, np.rad2deg(np.linalg.norm(sd.so3_log(A[:3, :3].T @ Bm[:3, :3])))


def structure_score(scheme, true_edges, observed):
    """Edges of the learned tree vs the real chain among observed parts (an unobserved part is bridged)."""
    chain = [k for k in range(max(max(e) for e in true_edges) + 1) if k in observed]
    want = {}
    for a, b in zip(chain[:-1], chain[1:]):
        joints = set()
        for k in range(a, b):
            joints |= set(true_edges[(k, k + 1)])
        want[(a, b)] = tuple(sorted(joints))
    got = {(min(p, c), max(p, c)): lm.A for p, c, lm, _ in scheme.edges}
    topo = sum(1 for e in want if e in got)
    joints = sum(1 for e, js in want.items() if e in got and set(js) <= set(got[e]))
    return dict(edges_true=len(want), edges_found=topo, edges_right_joints=joints,
                tree_correct=bool(topo == len(want) == len(got)),
                learned={f"{a}-{b}": list(A) for (a, b), A in got.items()}, true={f"{a}-{b}": list(j) for (a, b), j in want.items()})


def run_cell(body, seed, mode, cfg, log=print):
    t0 = time.time()
    rng = np.random.default_rng(100 + seed)
    lab = Lab(body, seed, mode)
    times = {}
    # ---------------------------------------------------------------- 1 learn
    t = time.time()
    A, Y, _, Q = lab.babble(cfg["n_learn"], rng)
    jumps = lab.last_jumps
    times["babble_s"] = time.time() - t
    seen = [float(np.mean(~np.isnan(y[:, 0, 0]))) for y in Y]
    observed = {k for k, s in enumerate(seen) if s > 0.3}
    Y_use = [y if k in observed else np.full_like(y, np.nan) for k, y in enumerate(Y)]
    t = time.time()
    scheme = B.learn(A, Y_use, root=0, seed=seed)
    times["learn_s"] = time.time() - t
    end = lab.n_parts - 1
    out = dict(body=body, seed=seed, perception=mode, cfg=cfg, detect_rate=seen,
               structure=structure_score(scheme, lab.true_edges, observed))
    # PoE baseline on the same frames (the end part's observed poses)
    ok = ~np.isnan(Y[end][:, 0, 0])
    idx = np.where(ok)[0]
    poe = None
    try:
        poe = sd.fit_poe(A[idx], Y[end][idx], "data", valid=(np.diff(idx) == 1) & ~jumps[idx[1:]], robust=True, absolute=True)
    except Exception as ex:   # too few consecutive pairs for its initial Jacobian
        out["poe_error"] = repr(ex)
    # ---------------------------------------------------------------- 2 evaluate on held-out poses (true pose, no noise)
    t = time.time()
    test_q = rng.uniform(lab.lo + 0.15 * (lab.hi - lab.lo), lab.hi - 0.15 * (lab.hi - lab.lo), (cfg["n_test"], len(lab.lo)))
    truth = []
    for q in test_q:
        lab.set_real(q)
        truth.append(lab.per.true_poses()[end])
    pred = scheme.forward_batch(test_q, end)
    fe = dict(bayes=[pose_err(P_, T_)[0] for P_, T_ in zip(pred, truth)],
              nominal=[pose_err(lab.end_pose_nominal(q), T_)[0] for q, T_ in zip(test_q, truth)])
    if poe:
        fe["poe"] = [pose_err(poe.fk(q), T_)[0] for q, T_ in zip(test_q, truth)]
    out["forward_mm"] = {k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95))) for k, v in fe.items()}
    # open-loop reach: solve on each model, run on the real arm, measure where the end part really is
    reach = {k: [] for k in fe}
    home = np.array(lab.spec["home"], float)
    for _ in range(cfg["n_reach"]):
        qt = np.clip(home + rng.normal(0, 0.4, len(home)), lab.lo + 0.1, lab.hi - 0.1)
        lab.set_real(qt)
        Tt = lab.per.true_poses()[end]
        sols = dict(bayes=scheme.inverse(end, Tt, home, lab.lo, lab.hi),
                    nominal=sd.ik(lab.end_pose_nominal, Tt, home, lab.lo, lab.hi))
        if poe:
            sols["poe"] = sd.ik(poe.fk, Tt, home, lab.lo, lab.hi)
        for k, qc in sols.items():
            lab.set_real(qc)
            reach[k].append(pose_err(lab.per.true_poses()[end], Tt))
    out["reach"] = {k: dict(success=int(sum(e[0] < 1000 * REACH_TOL[0] and e[1] < REACH_TOL[1] for e in v)),
                            success_10mm=int(sum(e[0] < 10 for e in v)), n=len(v),
                            median_mm=float(np.median([e[0] for e in v]))) for k, v in reach.items()}
    times["evaluate_s"] = time.time() - t
    # ---------------------------------------------------------------- 3 deploy with a disturbance on the encoders
    _, joints_used = scheme.chain_to(end)
    Ac, Yc, _, _ = lab.babble(cfg["n_infer"], np.random.default_rng(800 + seed), jump=0.3)   # a fresh clean session
    sigma = B.residual_sigma(scheme, end, Ac, Yc[end])
    base_surprise = B.surprise(scheme, end, Ac, Yc[end], sigma)
    out["sigma_obs_mm"] = sigma * 1000
    out["deploy"] = {}
    for kind in ("none", "constant", "linear"):
        t = time.time()
        drng = np.random.default_rng(900 + seed)
        m = len(lab.lo)
        c = drng.normal(0, DISTURB["constant"], m) if kind == "constant" else np.zeros(m)
        v = drng.normal(0, DISTURB["linear"], m) if kind == "linear" else np.zeros(m)
        dist = (lambda c=c, v=v: lambda s: c + v * s)()
        Ad, Yd, tau, Qd = lab.babble(cfg["n_deploy"], drng, disturb=dist, jump=0.3)   # task poses, more varied
        Ye = Yd[end]
        n_inf = cfg["n_infer"]
        sur = B.surprise(scheme, end, Ad[:n_inf], Ye[:n_inf], sigma)
        best, fits = B.select_disturbance(scheme, end, Ad[:n_inf], tau[:n_inf], Ye[:n_inf], joints=joints_used, sigma_obs=sigma)
        rest = slice(n_inf, None)
        truth_rest = []
        for q in Qd[rest]:
            lab.set_real(q)
            truth_rest.append(lab.per.true_poses()[end])
        raw = scheme.forward_batch(Ad[rest], end)
        corr = scheme.forward_batch(Ad[rest] - (best["c"] + np.outer(tau[rest], best["v"])), end)
        e_raw = [pose_err(P_, T_)[0] for P_, T_ in zip(raw, truth_rest)]
        e_cor = [pose_err(P_, T_)[0] for P_, T_ in zip(corr, truth_rest)]
        ju = list(joints_used)
        out["deploy"][kind] = dict(
            detected=bool(best["kind"] != "none"), surprise_mean=float(np.nanmean(sur)), surprise_clean=float(np.nanmean(base_surprise)),
            selected=best["kind"], log_evidence={k: f["log_evidence"] for k, f in fits.items()},
            c_true_deg=np.rad2deg(c[ju]).tolist(), c_est_deg=np.rad2deg(best["c"][ju]).tolist(),
            v_true_deg=np.rad2deg(v[ju]).tolist(), v_est_deg=np.rad2deg(best["v"][ju]).tolist(),
            posterior_sd_deg=np.rad2deg(best["sd"]).tolist(),
            end_error_mm_uncorrected=float(np.median(e_raw)), end_error_mm_corrected=float(np.median(e_cor)),
            seconds=time.time() - t)
    times["deploy_s"] = sum(d["seconds"] for d in out["deploy"].values())
    out["times"] = times
    out["seconds"] = time.time() - t0
    lab.close()
    log(f"{body} {mode} seed {seed}: tree {'ok' if out['structure']['tree_correct'] else 'WRONG'} "
        f"fwd bayes {out['forward_mm']['bayes']['median']:.1f} mm, nominal {out['forward_mm']['nominal']['median']:.1f} mm | "
        f"reach bayes {out['reach']['bayes']['success']}/{out['reach']['bayes']['n']} | "
        + " ".join(f"{k}: sel {d['selected']} {d['end_error_mm_uncorrected']:.1f}->{d['end_error_mm_corrected']:.1f} mm"
                   for k, d in out["deploy"].items()) + f" ({out['seconds']:.0f}s)")
    return out


def _job(args):
    body, seed, mode, cfg = args
    path = OUT / "cells" / f"{body}_{mode}_{seed}.json"
    if path.exists() and not cfg.get("_force"):
        return json.loads(path.read_text())
    r = run_cell(body, seed, mode, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=1, default=float))
    return r


def estimate(bodies, perceptions, seeds, workers):
    """Time a quick cell per arm and perception, scale each stage to the full sizes, and project the grid."""
    import psutil
    proc = psutil.Process()
    rows = []
    for body in bodies:
        for mode in perceptions:
            r = run_cell(body, 0, mode, dict(QUICK))
            tm = r["times"]
            # babbling and evaluation scale linearly with frames; GP learning with train size cubed (capped at 300)
            babble = tm["babble_s"] * FULL["n_learn"] / QUICK["n_learn"]
            learn = tm["learn_s"] * (min(FULL["n_learn"] * 0.75, 300) / (QUICK["n_learn"] * 0.75)) ** 2.4
            evaluate = tm["evaluate_s"] * (FULL["n_reach"] / QUICK["n_reach"])
            deploy = tm["deploy_s"] * (FULL["n_deploy"] / QUICK["n_deploy"]) * (FULL["n_infer"] / QUICK["n_infer"]) ** 0.5
            per_cell = babble + learn + evaluate + deploy
            rows.append(dict(body=body, perception=mode, quick_s=r["seconds"], full_cell_min=per_cell / 60))
    rss = proc.memory_info().rss / 2**30
    total = sum(r["full_cell_min"] for r in rows) * seeds
    print(json.dumps(rows, indent=1))
    print(f"peak memory of one worker ~{rss:.2f} GB")
    print(f"full grid: {len(bodies)} arms x {len(perceptions)} perception x {seeds} seeds = "
          f"{len(bodies) * len(perceptions) * seeds} cells, {total:.0f} min serial, ~{total / workers:.0f} min on {workers} workers")
    return rows, rss, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", default="franka,so101")
    ap.add_argument("--perception", default="camera,oracle")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args()
    bodies, perceptions = a.bodies.split(","), a.perception.split(",")
    if a.estimate:
        estimate(bodies, perceptions, a.seeds, a.workers)
        return
    cfg = dict(QUICK if a.quick else FULL)
    jobs = [(b, s, m, cfg) for b in bodies for m in perceptions for s in range(a.seeds)]
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        cells = pool.map(_job, jobs)
    summary = {}
    for b in bodies:
        for m in perceptions:
            cs = [c for c in cells if c["body"] == b and c["perception"] == m]
            summary[f"{b}_{m}"] = dict(
                trees_correct=sum(c["structure"]["tree_correct"] for c in cs), cells=len(cs),
                forward_mm={k: float(np.median([c["forward_mm"][k]["median"] for c in cs if k in c["forward_mm"]]))
                            for k in ("bayes", "nominal", "poe")},
                reach={k: sum(c["reach"][k]["success"] for c in cs if k in c["reach"]) for k in ("bayes", "nominal", "poe")},
                deploy={k: dict(selected_right=sum(c["deploy"][k]["selected"] == k for c in cs),
                                detected=sum(c["deploy"][k]["detected"] for c in cs),
                                error_mm_uncorrected=float(np.median([c["deploy"][k]["end_error_mm_uncorrected"] for c in cs])),
                                error_mm_corrected=float(np.median([c["deploy"][k]["end_error_mm_corrected"] for c in cs])))
                        for k in ("none", "constant", "linear")})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
