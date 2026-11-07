"""Ego-centric Bayesian self-model: the robot learns its own kinematic body from one on-body sensor.

The only things the robot perceives are its joint encoders a_1..a_m and the world pose of a sensor it carries
(a wrist mono or stereo camera localizing against a calibration board, or a 3D lidar localizing against a map of
the room: xbody/selfmodel_sensing.py). Every body part between base and sensor is hidden. This is the setting of
Bongard, Zykov and Lipson (Science 2006, "Resilient machines through continuous self-modeling"), where the robot
only has on-body sensors; the Bayesian-network form follows Sturm, Plagemann and Burgard (2008/2009).

Bayesian network
  observed   action signals a_j; sensor pose Y (noisy, sometimes missing)
  hidden     one link transform per action signal, L_k = exp([S_k] (a_pi(k) - a_ref)), chained base to sensor,
             and the sensor's mounting M:  X_sensor(a) = L_1 ... L_m M,   Y = X_sensor (+) noise
  structure  pi, the order in which the action signals drive the chain, and each link's type (revolute or
             prismatic). Nothing about the arm's layout is given; structure is chosen by Bayesian evidence.
  local      a screw per link (5 numbers revolute, 2 prismatic) and M (6). Hidden links are only identifiable
  models     through their effect on the one observed pose, so the local models are parametric screws; an optional
             Gaussian-process residual on top absorbs what screws cannot (gravity sag, backlash).
  inference  parameters by robust nonlinear least squares, Laplace posterior (Gauss-Newton Hessian), evidence
             log p(D | structure) by the Laplace approximation; posterior predictive covariance of the sensor pose.

Learning loop (estimation-exploration, after Bongard and Lipson): random babbling, or active babbling that sends
the arm where the posterior predictive is most uncertain and where competing structures disagree most.

Extension: a disturbance node on the encoders, a_true = a_observed - d(tau) with d = c (constant) or c + v tau
(linear drift), inferred after learning with the body model fixed (MAP, Laplace, Bayes-factor model choice).
"""
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
from scipy.optimize import least_squares

from xbody import selfmodel_diff as sd

L_ROT = sd.L_ROT          # metres per radian when position and rotation residuals are mixed
PRIOR_SD = 2.0            # weak Gaussian prior on every parameter (radians / metres), for the evidence


# ------------------------------------------------------------------ the chain model
def _axis(th, ph):
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def _twist_exp_batch(w, v, theta):
    """exp([S] theta) for a screw S = (w, v), batched over theta. w = 0 gives a prismatic joint along v."""
    N = len(theta)
    T = np.zeros((N, 4, 4))
    T[:, 3, 3] = 1
    if np.linalg.norm(w) < 1e-12:
        T[:, :3, :3] = np.eye(3)
        T[:, :3, 3] = theta[:, None] * v
        return T
    return sd.se3_exp_batch(np.r_[w, v], theta)


@dataclass
class Chain:
    """X_sensor(a) = prod_k exp([S_k](a_order[k] - a_ref)) M, screws in the world frame at a_ref."""
    order: tuple
    types: tuple                     # 'R' or 'P' per chain position
    a_ref: np.ndarray
    x: np.ndarray = None             # stacked link parameters, then M (rotation vector, position)
    sigma: float = None              # fitted noise per residual entry
    log_evidence: float = None
    cov: np.ndarray = None           # Laplace posterior covariance of x
    gp: object = None                # optional residual model
    info: dict = field(default_factory=dict)

    @property
    def sizes(self):
        return [5 if t == "R" else 2 for t in self.types]

    def screws(self, x=None):
        x = self.x if x is None else x
        out, i = [], 0
        for t in self.types:
            if t == "R":
                w = _axis(x[i], x[i + 1])
                out.append((w, -np.cross(w, x[i + 2:i + 5])))
                i += 5
            else:
                out.append((np.zeros(3), _axis(x[i], x[i + 1])))
                i += 2
        return out, x[i:i + 6]

    def fk(self, A, x=None, residual=True):
        A = np.atleast_2d(np.asarray(A, float))
        scr, mx = self.screws(x)
        T = np.broadcast_to(np.eye(4), (len(A), 4, 4)).copy()
        for (w, v), j in zip(scr, self.order):
            T = T @ _twist_exp_batch(w, v, A[:, j] - self.a_ref[j])
        M = sd.pose(mx[3:], sd.so3_exp(mx[:3]))
        X = T @ M
        if residual and self.gp is not None:
            X = X @ np.array([_exp6(d) for d in self.gp.predict(_embed(A))])
        return X

    def fk1(self, a):
        return self.fk(a[None])[0]


def _exp6(d):
    """Small correction (rotation vector scaled by L_ROT, translation) as a transform."""
    return sd.pose(d[3:], sd.so3_exp(d[:3] / L_ROT))


def _embed(A):
    return np.concatenate([np.sin(A), np.cos(A)], 1)


def pose_residuals(X, T):
    """(N, 6): rotation error (scaled to metres) and position error of predicted poses X against observed T."""
    Rt = np.transpose(T[:, :3, :3], (0, 2, 1))
    return np.concatenate([sd.rotvec_batch(Rt @ X[:, :3, :3]) * L_ROT, X[:, :3, 3] - T[:, :3, 3]], 1)


# ------------------------------------------------------------------ fitting and evidence
def initial_screws(Q, T, valid, radius=0.6):
    """Each action signal's screw at a_ref from the least-squares spatial Jacobian of consecutive motions
    (independent of the chain order: at a_ref the spatial Jacobian columns are the screws in any order).
    A Jacobian holds near one configuration, so only small consecutive motions within `radius` (rad, per joint)
    of a_ref = Q[0] are used, as the product-of-exponentials fit uses its first window."""
    near = np.all(np.abs(Q - Q[0]) < radius, 1)
    use = np.asarray(valid) & near[:-1] & near[1:]
    if use.sum() < 3 * Q.shape[1]:                    # too few: widen until there are enough pairs
        d = np.max(np.abs(Q - Q[0]), 1)
        r = np.sort(np.maximum(d[:-1], d[1:])[np.asarray(valid)])[min(3 * Q.shape[1], int(np.sum(valid)) - 1)]
        use = np.asarray(valid) & (np.maximum(d[:-1], d[1:]) <= r)
    J = sd.spatial_jacobian_ls(Q, T, use)
    out = []
    for i in range(J.shape[1]):
        w, v = J[:3, i] / L_ROT, J[3:, i]
        if np.linalg.norm(w) < 1e-3:
            w = np.array([0.0, 0.0, 1.0])
        nw = np.linalg.norm(w)
        w, v = w / nw, v / nw
        out.append((w, v, J[:, i]))
    return out


def pack_init(order, types, init, Q, T):
    """Parameter vector for a structure from the per-signal Jacobian screws, M from the absolute poses."""
    x = []
    for t, j in zip(types, order):
        w, v, col = init[j]
        if t == "R":
            p = np.cross(w, v)
            x += [np.arccos(np.clip(w[2], -1, 1)), np.arctan2(w[1], w[0]), *p]
        else:
            d = col[3:] / max(np.linalg.norm(col[3:]), 1e-9)
            x += [np.arccos(np.clip(d[2], -1, 1)), np.arctan2(d[1], d[0])]
    ch = Chain(tuple(order), tuple(types), Q[0].copy(), np.r_[x, np.zeros(6)])
    P = ch.fk(Q)
    A = np.einsum("nij,njk->nik", sd.inv_batch(P), T)
    A = A[np.all(np.isfinite(A.reshape(len(A), -1)), 1)]     # a diverged initial screw can blow up a few frames
    M = sd.pose(np.mean(A[:, :3, 3], 0), sd.project_rot(np.mean(A[:, :3, :3], 0))) if len(A) else np.eye(4)
    ch.x[-6:] = np.r_[sd.so3_log(M[:3, :3]), M[:3, 3]]
    return ch


def fit(chain, Q, T, inliers=None, iters=100):
    """Robust nonlinear least squares, then the Laplace evidence on the inlier frames (Gaussian likelihood with the
    noise level profiled out, weak Gaussian prior on every parameter)."""
    Q, T = np.asarray(Q), np.asarray(T)

    def res(x):
        return pose_residuals(chain.fk(Q, x, residual=False), T).ravel()

    sol = least_squares(res, chain.x, method="trf", loss="soft_l1", f_scale=0.005, max_nfev=iters)   # iterations
    chain.x = sol.x
    r = res(sol.x).reshape(-1, 6)
    if inliers is None:
        e = np.linalg.norm(r, axis=1)
        inliers = e <= np.median(e) + 5 * 1.4826 * np.median(np.abs(e - np.median(e)))
    ri = r[inliers].ravel()
    n, k = ri.size, len(sol.x)
    sigma2 = max(float(ri @ ri) / n, 1e-12)
    J = sol.jac.reshape(-1, 6, k)[inliers].reshape(-1, k)
    H = J.T @ J / sigma2 + np.eye(k) / PRIOR_SD ** 2
    sign, logdet = np.linalg.slogdet(H)
    log_lik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    log_prior = -0.5 * float(sol.x @ sol.x) / PRIOR_SD ** 2 - 0.5 * k * np.log(2 * np.pi * PRIOR_SD ** 2)
    chain.log_evidence = float(log_lik + log_prior + 0.5 * k * np.log(2 * np.pi) - 0.5 * logdet)
    chain.sigma = float(np.sqrt(sigma2))
    chain.cov = np.linalg.inv(H) if sign > 0 else None
    chain.info = dict(rms_mm=float(np.sqrt(np.mean(np.sum(r[inliers][:, 3:] ** 2, 1)))) * 1000, nfev=int(sol.nfev),
                      inliers=int(inliers.sum()), frames=len(inliers))
    return chain, inliers


def seed_order(init):
    """Initial guess of the chain order: joints whose screw axis lies far from the sensor move it most per radian
    and tend to sit nearer the base (largest lever arm first)."""
    lever = [np.linalg.norm(col[3:]) for _, _, col in init]
    return tuple(int(i) for i in np.argsort(lever)[::-1])


def structure_search(Q, T, valid, types=None, max_rounds=8, log=None, exhaustive_upto=4, quick_iters=40, refit_top=3):
    """Search the chain order by evidence: exhaustive for few signals, otherwise hill climbing over adjacent swaps
    and single moves from the lever-arm seed, each candidate fitted from the same Jacobian screws.
    Returns (best chain, every evaluated chain sorted by evidence)."""
    m = Q.shape[1]
    types = types or ("R",) * m
    init = initial_screws(Q, T, valid)
    seen = {}
    inliers = None

    def score(order):
        nonlocal inliers
        if order in seen:
            return seen[order]
        ch = pack_init(order, tuple(types[j] for j in order), init, Q, T)
        ch, inl = fit(ch, Q, T, inliers, iters=quick_iters)      # short fit to rank; winners refitted below
        if inliers is None:
            inliers = inl                  # one outlier mask for every structure, so evidences compare
        seen[order] = ch
        if log:
            log(f"order {order}: log evidence {ch.log_evidence:.1f}, rms {ch.info['rms_mm']:.2f} mm")
        return ch

    if m <= exhaustive_upto:
        for order in permutations(range(m)):
            score(order)
    else:
        best = score(seed_order(init))
        fit(best, Q, T, inliers)
        for _ in range(max_rounds):
            o = list(best.order)
            neighbours = [tuple(o[:i] + [o[i + 1], o[i]] + o[i + 2:]) for i in range(m - 1)]
            neighbours += [tuple(o[:i] + o[i + 1:j + 1] + [o[i]] + o[j + 1:]) for i in range(m) for j in range(i + 2, m)]
            neighbours += [tuple(o[:j] + [o[i]] + o[j:i] + o[i + 1:]) for i in range(m) for j in range(i - 1)]
            scored = sorted((score(nb) for nb in neighbours), key=lambda c: -c.log_evidence)
            for ch in scored[:refit_top]:            # short fits rank; contenders are fitted fully before comparing
                fit(ch, Q, T, inliers)
            cand = max(scored[:refit_top], key=lambda c: c.log_evidence)
            if cand.log_evidence <= best.log_evidence + 1e-6:
                break
            best = cand
    ranked = sorted(seen.values(), key=lambda c: -c.log_evidence)
    for ch in ranked[:refit_top]:
        fit(ch, Q, T, inliers)
    ranked = sorted(seen.values(), key=lambda c: -c.log_evidence)
    return ranked[0], ranked, inliers


def refit(models, Q, T, inliers=None, iters=20):
    """Warm-started refit of a few candidate structures on more data (active babbling between batches)."""
    for ch in models:
        fit(ch, Q, T, inliers, iters=iters)
    return sorted(models, key=lambda c: -c.log_evidence)


def choose_types(best, Q, T, inliers, log=None):
    """For each chain position, try a prismatic joint in place of the revolute one; keep it if the evidence rises."""
    init = None
    for k in range(len(best.order)):
        types = list(best.types)
        types[k] = "P" if types[k] == "R" else "R"
        ch = Chain(best.order, tuple(types), best.a_ref)
        # start from the current fit, converting only link k
        x, i = [], 0
        for kk, (t_old, t_new) in enumerate(zip(best.types, types)):
            n_old = 5 if t_old == "R" else 2
            seg = best.x[i:i + n_old]
            x += list(seg if t_old == t_new else seg[:2] if t_new == "P" else np.r_[seg[:2], 0, 0, 0])
            i += n_old
        ch.x = np.r_[x, best.x[-6:]]
        ch, _ = fit(ch, Q, T, inliers)
        if log:
            log(f"position {k}: {types[k]} log evidence {ch.log_evidence:.1f} vs {best.log_evidence:.1f}")
        if ch.log_evidence > best.log_evidence:
            best = ch
    return best


# ------------------------------------------------------------------ optional residual GP (non-geometric effects)
def fit_residual(chain, Q, T, inliers, Qv, Tv, max_train=300, seed=0):
    """GP on the 6-D pose residual over all joint angles (sin/cos). Kept only if it lowers held-out error."""
    from xbody.bodyscheme import GP
    X = chain.fk(Q, residual=False)
    Rt = sd.inv_batch(X) @ T                                     # what the screws miss, in the sensor frame
    D = np.concatenate([sd.rotvec_batch(Rt[:, :3, :3]) * L_ROT, Rt[:, :3, 3]], 1)[inliers]
    Qi = Q[inliers]
    if len(Qi) > max_train:
        idx = np.random.default_rng(seed).choice(len(Qi), max_train, replace=False)
        Qi, D = Qi[idx], D[idx]
    before = np.median(np.linalg.norm(pose_residuals(chain.fk(Qv, residual=False), Tv)[:, 3:], axis=1))
    chain.gp = GP(_embed(Qi), D, seed=seed)
    after = np.median(np.linalg.norm(pose_residuals(chain.fk(Qv), Tv)[:, 3:], axis=1))
    if after >= before:
        chain.gp = None
    chain.info["residual_gp"] = dict(kept=chain.gp is not None, heldout_mm_before=float(before) * 1000,
                                     heldout_mm_after=float(after) * 1000)
    return chain


# ------------------------------------------------------------------ posterior predictive, active babbling
def pose_jacobian_x(chain, A, eps=1e-6):
    """d(pose residual vector)/d(params) at each action vector: (N, 6, k), by forward differences."""
    X0 = chain.fk(A, residual=False)
    k = len(chain.x)
    J = np.zeros((len(A), 6, k))
    for i in range(k):
        x = chain.x.copy()
        x[i] += eps
        J[:, :, i] = pose_residuals(chain.fk(A, x, residual=False), X0) / eps
    return J


def predictive_var(chain, A):
    """Trace of the posterior predictive covariance of the sensor pose (m^2, rotation scaled), per action vector."""
    if chain.cov is None:
        return np.zeros(len(A))
    J = pose_jacobian_x(chain, A)
    return np.einsum("nik,kl,nil->n", J, chain.cov, J)


def choose_next(models, candidates, k=1, visible=None):
    """Estimation-exploration: the candidate poses where the posterior of the best model is most uncertain and the
    competing structures disagree most (Bongard and Lipson's disagreement, in metres)."""
    best = models[0]
    score = predictive_var(best, candidates)
    Xb = best.fk(candidates)
    for other in models[1:3]:
        d = pose_residuals(other.fk(candidates), Xb)
        score = score + np.sum(d ** 2, 1)
    if visible is not None:
        score = np.where(visible(Xb), score, -np.inf)
    return np.argsort(score)[::-1][:k]


# ------------------------------------------------------------------ disturbance node (extension)
KINDS = {"none": 0, "constant": 1, "linear": 2}


def infer_disturbance(chain, A_obs, tau, T_obs, kind, joints=None, sigma=0.003, prior_c=np.deg2rad(5),
                      prior_v=np.deg2rad(10)):
    """MAP of d(tau) on the encoders with the body model fixed; Gaussian likelihood (sigma per entry plus the
    model's own posterior predictive variance), Gaussian priors on c and v, optimized in prior-whitened units.
    Returns c, v, posterior standard deviations and the Laplace log evidence."""
    ok = np.array([t is not None and not np.isnan(np.asarray(t)[0, 0]) for t in T_obs])
    A, tau, T = A_obs[ok], tau[ok], np.array([t for t, o in zip(T_obs, ok) if o])
    m = A.shape[1]
    joints = list(range(m)) if joints is None else list(joints)
    nj = len(joints)
    kdim = KINDS[kind] * nj
    prior = np.r_[np.full(nj, prior_c), np.full(nj, prior_v)][:kdim]
    s2_model = predictive_var(chain, A) / 6

    def d_of(u):
        z = u * prior
        c, v = np.zeros(m), np.zeros(m)
        if kind != "none":
            c[joints] = z[:nj]
        if kind == "linear":
            v[joints] = z[nj:]
        return c, v

    def res(u):
        c, v = d_of(u)
        r = pose_residuals(chain.fk(A - c - np.outer(tau, v)), T)
        s = np.sqrt(sigma ** 2 + s2_model)[:, None]
        return np.r_[(r / s).ravel(), u]                    # prior as extra residuals (unit Gaussian in u)

    lognorm = -np.sum(np.log(np.sqrt(2 * np.pi) * np.sqrt(sigma ** 2 + s2_model))) * 6
    if kdim == 0:
        r = res(np.zeros(0))
        return dict(kind=kind, c=np.zeros(m), v=np.zeros(m), sd=np.zeros(0), log_evidence=float(lognorm - 0.5 * r @ r))
    sol = least_squares(res, np.zeros(kdim), method="lm", max_nfev=200 * kdim)
    r = sol.fun
    H = sol.jac.T @ sol.jac
    sign, logdet = np.linalg.slogdet(H)
    cov = np.linalg.inv(H) if sign > 0 else np.full((kdim, kdim), np.nan)
    c, v = d_of(sol.x)
    return dict(kind=kind, c=c, v=v, sd=np.sqrt(np.clip(np.diag(cov), 0, None)) * prior,
                log_evidence=float(lognorm - 0.5 * r @ r - 0.5 * logdet))


def select_disturbance(chain, A_obs, tau, T_obs, margin=3.0, holdout=0.4, gain=0.1, **kw):
    """Fit none, constant and linear; accept the more complex one only when its log evidence wins by `margin`
    AND, refitted on the earlier (1 - holdout) of the window, it predicts the later frames better by `gain`
    (a forward-in-time check: correlated sensor error can fool the evidence, but not a forecast)."""
    fits = {k: infer_disturbance(chain, A_obs, tau, T_obs, k, **kw) for k in KINDS}
    best = fits["none"]
    n = len(A_obs)
    cut = int(n * (1 - holdout))
    ok = np.array([t is not None and not np.isnan(np.asarray(t)[0, 0]) for t in T_obs])
    late = np.where(ok[cut:])[0] + cut
    Tl = np.array([T_obs[i] for i in late])

    def late_error(c, v):
        if not len(late):
            return np.inf
        X = chain.fk(A_obs[late] - c - np.outer(tau[late], v))
        return float(np.median(np.linalg.norm(pose_residuals(X, Tl)[:, 3:], axis=1)))

    base_err = late_error(np.zeros(A_obs.shape[1]), np.zeros(A_obs.shape[1]))
    checks = {}
    for k in ("constant", "linear"):
        if fits[k]["log_evidence"] <= best["log_evidence"] + margin:
            continue
        early = infer_disturbance(chain, A_obs[:cut], tau[:cut], T_obs[:cut], k, **kw)
        err = late_error(early["c"], early["v"])
        checks[k] = dict(heldout_mm=err * 1000, none_mm=base_err * 1000)
        if err < (1 - gain) * base_err:
            best = fits[k]
    best = dict(best, checks=checks)
    return best, fits
