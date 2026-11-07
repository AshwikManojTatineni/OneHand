"""Lipson-style self-model as a Bayesian network: the robot learns its own body scheme from visual self-perception.

Follows Sturm, Plagemann and Burgard (ICRA 2008, "Unsupervised body scheme learning through self-perception"; extended
in J. Physiology-Paris 103, 2009, "Body schema learning for robotic manipulators from visual self-perception"):

  nodes      action signals a_1..a_m (joint encoder readings), hidden body-part poses X_1..X_n, and their visual
             observations Y_1..Y_n (marker poses; a part may be unobserved on any frame)
  local      p(Delta_ij | A_ij) for every pair of parts and every small subset A_ij of action signals, where
  models     Delta_ij = Y_i^-1 Y_j; each is a Gaussian process over the 12 free entries of the 4x4 transform
  structure  the kinematic chain is the minimum spanning tree over the parts, edge cost = validation error of the
             best local model for that pair (nothing about the arm's layout is given)
  inference  forward model by composing local-model means along the tree (variances add), inverse model by
             gradient descent on the predicted pose

Extension (this project): a disturbance node D_j on each action signal, so the network reads
    a_j,true = a_j,observed - d_j(tau),   d_j(tau) = c_j  (constant)  or  c_j + v_j * tau  (linear drift)
with tau the time since deployment. After the body scheme is learned, D is inferred from new observations with
the local models held fixed: a MAP estimate under a Gaussian prior, a Laplace posterior for its uncertainty, and
model selection between none / constant / linear by the Laplace approximation to the evidence. A drop in predictive
likelihood flags that the body has changed.
"""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.sparse.csgraph import minimum_spanning_tree

ROT_SCALE = 0.1   # metres per unit of a rotation-matrix entry when rotation and translation errors are added


# ------------------------------------------------------------------ transforms
def inv(T):
    Ti = np.zeros_like(T)
    R = np.swapaxes(T[..., :3, :3], -1, -2)
    Ti[..., :3, :3] = R
    Ti[..., :3, 3] = -np.einsum("...ij,...j->...i", R, T[..., :3, 3])
    Ti[..., 3, 3] = 1
    return Ti


def to_vec(T):
    """The 12 free entries of a transform (rotation entries scaled to metres)."""
    return np.concatenate([T[..., :3, :3].reshape(*T.shape[:-2], 9) * ROT_SCALE, T[..., :3, 3]], -1)


def to_T(v):
    """Back to a valid transform: the nearest rotation to the predicted 3x3 block (Sturm's normalization step)."""
    R = v[:9].reshape(3, 3) / ROT_SCALE
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, v[9:]
    return T


# ------------------------------------------------------------------ Gaussian-process local model
class GP:
    """Multi-output GP with one squared-exponential ARD kernel shared by the 12 outputs (outputs standardized).
    Hyperparameters by maximum marginal likelihood with analytic gradients."""

    def __init__(self, X, Y, restarts=1, seed=0):
        X = np.asarray(X, float)
        self.X = X.reshape(len(X), -1) if X.size else np.zeros((len(Y), 0))
        Y = np.asarray(Y, float)
        self.mu, self.sd = Y.mean(0), Y.std(0) + 1e-6
        self.Y = (Y - self.mu) / self.sd
        self.k = self.X.shape[1]
        rng = np.random.default_rng(seed)
        best = None
        for r in range(restarts):
            th0 = np.r_[np.log(np.full(self.k, 0.8)) + (rng.normal(0, 0.3, self.k) if r else 0), np.log(1.0), np.log(0.05)]
            res = minimize(self._nll, th0, jac=True, method="L-BFGS-B",
                           bounds=[(-4, 3)] * self.k + [(-3, 3), (-9, 0)], options={"maxiter": 80})
            if best is None or res.fun < best.fun:
                best = res
        self.theta = best.x
        self.nll = float(best.fun)
        self._prepare()

    def _K(self, A, B, th):
        ell, sf = np.exp(th[: self.k]), np.exp(th[self.k])
        if self.k == 0:
            return np.full((len(A), len(B)), sf ** 2)
        d = (A[:, None, :] - B[None, :, :]) / ell
        return sf ** 2 * np.exp(-0.5 * np.sum(d * d, -1))

    def _nll(self, th):
        N, D = self.Y.shape
        sn2 = np.exp(2 * th[-1])
        Kf = self._K(self.X, self.X, th)
        K = Kf + (sn2 + 1e-8) * np.eye(N)
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(th)
        alpha = cho_solve(c, self.Y)
        logdet = 2 * np.sum(np.log(np.diag(c[0])))
        nll = 0.5 * np.sum(self.Y * alpha) + 0.5 * D * logdet + 0.5 * N * D * np.log(2 * np.pi)
        W = D * cho_solve(c, np.eye(N)) - alpha @ alpha.T          # dnll/dK = 0.5 * W
        g = np.zeros_like(th)
        ell = np.exp(th[: self.k])
        for i in range(self.k):
            dx = (self.X[:, None, i] - self.X[None, :, i]) ** 2 / ell[i] ** 2
            g[i] = 0.5 * np.sum(W * Kf * dx)
        g[self.k] = 0.5 * np.sum(W * 2 * Kf)
        g[-1] = 0.5 * np.trace(W) * 2 * sn2
        return nll, g

    def _prepare(self):
        N = len(self.X)
        K = self._K(self.X, self.X, self.theta) + (np.exp(2 * self.theta[-1]) + 1e-8) * np.eye(N)
        self.c = cho_factor(K, lower=True)
        self.alpha = cho_solve(self.c, self.Y)

    def predict(self, Xs, var=False):
        Xs = np.asarray(Xs, float)
        Xs = Xs.reshape(len(Xs), self.k) if Xs.ndim == 2 else Xs.reshape(-1, self.k) if self.k else Xs.reshape(-1, 0)
        Ks = self._K(Xs, self.X, self.theta)
        mean = Ks @ self.alpha * self.sd + self.mu
        if not var:
            return mean
        v = cho_solve(self.c, Ks.T)
        s2 = np.exp(2 * self.theta[self.k]) - np.sum(Ks * v.T, 1)     # latent function variance (no noise term)
        return mean, np.maximum(s2, 1e-12)[:, None] * self.sd ** 2


def embed(a, S):
    """Action signals are joint angles: each enters the GP as (sin, cos), so the kernel sees the circle."""
    a = np.atleast_2d(np.asarray(a, float))
    if not S:
        return np.zeros((len(a), 0))
    x = a[:, list(S)]
    return np.concatenate([np.sin(x), np.cos(x)], 1)


# ------------------------------------------------------------------ the body scheme network
@dataclass
class LocalModel:
    i: int
    j: int
    A: tuple            # indices of the action signals this transform depends on
    gp: GP
    rss: float          # mean squared validation error (m^2, rotation entries scaled)

    def delta(self, a, var=False):
        """Predicted Delta_ij at action vectors a (N, m)."""
        return self.gp.predict(embed(np.atleast_2d(a), self.A), var)


@dataclass
class BodyScheme:
    n_parts: int
    n_actions: int
    root: int
    edges: list = field(default_factory=list)       # (parent, child, LocalModel, inverted)
    root_pose: np.ndarray = None
    candidates: dict = field(default_factory=dict)  # (i, j) -> best LocalModel, for the report

    def forward(self, a, var=False):
        """Poses of every part for one action vector, composed from the root along the learned tree."""
        a = np.asarray(a, float)
        poses = {self.root: self.root_pose}
        var_pos = {self.root: 0.0}
        for parent, child, lm, inverted in self.edges:
            if var:
                v, s2 = lm.delta(a, True)
                s = float(np.sum(s2[0, 9:]))
            else:
                v, s = lm.delta(a), 0.0
            D = to_T(v[0])
            poses[child] = poses[parent] @ (inv(D) if inverted else D)
            var_pos[child] = var_pos[parent] + s
        return (poses, var_pos) if var else poses

    def forward_batch(self, A, part, var=False):
        """Pose of one part for many action vectors at once (N, 4, 4): one GP call per edge on the path. With
        var=True also the predictive variance per entry, summed along the chain (N,)."""
        A = np.atleast_2d(np.asarray(A, float))
        path, _ = self.chain_to(part)
        lms = {(p, c): (lm, inverted) for p, c, lm, inverted in self.edges}
        X = np.broadcast_to(self.root_pose, (len(A), 4, 4)).copy()
        s2 = np.zeros(len(A))
        for p, c in path:
            lm, inverted = lms[(p, c)]
            if var:
                mean, v = lm.delta(A, True)
                s2 += np.mean(v, 1)
            else:
                mean = lm.delta(A)
            D = np.array([to_T(v) for v in mean])
            X = X @ (inv(D) if inverted else D)
        return (X, s2) if var else X

    def chain_to(self, part):
        """Edges on the path from the root to `part`, and the action signals that path uses."""
        parent_of = {c: (p, lm) for p, c, lm, _ in self.edges}
        path, used, x = [], set(), part
        while x != self.root:
            p, lm = parent_of[x]
            path.append((p, x))
            used |= set(lm.A)
            x = p
        return path[::-1], sorted(used)

    def inverse(self, part, T_target, a0, lo, hi, iters=200, w_rot=0.2):
        """Gradient descent on || X_part(a) - T_target || (Sturm, section III-D), joint limits as box bounds."""
        def cost(a):
            X = self.forward_batch(a, part)[0]
            e = X[:3, 3] - T_target[:3, 3]
            r = (X[:3, :3] - T_target[:3, :3]).ravel() * w_rot
            return float(e @ e + r @ r)
        res = minimize(cost, np.clip(a0, lo, hi), method="L-BFGS-B", bounds=list(zip(lo, hi)), options={"maxiter": iters})
        return res.x


def pair_data(A, Y, i, j):
    """Action vectors and observed Delta_ij on frames where both parts were seen."""
    ok = ~np.isnan(Y[i][:, 0, 0]) & ~np.isnan(Y[j][:, 0, 0])
    return A[ok], to_vec(inv(Y[i][ok]) @ Y[j][ok]), ok


def learn(A, Y, root=0, max_order=2, widen=3, val_every=4, block=20, max_train=300, seed=0, log=None):
    """Learn the body scheme from motor babbling.

    A: (N, m) action signals (encoder readings). Y: list of (N, 4, 4) observed part poses, NaN where not seen.
    For each pair of parts: a GP with no action input, one per single action signal, and for the `widen` best
    single signals all pairs among them (Sturm's search ordered by complexity, pruned). Edge cost = mean squared
    validation error of the best one; the chain is the minimum spanning tree.
    """
    A = np.asarray(A, float)
    n, m = len(Y), A.shape[1]
    best = {}
    rng = np.random.default_rng(seed)
    for i, j in combinations(range(n), 2):
        Aij, D, _ = pair_data(A, Y, i, j)
        if len(D) < 30:
            continue
        # validation on whole stretches of the walk (Sturm's separate validation set): interleaved frames of a
        # random walk sit next to training frames and would let spurious action inputs through
        val = (np.arange(len(D)) // block) % val_every == 0
        tr = np.where(~val)[0]
        if len(tr) > max_train:
            tr = np.sort(rng.choice(tr, max_train, replace=False))
        Xtr, Dtr, Xva, Dva = Aij[tr], D[tr], Aij[val], D[val]

        def fit(S):
            gp = GP(embed(Xtr, S), Dtr, seed=seed)
            # robust refit: a vision outlier (a misread marker) pulls a Gaussian-noise GP off; drop training points
            # beyond 4 robust standard deviations of the first fit's residuals and fit again
            r = np.sqrt(np.sum((gp.predict(embed(Xtr, S)) - Dtr) ** 2, 1))
            keep = r <= np.median(r) + 4 * 1.4826 * np.median(np.abs(r - np.median(r)))
            if keep.sum() < len(keep) and keep.sum() >= 20:
                gp = GP(embed(Xtr[keep], S), Dtr[keep], seed=seed)
            pred = gp.predict(embed(Xva, S))
            e = np.sum((pred - Dva) ** 2, 1)
            return LocalModel(i, j, tuple(S), gp, float(np.mean(np.minimum(e, 9 * np.median(e)))))   # capped

        models = [fit(())] + [fit((s,)) for s in range(m)]
        if max_order >= 2:
            top = [lm.A[0] for lm in sorted(models[1:], key=lambda x: x.rss)[:widen]]
            models += [fit(S) for S in combinations(sorted(top), 2)]
        b = min(models, key=lambda x: x.rss)
        best[(i, j)] = b
        if log:
            log(f"pair {i}-{j}: best A={b.A} rss={b.rss:.2e} from {len(models)} models")
    W = np.zeros((n, n))
    for (i, j), lm in best.items():
        W[i, j] = lm.rss + 1e-12
    tree = minimum_spanning_tree(W).toarray()
    adj = {k: [] for k in range(n)}
    for i, j in zip(*np.nonzero(tree)):
        adj[i].append(j); adj[j].append(i)
    edges, seen, queue = [], {root}, [root]
    while queue:                                    # orient the tree away from the root
        p = queue.pop(0)
        for c in adj[p]:
            if c not in seen:
                lm = best[(min(p, c), max(p, c))]
                edges.append((p, c, lm, p > c))
                seen.add(c); queue.append(c)
    Yr = Y[root][~np.isnan(Y[root][:, 0, 0])]
    root_pose = to_T(np.mean(to_vec(Yr), 0))
    return BodyScheme(n, m, root, edges, root_pose, best)


# ------------------------------------------------------------------ disturbance node (extension)
KINDS = {"none": 0, "constant": 1, "linear": 2}


def _residuals(scheme, part, A_obs, tau, Y_obs, d, var=False):
    """Predicted minus observed pose of `part` (12 entries) with actions corrected by the disturbance d(tau); with
    var=True also each frame's model variance (GP predictive variance along the chain)."""
    ok = ~np.isnan(Y_obs[:, 0, 0])
    A = A_obs[ok] - np.array([np.broadcast_to(d(t), A_obs.shape[1]) for t in tau[ok]]).reshape(-1, A_obs.shape[1])
    if var:
        X, s2 = scheme.forward_batch(A, part, var=True)
        return to_vec(X) - to_vec(Y_obs[ok]), s2
    return to_vec(scheme.forward_batch(A, part)) - to_vec(Y_obs[ok])


def residual_sigma(scheme, part, A_obs, Y_obs, floor=5e-4):
    """Observation noise per entry, from clean frames: the RMS prediction error minus the part the network's own
    predictive variance already explains."""
    r, s2m = _residuals(scheme, part, A_obs, np.zeros(len(A_obs)), Y_obs, lambda t: 0.0, var=True)
    return float(np.sqrt(max(np.mean(r ** 2) - np.mean(s2m), floor ** 2)))


def infer_disturbance(scheme, part, A_obs, tau, Y_obs, kind, joints=None, sigma_obs=0.003, prior_c=np.deg2rad(5),
                      prior_v=np.deg2rad(10)):
    """MAP estimate of the disturbance d_j(tau) on the action signals of `joints` from new observations of `part`,
    with the learned local models fixed. Gaussian likelihood (sigma_obs per entry), Gaussian priors on c and v.
    Optimized in prior-whitened units u = z / prior. Returns c, v, their Laplace standard deviations and the log
    evidence (Laplace approximation)."""
    m = A_obs.shape[1]
    joints = list(range(m)) if joints is None else list(joints)
    nj = len(joints)
    k = KINDS[kind] * nj
    prior = np.r_[np.full(nj, prior_c), np.full(nj, prior_v)][:k]

    def unpack(z):
        c = np.zeros(m); v = np.zeros(m)
        if kind != "none":
            c[joints] = z[:nj]
        if kind == "linear":
            v[joints] = z[nj:]
        return c, v

    def neg_log_post(u):
        # Gaussian likelihood per entry with variance sigma^2 + the network's own predictive variance at the
        # corrected actions (Sturm propagates GP variances along the chain): moving the actions into a region the
        # network never learned costs likelihood instead of buying it.
        z = u * prior
        c, v = unpack(z)
        r, s2m = _residuals(scheme, part, A_obs, tau, Y_obs, lambda t: c + v * t, var=True)
        s2 = sigma_obs ** 2 + s2m[:, None]
        return 0.5 * np.sum(r ** 2 / s2 + np.log(2 * np.pi * s2)) + 0.5 * float(u @ u), r.size

    f0, nres = neg_log_post(np.zeros(k))
    const = 0.0
    if k == 0:
        return dict(kind=kind, c=np.zeros(m), v=np.zeros(m), sd=np.zeros(0), log_evidence=float(const - f0), nll=float(f0))
    res = minimize(lambda u: neg_log_post(u)[0], np.zeros(k), method="BFGS", options={"maxiter": 300, "gtol": 1e-6})
    u = res.x
    h = 1e-3                                       # Hessian in whitened units by central differences
    H = np.zeros((k, k))
    for a_ in range(k):
        for b_ in range(a_, k):
            e_a, e_b = np.eye(k)[a_] * h, np.eye(k)[b_] * h
            H[a_, b_] = H[b_, a_] = (neg_log_post(u + e_a + e_b)[0] - neg_log_post(u + e_a - e_b)[0]
                                     - neg_log_post(u - e_a + e_b)[0] + neg_log_post(u - e_a - e_b)[0]) / (4 * h * h)
    H = 0.5 * (H + H.T) + 1e-9 * np.eye(k)
    sign, logdet = np.linalg.slogdet(H)
    cov_u = np.linalg.inv(H) if sign > 0 else np.full((k, k), np.nan)
    # evidence: the prior is a unit Gaussian in u, so log Z = log L(u*) - |u*|^2/2 - log det(H)/2
    log_ev = const - res.fun - 0.5 * logdet
    c, v = unpack(u * prior)
    return dict(kind=kind, c=c, v=v, sd=np.sqrt(np.clip(np.diag(cov_u), 0, None)) * prior,
                log_evidence=float(log_ev), nll=float(res.fun))


def select_disturbance(scheme, part, A_obs, tau, Y_obs, joints=None, margin=3.0, **kw):
    """Fit none, constant and linear; accept a more complex disturbance only when its Laplace log evidence beats the
    simpler one by `margin` (3 = a Bayes factor of about 20, "strong" on the Kass-Raftery scale)."""
    fits = {kind: infer_disturbance(scheme, part, A_obs, tau, Y_obs, kind, joints, **kw) for kind in KINDS}
    best = fits["none"]
    for kind in ("constant", "linear"):
        if fits[kind]["log_evidence"] > best["log_evidence"] + margin:
            best = fits[kind]
    return best, fits


def surprise(scheme, part, A_obs, Y_obs, sigma_obs=0.003):
    """Per-frame negative log predictive likelihood of the observed part pose (NaN where unseen): the change
    detector. Same likelihood as the disturbance inference, with no disturbance."""
    out = np.full(len(A_obs), np.nan)
    ok = ~np.isnan(Y_obs[:, 0, 0])
    r, s2m = _residuals(scheme, part, A_obs, np.zeros(len(A_obs)), Y_obs, lambda t: 0.0, var=True)
    s2 = sigma_obs ** 2 + s2m[:, None]
    out[ok] = 0.5 * np.sum(r ** 2 / s2 + np.log(2 * np.pi * s2), 1)
    return out
