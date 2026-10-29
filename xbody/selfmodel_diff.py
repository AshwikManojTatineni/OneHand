"""Differential self-model: identify an arm's kinematics from teleoperation logs, without its model file.

The robot is moved (teleoperated) while it logs two streams: joint encoder readings q_k and the measured pose
T_k of a frame on the hand (a wrist camera's extrinsics, or a tracked marker). The self-model only ever sees
those two streams.

Fitted model: product of exponentials (PoE) about the first logged configuration q_ref,
    T(q) = exp([S_1](q_1 - q_ref,1)) ... exp([S_n](q_n - q_ref,n)) M.
The joint screw axes S_i (world frame, at q_ref) are fitted from differential changes only, i.e. relative
transforms D_k = T_{k+1} T_k^-1 = P(q_{k+1}) P(q_k)^-1, in which M cancels. M is then the average of
P(q_k)^-1 T_k. Fitting q - q_ref absorbs constant encoder offsets. This is classical kinematic calibration
(nonlinear least squares on 6 numbers per joint), not training.

Also: LocalJacobian, a model-free baseline that tracks dT = J dq online (recursive least squares), with no
kinematic structure at all.
"""
import numpy as np
from scipy.optimize import least_squares

L_ROT = 0.2   # metres per radian when position and rotation residuals are mixed


# ---------- SE(3) helpers ----------
def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def so3_exp(w):
    th = np.linalg.norm(w)
    K = hat(w)
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-7:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if np.pi - th < 1e-4:   # near pi: take the axis from the symmetric part
        A = (R + np.eye(3)) / 2
        ax = A[:, np.argmax(np.diag(A))]
        return th * ax / np.linalg.norm(ax)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def se3_exp(S, th):
    """Revolute screw S = (w, v) with |w| = 1, rotated by angle th."""
    w, v = S[:3], S[3:]
    R = so3_exp(w * th)
    K = hat(w)
    G = np.eye(3) * th + (1 - np.cos(th)) * K + (th - np.sin(th)) * K @ K
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, G @ v
    return T


def pose(p, R):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def twist_of(D):
    """(rotation vector, translation) of a relative transform: a 6-vector with rotation scaled by L_ROT."""
    return np.concatenate([so3_log(D[:3, :3]) * L_ROT, D[:3, 3]])


def se3_exp_batch(S, th):
    w, v = S[:3], S[3:]
    K = hat(w)
    K2 = K @ K
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    t = th[:, None, None]
    T = np.zeros((len(th), 4, 4))
    T[:, :3, :3] = np.eye(3) + s * K + (1 - c) * K2
    T[:, :3, 3] = (np.eye(3) * t + (1 - c) * K + (t - s) * K2) @ v
    T[:, 3, 3] = 1
    return T


def inv_batch(T):
    Ti = np.zeros_like(T)
    Rt = np.transpose(T[:, :3, :3], (0, 2, 1))
    Ti[:, :3, :3] = Rt
    Ti[:, :3, 3] = -np.einsum("nij,nj->ni", Rt, T[:, :3, 3])
    Ti[:, 3, 3] = 1
    return Ti


def rotvec_batch(R):
    """Rotation vectors of a stack of rotations whose angles stay well below pi (relative motions)."""
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    a = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    f = np.where(th < 1e-7, 0.5, th / (2 * np.sin(np.maximum(th, 1e-7))))
    return a * f[:, None]


def twists_batch(D):
    return np.concatenate([rotvec_batch(D[:, :3, :3]) * L_ROT, D[:, :3, 3]], 1)


def project_rot(R):
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# ---------- PoE self-model ----------
class PoEModel:
    """Screws are stored as (axis angles theta, phi; point on axis p) per joint."""

    def __init__(self, n, q_ref):
        self.n, self.q_ref = n, np.asarray(q_ref, float)
        self.x = np.zeros(5 * n)
        self.M = np.eye(4)

    @staticmethod
    def screw(xi):
        th, ph, p = xi[0], xi[1], xi[2:5]
        w = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        return np.concatenate([w, -np.cross(w, p)])

    @staticmethod
    def encode(S):
        w = S[:3] / np.linalg.norm(S[:3])
        p = np.cross(w, S[3:])          # point on the axis closest to the origin
        return np.array([np.arccos(np.clip(w[2], -1, 1)), np.arctan2(w[1], w[0]), *p])

    def screws(self, x=None):
        x = self.x if x is None else x
        return [self.screw(x[5 * i:5 * i + 5]) for i in range(self.n)]

    def product(self, q, x=None):
        T = np.eye(4)
        for S, t in zip(self.screws(x), np.asarray(q) - self.q_ref):
            T = T @ se3_exp(S, t)
        return T

    def fk(self, q):
        return self.product(q) @ self.M

    def products(self, Q, x=None):
        """Batched P(q) for an (N, n) array of configurations."""
        TH = np.asarray(Q) - self.q_ref
        P = np.broadcast_to(np.eye(4), (len(TH), 4, 4)).copy()
        for i, S in enumerate(self.screws(x)):
            P = P @ se3_exp_batch(S, TH[:, i])
        return P

    def fit(self, Q, T, x0, valid=None, iters=100):
        """Differential fit of the screws on consecutive pairs (valid[k]: k -> k+1 is one motion, False across
        separate recordings), then M from the absolute poses."""
        Q, T = np.asarray(Q), np.asarray(T)
        valid = np.ones(len(Q) - 1, bool) if valid is None else np.asarray(valid)
        meas = twists_batch(T[1:] @ inv_batch(T[:-1]))[valid]

        def res(x):
            P = self.products(Q, x)
            return (twists_batch(P[1:] @ inv_batch(P[:-1]))[valid] - meas).ravel()

        sol = least_squares(res, x0, method="lm", max_nfev=iters * len(x0))
        self.x = sol.x
        self.set_M(Q, T)
        self.cost = float(sol.cost)
        return self

    def set_M(self, Q, T):
        A = inv_batch(self.products(Q)) @ T
        self.M = pose(np.mean([a[:3, 3] for a in A], 0), project_rot(np.mean([a[:3, :3] for a in A], 0)))
        return self

    def refine_absolute(self, Q, T, iters=100):
        """Second stage: refine screws and M together on every logged absolute pose. Differences between noisy
        poses carry the noise twice and say nothing about drift along the chain; absolute poses pin it."""
        Q, T = np.asarray(Q), np.asarray(T)
        Rt = np.transpose(T[:, :3, :3], (0, 2, 1))

        def unpack(z):
            return z[:-6], pose(z[-3:], so3_exp(z[-6:-3]) @ self.M[:3, :3])

        def res(z):
            x, M = unpack(z)
            P = self.products(Q, x) @ M
            return np.concatenate([(P[:, :3, 3] - T[:, :3, 3]),
                                   rotvec_batch(Rt @ P[:, :3, :3]) * L_ROT], 1).ravel()

        z0 = np.concatenate([self.x, np.zeros(3), self.M[:3, 3]])
        sol = least_squares(res, z0, method="lm", max_nfev=iters * len(z0))
        self.x, self.M = unpack(sol.x)
        self.cost_abs = float(sol.cost)
        return self

    def init_from_jacobian(self, J):
        """Screw initial guess from a 6xn spatial Jacobian estimate at q_ref (rotation rows scaled by L_ROT)."""
        x = []
        for i in range(self.n):
            w, v = J[:3, i] / L_ROT, J[3:, i]
            if np.linalg.norm(w) < 1e-3:
                w = np.array([0.0, 0.0, 1.0])
            nw = np.linalg.norm(w)
            w, v = w / nw, v / nw
            x.append(self.encode(np.concatenate([w, v])))
        return np.concatenate(x)


def spatial_jacobian_ls(Q, T, valid=None, ridge=1e-6):
    """Least-squares 6xn Jacobian that maps dq to the measured spatial twists (rotation rows scaled)."""
    T = np.asarray(T)
    valid = np.ones(len(Q) - 1, bool) if valid is None else np.asarray(valid)
    dQ = np.diff(np.asarray(Q), axis=0)[valid]
    # For a small relative motion D = T_{k+1} T_k^-1 = I + [xi], the spatial twist is (log R, t): t is how far
    # the point at the world origin moves, which is what a PoE spatial Jacobian column describes.
    X = twists_batch(T[1:] @ inv_batch(T[:-1]))[valid]
    A = dQ.T @ dQ + ridge * np.eye(dQ.shape[1])
    return np.linalg.solve(A, dQ.T @ X).T


def fit_poe(Q, T, init="data", nominal_J=None, valid=None, window=100, absolute=False):
    """Blind fit (screws initialised from a local Jacobian over the first `window` steps) or calibration
    (initialised from a nominal model's spatial Jacobian at q_ref). absolute=True adds the second stage."""
    m = PoEModel(len(Q[0]), Q[0])
    if init == "data":
        v = None if valid is None else np.asarray(valid)[:window]
        J = spatial_jacobian_ls(Q[: window + 1], T[: window + 1], v)
    else:
        J = nominal_J
    m.fit(Q, T, m.init_from_jacobian(J), valid)
    return m.refine_absolute(Q, T) if absolute else m


# ---------- model-free baseline ----------
class LocalJacobian:
    """Recursive least squares for x_{k+1} - x_k = J dq (x = TCP position and scaled rotation vector), with
    forgetting. Predicts one step ahead; it has no notion of the arm's global shape."""

    def __init__(self, n, lam=0.98, p0=10.0):
        self.J = np.zeros((6, n))
        self.P = np.eye(n) * p0
        self.lam = lam

    def predict(self, dq):
        return self.J @ dq

    def update(self, dq, dx):
        Pd = self.P @ dq
        g = Pd / (self.lam + dq @ Pd)
        self.J += np.outer(dx - self.J @ dq, g)
        self.P = (self.P - np.outer(g, Pd)) / self.lam


def delta_x(Ta, Tb):
    """Change of TCP position and world-frame rotation vector (scaled) from pose Ta to Tb."""
    return np.concatenate([Tb[:3, 3] - Ta[:3, 3], so3_log(Tb[:3, :3] @ Ta[:3, :3].T) * L_ROT])


# ---------- IK on any model with an fk(q) -> 4x4 ----------
def ik(fk, T_target, q0, lo, hi, iters=60):
    def res(q):
        T = fk(q)
        return np.concatenate([(T[:3, 3] - T_target[:3, 3]), so3_log(T_target[:3, :3].T @ T[:3, :3]) * L_ROT])
    sol = least_squares(res, np.clip(q0, lo, hi), bounds=(lo, hi), max_nfev=iters * len(q0))
    return sol.x
