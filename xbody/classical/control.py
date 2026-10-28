"""Decision-tree layers 7 (reference conversion), 8 (feedback / dynamic control) and 9 (actuator interface).

Layer 7 maps a task-space twist to joint velocity:
  dls, pinv, jinv (square Jacobian only), jtrans, qp (limits + collision rate), hqp (position, then orientation),
  cbf (QP with barrier constraints on clearance and table height)
The task is 6-D on arms with >= 6 joints and 5-D on the 5-DOF SO-101 (position + the two approach-axis
components), so the Jacobian inverse is square only on the SO-101.

Layer 8, joint-space laws (reference q, qd, qdd):
  p, pd, pid, pdg (PD + gravity), ct (computed torque), lqr, lqg, mpc, nmpc, smc, hinf, backstep, slotine
  (adaptive, diagonal regressor), mrac, jimp (joint impedance), passivity
Layer 8, task-space torque laws (Cartesian torque tracking): osc, cimp, force, hybrid, ptask
Layer 9: "torque" sends joint torques; "cascade" sends position setpoints to the embedded servo (p/pd/pid only).
"""
import mujoco
import numpy as np
import osqp
import scipy.linalg as sla
from scipy import sparse

from xbody.kinematics import rot_err


# ======================================================================== robot dynamics helpers
class Dyn:
    def __init__(self, model, kin):
        self.m, self.kin = model, kin
        self.d = mujoco.MjData(model)
        self.dof = kin.dadr

    def sync(self, data):
        self.d.qpos[:] = data.qpos; self.d.qvel[:] = data.qvel
        mujoco.mj_forward(self.m, self.d)

    def M(self):
        full = np.zeros((self.m.nv, self.m.nv))
        mujoco.mj_fullM(self.m, self.d, full)          # MuJoCo 3.17 signature: (model, data, dst)
        return full[np.ix_(self.dof, self.dof)]

    def h(self):
        return self.d.qfrc_bias[self.dof].copy()

    def g(self):
        v = self.d.qvel.copy(); self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        out = self.d.qfrc_bias[self.dof].copy()
        self.d.qvel[:] = v; mujoco.mj_forward(self.m, self.d)
        return out

    def passive(self):
        return self.d.qfrc_passive[self.dof].copy()


# ======================================================================== layer 7: reference conversion
def task_jacobian(kin, q, n_task):
    p, R, Jp, Jr = kin.jacobian(q)
    if n_task == 6:
        return p, R, np.vstack([Jp, Jr])
    z = R[:, 2]
    B = np.linalg.svd(np.eye(3) - np.outer(z, z))[0][:, :2]    # two directions perpendicular to the approach
    return p, R, np.vstack([Jp, B.T @ Jr]), B


def task_error(kin, q, p_ref, R_ref, n_task):
    p, R = kin.fk(q)
    if n_task == 6:
        return np.r_[p_ref - p, rot_err(R_ref, R)]
    z, zt = R[:, 2], R_ref[:, 2]
    w = np.cross(z, zt)
    out = task_jacobian(kin, q, 5)
    B = out[3]
    return np.r_[p_ref - p, B.T @ w]


class Converter:
    def __init__(self, kind, kin, clearance_grad=None, lam=0.02):
        self.kind, self.kin, self.lam = kind, kin, lam
        self.n_task = 6 if kin.n >= 6 else 5
        self.clear = clearance_grad

    def __call__(self, q, v, dt):
        """v: desired task velocity (n_task). Returns joint velocity, or None when the method cannot apply."""
        k = self.kind
        out = task_jacobian(self.kin, q, self.n_task)
        J = out[2]
        n = self.kin.n
        if k == "dls":
            return J.T @ np.linalg.solve(J @ J.T + self.lam ** 2 * np.eye(len(v)), v)
        if k == "pinv":
            return np.linalg.pinv(J) @ v
        if k == "jinv":
            if J.shape[0] != J.shape[1]:
                return None                      # needs a square Jacobian: not this body
            try:
                return np.linalg.solve(J, v)
            except np.linalg.LinAlgError:
                return None
        if k == "jtrans":
            return 2.0 * J.T @ v
        lo = (self.kin.lo - q) / dt; hi = (self.kin.hi - q) / dt
        lo = np.maximum(lo, -2.0); hi = np.minimum(hi, 2.0)
        lo = np.minimum(lo, 0.0); hi = np.maximum(hi, 0.0)    # a joint slightly past its limit may still stop
        A_rows, l_rows, u_rows = [np.eye(n)], [lo], [hi]
        if k in ("qp", "cbf") and self.clear is not None:
            c, gq = self.clear(q)
            gamma = 5.0 if k == "cbf" else 1.0 / dt
            A_rows.append(gq[None]); l_rows.append([-gamma * (c - 0.01)]); u_rows.append([np.inf])
        if k == "cbf":
            p, _, Jp, _ = self.kin.jacobian(q)
            A_rows.append(Jp[2][None]); l_rows.append([-5.0 * (p[2] - 0.004)]); u_rows.append([np.inf])
        if k == "hqp":
            Jp = J[:3]
            x1 = _qp(Jp, v[:3], [np.eye(n)], [lo], [hi])
            if x1 is None:
                return None
            return _qp(J[3:], v[3:], [np.eye(n), Jp], [lo, Jp @ x1 - 1e-6], [hi, Jp @ x1 + 1e-6])
        return _qp(J, v, A_rows, l_rows, u_rows)


def _qp(J, v, A_rows, l_rows, u_rows, reg=1e-4):
    n = J.shape[1]
    P = sparse.csc_matrix(J.T @ J + reg * np.eye(n))
    qv = -J.T @ v
    A = sparse.csc_matrix(np.vstack(A_rows))
    lo = np.concatenate([np.atleast_1d(x) for x in l_rows]); hi = np.concatenate([np.atleast_1d(x) for x in u_rows])
    hi = np.maximum(hi, lo)
    try:
        s = osqp.OSQP()
        s.setup(P, qv, A, lo, hi, verbose=False, polish=False, max_iter=400)
        r = s.solve()
    except Exception:
        return np.zeros(n)                     # solver failure: hold still (counted as the method's behaviour)
    return np.zeros(n) if r.x is None or np.any(np.isnan(r.x)) else r.x


# ======================================================================== layer 8: joint-space laws
class JointLaw:
    def __init__(self, kind, dyn, dt, interface, rng=None):
        self.kind, self.dyn, self.dt, self.iface = kind, dyn, dt, interface
        n = len(dyn.dof)
        self.n = n
        dyn.sync(dyn.d)
        self.Mdiag = None
        self.int_e = np.zeros(n)
        self.rng = rng or np.random.default_rng(0)
        self.theta = np.zeros((n, 3))           # adaptive parameters (slotine / mrac)
        self.K_lqr, self.k_since = None, 10 ** 9
        self.kf_x, self.kf_P = None, None
        self.mpc_u, self.mpc_k = None, 10 ** 9
        self.qm = self.qdm = None               # MRAC reference model state
        self.ptank = 1.0

    def gains(self, M, w=25.0, z=1.0):
        Md = np.maximum(np.diag(M), 1e-3)
        return w * w * Md, 2 * z * w * Md

    def __call__(self, data, q_ref, qd_ref, qdd_ref):
        d, k = self.dyn, self.kind
        d.sync(data)
        q, qd = data.qpos[d.kin.qadr].copy(), data.qvel[d.dof].copy()
        e, ed = q_ref - q, qd_ref - qd
        if self.iface == "cascade":
            # embedded position servo; the host adds an outer loop on top of the reference
            if k == "p":
                return q_ref + 0.5 * e
            if k == "pd":
                return q_ref + 0.5 * e + 0.02 * ed
            if k == "pid":
                self.int_e = np.clip(self.int_e + e * self.dt, -0.2, 0.2)
                return q_ref + 0.5 * e + 0.02 * ed + 2.0 * self.int_e
            raise ValueError(f"{k} needs a torque interface")
        M, h = d.M(), d.h()
        Kp, Kd = self.gains(M)
        if k == "p":
            return Kp * e
        if k == "pd":
            return Kp * e + Kd * ed
        if k == "pid":
            self.int_e = np.clip(self.int_e + e * self.dt, -0.5, 0.5)
            return Kp * e + Kd * ed + Kp * 5.0 * self.int_e
        if k == "pdg":
            return Kp * e + Kd * ed + d.g()
        if k == "ct":
            return M @ (qdd_ref + 2 * 25.0 * ed + 625.0 * e) + h
        if k == "jimp":
            Kp2, Kd2 = self.gains(M, w=12.0, z=0.9)
            return Kp2 * e + Kd2 * ed + d.g()
        if k == "smc":
            lam, phi = 20.0, 0.1
            s = ed + lam * e
            return M @ (qdd_ref + lam * ed) + h + 10.0 * np.maximum(np.diag(M), 1e-3) * np.clip(s / phi, -1, 1)
        if k == "backstep":
            k1, k2 = 20.0, 20.0
            z1 = q - q_ref
            alpha = qd_ref - k1 * z1
            z2 = qd - alpha
            alpha_dot = qdd_ref - k1 * (qd - qd_ref)
            return M @ (alpha_dot - z1 - k2 * z2) + h
        if k == "passivity":
            lam = 20.0
            qdr = qd_ref + lam * e
            qddr = qdd_ref + lam * ed
            s = qd - qdr
            return M @ qddr + h - 50.0 * np.maximum(np.diag(M), 1e-3) * s
        if k == "hinf":
            if self.K_lqr is None:
                self.K_lqr = _hinf_gain(gamma=2.0)
            u = -(self.K_lqr @ np.vstack([-e, -ed])).reshape(-1)
            return M @ (qdd_ref + u) + h
        if k in ("lqr", "lqg", "mpc"):
            return self._linear(data, q, qd, q_ref, qd_ref, qdd_ref, M, h)
        if k == "nmpc":
            return self._nmpc(data, q_ref, qd_ref, qdd_ref, M, h)
        if k == "slotine":
            lam, Kd_s, Gamma = 15.0, 2.0, np.array([0.5, 0.5, 20.0])
            qdr, qddr = qd_ref + lam * e, qdd_ref + lam * ed
            s = qd - qdr
            Y = np.stack([qddr, qdr, np.ones(self.n)], 1)
            self.theta -= self.dt * Gamma * Y * s[:, None]
            Kd = 10 * Kd_s * np.maximum(np.diag(M), 1e-3)          # gains scaled once by link inertia (tuning)
            return np.sum(Y * self.theta, 1) - Kd * s
        if k == "mrac":
            if self.qm is None:
                self.qm, self.qdm = q.copy(), qd.copy()
            wn = 15.0
            qddm = qdd_ref + 2 * wn * (qd_ref - self.qdm) + wn * wn * (q_ref - self.qm)
            self.qdm = self.qdm + qddm * self.dt; self.qm = self.qm + self.qdm * self.dt
            em, edm = self.qm - q, self.qdm - qd
            phi = np.stack([qddm + 20 * edm + 100 * em, qd, np.ones(self.n)], 1)
            sig = edm + 5 * em
            self.theta += self.dt * np.array([2.0, 0.5, 20.0]) * phi * sig[:, None]
            return np.sum(phi * self.theta, 1) + 50.0 * np.maximum(np.diag(M), 1e-3) * sig
        raise ValueError(k)

    # ---- linear-model laws: the model is linearized by finite differences around the current state
    def _linearize(self, data):
        m = self.dyn.m
        nv = m.nv
        A = np.zeros((2 * nv, 2 * nv)); B = np.zeros((2 * nv, m.nu))
        dtmp = mujoco.MjData(m)
        dtmp.qpos[:], dtmp.qvel[:], dtmp.ctrl[:] = data.qpos, data.qvel, data.ctrl
        mujoco.mjd_transitionFD(m, dtmp, 1e-6, True, A, B, None, None)
        dof = self.dyn.dof
        idx = np.r_[dof, nv + dof]
        arm_act = self.arm_act
        return A[np.ix_(idx, idx)], B[np.ix_(idx, arm_act)]

    def _linear(self, data, q, qd, q_ref, qd_ref, qdd_ref, M, h):
        k = self.kind
        u_ff = M @ qdd_ref + h
        x_err = np.r_[q - q_ref, qd - qd_ref]
        n = self.n
        if self.K_lqr is None or self.k_since > 100:
            A, B = self._linearize(data)
            Q = np.diag(np.r_[np.full(n, 1e4), np.full(n, 10.0)])
            R = np.diag(1.0 / np.maximum(self.umax, 1e-3) ** 2) * 10
            self.A, self.B = A, B
            try:
                P = sla.solve_discrete_are(A, B, Q, R)
                self.K_lqr = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
                self.P_term = P
            except (np.linalg.LinAlgError, ValueError):
                self.K_lqr = np.zeros((n, 2 * n)); self.P_term = Q
            self.k_since = 0
        self.k_since += 1
        if k == "lqr":
            return u_ff - self.K_lqr @ x_err
        if k == "lqg":
            y = q + self.rng.normal(0, 1e-3, n)                   # noisy encoder; velocity is not measured
            dt = self.dt
            if self.kf_x is None:
                self.kf_x = np.r_[q, np.zeros(n)]; self.kf_P = np.tile(np.eye(2) * 1e-4, (n, 1, 1))
                self.kf_u = np.zeros(n)
            Fk = np.array([[1, dt], [0, 1.0]]); Qk = np.diag([1e-9, 1e-3]); Rk = 1e-6
            acc = np.linalg.solve(M, self.kf_u - h)               # model acceleration from the last torque
            xq, xv = self.kf_x[:n].copy(), self.kf_x[n:].copy()
            for j in range(n):                                    # per-joint constant-velocity filter
                xj = Fk @ np.array([xq[j], xv[j]]) + np.array([0.5 * dt * dt, dt]) * acc[j]
                Pj = Fk @ self.kf_P[j] @ Fk.T + Qk
                Kj = Pj[:, 0] / (Pj[0, 0] + Rk)
                xj = xj + Kj * (y[j] - xj[0])
                self.kf_P[j] = Pj - np.outer(Kj, Pj[0])
                xq[j], xv[j] = xj
            self.kf_x = np.r_[xq, xv]
            xe = self.kf_x - np.r_[q_ref, qd_ref]
            u = u_ff - self.K_lqr @ xe
            self.kf_u = u
            return u
        # linear MPC: box-constrained QP over a short horizon on the linearized model, re-solved every 5 steps
        if self.mpc_u is None or self.mpc_k >= 2:
            self.mpc_u = self._mpc_solve(x_err, u_ff)
            self.mpc_k = 0
        self.mpc_k += 1
        return u_ff + self.mpc_u

    def _mpc_solve(self, x0, u_ff, N=15):
        A, B, n = self.A, self.B, self.n
        nx, nu = 2 * n, n
        Q = np.diag(np.r_[np.full(n, 1e4), np.full(n, 10.0)])
        R = np.diag(1.0 / np.maximum(self.umax, 1e-3) ** 2) * 10
        Phi = np.zeros((N * nx, nx)); Gam = np.zeros((N * nx, N * nu))
        Ak = np.eye(nx)
        for i in range(N):
            Ak = A @ Ak
            Phi[i * nx:(i + 1) * nx] = Ak
            for j in range(i + 1):
                Gam[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu] = np.linalg.matrix_power(A, i - j) @ B
        Qb = np.kron(np.eye(N), Q); Rb = np.kron(np.eye(N), R)
        Qb[-nx:, -nx:] = getattr(self, "P_term", Q)               # LQR cost-to-go as terminal cost
        H = Gam.T @ Qb @ Gam + Rb
        f = Gam.T @ Qb @ Phi @ x0
        lo = np.tile(-self.umax - u_ff, N); hi = np.tile(self.umax - u_ff, N)
        s = osqp.OSQP()
        s.setup(sparse.csc_matrix(H), f, sparse.eye(N * nu, format="csc"), lo, hi, verbose=False, max_iter=200)
        r = s.solve()
        return np.zeros(nu) if r.x is None else r.x[:nu]

    def _nmpc(self, data, q_ref, qd_ref, qdd_ref, M, h, K=12, H=8, every=10):
        """Sampling-based nonlinear MPC (MPPI) with MuJoCo rollouts of the full model."""
        if not hasattr(self, "_nm_k"):
            self._nm_k, self._nm_u, self._roll = every, np.zeros((H, self.n)), mujoco.MjData(self.dyn.m)
        if self._nm_k >= every:
            m, r = self.dyn.m, self._roll
            u_ff = M @ qdd_ref + h
            Kp, Kd = self.gains(M)
            base = self._nm_u.copy()
            noise = self.rng.normal(0, 1, (K, H, self.n)) * 0.03 * self.umax
            costs = np.zeros(K)
            for kk in range(K):
                r.qpos[:], r.qvel[:], r.ctrl[:] = data.qpos, data.qvel, data.ctrl
                for t in range(H):
                    qq, qv = r.qpos[self.dyn.kin.qadr], r.qvel[self.dyn.dof]
                    tau = u_ff + Kp * (q_ref + qd_ref * t * self.dt * every - qq) + Kd * (qd_ref - qv) + base[t] + noise[kk, t]
                    r.ctrl[self.arm_act] = np.clip(tau, -self.umax, self.umax)
                    for _ in range(every):
                        mujoco.mj_step(m, r)
                    costs[kk] += np.sum((r.qpos[self.dyn.kin.qadr] - (q_ref + qd_ref * (t + 1) * self.dt * every)) ** 2)
            w = np.exp(-(costs - costs.min()) / (costs.std() + 1e-9)); w /= w.sum()
            self._nm_u = base + np.tensordot(w, noise, 1)
            self._nm_u = np.r_[self._nm_u[1:], self._nm_u[-1:]]
            self._nm_k = 0
        self._nm_k += 1
        Kp, Kd = self.gains(M)
        q, qd = data.qpos[self.dyn.kin.qadr], data.qvel[self.dyn.dof]
        return M @ qdd_ref + h + Kp * (q_ref - q) + Kd * (qd_ref - qd) + self._nm_u[0]


def _hinf_gain(gamma, w=25.0):
    """State feedback for the feedback-linearized joint error (double integrator) with disturbance attenuation
    gamma: game Riccati equation via the indefinite-R form of the CARE."""
    A = np.array([[0, 1], [0, 0.0]]); B = np.array([[0], [1.0]]); Bw = np.array([[0], [1.0]])
    Q = np.diag([w ** 4, 2 * w ** 2])
    Baug = np.hstack([B, Bw]); Raug = np.diag([1.0, -gamma ** 2])
    P = sla.solve_continuous_are(A, Baug, Q, Raug)
    return (B.T @ P).reshape(1, 2)   # applied per joint: u = -K [e, ed] with e = q - q_ref


# ======================================================================== layer 8: task-space torque laws
class TaskLaw:
    def __init__(self, kind, dyn, kin, dt):
        self.kind, self.dyn, self.kin, self.dt = kind, dyn, kin, dt
        self.n_task = 6 if kin.n >= 6 else 5
        self.int_f = 0.0
        self.tank = 2.0

    def __call__(self, data, p_ref, R_ref, v_ref, q_home, wrench_z=0.0, contact_phase=False):
        d, kin = self.dyn, self.kin
        d.sync(data)
        q, qd = data.qpos[kin.qadr].copy(), data.qvel[d.dof].copy()
        out = task_jacobian(kin, q, self.n_task)
        J = out[2]
        e = task_error(kin, q, p_ref, R_ref, self.n_task)
        xd = J @ qd
        ed = v_ref - xd
        M, h, g = d.M(), d.h(), d.g()
        Minv = np.linalg.inv(M)
        Lam = np.linalg.pinv(J @ Minv @ J.T, rcond=1e-3)
        Kx = np.r_[np.full(3, 400.0), np.full(self.n_task - 3, 60.0)]
        Dx = 2 * np.sqrt(Kx)
        N = np.eye(kin.n) - J.T @ Lam @ J @ Minv                   # dynamically consistent null space
        Mn = np.maximum(np.diag(M), 1e-3)
        tau_null = N @ (Mn * (25.0 * (q_home - q) - 10.0 * qd))
        k = self.kind
        if k == "osc":
            return J.T @ (Lam @ (Kx * e + Dx * ed)) + h + tau_null
        if k == "cimp":
            Kc = Kx * np.r_[np.full(3, 0.5), np.ones(self.n_task - 3)]
            return J.T @ (Lam @ (Kc * e + 2 * np.sqrt(Kc) * ed)) + g + tau_null
        F = Lam @ (Kx * e + Dx * ed)
        if k in ("force", "hybrid") and contact_phase and abs(wrench_z) > 0.5:   # force loop only once in contact
            f_des = -3.0                                            # press 3 N into the table/cube while in contact
            self.int_f += (f_des - wrench_z) * self.dt
            fz = f_des + 0.5 * (f_des - wrench_z) + 5.0 * self.int_f
            if k == "force":
                F[2] = fz                                           # z purely force-controlled
            else:
                S = np.ones(self.n_task); S[2] = 0                  # selection: xy/orientation in position
                F = S * F; F[2] = fz
            return J.T @ F + h + tau_null
        if k in ("force", "hybrid"):
            return J.T @ F + h + tau_null
        if k == "ptask":
            # Cartesian impedance whose injected power is drawn from an energy tank (passivity-preserving)
            Fk = Lam @ (Kx * e); Fd = Lam @ (Dx * ed)
            tau = J.T @ (Fk + Fd) + g + tau_null
            p_stiff = float(xd @ Fk)                               # power the spring injects
            p_damp = float(xd @ (Lam @ (Dx * xd)))                 # power the damper dissipates refills the tank
            self.tank = np.clip(self.tank + (p_damp - p_stiff) * self.dt, 0.0, 5.0)
            if self.tank <= 0.01 and p_stiff > 0:
                tau = J.T @ Fd + g + tau_null                      # tank empty: only passive damping
            return tau
        raise ValueError(k)


JOINT_LAWS = ["p", "pd", "pid", "pdg", "ct", "lqr", "lqg", "mpc", "nmpc", "smc", "hinf", "backstep", "slotine",
              "mrac", "jimp", "passivity"]
TASK_LAWS = ["osc", "cimp", "force", "hybrid", "ptask"]
CONVERTERS = ["dls", "pinv", "jinv", "jtrans", "qp", "hqp", "cbf"]
