"""Decision-tree layer 3 (endpoint selection): every option turns a TCP target into a joint goal.

  analytic     closed form where the geometry allows: the SO-101 (base pan + planar three-link chain + wrist roll).
               The Franka has no closed form without a free parameter, so it fails by construction.
  newton       undamped Newton-Raphson on the full pose
  lm           Levenberg-Marquardt: damped full-pose steps with adaptive damping
  ccd          cyclic coordinate descent, one joint at a time
  constrained  pose IK as constrained optimization (SLSQP): position as an equality, joint limits as bounds,
               orientation error minimized
  taught       given joint goals, taught once at the nominal layout and replayed (pose IK bypassed)
"""
import mujoco
import numpy as np
from scipy.optimize import minimize

from xbody.kinematics import rot_err, solve_ik


def _pose_err(body, q, p_t, R_t, w_rot=0.3):
    p, R = body.fk(q)
    return np.r_[p_t - p, w_rot * rot_err(R_t, R)]


def newton(body, p_t, R_t, q0, iters=200):
    q, info = solve_ik(body, p_t, R_t, q0=q0, method="newton", iters=iters)
    return q, _info(body, q, p_t, R_t) | ({"failed": info["failed"]} if "failed" in info else {})


def lm(body, p_t, R_t, q0, iters=200, w_rot=0.3):
    q, lam = np.clip(q0, body.lo, body.hi), 1e-2
    e = _pose_err(body, q, p_t, R_t, w_rot)
    for _ in range(iters):
        p, R, Jp, Jr = body.jacobian(q)
        J = np.vstack([Jp, w_rot * Jr])
        A = J.T @ J
        dq = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), J.T @ e)
        q_new = np.clip(q + dq, body.lo, body.hi)
        e_new = _pose_err(body, q_new, p_t, R_t, w_rot)
        if e_new @ e_new < e @ e:
            q, e, lam = q_new, e_new, max(lam / 3, 1e-6)
        else:
            lam = min(lam * 4, 1e4)
        if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
            break
    return q, _info(body, q, p_t, R_t)


def ccd(body, p_t, R_t, q0, sweeps=60):
    """Each joint in turn rotates to bring the TCP toward the target (closed-form per joint), last joints also
    align the approach axis."""
    q = np.clip(np.array(q0, float), body.lo, body.hi)
    m, d = body.model, body.data
    for _ in range(sweeps):
        for k in reversed(range(body.n)):
            p, R = body.fk(q)
            jid = body.jids[k]
            anchor, axis = d.xanchor[jid].copy(), d.xaxis[jid].copy()
            a = p - anchor; b = p_t - anchor
            a -= axis * (a @ axis); b -= axis * (b @ axis)
            if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
                continue
            ang_p = np.arctan2(axis @ np.cross(a, b), a @ b)
            # orientation pull on the wrist joints, weighted by how far the position already is
            z, zt = R[:, 2], R_t[:, 2]
            zt_p = zt - axis * (zt @ axis); z_p = z - axis * (z @ axis)
            ang_o = np.arctan2(axis @ np.cross(z_p, zt_p), z_p @ zt_p) if np.linalg.norm(z_p) > 1e-6 else 0.0
            w = 0.5 if k >= body.n - 2 else 0.1
            q[k] = np.clip(q[k] + (1 - w) * ang_p + w * ang_o, body.lo[k], body.hi[k])
        if np.linalg.norm(body.fk(q)[0] - p_t) < 1e-4:
            break
    return q, _info(body, q, p_t, R_t)


def constrained(body, p_t, R_t, q0, axis_only=False):
    def f(q):
        p, R = body.fk(q)
        if axis_only:
            # approach axis first; yaw as a soft preference (a 5-DOF arm sets it with the wrist roll when it can)
            return (1 - R[:, 2] @ R_t[:, 2]) + 0.02 * float(np.sum(rot_err(R_t, R) ** 2))
        return float(np.sum(rot_err(R_t, R) ** 2))

    def ceq(q):
        return body.fk(q)[0] - p_t

    q, _ = solve_ik(body, p_t, R_t, q0=q0, method="hqp", orient="axis", iters=60)   # warm start
    res = minimize(f, q, method="SLSQP", bounds=list(zip(body.lo, body.hi)),
                   constraints=[{"type": "eq", "fun": ceq}], options={"maxiter": 100, "ftol": 1e-10})
    return np.clip(res.x, body.lo, body.hi), _info(body, res.x, p_t, R_t)


class AnalyticSO101:
    """Closed-form IK for the SO-101, with link parameters read off the model at the zero pose."""

    def __init__(self, body):
        self.b = body
        m, d = body.model, body.data
        q0 = np.zeros(body.n)
        p_tcp0, R0 = body.fk(q0)
        anc = [d.xanchor[j].copy() for j in body.jids]
        ax = [d.xaxis[j].copy() for j in body.jids]
        self.base = anc[0]
        self.zaxis = ax[0] / np.linalg.norm(ax[0])
        h = p_tcp0 - self.base
        e1 = h - self.zaxis * (h @ self.zaxis); e1 /= np.linalg.norm(e1)   # arm-plane direction at pan = 0
        n = np.cross(self.zaxis, e1)                           # plane normal, so that n x e1 = -z
        self.sign = [float(np.sign(a @ n)) for a in ax[1:4]]   # lift / elbow / wrist-flex axes are +-n
        self.e1, self.n0 = e1, n
        to2 = lambda P: np.array([(P - self.base) @ e1, (P - self.base) @ self.zaxis])
        # The TCP's offset along n is what pan must account for; e1 runs through the zero-pose TCP, so it is 0 and
        # joints about +-n never change it (wrist roll can, if the TCP is off the roll axis).
        self.lat = float((p_tcp0 - self.base) @ n)
        self.S, self.E, self.W = to2(anc[1]), to2(anc[2]), to2(anc[3])
        self.T0 = to2(p_tcp0)
        z0 = R0[:, 2]
        self.phi0 = np.arctan2(z0 @ self.zaxis, z0 @ e1)        # approach angle in the plane at zero
        self.a1, self.a2, self.v3 = self.E - self.S, self.W - self.E, self.T0 - self.W

    @staticmethod
    def _rot(v, t):
        c, s = np.cos(t), np.sin(t)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    def _solve_once(self, p_t, R_t, q_seed):
        """Pan from the target's bearing (minus the arm plane's lateral offset), then a two-link solution for
        shoulder lift and elbow that puts the wrist where the requested approach angle needs it, then wrist flex
        for that angle and wrist roll for the jaw yaw. In-plane angles run from e1 toward z; a joint whose axis is
        +n turns the plane by -q, so q = -angle * sign."""
        b = self.b
        rel = p_t - self.base
        x, y = rel @ self.e1, rel @ self.n0
        rho = np.hypot(x, y)
        if rho <= abs(self.lat):
            return q_seed, _info(b, q_seed, p_t, R_t) | {"failed": "inside pan offset"}
        r = np.sqrt(rho**2 - self.lat**2)
        pan = np.arctan2(y, x) - np.arctan2(self.lat, r)
        c, s_ = np.cos(pan), np.sin(pan)
        e1p = c * self.e1 + s_ * self.n0                         # arm-plane horizontal direction after pan
        T = np.array([r, rel @ self.zaxis])
        zt = R_t[:, 2]
        phi_t = np.arctan2(zt @ self.zaxis, zt @ e1p)
        tsum = phi_t - self.phi0
        Wt = T - self._rot(self.v3, tsum)
        dW = Wt - self.S
        L1, L2 = np.linalg.norm(self.a1), np.linalg.norm(self.a2)
        c2 = (dW @ dW - L1**2 - L2**2) / (2 * L1 * L2)
        if abs(c2) > 1:
            return q_seed, _info(b, q_seed, p_t, R_t) | {"failed": "out of reach"}
        g1, g2 = np.arctan2(self.a1[1], self.a1[0]), np.arctan2(self.a2[1], self.a2[0])
        best = None
        for elbow in (1, -1):
            b2 = elbow * np.arccos(c2)                           # angle between link 1 and link 2 directions
            b1 = np.arctan2(dW[1], dW[0]) - np.arctan2(L2 * np.sin(b2), L1 + L2 * np.cos(b2))
            t1 = b1 - g1                                         # rotation of link 1 from its zero direction
            t2 = (b1 + b2) - g2 - t1                             # extra rotation of link 2
            t3 = tsum - t1 - t2
            q = np.array(q_seed, float)
            q[0] = pan
            q[1:4] = -np.array([t1, t2, t3]) * np.array(self.sign)
            q[1:4] = (q[1:4] + np.pi) % (2 * np.pi) - np.pi
            if np.any(q[:4] < b.lo[:4] - 1e-6) or np.any(q[:4] > b.hi[:4] + 1e-6):
                continue
            info = _info(b, q, p_t, R_t)
            if best is None or info["pos_err_m"] < best[1]["pos_err_m"]:
                best = (q, info)
        if best is None:
            return q_seed, _info(b, q_seed, p_t, R_t) | {"failed": "joint limits"}
        q, _ = best
        _, R = b.fk(q)
        yaw_err = np.arctan2(R[:, 0] @ R_t[:, 1] - R[:, 1] @ R_t[:, 0], R[:, 0] @ R_t[:, 0] + R[:, 1] @ R_t[:, 1])
        for cand in (q[4] - yaw_err, q[4] + yaw_err):
            qq = q.copy(); qq[4] = np.clip(cand, b.lo[4], b.hi[4])
            if _info(b, qq, p_t, R_t)["rot_err_rad"] < _info(b, q, p_t, R_t)["rot_err_rad"]:
                q = qq
        return q, _info(b, q, p_t, R_t)

    def solve(self, p_t, R_t, q_seed, passes=6):
        """The SO-101 TCP sits off the wrist-roll axis (one fixed jaw, one moving jaw), so setting the roll for the
        jaw yaw moves the TCP. Fixed-point loop: closed-form solve, then shift the target by the remaining error."""
        target = np.array(p_t, float)
        q, info = self._solve_once(target, R_t, q_seed)
        if "failed" in info:
            # enumerate approach tilts about the arm-plane normal until a branch respects the joint limits
            rel = p_t - self.base
            e1p = rel - self.zaxis * (rel @ self.zaxis); e1p /= max(np.linalg.norm(e1p), 1e-9)
            axis = np.cross(self.zaxis, e1p)
            for deg in (2, -2, 4, -4, 6, -6, 8, -8, 10, -10):
                t = np.radians(deg); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                Rtilt = (np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K) @ R_t
                q, info = self._solve_once(target, Rtilt, q_seed)
                if "failed" not in info:
                    R_t = Rtilt
                    break
        for _ in range(passes):
            if "failed" in info or info["pos_err_m"] < 2e-4:
                break
            target = target + (p_t - self.b.fk(q)[0])
            q_new, info_new = self._solve_once(target, R_t, q)
            if "failed" in info_new:
                break
            q = q_new
            info = _info(self.b, q, p_t, R_t)
        return q, _info(self.b, q, p_t, R_t) | ({"failed": info["failed"]} if "failed" in info else {})


def _info(body, q, p_t, R_t):
    p, R = body.fk(q)
    return {"pos_err_m": float(np.linalg.norm(p_t - p)), "rot_err_rad": float(np.linalg.norm(rot_err(R_t, R))),
            "axis_err_rad": float(np.arccos(np.clip(R[:, 2] @ R_t[:, 2], -1, 1)))}


class EndpointSolver:
    def __init__(self, body, method, body_name):
        self.b, self.method, self.body_name = body, method, body_name
        self.analytic = AnalyticSO101(body) if (method == "analytic" and body_name == "so101") else None
        self.taught = {}

    def __call__(self, p_t, R_t, q0, key=None, nominal=None):
        b, m = self.b, self.method
        if m == "analytic":
            if self.analytic is None:
                return q0, _info(b, q0, p_t, R_t) | {"failed": "no closed form for this geometry"}
            return self.analytic.solve(p_t, R_t, q0)
        if m == "newton":
            return newton(b, p_t, R_t, q0)
        if m == "lm":
            return lm(b, p_t, R_t, q0)
        if m == "ccd":
            return ccd(b, p_t, R_t, q0)
        if m == "constrained":
            # a 5-DOF arm cannot match a full 6D target in general: constrain position, minimize approach-axis error
            return constrained(b, p_t, R_t, q0, axis_only=(b.n < 6))
        if m == "taught":
            if key not in self.taught:
                q, _ = constrained(b, nominal, R_t, q0, axis_only=(b.n < 6))
                self.taught[key] = q
            q = self.taught[key]
            return q, _info(b, q, p_t, R_t)
        raise ValueError(m)
