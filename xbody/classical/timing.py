"""Decision-tree layer 5 (timing): a joint path (waypoints) becomes a timed reference q(t), qd(t), qdd(t) within
each body's velocity, acceleration and jerk limits.

  toppra     TOPP-RA (reachability analysis, toppra library) on a cubic spline through the waypoints
  topp       classic TOPP: phase-plane numerical integration (max-acceleration forward, max-deceleration backward)
             on the same spline
  totg       Kunz-Stilman style: corners replaced by bounded-deviation blends (quadratic Bezier in place of the
             circular arc), then the phase-plane profile along the blended path
  iptp       iterative parabolic time parameterization (MoveIt): segment durations scaled until accelerations fit
  trapezoid  stop at every waypoint, synchronized trapezoidal velocity per segment
  scurve     stop at every waypoint, synchronized jerk-limited (double-S) profile per segment
  ruckig     Ruckig jerk-limited online generation through the waypoints with IPTP waypoint velocities
"""
import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline

DT = 0.01


class Traj:
    def __init__(self, t, q, qd=None, qdd=None):
        self.t, self.q = np.asarray(t), np.asarray(q)
        self.qd = np.gradient(self.q, self.t, axis=0) if qd is None else np.asarray(qd)
        self.qdd = np.gradient(self.qd, self.t, axis=0) if qdd is None else np.asarray(qdd)

    @property
    def duration(self):
        return float(self.t[-1])

    def at(self, t):
        t = np.clip(t, 0, self.t[-1])
        f = lambda A: np.array([np.interp(t, self.t, A[:, j]) for j in range(A.shape[1])])
        return f(self.q), f(self.qd), f(self.qdd)


def _dedupe(path):
    out = [np.asarray(path[0], float)]
    for q in path[1:]:
        if np.linalg.norm(q - out[-1]) > 1e-6:
            out.append(np.asarray(q, float))
    if len(out) == 1:
        out.append(out[0] + 1e-6)
    return np.array(out)


def _arc_param(P):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return s / s[-1], s[-1]


# ---------------------------------------------------------------- phase-plane profile (TOPP core)
def phase_plane(qs, qs_d, qs_dd, ds, vmax, amax):
    """Time-optimal s(t) along a path with q'(s), q''(s): numerical integration of the max-acceleration curve
    forward and the max-deceleration curve backward, capped by the velocity limit curve."""
    N = len(qs)
    with np.errstate(divide="ignore", invalid="ignore"):
        sd_vel = np.nanmin(np.where(np.abs(qs_d) > 1e-9, vmax / np.abs(qs_d), np.inf), axis=1)
        # acceleration-limited max speed from the centripetal term |q'' sd^2| <= amax
        sd_acc = np.nanmin(np.where(np.abs(qs_dd) > 1e-9, np.sqrt(amax / np.abs(qs_dd)), np.inf), axis=1)
    mvc = np.minimum(sd_vel, sd_acc)
    mvc = np.minimum(mvc, 1e3)

    def sdd_bounds(i, sd):
        lo, hi = -np.inf, np.inf
        for j in range(qs.shape[1]):
            a, b = qs_d[i, j], qs_dd[i, j] * sd * sd
            if abs(a) < 1e-9:
                continue
            l, h = (-amax - b) / a, (amax - b) / a
            if a < 0:
                l, h = h, l
            lo, hi = max(lo, l), min(hi, h)
        return lo, hi

    fwd = np.zeros(N)
    for i in range(N - 1):
        _, hi = sdd_bounds(i, fwd[i])
        fwd[i + 1] = min(np.sqrt(max(fwd[i] ** 2 + 2 * max(hi, 0) * ds, 0)), mvc[i + 1])
    bwd = np.zeros(N)
    for i in range(N - 1, 0, -1):
        lo, _ = sdd_bounds(i, bwd[i])
        bwd[i - 1] = min(np.sqrt(max(bwd[i] ** 2 - 2 * min(lo, 0) * ds, 0)), mvc[i - 1])
    sd = np.minimum(fwd, bwd)
    sd = np.maximum(sd, 1e-4)
    t = np.r_[0, np.cumsum(2 * ds / (sd[:-1] + sd[1:]))]
    return t


def _resample(t_knots, q_knots):
    T = np.arange(0, t_knots[-1] + DT, DT)
    q = np.array([np.interp(T, t_knots, q_knots[:, j]) for j in range(q_knots.shape[1])]).T
    return Traj(T, q)


def _spline_profile(P, vmax, amax, N=800):
    u, L = _arc_param(P)
    cs = CubicSpline(u, P, bc_type="clamped")
    s = np.linspace(0, 1, N)
    return phase_plane(cs(s), cs(s, 1), cs(s, 2), s[1] - s[0], vmax, amax), cs(s)


def topp(path, lim):
    P = _dedupe(path)
    t, q = _spline_profile(P, lim["vmax"], lim["amax"])
    return _resample(t, q)


def toppra_(path, lim):
    import toppra as ta
    import toppra.algorithm as algo
    import toppra.constraint as constraint
    ta.setup_logging("CRITICAL")
    P = _dedupe(path)
    u, _ = _arc_param(P)
    n = P.shape[1]
    pth = ta.SplineInterpolator(u, P)
    vl = np.c_[-np.full(n, lim["vmax"]), np.full(n, lim["vmax"])]
    al = np.c_[-np.full(n, lim["amax"]), np.full(n, lim["amax"])]
    inst = algo.TOPPRA([constraint.JointVelocityConstraint(vl), constraint.JointAccelerationConstraint(al)], pth,
                       parametrizer="ParametrizeConstAccel")
    jt = inst.compute_trajectory()
    if jt is None:
        return topp(path, lim)
    T = np.arange(0, jt.duration + DT, DT)
    return Traj(T, jt(T), jt(T, 1), jt(T, 2))


def totg(path, lim, max_dev=0.05):
    """Blend each corner with a circular arc whose deviation from the corner is at most max_dev, then profile."""
    P = _dedupe(path)
    if len(P) < 3:
        return topp(P, lim)
    pts = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        u1, u2 = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
        ang = np.arccos(np.clip(u1 @ u2, -1, 1))
        if ang < 1e-3:
            pts.append(b); continue
        # quadratic Bezier with the corner as control point deviates 0.25 * d * |u2 - u1| from the corner
        d = min(4 * max_dev / np.linalg.norm(u2 - u1), 0.5 * np.linalg.norm(b - a), 0.5 * np.linalg.norm(c - b))
        p1, p2 = b - u1 * d, b + u2 * d
        for w in np.linspace(0, 1, 7):
            pts.append((1 - w) ** 2 * p1 + 2 * (1 - w) * w * b + w ** 2 * p2)
    pts.append(P[-1])
    Q = _dedupe(pts)
    u, _ = _arc_param(Q)
    cs = CubicSpline(u, Q, bc_type="clamped")
    s = np.linspace(0, 1, 800)
    t = phase_plane(cs(s), cs(s, 1), cs(s, 2), s[1] - s[0], lim["vmax"], lim["amax"])
    return _resample(t, cs(s))


def _iptp_times(P, vmax, amax, iters=100):
    seg = np.diff(P, axis=0)
    dt = np.maximum(np.max(np.abs(seg), axis=1) / vmax, 1e-3)
    for _ in range(iters):
        v = seg / dt[:, None]
        changed = False
        for i in range(1, len(P) - 1):
            acc = np.abs(v[i] - v[i - 1]) / (0.5 * (dt[i - 1] + dt[i]))
            if np.max(acc) > amax * 1.0001:
                f = np.sqrt(np.max(acc) / amax)
                dt[i - 1] *= f ** 0.5; dt[i] *= f ** 0.5; changed = True
        a0 = np.max(np.abs(v[0])) / dt[0]; a1 = np.max(np.abs(v[-1])) / dt[-1]
        if a0 > amax:
            dt[0] *= np.sqrt(a0 / amax); changed = True
        if a1 > amax:
            dt[-1] *= np.sqrt(a1 / amax); changed = True
        if not changed:
            break
    t = np.r_[0, np.cumsum(dt)]
    v = seg / dt[:, None]
    vel = np.vstack([np.zeros(P.shape[1]), 0.5 * (v[:-1] + v[1:]), np.zeros(P.shape[1])])
    return t, vel


def iptp(path, lim):
    """MoveIt-style IPTP: waypoint times from iterative scaling, waypoint velocities from neighbouring segments,
    cubic interpolation through every waypoint. Like the original, it can exceed limits between waypoints."""
    P = _dedupe(path)
    t, vel = _iptp_times(P, lim["vmax"], lim["amax"])
    h = CubicHermiteSpline(t, P, vel)
    T = np.arange(0, t[-1] + DT, DT)
    return Traj(T, h(T), h(T, 1), h(T, 2))


def _segment_profile(dist, vmax, amax, jmax=None):
    """Normalized progress u(t) in [0,1] for moving `dist` (the limiting joint) under v/a(/j) limits."""
    if dist < 1e-9:
        return np.array([0.0, DT]), np.array([1.0, 1.0])
    if jmax is None:   # trapezoid
        ta = vmax / amax
        if dist < vmax * ta:
            ta = np.sqrt(dist / amax); vp = amax * ta; tc = 0.0
        else:
            vp = vmax; tc = (dist - vmax * ta) / vmax
        T = 2 * ta + tc
        t = np.arange(0, T + DT, DT)
        s = np.where(t < ta, 0.5 * amax * t ** 2,
                     np.where(t < ta + tc, 0.5 * amax * ta ** 2 + vp * (t - ta),
                              dist - 0.5 * amax * np.maximum(T - t, 0) ** 2))
        return t, np.clip(s / dist, 0, 1)
    # double-S (Biagiotti and Melchiorri): jerk phases of length tj, constant-acceleration and cruise phases
    def phases(v):
        tj = min(amax / jmax, np.sqrt(v / jmax))
        ta = tj + max(v / min(amax, jmax * tj) - tj, 0.0) if tj > 0 else 0.0
        return tj, max(ta, 2 * tj)
    v = vmax
    tj, ta = phases(v)
    if v * ta > dist:                       # cannot reach vmax: bisect the peak speed
        lo, hi = 1e-6, vmax
        for _ in range(60):
            v = 0.5 * (lo + hi); tj, ta = phases(v)
            lo, hi = (v, hi) if v * ta < dist else (lo, v)
        tj, ta = phases(v)
    tv = max((dist - v * ta) / v, 0.0)
    T = 2 * ta + tv
    t = np.arange(0, T + DT, DT)
    a_pk = v / (ta - tj) if ta > tj else jmax * tj

    def acc(x):
        if x < tj: return jmax * x
        if x < ta - tj: return a_pk
        if x < ta: return jmax * (ta - x)
        return 0.0
    a = np.array([acc(x) if x < ta else (0.0 if x < ta + tv else -acc(x - ta - tv)) for x in t])
    vel = np.cumsum(a) * DT
    s = np.cumsum(vel) * DT
    return t, np.clip(s / max(s[-1], 1e-12), 0, 1)


def _stop_and_go(path, lim, jerk):
    P = _dedupe(path)
    T, Q, t0 = [0.0], [P[0]], 0.0
    for a, b in zip(P[:-1], P[1:]):
        d = np.max(np.abs(b - a))
        t, u = _segment_profile(d, lim["vmax"], lim["amax"], lim["jmax"] if jerk else None)
        for ti, ui in zip(t[1:], u[1:]):
            T.append(t0 + ti); Q.append(a + ui * (b - a))
        t0 += t[-1]
    return _resample(np.array(T), np.array(Q))


def trapezoid(path, lim):
    return _stop_and_go(path, lim, jerk=False)


def scurve(path, lim):
    return _stop_and_go(path, lim, jerk=True)


def ruckig_(path, lim):
    from ruckig import InputParameter, OutputParameter, Result, Ruckig
    P = _dedupe(path)
    n = P.shape[1]
    _, vel = _iptp_times(P, lim["vmax"], lim["amax"])
    otg = Ruckig(n, DT)
    inp, out = InputParameter(n), OutputParameter(n)
    inp.max_velocity, inp.max_acceleration, inp.max_jerk = [lim["vmax"]] * n, [lim["amax"]] * n, [lim["jmax"]] * n
    inp.current_position, inp.current_velocity, inp.current_acceleration = list(P[0]), [0.0] * n, [0.0] * n
    T, Q, t = [0.0], [P[0]], 0.0
    for k in range(1, len(P)):
        inp.target_position = list(P[k])
        inp.target_velocity = list(np.clip(vel[k], -lim["vmax"], lim["vmax"]))
        inp.target_acceleration = [0.0] * n
        res = Result.Working
        steps = 0
        while res == Result.Working and steps < 20000:
            res = otg.update(inp, out)
            t += DT; steps += 1
            T.append(t); Q.append(np.array(out.new_position))
            out.pass_to_input(inp)
    return _resample(np.array(T), np.array(Q))


TIMING = {"toppra": toppra_, "topp": topp, "totg": totg, "iptp": iptp, "trapezoid": trapezoid, "scurve": scurve,
          "ruckig": ruckig_}
