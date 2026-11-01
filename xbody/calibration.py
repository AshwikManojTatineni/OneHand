"""Calibration errors on a robot inside any MuJoCo scene, and their identification back into the model file.

perturb(m, ...)      turns a model into a "real" arm: link offsets and tilts on every body along the chain
                     (base, one per arm joint, the TCP body), and encoder zero offsets, applied as a rotation of
                     each joint body about its own axis, so qpos is what the encoders read.
calibrate(m, Q, T)   the self-model in model-file form: per-body position and rotation corrections fitted by
                     nonlinear least squares to logged encoder readings Q and measured TCP poses T (world frame).
                     The result is an ordinary MjModel the classical pipeline can plan with.
"""
import copy

import mujoco
import numpy as np
from scipy.optimize import least_squares

from xbody import selfmodel_diff as sd
from xbody.bodies import robot_body_ids, tcp_frame

ERRORS = {   # same magnitudes as the teleop pilots, plus base placement
    "franka": dict(link_mm=2.0, axis_deg=0.2, enc_deg=0.5, base_mm=3.0, base_deg=0.3),
    "so101": dict(link_mm=1.0, axis_deg=0.5, enc_deg=1.5, base_mm=3.0, base_deg=0.3),
}


def chain_bodies(m, spec):
    """Root robot body, the body of each arm joint, and the TCP body (deduplicated, base to tip)."""
    joint_bodies = [int(m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in spec["arm"]]
    root = joint_bodies[0]
    while m.body_parentid[root] != 0:
        root = int(m.body_parentid[root])
    bn, _, _ = tcp_frame(m, spec)
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
    return list(dict.fromkeys([root, *joint_bodies, tcp])), joint_bodies, root


def _mulq(a, b):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.asarray(a, float), np.asarray(b, float))
    return q / np.linalg.norm(q)


def _quat_rv(rv):
    q = np.zeros(4)
    th = np.linalg.norm(rv)
    mujoco.mju_axisAngle2Quat(q, rv / th if th > 1e-12 else np.array([1.0, 0, 0]), th)
    return q


def perturb(m, spec, seed, errors):
    """In place. Returns the sampled encoder offsets (rad) for reference."""
    rng = np.random.default_rng(seed)
    bodies, joint_bodies, root = chain_bodies(m, spec)
    for b in bodies:
        big = b == root
        m.body_pos[b] += rng.normal(0, (errors["base_mm"] if big else errors["link_mm"]) / 1000, 3)
        ax = rng.normal(size=3)
        ang = np.deg2rad(errors["base_deg"] if big else errors["axis_deg"]) * rng.normal()
        m.body_quat[b] = _mulq(m.body_quat[b], _quat_rv(ax / np.linalg.norm(ax) * ang))
    enc = np.deg2rad(rng.normal(0, errors["enc_deg"], len(spec["arm"])))
    for j, b, off in zip(spec["arm"], joint_bodies, enc):
        axis = m.jnt_axis[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
        # Body frame then joint rotation: R(axis, q - off) means the encoder reads q when the joint is at q - off.
        m.body_quat[b] = _mulq(m.body_quat[b], _quat_rv(-axis * off))
    return enc


class ModelFK:
    def __init__(self, m, spec):
        self.m, self.d = m, mujoco.MjData(m)
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in spec["arm"]])
        bn, off, q = tcp_frame(m, spec)
        self.bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
        self.off = np.asarray(off, float)
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, np.asarray(q if q is not None else [1, 0, 0, 0], float))
        self.Roff = Rq.reshape(3, 3)

    def __call__(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        R = self.d.xmat[self.bid].reshape(3, 3)
        return sd.pose(self.d.xpos[self.bid] + R @ self.off, R @ self.Roff)


def calibrate(m_nominal, spec, Q, T, iters=60):
    """Returns (calibrated MjModel copy, fit info). Corrections: 3 position + 3 rotation numbers per chain body."""
    m = copy.deepcopy(m_nominal)
    bodies, _, _ = chain_bodies(m, spec)
    pos0, quat0 = m.body_pos[bodies].copy(), m.body_quat[bodies].copy()
    fk = ModelFK(m, spec)
    Q, T = np.asarray(Q), np.asarray(T)

    def apply(x):
        x = x.reshape(len(bodies), 6)
        for i, b in enumerate(bodies):
            m.body_pos[b] = pos0[i] + x[i, :3]
            m.body_quat[b] = _mulq(quat0[i], _quat_rv(x[i, 3:]))

    def res(x):
        apply(x)
        out = np.empty((len(Q), 6))
        for k, q in enumerate(Q):
            P = fk(q)
            out[k, :3] = P[:3, 3] - T[k, :3, 3]
            out[k, 3:] = sd.so3_log(T[k, :3, :3].T @ P[:3, :3]) * sd.L_ROT
        return out.ravel()

    x0 = np.zeros(6 * len(bodies))
    r0 = res(x0)
    sol = least_squares(res, x0, method="trf", loss="soft_l1", f_scale=0.005, max_nfev=iters,
                        x_scale=np.tile([0.002, 0.002, 0.002, 0.01, 0.01, 0.01], len(bodies)))
    apply(sol.x)
    rms = lambda r: float(np.sqrt((r.reshape(-1, 6)[:, :3] ** 2).sum(1).mean()) * 1000)
    return m, {"rms_mm_before": rms(r0), "rms_mm_after": rms(sol.fun), "nfev": int(sol.nfev)}
