"""Interface K: body-agnostic retargeting of end-effector targets onto any arm (decision-tree layers 3 and 7).

Every solver reads the body from its MuJoCo model only: arm joints, joint limits and a TCP frame.
Conversion variants: "dls" (damped least squares on the full task) and "hqp" (prioritized: position first,
orientation in the position task's null space). The prioritized solver is a null-space stand-in for a
hierarchical QP; joint limits are enforced by clamping after each step.
"""
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Body:
    model: mujoco.MjModel
    arm_joints: list           # joint names, base to tip
    tcp_body: str              # body carrying the TCP
    tcp_offset: np.ndarray     # TCP position in that body's frame
    tcp_quat_offset: np.ndarray = None  # TCP orientation in that body's frame (w, x, y, z); identity if None

    def __post_init__(self):
        m = self.model
        self.jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.arm_joints]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([m.jnt_dofadr[j] for j in self.jids])
        self.lo = m.jnt_range[self.jids, 0].copy()
        self.hi = m.jnt_range[self.jids, 1].copy()
        self.bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.tcp_body)
        self.tcp_offset = np.asarray(self.tcp_offset, float)
        q = self.tcp_quat_offset if self.tcp_quat_offset is not None else np.array([1.0, 0, 0, 0])
        self.R_off = np.zeros(9)
        mujoco.mju_quat2Mat(self.R_off, np.asarray(q, float))
        self.R_off = self.R_off.reshape(3, 3)
        self.data = mujoco.MjData(m)

    @property
    def n(self):
        return len(self.arm_joints)

    def fk(self, q):
        d = self.data
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        R = d.xmat[self.bid].reshape(3, 3)
        return d.xpos[self.bid] + R @ self.tcp_offset, R @ self.R_off

    def jacobian(self, q):
        p, R = self.fk(q)
        mujoco.mj_comPos(self.model, self.data)
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jp, jr, p, self.bid)
        return p, R, jp[:, self.dadr], jr[:, self.dadr]


def rot_err(R_target, R):
    """Axis-angle error (world frame) that rotates R toward R_target."""
    E = R_target @ R.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, E.reshape(-1))
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def solve_ik(body, p_target, R_target=None, q0=None, method="hqp", w_rot=0.3, damping=1e-2,
             iters=200, tol_p=1e-4, step=1.0, max_step=0.2, orient="full"):
    """Returns (q, info). Works for any joint count: redundant, 6-DOF or deficient (e.g. 5-DOF SO-101).

    method: "hqp" prioritized position > orientation; "dls" weighted full task; "newton" undamped full task
    (a deliberate probe that stalls when the 6D target is unreachable).
    """
    q = np.clip(np.zeros(body.n) if q0 is None else np.array(q0, float), body.lo, body.hi)
    I = np.eye(body.n)
    for it in range(iters):
        p, R, Jp, Jr = body.jacobian(q)
        ep = p_target - p
        er = rot_err(R_target, R) if R_target is not None else np.zeros(3)
        if R_target is not None and orient == "axis":
            # Approach-axis task only: rotation about the approach axis is left free (5-DOF projection).
            z, zt = R[:, 2], R_target[:, 2]
            er = np.cross(z, zt)
            P = np.eye(3) - np.outer(z, z)
            Jr = P @ Jr
        if np.linalg.norm(ep) < tol_p and (R_target is None or method == "hqp" and it > 20):
            break
        if method == "hqp":
            Jp_pinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + damping**2 * np.eye(3))
            dq = Jp_pinv @ ep
            if R_target is not None:
                N = I - Jp_pinv @ Jp                     # null space of the position task
                JrN = Jr @ N
                dq += N @ (JrN.T @ np.linalg.solve(JrN @ JrN.T + damping * np.eye(3), er - Jr @ dq))
        elif method in ("dls", "newton"):
            J = Jp if R_target is None else np.vstack([Jp, w_rot * Jr])
            e = ep if R_target is None else np.concatenate([ep, w_rot * er])
            lam = damping if method == "dls" else 1e-9
            try:
                dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(J.shape[0]), e)
            except np.linalg.LinAlgError:   # rank-deficient task, e.g. a 6D target on a 5-DOF arm
                p, R = body.fk(q)
                return q, {"pos_err_m": float(np.linalg.norm(p_target - p)), "iters": it + 1,
                           "rot_err_rad": float(np.linalg.norm(rot_err(R_target, R))), "failed": "singular"}
        else:
            raise ValueError(method)
        dq = step * dq
        norm = np.linalg.norm(dq)
        if norm > max_step:            # trust region: large orientation errors must not drag the position
            dq *= max_step / norm
        q = np.clip(q + dq, body.lo, body.hi)
    if method == "hqp" and R_target is not None:
        # Strict priority: joint-limit clamping can let the secondary task drag the position, which a true
        # hierarchical QP forbids. Finish with position-only steps so position always wins.
        for _ in range(100):
            p, _, Jp, _ = body.jacobian(q)
            ep = p_target - p
            if np.linalg.norm(ep) < tol_p:
                break
            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + damping**2 * np.eye(3), ep)
            n = np.linalg.norm(dq)
            q = np.clip(q + (dq * max_step / n if n > max_step else dq), body.lo, body.hi)
    p, R = body.fk(q)
    if R_target is None:
        rerr = 0.0
    elif orient == "axis":
        rerr = float(np.arccos(np.clip(R[:, 2] @ R_target[:, 2], -1, 1)))
    else:
        rerr = float(np.linalg.norm(rot_err(R_target, R)))
    info = {"pos_err_m": float(np.linalg.norm(p_target - p)), "iters": it + 1, "rot_err_rad": rerr}
    return q, info


# Gripper mapping: the source policy commands an aperture in metres; each body maps it to its own actuator.
def franka_width_to_ctrl(width_m):          # Menagerie Panda tendon actuator: 0..255 for 0..0.08 m
    return float(np.clip(width_m / 0.08, 0, 1) * 255)


def so101_width_to_ctrl(width_m, closed=-0.17453, open_=1.74533, max_width_m=0.07):
    """Linear jaw-angle map; max_width_m is the jaw opening at full open, to be measured in Phase 0."""
    return float(closed + np.clip(width_m / max_width_m, 0, 1) * (open_ - closed))
