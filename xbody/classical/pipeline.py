"""One episode of the full classical stack (S1): every decision-tree layer is a swappable component.

cfg keys (defaults in DEFAULT): input, world, endpoint, planner, timing, tracking, conversion, control, actuator.
Layer dependencies follow the tree: conversion variants run under Cartesian tracking; task-space torque laws run
under Cartesian torque tracking on a torque interface; joint laws other than p/pd/pid need a torque interface;
input "camera" is the visual-servo branch (no map, no planner, direct visual approach).
"""
import os
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from xbody.bodies import SPECS, robot_body_ids, tcp_frame  # noqa: E402
from xbody.classical import control as C  # noqa: E402
from xbody.classical.ik import EndpointSolver  # noqa: E402
from xbody.classical.perception import Perceiver, estimate_cube_pose, obstacle_points  # noqa: E402
from xbody.classical.planning import PLANNERS, Space, path_length, shortcut  # noqa: E402
from xbody.classical.timing import TIMING, Traj  # noqa: E402
from xbody.classical.worldmodel import RobotPoints, WorldModel  # noqa: E402
from xbody.kinematics import Body  # noqa: E402
from xbody.scene import CLASSICAL, build_task  # noqa: E402
from xbody.stages import StageScorer, StepState  # noqa: E402

DEFAULT = dict(input="pointcloud", world="primitives", endpoint="constrained", planner="rrtconnect", timing="toppra",
               tracking="joint", conversion="hqp", control=None, actuator=None)
# control=None resolves by interface: PD + gravity compensation on a torque interface (as every torque-controlled
# arm's joint controller does), servo PD on an embedded position servo. Plain "pd" on a torque arm is a variant.
R_DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
TRANSIT_Z, PLACE_Z = 0.09, 0.022


def real_interface(body):
    return "torque" if SPECS[body]["torque_capable"] else "cascade"


class Episode:
    def __init__(self, body, cfg, seed, scorer_cfg):
        self.body, self.spec = body, SPECS[body]
        self.cfg = {**DEFAULT, **cfg}
        if self.cfg["actuator"] is None:
            self.cfg["actuator"] = real_interface(body)
        c = self.cfg
        if c["control"] is None:
            c["control"] = "pdg" if c["actuator"] == "torque" else "pd"
        if c["control"] not in ("p", "pd", "pid") or c["tracking"] == "carttorque":
            c["actuator"] = "torque"
        self.m = build_task(body, torque=(c["actuator"] == "torque"))
        self.d = mujoco.MjData(self.m)
        bn, off, q = tcp_frame(self.m, self.spec)
        self.kin = Body(self.m, self.spec["arm"], bn, off, q)
        self.rb = robot_body_ids(self.m, self.spec)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.scorer = StageScorer(scorer_cfg)
        self.metrics = dict(ik_fail=0, plan_fail=0, plan_time=0.0, path_len=0.0, post_contact=False, sat_frac=0.0,
                            track_rms=0.0, n_steps=0, not_applicable="")
        m = self.m
        self.arm_act = [i for i in range(m.nu) if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                        and m.actuator_trnid[i, 0] in self.kin.jids]
        self.arm_act.sort(key=lambda i: self.kin.jids.index(m.actuator_trnid[i, 0]))
        self.grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, self.spec["gripper_actuator"])
        self.cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.post = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "post")
        self.cube_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
        self.plate = m.site_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "plate")][:2].copy()
        jaw_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.spec["jaw_bodies"]]
        self.jaw_geoms = [set(np.where(m.geom_bodyid == j)[0]) for j in jaw_ids]
        self.robot_geoms = set(np.where(np.isin(m.geom_bodyid, list(self.rb)))[0])
        self.lim = dict(vmax=self.spec["vmax"], amax=self.spec["amax"], jmax=self.spec["jmax"])
        self.umax = m.actuator_ctrlrange[self.arm_act, 1].copy() if c["actuator"] == "torque" else None
        self.t = 0.0
        self.dyn = C.Dyn(m, self.kin)

    # ------------------------------------------------------------------ setup
    def reset(self, cube_xy):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[self.kin.qadr] = self.spec["home"]
        d.qpos[self.cube_q:self.cube_q + 3] = [cube_xy[0], cube_xy[1], 0.015]
        d.qpos[self.cube_q + 3:self.cube_q + 7] = [1, 0, 0, 0]
        d.ctrl[self.grip] = self.spec["gripper_open"]
        mujoco.mj_forward(m, d)
        c = self.cfg
        self.law = None
        if c["tracking"] == "carttorque":
            self.law = C.TaskLaw(c["control"] if c["control"] in C.TASK_LAWS else "osc", self.dyn, self.kin, m.opt.timestep)
        else:
            self.law = C.JointLaw(c["control"], self.dyn, m.opt.timestep, c["actuator"], rng=self.rng)
            self.law.arm_act, self.law.umax = self.arm_act, (self.umax if self.umax is not None else np.ones(self.kin.n))
        self.q_hold = np.array(self.spec["home"], float)

    # ------------------------------------------------------------------ perception and world model (layers 1-2)
    def perceive(self):
        c, m, d = self.cfg, self.m, self.d
        off = np.r_[self.spec["layout_offset"], 0]
        bounds = ([off[0] - 0.05, -0.25, 0.0], [off[0] + 0.45, 0.40, 0.35])
        self.rp = RobotPoints(m, self.rb)
        if c["input"] == "known":
            cube = d.xpos[self.cube].copy()
            ox, oy = CLASSICAL["post_xy"]
            th = np.linspace(0, 2 * np.pi, 40); zz = np.linspace(0.01, CLASSICAL["post_h"], 15)
            obs = np.array([[ox + off[0] + CLASSICAL["post_r"] * np.cos(a), oy + CLASSICAL["post_r"] * np.sin(a), z]
                            for a in th for z in zz])
            frames = None
        else:
            per = Perceiver(m, d, self.rb)
            fr = per.frames()
            P, red = per.point_cloud(fr)
            cube = estimate_cube_pose(P, red)
            obs = obstacle_points(P, red, cube)
            frames = []
            for cam, rgb, depth, seg, Pw in fr:
                cid = per.cam_ids[cam]
                Rc, tc = d.cam_xmat[cid].reshape(3, 3).copy(), d.cam_xpos[cid].copy()
                f = 0.5 * per.H / np.tan(np.radians(m.cam_fovy[cid]) / 2)
                robot = np.isin(seg, list(per.robot_geoms))
                keep = (Pw[:, 2] > 0.008) & ~robot.reshape(-1) & (depth.reshape(-1) < 2)
                frames.append((cam, rgb, np.where(robot, np.inf, depth), seg, Pw[keep],
                               tc if c["world"] == "octomap" else (Rc, tc, f, per.H, per.W)))
            per.close()
            if cube is None:
                return False
        self.cube_est = cube
        if c["input"] == "camera":
            return True
        kind = c["world"]
        self.wm = WorldModel(kind, obstacle_pts=obs, frames=frames, model=m, data=d, post_geom=self.post,
                             bounds=bounds, robot_points=self.rp)
        self.wm.cube_geom = self.cube_geom
        return True

    def clearance(self, q_arm, holding=False):
        x = self.d.qpos.copy(); x[self.kin.qadr] = q_arm
        if holding and self.wm.kind == "mesh":
            # the held cube moves with the TCP: place it there for the exact-geometry check
            p, _ = self.kin.fk(q_arm)
            x[self.cube_q:self.cube_q + 3] = p
        return self.wm.clearance_q(x)

    # ------------------------------------------------------------------ endpoints (layer 3)
    def endpoints(self):
        c, kin = self.cfg, self.kin
        ik = EndpointSolver(kin, c["endpoint"], self.body)
        if c["endpoint"] == "analytic" and ik.analytic is None:
            self.metrics["not_applicable"] = "analytic IK: no closed form for this arm"
            return None
        cube = self.cube_est
        off = np.r_[self.spec["layout_offset"], 0]
        nominal = {"pre": np.r_[0.22, 0.0, TRANSIT_Z] + off, "grasp": np.r_[0.22, 0.0, 0.015] + off}
        plate = np.r_[self.plate, 0]
        best = None
        q0 = np.array(self.spec["home"], float)
        for yaw in (0, 90, 180, 270):   # the cube is symmetric; the arm's hand is not
            t = np.radians(yaw)
            Rt = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]) @ R_DOWN
            targets = [("pre", np.r_[cube[:2], TRANSIT_Z]), ("grasp", np.r_[cube[:2], cube[2]]),
                       ("prep", plate + [0, 0, TRANSIT_Z]), ("place", plate + [0, 0, PLACE_Z])]
            qs, errs, qq = {}, [], q0
            for name, p in targets:
                qq, info = ik(p, Rt, qq, key=(name, yaw), nominal=nominal.get(name, p))
                qs[name] = qq; errs.append(info["pos_err_m"])
            score = min(self.clearance(qs["pre"]), self.clearance(qs["grasp"]),
                        self.clearance(qs["prep"], True), self.clearance(qs["place"], True)) - 10 * max(errs)
            if best is None or score > best[0]:
                best = (score, qs, errs, Rt)
        _, qs, errs, Rt = best
        self.metrics["ik_fail"] = int(sum(e > 0.005 for e in errs))
        self.metrics["ik_err_mm"] = 1000 * float(max(errs))
        self.R_task = Rt
        return qs

    # ------------------------------------------------------------------ planning + timing (layers 4-5)
    def plan(self, qa, qb, holding):
        c = self.cfg
        if holding:
            self.rp.attached = (self.kin.bid, (np.random.default_rng(0).uniform(-0.015, 0.015, (30, 3))
                                               + self.kin.tcp_offset))
        sp = Space(self.kin.lo, self.kin.hi, lambda q: self.clearance(q, holding), budget_s=3.0, seed=self.seed)
        t0 = time.time()
        try:
            path = PLANNERS[c["planner"]](sp, qa, qb)
        except Exception:
            path = None
        if path is not None:
            path = shortcut(sp, path)          # with the held cube still attached to the robot model
        self.metrics["plan_time"] += time.time() - t0
        self.rp.attached = None
        if path is None:
            self.metrics["plan_fail"] += 1
            return None
        self.metrics["path_len"] += path_length(path)
        return path

    def timed(self, path):
        return TIMING[self.cfg["timing"]](path, self.lim)

    # ------------------------------------------------------------------ execution (layers 6-9)
    def _wrench_z(self):
        f = np.zeros(6); fz = 0.0
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            if con.geom1 in self.robot_geoms or con.geom2 in self.robot_geoms:
                mujoco.mj_contactForce(self.m, self.d, i, f)
                fz += abs(f[0]) * con.frame[2] * (1 if con.geom1 in self.robot_geoms else -1)
        return fz

    def _step(self, ctrl_arm):
        d = self.d
        if self.cfg["actuator"] == "torque":
            ctrl_arm = np.clip(ctrl_arm, -self.umax, self.umax)
            self.metrics["sat_frac"] += float(np.mean(np.abs(ctrl_arm) >= 0.999 * self.umax))
        d.ctrl[self.arm_act] = ctrl_arm
        mujoco.mj_step(self.m, d)
        self.t += self.m.opt.timestep
        self.metrics["n_steps"] += 1
        for i in range(d.ncon):
            con = d.contact[i]
            if self.post in (con.geom1, con.geom2) and (con.geom1 in self.robot_geoms or con.geom2 in self.robot_geoms
                                                        or self.cube_geom in (con.geom1, con.geom2)):
                self.metrics["post_contact"] = True
        self.scorer.update(self.state())

    def state(self):
        d = self.d
        tcp, _ = self.kin.fk(d.qpos[self.kin.qadr])
        touch = [False, False]
        for i in range(d.ncon):
            con = d.contact[i]
            pair = {con.geom1, con.geom2}
            if self.cube_geom in pair:
                other = (pair - {self.cube_geom}).pop() if len(pair) == 2 else -1
                for k in (0, 1):
                    touch[k] = touch[k] or other in self.jaw_geoms[k]
        return StepState(t=self.t, tcp=tcp, obj=d.xpos[self.cube].copy(), obj_vel=float(np.linalg.norm(d.cvel[self.cube][3:])),
                         both_jaws_touch=all(touch), jaws_open=bool(d.ctrl[self.grip] == self.spec["gripper_open"]),
                         table_z=0.015, container_xy=self.plate, container_r=0.05)

    def hold(self, seconds, grip=None):
        if grip is not None:
            self.d.ctrl[self.grip] = self.spec["gripper_open" if grip == "open" else "gripper_closed"]
        n = int(seconds / self.m.opt.timestep)
        tr = Traj(np.array([0, seconds + 1]), np.vstack([self.q_hold, self.q_hold]))
        return self.track(tr, n_override=n)

    def track(self, tr, n_override=None, contact=False):
        """Follow a timed joint reference with the configured tracking mode. Returns False if a layer cannot run."""
        c, d, kin = self.cfg, self.d, self.kin
        dt = self.m.opt.timestep
        n = n_override or int(tr.duration / dt) + int(0.3 / dt)
        mode = c["tracking"]
        conv = None
        if mode in ("cart", "admittance") or c["tracking"] in ("ibvs", "pbvs", "vs25"):
            conv = C.Converter(c["conversion"], kin, clearance_grad=self._clear_grad if c["conversion"] in ("qp", "cbf") else None)
        q_cmd = d.qpos[kin.qadr].copy()
        z_adm, zd_adm = 0.0, 0.0
        err2 = 0.0
        for i in range(n):
            t = i * dt
            q_ref, qd_ref, qdd_ref = tr.at(t)
            q = d.qpos[kin.qadr].copy()
            err2 += float(np.sum((q_ref - q) ** 2))
            if mode == "joint" or (mode in ("ibvs", "pbvs", "vs25")):
                u = self.law(d, q_ref, qd_ref, qdd_ref)
            elif mode == "carttorque":
                p_ref, R_ref = kin.fk(q_ref)
                p_nxt, _ = kin.fk(tr.at(t + dt)[0])
                v = np.r_[(p_nxt - p_ref) / dt, np.zeros(self.law.n_task - 3)]
                u = self.law(d, p_ref, R_ref, v, self.q_hold, self._wrench_z(), contact)
            else:
                p_ref, R_ref = kin.fk(q_ref)
                if mode == "admittance":
                    fz = self._wrench_z()
                    zdd = (fz - 40.0 * z_adm - 12.0 * zd_adm) / 0.5
                    zd_adm += zdd * dt; z_adm = np.clip(z_adm + zd_adm * dt, -0.03, 0.03)
                    p_ref = p_ref + [0, 0, z_adm]
                p_nxt, _ = kin.fk(tr.at(t + dt)[0])
                e = C.task_error(kin, q, p_ref, R_ref, conv.n_task)
                v = np.r_[(p_nxt - kin.fk(q_ref)[0]) / dt, np.zeros(conv.n_task - 3)] + 20.0 * e
                qd_c = conv(q, v, dt)
                if qd_c is None:
                    self.metrics["not_applicable"] = f"conversion {c['conversion']} cannot run on this arm"
                    return False
                qd_c = np.clip(qd_c, -3.0, 3.0)
                q_cmd = np.clip(q + qd_c * dt, kin.lo, kin.hi)
                u = self.law(d, q_cmd, qd_c, np.zeros(kin.n))
            self._step(u)
            if self.scorer.stage == 5:
                pass
        self.metrics["track_rms"] += err2
        self.q_hold = tr.at(tr.duration)[0]
        return True

    def _clear_grad(self, q):
        c0 = self.clearance(q)
        g = np.zeros(self.kin.n)
        for i in range(self.kin.n):
            dq = q.copy(); dq[i] += 1e-3
            g[i] = (self.clearance(dq) - c0) / 1e-3
        return c0, g

    # ------------------------------------------------------------------ visual servoing (layer 6 VS branch)
    def servo(self, target_color, final_z, kind, timeout=5.0):
        """Drive the TCP down onto the target seen by the wrist camera (cube) using image features."""
        from xbody.classical.perception import Perceiver
        kin, d = self.kin, self.d
        per = Perceiver(self.m, d, self.rb, cams=("wrist_cam",))
        conv = C.Converter("dls", kin)
        dt = self.m.opt.timestep
        every = max(1, int(0.05 / dt))
        v_cmd = np.zeros(conv.n_task)
        q_cmd = d.qpos[kin.qadr].copy()
        for i in range(int(timeout / dt)):
            q = d.qpos[kin.qadr].copy()
            p, R = kin.fk(q)
            if i % every == 0:
                feat = per.image_features("wrist_cam")
                if feat is None:
                    break
                cid = per.cam_ids["wrist_cam"]
                Rc = d.cam_xmat[cid].reshape(3, 3)
                xn = (feat["u"] - feat["W"] / 2) / feat["f"]; yn = -(feat["v"] - feat["H"] / 2) / feat["f"]
                Z = feat["depth"]
                if kind == "pbvs":
                    target_cam = np.array([xn * Z, yn * Z, -Z])
                    target = d.cam_xpos[cid] + Rc @ target_cam
                    v_lin = 2.0 * (np.r_[target[:2], max(target[2], final_z)] - p)
                elif kind == "ibvs":
                    # point-feature interaction matrix (translation part); depth from the depth image
                    L = np.array([[-1 / Z, 0, xn / Z], [0, -1 / Z, yn / Z]])
                    s_err = np.array([xn, yn])
                    v_cam = -1.5 * np.linalg.pinv(L) @ s_err
                    v_lin = Rc @ np.array([v_cam[0], v_cam[1], 0.0])
                    v_lin[2] = 1.5 * (final_z - p[2]) * (np.linalg.norm(s_err) < 0.15)
                else:   # 2.5-D: image point error for x-y, log depth ratio for z
                    L = np.array([[-1 / Z, 0], [0, -1 / Z]])
                    v_xy = -1.5 * np.linalg.pinv(L) @ np.array([xn, yn])
                    v_lin = Rc @ np.array([v_xy[0], v_xy[1], 0.0])
                    v_lin[2] = -1.5 * np.log(max(p[2], 1e-3) / max(final_z, 1e-3)) * (np.hypot(xn, yn) < 0.15)
                v_cmd = np.r_[np.clip(v_lin, -0.15, 0.15), np.zeros(conv.n_task - 3)]
                if abs(p[2] - final_z) < 0.004 and np.hypot(xn, yn) < 0.03:
                    break
            e_rot = C.task_error(kin, q, p, self.R_task, conv.n_task)[3:]
            v = v_cmd.copy(); v[3:] = 5.0 * e_rot
            qd_c = np.clip(conv(q, v, dt), -2, 2)
            q_cmd = np.clip(q + qd_c * dt, kin.lo, kin.hi)
            self._step(self.law(d, q_cmd, qd_c, np.zeros(kin.n)))
        per.close()
        self.q_hold = d.qpos[kin.qadr].copy()
        return True

    # ------------------------------------------------------------------ the episode
    def run(self, cube_xy):
        self.reset(cube_xy)
        c = self.cfg
        if not self.perceive():
            self.metrics["fail_layer"] = "perception"
            return self.result()
        if c["input"] == "camera":
            return self.run_vs_branch()
        qs = self.endpoints()
        if qs is None:
            return self.result()
        q_home = np.array(self.spec["home"], float)
        p1 = self.plan(q_home, qs["pre"], holding=False)
        if p1 is None:
            self.metrics["fail_layer"] = "planning"
            return self.result()
        steps = [("move", p1, None), ("down", [qs["pre"], qs["grasp"]], None), ("hold", 0.6, "close"),
                 ("move", [qs["grasp"], qs["pre"]], None), ("hold", 0.6, None), ("plan", None, None),
                 ("down", [qs["prep"], qs["place"]], None), ("hold", 0.5, "open"), ("move", [qs["place"], qs["prep"]], None)]
        for kind, arg, grip in steps:
            ok = True
            if kind == "hold":
                ok = self.hold(arg, grip)
            elif kind == "plan":
                p2 = self.plan(qs["pre"], qs["prep"], holding=True)
                if p2 is None:
                    self.metrics["fail_layer"] = "planning"
                    return self.result()
                ok = self.track(self.timed(p2))
            elif kind == "down" and c["tracking"] in ("ibvs", "pbvs", "vs25") and arg[1] is qs["grasp"]:
                ok = self.servo("red", self.cube_est[2], c["tracking"])
            else:
                ok = self.track(self.timed(arg), contact=(kind == "down"))
            if not ok:
                self.metrics["fail_layer"] = "execution"
                return self.result()
        self.hold(0.6)
        return self.result()

    def run_vs_branch(self):
        """Input "camera": image features only. No map, no planner: servo onto the cube with the wrist camera, lift
        straight up, then move straight to the plate seen by the scene camera (nothing avoids the post)."""
        kin, d = self.kin, self.d
        self.R_task = R_DOWN
        q0 = d.qpos[kin.qadr].copy()
        ik = EndpointSolver(kin, "constrained", self.body)
        # first view: hover above the cube's image position at transit height (direct visual approach)
        q_pre, _ = ik(np.r_[self.cube_est[:2], TRANSIT_Z], R_DOWN, q0)
        self.track(TIMING["trapezoid"]([q0, q_pre], self.lim))
        self.servo("red", 0.015, "ibvs")
        self.hold(0.6, "close")
        q = d.qpos[kin.qadr].copy()
        q_up, _ = ik(np.r_[kin.fk(q)[0][:2], TRANSIT_Z], R_DOWN, q)
        self.track(TIMING["trapezoid"]([q, q_up], self.lim)); self.hold(0.6)
        q_pp, _ = ik(np.r_[self.plate, TRANSIT_Z], R_DOWN, q_up)
        q_pl, _ = ik(np.r_[self.plate, PLACE_Z], R_DOWN, q_pp)
        self.track(TIMING["trapezoid"]([q_up, q_pp], self.lim))
        self.track(TIMING["trapezoid"]([q_pp, q_pl], self.lim))
        self.hold(0.5, "open")
        self.track(TIMING["trapezoid"]([q_pl, q_pp], self.lim)); self.hold(0.6)
        return self.result()

    def result(self):
        mt = dict(self.metrics)
        n = max(mt.pop("n_steps"), 1)
        mt["sat_frac"] = mt["sat_frac"] / n
        mt["track_rms_rad"] = float(np.sqrt(mt.pop("track_rms") / n / self.kin.n))
        mt["stage"] = int(self.scorer.stage)
        mt["success"] = bool(self.scorer.stage == 5)
        mt["sim_time"] = self.t
        return mt
