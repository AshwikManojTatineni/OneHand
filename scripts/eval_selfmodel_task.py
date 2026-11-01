"""Does self-modeling change task success? Classical pick-and-place on a mis-calibrated arm.

For each sampled real arm (calibration errors from xbody/calibration.ERRORS): teleoperate it through its nominal
model for 400 Cartesian poses, logging encoder readings and the TCP pose measured in the world frame (1 mm / 0.3
deg noise, the oracle-sensor level; the sensor-swap benchmark shows monocular ChArUco PnP gets close to it).
Calibrate the model file from those logs, then run the classical default pipeline on the 20 classical layouts:

  ideal        the arm matches its model file (the classical benchmark's default cell)
  nominal      real arm, pipeline plans with the nominal model file
  self-model   real arm, pipeline plans with the calibrated model file

Usage: MUJOCO_GL=osmesa python scripts/eval_selfmodel_task.py <franka|so101> [arms] [layouts]
Writes results/selfmodel_task/<body>.json.
"""
import json
import os
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml

BODY = sys.argv[1] if len(sys.argv) > 1 else "franka"
os.environ["SELFMODEL_BODY"] = BODY
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import selfmodel_teleop as tele  # noqa: E402
from xbody import selfmodel_diff as sd  # noqa: E402
from xbody.bodies import SPECS  # noqa: E402
from xbody.calibration import ERRORS, ModelFK, calibrate, perturb  # noqa: E402
from xbody.classical.pipeline import Episode, real_interface  # noqa: E402
from xbody.scene import build_task, sample_layout  # noqa: E402
from xbody.stats import mcnemar_exact, wilson  # noqa: E402

SPEC = SPECS[BODY]
PROTO = yaml.safe_load((ROOT / "protocol.yaml").read_text())["stages"]
TORQUE = real_interface(BODY) == "torque"


def layouts(n):
    return [sample_layout(np.random.default_rng(20261028 + ep), BODY, region_y=(-0.08, 0.06)) for ep in range(n)]


def run_episodes(cfg, n):
    out = []
    for ep, xy in enumerate(layouts(n)):
        r = Episode(BODY, cfg, 20261028 + ep, PROTO).run(xy)
        out.append({"stage": int(r["stage"]), "post_contact": bool(r["post_contact"])})
    return out


def fk_error(fk_model, fk_true, lo, hi, n=300, seed=0):
    rng = np.random.default_rng(seed)
    e = []
    for q in rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), (n, len(lo))):
        e.append(np.linalg.norm(fk_model(q)[:3, 3] - fk_true(q)[:3, 3]) * 1000)
    return float(np.median(e)), float(np.percentile(e, 95))


def arm_job(args):
    seed, n = args
    t0 = time.time()
    err = ERRORS[BODY]
    nominal = build_task(BODY, torque=TORQUE)
    real = build_task(BODY, torque=TORQUE)
    perturb(real, SPEC, seed, err)
    rng = np.random.default_rng(1000 + seed)
    cmd = tele.teleop_commands("cartesian", 400, rng, tele.nominal_body())
    fk_real = ModelFK(real, SPEC)
    Q = cmd + rng.normal(0, tele.ENC_NOISE, cmd.shape)      # encoders read the commanded joints
    T = np.array([tele.measure(rng, fk_real(q), "mid") for q in cmd])
    cal, info = calibrate(nominal, SPEC, Q, T)
    fk_nom, fk_cal = ModelFK(nominal, SPEC), ModelFK(cal, SPEC)
    lo, hi = fk_real.m.jnt_range[[int(j) for j in _jids(real)]].T
    info["fk_nominal_mm"] = fk_error(fk_nom, fk_real, lo, hi)
    info["fk_selfmodel_mm"] = fk_error(fk_cal, fk_real, lo, hi)
    res = {"seed": seed, "calibration": info,
           "nominal": run_episodes({"real_arm": (seed, err)}, n),
           "self-model": run_episodes({"real_arm": (seed, err), "belief_model": cal}, n),
           "seconds": time.time() - t0}
    print(f"{BODY} arm {seed}: FK nominal {info['fk_nominal_mm'][0]:.1f} mm -> self-model {info['fk_selfmodel_mm'][0]:.1f} mm | "
          f"success nominal {sum(r['stage'] == 5 for r in res['nominal'])}/{n}, "
          f"self-model {sum(r['stage'] == 5 for r in res['self-model'])}/{n} ({res['seconds']:.0f}s)", flush=True)
    return res


def _jids(m):
    import mujoco
    return [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in SPEC["arm"]]


def ideal_job(n):
    return {"ideal": run_episodes({}, n)}


def main():
    arms = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    with Pool(4) as pool:
        ideal = pool.apply_async(ideal_job, (n,))
        arm_res = pool.map(arm_job, [(s, n) for s in range(arms)])
        ideal = ideal.get()["ideal"]
    succ = lambda rows: np.array([r["stage"] == 5 for r in rows])
    nom = np.concatenate([succ(a["nominal"]) for a in arm_res])
    sm = np.concatenate([succ(a["self-model"]) for a in arm_res])
    out = {"body": BODY, "errors": ERRORS[BODY], "arms": arms, "layouts": n,
           "ideal": {"success": int(succ(ideal).sum()), "n": n, "ci": wilson(int(succ(ideal).sum()), n),
                     "stages": [r["stage"] for r in ideal]},
           "nominal": {"success": int(nom.sum()), "n": len(nom), "ci": wilson(int(nom.sum()), len(nom))},
           "self-model": {"success": int(sm.sum()), "n": len(sm), "ci": wilson(int(sm.sum()), len(sm))},
           "nominal_vs_selfmodel_mcnemar_p": mcnemar_exact(nom, sm), "per_arm": arm_res}
    od = ROOT / "results/selfmodel_task"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{BODY}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_arm"}))


if __name__ == "__main__":
    main()
