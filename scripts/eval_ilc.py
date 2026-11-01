"""Iterative learning control (decision-tree layer 8, the last classical method): repeated trials per layout.

ILC needs repetition, so each classical layout (same seeds as scripts/eval_classical.py) is run K times with the
learned feedforward carried from trial to trial. Trial 1 equals the base law (pdg on the Franka's torque interface,
pd on the SO-101's servos). Reports success and joint tracking RMS per trial; the final trial is the ILC result.

Usage: python scripts/eval_ilc.py <body> <episodes> [trials]
Writes results/classical/<body>__control-ilc.json (classical format, final trial) with the per-trial curve.
"""
import json
import pathlib
import sys
import time

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody.classical.control import ILCMemory  # noqa: E402
from xbody.classical.pipeline import DEFAULT, Episode  # noqa: E402
from xbody.scene import sample_layout  # noqa: E402
from xbody.stats import mcnemar_exact, wilson  # noqa: E402


def main():
    body, n = sys.argv[1], int(sys.argv[2])
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    proto = yaml.safe_load((ROOT / "protocol.yaml").read_text())
    per_trial = [[] for _ in range(trials)]
    t0 = time.time()
    for ep in range(n):
        seed = 20261028 + ep
        cube_xy = sample_layout(np.random.default_rng(seed), body, region_y=(-0.08, 0.06))
        mem = ILCMemory(None, 7 if body == "franka" else 5, None, None)
        for k in range(trials):
            e = Episode(body, {"control": "ilc", "ilc_memory": mem}, seed, proto["stages"])
            mem.base = e.cfg["ilc_base"]
            r = e.run(cube_xy)
            r["ilc_rms_rad"] = mem.end_trial() if mem.e else float("nan")
            per_trial[k].append(r)
            print(f"{body} ep {ep} trial {k + 1}: stage {r['stage']} track_rms {r['track_rms_rad']:.5f}", flush=True)
    final = per_trial[-1]
    succ = sum(r["success"] for r in final)
    first = np.array([r["stage"] == 5 for r in per_trial[0]])
    last = np.array([r["stage"] == 5 for r in final])
    out = {"stack": "S1-full", "body": body, "cfg": {**DEFAULT, "control": "ilc"}, "resolved": {"actuator": e.cfg["actuator"],
           "control": "ilc", "ilc_base": e.cfg["ilc_base"]}, "changed": {"control": "ilc"}, "episodes": n,
           "success": succ, "rate": succ / n, "wilson95": wilson(succ, n),
           "mean_stage": float(np.mean([r["stage"] for r in final])), "stages": [r["stage"] for r in final],
           "post_contact_rate": float(np.mean([r["post_contact"] for r in final])),
           "track_rms_rad": float(np.mean([r["track_rms_rad"] for r in final])),
           "not_applicable": final[0]["not_applicable"], "trials": trials,
           "per_trial": [{"trial": k + 1, "success": int(sum(r["success"] for r in rows)),
                          "track_rms_rad": float(np.mean([r["track_rms_rad"] for r in rows]))}
                         for k, rows in enumerate(per_trial)],
           "first_vs_last_mcnemar_p": mcnemar_exact(first, last), "sec_per_episode": (time.time() - t0) / n}
    (ROOT / f"results/classical/{body}__control-ilc.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "stages"}))


if __name__ == "__main__":
    main()
