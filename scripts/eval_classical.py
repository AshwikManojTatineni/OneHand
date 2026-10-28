"""Full classical (S1) benchmark: one decision-tree configuration on one body, N seeded episodes.

Usage: python scripts/eval_classical.py <franka|so101> <episodes> [layer=method ...] [--tag NAME]
Example: python scripts/eval_classical.py so101 20 planner=prmstar
"""
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody.classical.pipeline import DEFAULT, Episode  # noqa: E402
from xbody.scene import sample_layout  # noqa: E402
from xbody.stats import wilson  # noqa: E402


def main():
    body, n = sys.argv[1], int(sys.argv[2])
    cfg = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a and not a.startswith("--"))
    proto = yaml.safe_load((ROOT / "protocol.yaml").read_text())
    rows, t0 = [], time.time()
    for ep in range(n):
        seed = 20261028 + ep
        rng = np.random.default_rng(seed)
        cube_xy = sample_layout(rng, body, region_y=(-0.08, 0.06))
        e = Episode(body, cfg, seed, proto["stages"])
        r = e.run(cube_xy)
        r["ep"] = ep
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    k = sum(r["success"] for r in rows)
    full = {**DEFAULT, **cfg}
    tag = "_".join(f"{key}-{val}" for key, val in cfg.items()) or "default"
    out = {"stack": "S1-full", "body": body, "cfg": full, "resolved": e.cfg, "changed": cfg, "episodes": n, "success": k,
           "rate": k / n, "wilson95": wilson(k, n), "mean_stage": float(np.mean([r["stage"] for r in rows])),
           "stages": [r["stage"] for r in rows], "post_contact_rate": float(np.mean([r["post_contact"] for r in rows])),
           "plan_fail_rate": float(np.mean([r["plan_fail"] > 0 for r in rows])),
           "plan_time_s": float(np.mean([r["plan_time"] for r in rows])),
           "path_len_rad": float(np.mean([r["path_len"] for r in rows])),
           "ik_err_mm": float(np.mean([r.get("ik_err_mm", np.nan) for r in rows])),
           "track_rms_rad": float(np.mean([r["track_rms_rad"] for r in rows])),
           "sat_frac": float(np.mean([r["sat_frac"] for r in rows])),
           "not_applicable": rows[0]["not_applicable"],
           "fail_layers": {fl: sum(r.get("fail_layer") == fl for r in rows) for fl in ("perception", "planning", "execution")},
           "sec_per_episode": (time.time() - t0) / n}
    od = ROOT / "results/classical"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{body}__{tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k2: v for k2, v in out.items() if k2 != "stages"}))


if __name__ == "__main__":
    main()
