"""Summary of the sensor-swap self-modeling benchmark: results/selfmodel_sensors/SUMMARY.md.

Usage: python scripts/summarize_selfmodel_sensors.py [cells.json]
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody.stats import mcnemar_exact, wilson  # noqa: E402

OUT = ROOT / "results/selfmodel_sensors"
SENSORS = ["oracle", "mono", "stereo", "lidar"]
FRAME = {"oracle": "cam", "mono": "cam", "stereo": "cam", "lidar": "lidar"}
LEVELS = {"R0": "ideal renders", "R1": "sensor noise and calibration error", "R2": "R1 + motion blur and 15 ms latency",
          "R3": "R2 + joint compliance"}
LABEL = {"oracle": "Oracle (true pose + noise)", "mono": "Monocular camera", "stereo": "Stereo camera", "lidar": "Lidar"}


def cell_rows(cs, s):
    """Pooled numbers for one sensor over a set of arms."""
    valid = sum(c[s]["valid"] for c in cs) / sum(c[s]["frames"] for c in cs)
    sens = [c[s]["sensor_pos_mm_median"] for c in cs if "sensor_pos_mm_median" in c[s]]
    fits = [c[s]["diff_abs"] for c in cs if "fk" in c[s].get("diff_abs", {})]
    dif = [c[s]["diff_only"] for c in cs if "fk" in c[s].get("diff_only", {})]
    return valid, sens, fits, dif


def main(path=None):
    path = pathlib.Path(path) if path else OUT / "cells.json"
    cells = json.loads(path.read_text())
    md = ["# Sensor-swap self-modeling benchmark (Franka, rendered lab)", "",
          "One arm with sampled calibration errors (links, joint axes, encoder offsets, base placement, sensor mount)",
          "is teleoperated through its nominal model. Only the sensor that measures the hand changes. Each sensor's pose",
          "comes from its own pipeline on rendered data: ChArUco PnP (mono), triangulated corners (stereo), scan-to-map",
          "ICP (lidar). The self-model (differential fit, then absolute refinement, robust loss) never sees the model file.",
          "FK error: the sensor frame over 300 random configurations of the whole joint range. Reach: IK on the model,",
          "run on the real arm, success within 5 mm and 2 deg (40 targets per arm, the same targets for every model).",
          "Camera sensors model the camera frame, lidar the lidar frame; each is compared with the nominal model of the",
          "same frame.", ""]
    for level in LEVELS:
        for mode in ("cartesian", "joint"):
            cs = sorted([c for c in cells if c["level"] == level and c["mode"] == mode], key=lambda c: c["seed"])
            if not cs:
                continue
            md += [f"## {level} ({LEVELS[level]}), {mode} teleop: {len(cs)} arms x {cs[0]['steps']} frames", "",
                   "| Sensor | Valid frames | Sensor pose error (mm, median) | Self-model FK (mm, median / p95) | "
                   "Self-model reach | Differential-only reach | Nominal reach (same frame) | p vs nominal |",
                   "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for s in SENSORS:
                valid, sens, fits, dif = cell_rows(cs, s)
                nom = [c[f"nominal_{FRAME[s]}"] for c in cs]
                nh = np.concatenate([n["reach"]["hits"] for n in nom])
                if fits:
                    used = [c for c in cs if "fk" in c[s].get("diff_abs", {})]
                    hits = np.concatenate([c[s]["diff_abs"]["reach"]["hits"] for c in used])
                    nh_u = np.concatenate([c[f"nominal_{FRAME[s]}"]["reach"]["hits"] for c in used])
                    lo, hi = wilson(int(hits.sum()), len(hits))
                    fk = f"{np.median([f['fk']['pos_mm_median'] for f in fits]):.1f} / {np.median([f['fk']['pos_mm_p95'] for f in fits]):.1f}"
                    rc = f"{int(hits.sum())}/{len(hits)} ({100 * lo:.0f}-{100 * hi:.0f}%)"
                    p = f"{mcnemar_exact(nh_u, hits):.2g}"
                else:
                    fk, rc, p = "not enough valid frames", "-", ""
                dr = "-" if not dif else f"{sum(d['reach']['success'] for d in dif)}/{40 * len(dif)}"
                md.append(f"| {LABEL[s]} | {100 * valid:.0f}% | {np.median(sens):.2f} | {fk} | {rc} | {dr} | "
                          f"{int(nh.sum())}/{len(nh)} | {p} |" if sens else
                          f"| {LABEL[s]} | {100 * valid:.0f}% | - | {fk} | {rc} | {dr} | {int(nh.sum())}/{len(nh)} | |")
            st = {}
            for c in cs:
                for k, v in c["lidar_status"].items():
                    st[k] = st.get(k, 0) + v
            md += ["", "Lidar frames by outcome: " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())), ""]
    md += ["## Realism ladder at a glance (self-model reach successes, all arms pooled)", "",
           "| Teleop | Sensor | " + " | ".join(LEVELS) + " |", "| --- | --- |" + " --- |" * len(LEVELS)]
    for mode in ("cartesian", "joint"):
        for s in SENSORS:
            row = []
            for level in LEVELS:
                cs = [c for c in cells if c["level"] == level and c["mode"] == mode]
                fits = [c[s]["diff_abs"] for c in cs if "fk" in c[s].get("diff_abs", {})]
                row.append("-" if not cs else (f"{sum(f['reach']['success'] for f in fits)}/{40 * len(cs)}" if fits else "0 fits"))
            md.append(f"| {mode} | {LABEL[s]} | " + " | ".join(row) + " |")
    md.append("")
    (OUT / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
