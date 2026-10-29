"""Summary of the teleop self-model pilot (scripts/selfmodel_teleop.py): results/selfmodel_teleop/SUMMARY.md."""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody.stats import mcnemar_exact, wilson  # noqa: E402

OUT = ROOT / "results/selfmodel_teleop"
MODELS = ["nominal", "diff_only", "diff_abs", "calib"]
LABEL = {"nominal": "Nominal model file", "diff_only": "Differential only (blind)",
         "diff_abs": "Differential + absolute (blind)", "calib": "Calibrated nominal", "local_jac": "Local Jacobian (model-free)"}


def pick(cells, mode, steps, noise):
    return sorted([c for c in cells if c["mode"] == mode and c["steps"] == steps and c["noise"] == noise],
                  key=lambda c: c["seed"])


def main():
    cells = json.loads((OUT / "cells.json").read_text())
    md = ["# Teleop self-model pilot (Franka): kinematics from encoder and extrinsic changes", "",
          "The arm is teleoperated through its nominal kinematic model, but differs from it by sampled calibration",
          "errors (2 mm link offsets, 0.2 deg axis tilts, 0.5 deg encoder offsets, 1 mm tool offset). It logs encoder",
          "readings and the measured pose of a frame on the hand. The self-models rebuild the kinematics from those logs",
          "alone. 5 sampled arms per row. FK error is measured on 300 random configurations over the whole joint range.",
          "Reach: IK on the model, run on the real arm, success = within 5 mm and 2 deg (40 targets per arm,",
          "the same targets for every model; p = exact McNemar against the nominal model).", ""]
    for mode, title in (("joint", "Joint-space teleop (wiggle each joint, then random walk)"),
                        ("cartesian", "Cartesian teleop (gripper moved between waypoints, pointing down)")):
        cs = pick(cells, mode, 400, "mid")
        if not cs:
            continue
        md += [f"## {title}: 400 logged poses, sensor noise 1 mm / 0.3 deg", "",
               "| Model | FK median (mm) | FK p95 (mm) | Rotation median (deg) | One-step relative error | Reach | p vs nominal |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        nom_hits = np.concatenate([c["nominal"]["reach"]["hits"] for c in cs])
        for m in MODELS:
            hits = np.concatenate([c[m]["reach"]["hits"] for c in cs])
            lo, hi = wilson(int(hits.sum()), len(hits))
            p = "" if m == "nominal" else f"{mcnemar_exact(nom_hits, hits):.2g}"
            md.append(f"| {LABEL[m]} | {np.median([c[m]['fk']['pos_mm_median'] for c in cs]):.2f} | "
                      f"{np.median([c[m]['fk']['pos_mm_p95'] for c in cs]):.2f} | "
                      f"{np.median([c[m]['fk']['rot_deg_median'] for c in cs]):.2f} | "
                      f"{np.mean([c[m]['one_step']['rel_pos'] for c in cs]):.4f} | "
                      f"{int(hits.sum())}/{len(hits)} ({100 * lo:.0f}-{100 * hi:.0f}%) | {p} |")
        md.append(f"| {LABEL['local_jac']} | - | - | - | {np.mean([c['local_jac']['one_step']['rel_pos'] for c in cs]):.4f} | - | |")
        md.append("")
    for key, values, title in (("steps", (25, 50, 100, 200, 400, 800), "How much teleop is needed (noise 1 mm / 0.3 deg)"),
                               ("noise", ("none", "low", "mid", "high"), "Sensor noise (400 logged poses)")):
        md += [f"## {title}", "", "FK median error in mm (median over arms), reach successes out of 200.", "",
               "| Teleop | " + " | ".join(str(v) for v in values) + " |", "| --- |" + " --- |" * len(values)]
        for mode in ("joint", "cartesian"):
            for m in MODELS:
                row = []
                for v in values:
                    cs = pick(cells, mode, v if key == "steps" else 400, v if key == "noise" else "mid")
                    row.append("-" if not cs else f"{np.median([c[m]['fk']['pos_mm_median'] for c in cs]):.1f} mm, "
                               f"{sum(c[m]['reach']['success'] for c in cs)}/{40 * len(cs)}")
                md.append(f"| {mode}: {LABEL[m]} | " + " | ".join(row) + " |")
        md.append("")
    ms_path = OUT / "maniskill_same_data.json"
    if ms_path.exists():
        ms = json.loads(ms_path.read_text())
        md += ["## Same data as the current self-model (20 recorded ManiSkill Franka pick-cube episodes)", "",
               "Fit on 10 episodes, test on the other 10, both ways (480 test transitions). Relative error = mean squared",
               "error of the predicted TCP change / mean squared TCP change, the current readout's normalisation.",
               "The current readout (frozen TD-MPC2 latent) predicts from the commanded action; the differential",
               "self-models predict from the measured joint change. They answer different questions: \"what will",
               "this command do\" against \"what does this joint change do to my hand\".", "",
               "| Sensor noise | Differential only | Differential + absolute | Local Jacobian | Current readout (latent) |",
               "| --- | --- | --- | --- | --- |"]
        for r in ms:
            f = lambda k: np.mean([r[x][k] for x in ("fold_a", "fold_b")])
            md.append(f"| {r['noise']} | {f('diff_only_rel_pos'):.2g} | {f('diff_abs_rel_pos'):.2g} | "
                      f"{f('local_jac_rel_pos'):.2g} | {ms[0].get('newt_latent_rel', float('nan')):.3f} |")
        md.append("")
    (OUT / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
