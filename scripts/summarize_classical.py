"""Summary of the full classical grid: per decision-tree layer, every method's success on the Franka and the
SO-101, retention (SO-101 / Franka), stage reached, post contacts, and paired tests against each body's default.

Writes results/classical/SUMMARY.md and results/classical/summary.csv. Usage: python scripts/summarize_classical.py
"""
import csv
import glob
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody.stats import holm, mcnemar_exact, wilson  # noqa: E402

LAYERS = [("input", "1 Input source"), ("world", "2 World model"), ("endpoint", "3 Endpoint IK"),
          ("planner", "4 Motion planning"), ("timing", "5 Timing"), ("tracking", "6 Tracking"),
          ("conversion", "7 Conversion (Cartesian tracking)"), ("control", "8 Control"), ("actuator", "9 Actuator")]


def layer_of(changed):
    keys = set(changed)
    if "conversion" in keys:
        return "conversion", changed["conversion"]
    if "control" in keys:
        return "control", changed["control"] + (" (task space)" if changed.get("tracking") == "carttorque" else "")
    if not keys:
        return "default", "default"
    k = next(iter(keys))
    return k, changed[k]


def main():
    R = {}
    for f in glob.glob(str(ROOT / "results/classical/*.json")):
        d = json.load(open(f))
        if "stages" not in d:
            continue
        lay, meth = layer_of(d["changed"])
        R.setdefault((lay, meth), {})[d["body"]] = d
    base = R.get(("default", "default"), {})
    md = ["# Full classical benchmark (S1): every decision-tree method on the Franka and the SO-101", "",
          "One-layer swaps from the default pipeline; same seeded layouts on both bodies. Success = the protocol's",
          "stage 5 (cube resting on the plate, jaws open). Retention = SO-101 success / Franka success.", ""]
    if base:
        for b in ("franka", "so101"):
            if b in base:
                d = base[b]
                lo, hi = wilson(d["success"], d["episodes"])
                md.append(f"- Default on {b}: {d['success']}/{d['episodes']} ({100 * lo:.0f}-{100 * hi:.0f}%), "
                          f"resolved config `{d.get('resolved', d['cfg'])}`")
        md.append("")
    rows = []
    for key, title in LAYERS:
        items = sorted([(m, v) for (lay, m), v in R.items() if lay == key], key=lambda x: x[0])
        if not items:
            continue
        md += [f"## {title}", "", "| Method | Franka | SO-101 | Retention | Mean stage F / S | Post contact F / S | "
               "vs default p (F / S) | Note |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for meth, v in items:
            cells, ps = [], []
            for b in ("franka", "so101"):
                d = v.get(b)
                if d is None:
                    cells.append(None); ps.append(None); continue
                cells.append(d)
                if b in base:
                    n = min(len(d["stages"]), len(base[b]["stages"]))
                    ps.append(mcnemar_exact(np.array(base[b]["stages"][:n]) == 5, np.array(d["stages"][:n]) == 5))
                else:
                    ps.append(None)
            f, s = cells
            fmt = lambda d: "-" if d is None else (f"n/a" if d["not_applicable"] else f"{d['success']}/{d['episodes']}")
            ret = "-"
            if f and s and not f["not_applicable"] and not s["not_applicable"] and f["rate"] > 0:
                ret = f"{s['rate'] / f['rate']:.2f}"
            note = "; ".join(sorted({d["not_applicable"] for d in cells if d and d["not_applicable"]}))
            if s and s.get("resolved", {}).get("actuator") == "torque" and key in ("control", "actuator", "tracking"):
                note = (note + "; " if note else "") + "SO-101 torque: sim-only counterfactual"
            pf = lambda p: "" if p is None else f"{p:.2g}"
            md.append(f"| {meth} | {fmt(f)} | {fmt(s)} | {ret} | "
                      f"{f['mean_stage']:.1f} / {s['mean_stage']:.1f} | " if f and s else f"| {meth} | {fmt(f)} | {fmt(s)} | {ret} | - | ")
            md[-1] += (f"{f['post_contact_rate']:.2f} / {s['post_contact_rate']:.2f} | " if f and s else "- | ")
            md[-1] += f"{pf(ps[0])} / {pf(ps[1])} | {note} |"
            rows.append({"layer": title, "method": meth,
                         **{f"{b}_{k}": (v[b][k] if b in v else None) for b in ("franka", "so101")
                            for k in ("success", "episodes", "rate", "mean_stage", "post_contact_rate", "not_applicable")},
                         "retention": ret})
        md.append("")
    (ROOT / "results/classical/SUMMARY.md").write_text("\n".join(md) + "\n")
    with open(ROOT / "results/classical/summary.csv", "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
