"""Phase 0 summary: retention, paired tests (same seeds) and the self-model correlation, from the result files.

Writes results/phase0/SUMMARY.md and results/phase0/summary.json. Usage: python scripts/summarize_phase0.py
"""
import glob
import json
import pathlib
import re
import sys

import numpy as np
from scipy import stats as sps

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody.stats import holm, mcnemar_exact, retention, stage_retention, wilcoxon_stage, wilson  # noqa: E402

R = ROOT / "results/phase0"


def episodes_from_log(path):
    """Per-episode (success, stage) from a route B log, in seed order."""
    out = []
    for line in open(path):
        if line.startswith('{"ep"'):
            d = json.loads(line)
            out.append((bool(d["success"]), int(d["stage"])))
        m = re.match(r"\S+ ep\s+(\d+) success=(True|False)", line)
        if m:
            ok = m.group(2) == "True"
            out.append((ok, 4 if ok else 0))
    return out


def cell(name, eps, max_stage):
    s = np.array([e[0] for e in eps]); st = np.array([e[1] for e in eps])
    lo, hi = wilson(int(s.sum()), len(s))
    return {"cell": name, "n": len(s), "success": int(s.sum()), "rate": float(s.mean()), "wilson95": [lo, hi],
            "mean_stage": float(st.mean()), "max_stage": max_stage, "_s": s, "_st": st}


def fmt_ci(c):
    return f"{c['success']}/{c['n']} ({100 * c['wilson95'][0]:.0f}-{100 * c['wilson95'][1]:.0f}%)"


def main():
    md = ["# Phase 0 summary (CPU-only comparisons)", "",
          "All policies frozen; every comparison is paired on the same seeds and layouts.", ""]
    out = {}

    # ---- Route B: frozen Franka TD-MPC2 on its own body, the xArm6 (R3) and the SO100 (R6)
    B = {
        "R0 Franka home (prior)": "log_franka_prior.txt",
        "R3 xArm6 K (prior)": "log_r3_K--no-mpc.txt",
        "R3 xArm6 K+P (prior)": "log_r3_K+P--no-mpc.txt",
        "R3 xArm6 K+P naive gripper (prior)": "log_r3_K+P_naivegrip--no-mpc.txt",
        "R6 SO100 K dls (prior)": "log_transfer_K_dls--no-mpc.txt",
        "R6 SO100 K+P dls (prior)": "log_transfer_K+P_dls--no-mpc.txt",
        "R6 SO100 K hqp-axis (prior)": "log_transfer_K_hqp-axis--no-mpc.txt",
        "R6 SO100 K+P hqp-axis (prior)": "log_transfer_K+P_hqp-axis--no-mpc.txt",
        "R6 SO100 K+P dls no frame shift (prior)": "log_transfer_K+P_dls_noshift--no-mpc.txt",
        "R6 SO100 K dls (planning on)": "log_transfer_K_dls_mpc20.txt",
        "R6 SO100 K+P dls (planning on)": "log_transfer_K+P_dls_mpc20.txt",
        "R3 xArm6 K (planning on)": "log_r3_K_mpc.txt",
        "R3 xArm6 K+P (planning on)": "log_r3_K+P_mpc.txt",
        "R3 xArm6 K+P naive gripper (planning on)": "log_r3_K+P_--naive-gripper_mpc.txt",
    }
    cells = {k: cell(k, episodes_from_log(R / v), 4) for k, v in B.items() if (R / v).exists() and episodes_from_log(R / v)}
    src = cells["R0 Franka home (prior)"]
    md += ["## Route B: frozen Franka TD-MPC2 PickCube policy", "",
           "Stages: 0 none, 1 reach, 2 grasp, 3 at goal while grasped, 4 success.", "",
           "| Cell | Success (Wilson 95%) | Mean stage | Retention | Stage retention |", "| --- | --- | --- | --- | --- |"]
    for k, c in cells.items():
        c["retention"] = retention(c["rate"], src["rate"])
        c["stage_retention"] = stage_retention(c["_st"], src["_st"])
        md.append(f"| {k} | {fmt_ci(c)} | {c['mean_stage']:.2f} | {c['retention']:.2f} | {c['stage_retention']:.2f} |")

    pairs = [("R6 SO100 K dls (prior)", "R6 SO100 K+P dls (prior)", "P (virtual twin), R6"),
             ("R6 SO100 K+P dls (prior)", "R6 SO100 K+P hqp-axis (prior)", "IK conversion dls vs hqp-axis, R6"),
             ("R6 SO100 K+P dls (prior)", "R6 SO100 K+P dls no frame shift (prior)", "Frame alignment, R6"),
             ("R3 xArm6 K (prior)", "R3 xArm6 K+P (prior)", "P (virtual twin), R3"),
             ("R3 xArm6 K+P (prior)", "R3 xArm6 K+P naive gripper (prior)", "Gripper map in K, R3"),
             ("R3 xArm6 K+P (prior)", "R6 SO100 K+P dls (prior)", "Rung R3 vs R6 (different envs: unpaired tests)"),
             ("R6 SO100 K+P dls (prior)", "R6 SO100 K+P dls (planning on)", "Planning on vs off, R6 (first 20 seeds)"),
             ("R3 xArm6 K (prior)", "R3 xArm6 K (planning on)", "Planning on vs off, R3"),
             ("R3 xArm6 K+P (prior)", "R3 xArm6 K+P (planning on)", "Planning on vs off, R3 with P"),
             ("R3 xArm6 K (planning on)", "R3 xArm6 K+P (planning on)", "P (virtual twin), R3 with planning")]
    rows, pv = [], []
    for a, b, label in pairs:
        if a in cells and b in cells:
            n = min(cells[a]["n"], cells[b]["n"])
            sa, sb = cells[a]["_s"][:n], cells[b]["_s"][:n]
            ta, tb = cells[a]["_st"][:n], cells[b]["_st"][:n]
            if "unpaired" in label:   # seeds do not give matching layouts across different envs
                p_s = float(sps.fisher_exact([[sa.sum(), n - sa.sum()], [sb.sum(), n - sb.sum()]])[1])
                p_t = float(sps.mannwhitneyu(ta, tb).pvalue)
            else:
                p_s, p_t = mcnemar_exact(sa, sb), wilcoxon_stage(ta, tb)
            rows.append([label, n, f"{sa.mean():.2f} vs {sb.mean():.2f}", f"{ta.mean():.2f} vs {tb.mean():.2f}", p_s, p_t])
            pv.append(p_t)
    adj = holm(np.array(pv)) if pv else []
    md += ["", "Tests (exact McNemar on success and Wilcoxon on stage reached for paired rows; Fisher and Mann-Whitney for the unpaired row; Holm-adjusted stage p):", "",
           "| Comparison | Paired episodes | Success | Mean stage | McNemar p | Wilcoxon p (Holm) |",
           "| --- | --- | --- | --- | --- | --- |"]
    for r, a in zip(rows, adj):
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.3g} | {r[5]:.3g} ({a:.3g}) |")
    out["route_b"] = {k: {kk: vv for kk, vv in c.items() if not kk.startswith("_")} for k, c in cells.items()}

    # ---- S1 classical variants
    s1 = {}
    for f in sorted(glob.glob(str(R / "s1_*.json"))):
        d = json.load(open(f))
        if "stages" not in d:
            continue
        key = (d["ik"], d["tracking"], d["timing"], d["kp_scale"])
        s1.setdefault(key, {})[d["body"]] = d
    default = ("hqp-axis", "joint", "trapezoid", 1.0)
    md += ["", "## S1 classical variants (MuJoCo, Menagerie Franka and SO-101)", "",
           "Stages: 0 none, 1 reach, 2 grasp, 3 lift, 4 transport, 5 place. Retention = SO-101 / Franka.", "",
           "| Variant | Franka | SO-101 | Retention | SO-101 vs default: McNemar p | SO-101 IK misses per episode |",
           "| --- | --- | --- | --- | --- | --- |"]
    base_so = np.array(s1[default]["so101"]["stages"])
    for key, v in s1.items():
        if "franka" not in v or "so101" not in v:
            continue
        fr, so = v["franka"], v["so101"]
        label = "default" if key == default else ", ".join(
            x for x, dflt in zip([f"ik {key[0]}", f"tracking {key[1]}", f"timing {key[2]}", f"servo stiffness x{key[3]:g}"],
                                 default) if not x.endswith(str(dflt)) and x.split()[-1] != str(dflt).replace("1.0", "x1"))
        so_s = np.array(so["stages"]) == 5
        p = mcnemar_exact(base_so == 5, so_s) if key != default else float("nan")
        ret = retention(so["rate"], fr["rate"])
        md.append(f"| {label} | {fr['success']}/{fr['episodes']} | {so['success']}/{so['episodes']} | "
                  f"{'n/a' if np.isnan(ret) else f'{ret:.2f}'} | {'' if np.isnan(p) else f'{p:.3g}'} | {so['ik_failures_per_episode']:.1f} |")
    out["s1"] = {"|".join(map(str, k)): {b: {kk: vv for kk, vv in d.items() if kk != "stages"} for b, d in v.items()}
                 for k, v in s1.items()}

    # ---- Self-model readout vs transfer
    sm_path = R / "selfmodel_readout_route_b.json"
    if sm_path.exists():
        sm = json.load(open(sm_path))
        link = {"franka_home": "R0 Franka home (prior)", "xarm6_R3_KP": "R3 xArm6 K+P (prior)",
                "so100_R6_KP": "R6 SO100 K+P dls (prior)", "so100_R6_K": "R6 SO100 K dls (prior)",
                "so100_R6_KP_noshift": "R6 SO100 K+P dls no frame shift (prior)"}
        md += ["", "## Self-model readout vs transfer (route B, training-free)", "",
               "Frozen Franka world model, one-step latent prediction on each body. Relative error divides by a "
               "no-change baseline; lower is a better self-model.", "",
               "| Body / interface | Relative error | Self-prediction ratio | Mean stage reached |", "| --- | --- | --- | --- |"]
        xs, ys = [], []
        for k, v in sm.items():
            c = cells.get(link.get(k, ""), None)
            ms = c["mean_stage"] if c else float("nan")
            md.append(f"| {k} | {v['relative_error']:.3f} | {v['self_prediction_ratio']:.1f} | {ms:.2f} |")
            if c:
                xs.append(v["relative_error"]); ys.append(ms)
        if len(xs) >= 3:
            rho, p = sps.spearmanr(xs, ys)
            md += ["", f"Spearman correlation between relative error and mean stage: rho = {rho:.2f} (p = {p:.2g}, "
                       f"n = {len(xs)} cells; too few points for a firm claim)."]
            out["selfmodel_spearman"] = {"rho": float(rho), "p": float(p), "n": len(xs)}

    (R / "SUMMARY.md").write_text("\n".join(md) + "\n")
    (R / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
