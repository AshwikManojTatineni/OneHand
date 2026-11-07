"""Re-run teleop self-model cells with their stored seeds and print the numbers to compare with results/selfmodel_teleop*/cells.json."""
import json, os, sys, pathlib
body = sys.argv[1]
os.environ["SELFMODEL_BODY"] = body
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import selfmodel_teleop as t
out = []
for seed in (0, 1):
    r = t.run_cell(seed, "cartesian", 400, "mid", t.nominal_body())
    out.append({k: (r[k]["fk"]["pos_mm_median"], r[k]["reach"]["success"]) for k in ("nominal", "diff_only", "diff_abs", "calib")} | {"seed": seed})
    print(json.dumps(out[-1]), flush=True)
(pathlib.Path(__file__).parent / f"teleop_{body}.json").write_text(json.dumps(out, indent=1))
