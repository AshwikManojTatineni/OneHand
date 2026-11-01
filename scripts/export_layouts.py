"""Export the classical benchmark's evaluation layouts (cube positions per seed) so every simulator, MJX or Isaac,
starts from exactly the same scenes. Usage: python scripts/export_layouts.py [episodes]
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody.bodies import SPECS  # noqa: E402
from xbody.scene import CLASSICAL, sample_layout  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
for body in ("franka", "so101"):
    off = np.array(SPECS[body]["layout_offset"])
    out = {"body": body, "seed_base": 20261028, "frame": "robot base, metres",
           "cube_edge": 0.03, "cube_mass": 0.03, "cube_friction": 1.5, "table_z": 0.0,
           "plate_xy": (np.array(CLASSICAL["plate_xy"]) + off).tolist(), "plate_r": 0.05,
           "post_xy": (np.array(CLASSICAL["post_xy"]) + off).tolist(), "post_r": CLASSICAL["post_r"],
           "post_h": CLASSICAL["post_h"], "home": SPECS[body]["home"],
           "cube_xy": [sample_layout(np.random.default_rng(20261028 + ep), body, region_y=(-0.08, 0.06)).tolist()
                       for ep in range(n)]}
    p = ROOT / f"results/learned/layouts_{body}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(p, len(out["cube_xy"]), "layouts")
