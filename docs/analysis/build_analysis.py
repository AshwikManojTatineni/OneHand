"""Builds docs/analysis/onehand-analysis.html: a graphical, evidence-based analysis of the OneHand codebase.

Sections: the robot models as loaded (side views to scale, joint and actuator tables), the reproduction check, the
repository graph (layers, hubs, which script writes which result), Franka vs SO-101 through the code, blocks that
do the same job in different files, and one block diagram per file (its functions and classes, the calls between
them, and what each block does).

Run with the project venv (Python 3.11, mujoco): .venv/Scripts/python.exe docs/analysis/build_analysis.py
"""
import ast
import html
import json
import pathlib
import re
import subprocess
from collections import Counter, defaultdict

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
KG = json.loads((ROOT / ".ua/knowledge-graph.json").read_text(encoding="utf-8"))
MAP = json.loads((ROOT / "docs/anatomy/codemap.json").read_text(encoding="utf-8"))
esc = html.escape


# ---------------------------------------------------------------- 1. code blocks per file (AST)
SUMMARY = {}
for n in KG["nodes"]:
    if n["type"] in ("function", "class") and n.get("filePath"):
        SUMMARY[(n["filePath"], n["name"])] = n.get("summary", "")
LAYER = {}
for layer in KG["layers"]:
    for nid in layer["nodeIds"]:
        node = next((x for x in KG["nodes"] if x["id"] == nid), None)
        if node:
            LAYER[node["filePath"]] = layer["name"]


def first_line(doc):
    return (doc or "").strip().split("\n")[0].strip()


def role(path, name, doc):
    s = SUMMARY.get((path, name)) or SUMMARY.get((path, name.split(".")[-1])) or first_line(doc)
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= 90 else s[:87].rsplit(" ", 1)[0] + "…"


def blocks_of(path):
    src = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = []      # {name, kind, lines, calls:set, cls}
    setup = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
             and not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))]
    imports = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                imports[a.asname or a.name] = (n.module, a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                imports[a.asname or a.name.split(".")[0]] = (a.name, None)

    def calls(node):
        names = set()
        for c in ast.walk(node):
            if isinstance(c, ast.Call):
                f = c.func
                if isinstance(f, ast.Name):
                    names.add(f.id)
                elif isinstance(f, ast.Attribute):
                    base = f.value
                    if isinstance(base, ast.Name):
                        names.add(f"{base.id}.{f.attr}")
                    names.add("." + f.attr)
            if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Load):
                names.add(c.id)
        return names

    if setup:
        out.append(dict(name="module setup", kind="setup", lines=(setup[0].lineno, setup[-1].end_lineno),
                        count=sum(n.end_lineno - n.lineno + 1 for n in setup),
                        calls=set().union(*[calls(n) for n in setup]), cls=None,
                        role="Imports, constants and configuration read at import time"))
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(dict(name=n.name, kind="def", lines=(n.lineno, n.end_lineno), calls=calls(n), cls=None,
                            role=role(path, n.name, ast.get_docstring(n))))
        elif isinstance(n, ast.ClassDef):
            out.append(dict(name=n.name, kind="class", lines=(n.lineno, n.end_lineno), calls=set(), cls=None,
                            role=role(path, n.name, ast.get_docstring(n))))
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(dict(name=f"{n.name}.{m.name}", kind="method", lines=(m.lineno, m.end_lineno),
                                    calls=calls(m), cls=n.name, role=role(path, f"{n.name}.{m.name}", ast.get_docstring(m))))
    return out, imports, len(src.splitlines())


PY = [f["path"] for f in MAP["files"] if f["path"].endswith(".py")]
FILES = {}
for p in PY:
    blocks, imports, n = blocks_of(p)
    FILES[p] = dict(blocks=blocks, imports=imports, lines=n)

MODULE_OF = {p: p[:-3].replace("/", ".") for p in PY}
PATH_OF_MODULE = {v: k for k, v in MODULE_OF.items()}
for p in PY:                              # scripts import each other as top-level modules (sys.path trick)
    if p.startswith("scripts/"):
        PATH_OF_MODULE.setdefault(pathlib.Path(p).stem, p)
DEFS = {p: {b["name"]: b for b in FILES[p]["blocks"]} for p in PY}


def resolve_edges(p):
    """Internal edges (block -> block in this file) and external ones (block -> file:block)."""
    f = FILES[p]
    names = DEFS[p]
    short = defaultdict(list)
    for b in f["blocks"]:
        short[b["name"].split(".")[-1]].append(b["name"])
    internal, external = set(), set()
    for b in f["blocks"]:
        for c in b["calls"]:
            if c in names and c != b["name"]:
                internal.add((b["name"], c))
            elif c.startswith(".") and b["cls"] and f"{b['cls']}{c}" in names and f"{b['cls']}{c}" != b["name"]:
                internal.add((b["name"], f"{b['cls']}{c}"))
            elif c.startswith("self."):
                pass
            else:
                head, _, attr = c.partition(".")
                if head in f["imports"]:
                    mod, obj = f["imports"][head]
                    target = None
                    if obj and mod in PATH_OF_MODULE and obj in DEFS[PATH_OF_MODULE[mod]]:
                        target = (PATH_OF_MODULE[mod], obj)
                    elif obj is None and mod in PATH_OF_MODULE and attr in DEFS[PATH_OF_MODULE[mod]]:
                        target = (PATH_OF_MODULE[mod], attr)
                    elif obj and f"{mod}.{obj}" in PATH_OF_MODULE and attr in DEFS[PATH_OF_MODULE[f"{mod}.{obj}"]]:
                        target = (PATH_OF_MODULE[f"{mod}.{obj}"], attr)
                    elif obj and mod in PATH_OF_MODULE and obj in DEFS[PATH_OF_MODULE[mod]]:
                        target = (PATH_OF_MODULE[mod], obj)
                    if target:
                        external.add((b["name"], target))
    return internal, external


EDGES = {p: resolve_edges(p) for p in PY}


# ---------------------------------------------------------------- 2. mermaid helpers
def nid(s):
    return "n" + re.sub(r"\W", "_", s)


def lab(s):
    return s.replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;")


def file_diagram(p):
    f = FILES[p]
    internal, external = EDGES[p]
    direction = "TD" if len(f["blocks"]) > 9 else "LR"
    out = [f"flowchart {direction}"]
    classes = [b for b in f["blocks"] if b["kind"] == "class"]
    def node(b):
        a, z = b["lines"]
        r = b["role"]
        short = b["name"].split(".")[-1] if b["kind"] == "method" else b["name"]
        span = f"{b['count']} top-level lines" if b["kind"] == "setup" else f"L{a}–{z} · {z - a + 1} lines"
        text = f"<b>{lab(short)}</b><br/><small>{span}</small>"
        if r:
            text += f"<br/><small>{lab(r)}</small>"
        shape = ('[("', '")]') if b["kind"] == "setup" else ('["', '"]')
        return f'  {nid(b["name"])}{shape[0]}{text}{shape[1]}'
    for b in f["blocks"]:
        if b["kind"] in ("def", "setup"):
            out.append(node(b))
    for c in classes:
        a, z = c["lines"]
        out.append(f'  subgraph {nid(c["name"])}["class {lab(c["name"])} · L{a}–{z}"]')
        meths = [b for b in f["blocks"] if b["cls"] == c["name"]]
        if not meths:
            out.append(f'    {nid(c["name"])}_body["<small>{lab(c["role"] or "data class")}</small>"]')
        for b in meths:
            out.append("  " + node(b))
        out.append("  end")
    for a, b in sorted(internal):
        out.append(f"  {nid(a)} --> {nid(b)}")
    ext = defaultdict(set)
    for a, (tp, tb) in external:
        ext[tp].add((a, tb))
    for tp, pairs in sorted(ext.items()):
        targets = sorted({tb for _, tb in pairs})
        key = nid("ext_" + tp)
        out.append(f'  {key}{{{{"<small>{lab(tp)}</small><br/>{lab(", ".join(targets[:6]))}{"…" if len(targets) > 6 else ""}"}}}}')
        for a in sorted({a for a, _ in pairs}):
            out.append(f"  {nid(a)} -.-> {key}")
        out.append(f"  class {key} ext")
    out.append("  classDef ext stroke-dasharray:4 3")
    return "\n".join(out)


# ---------------------------------------------------------------- 3. the two robots, loaded from their model files
ROBOTS = [
    ("Franka Emika Panda", "franka", "assets/mujoco_menagerie/franka_emika_panda/panda.xml", [f"joint{i}" for i in range(1, 8)],
     [0, -0.4, 0, -2.4, 0, 2.0, 0.785], "hand", [0, 0, 0.1034]),
    ("SO-101", "so101", "assets/mujoco_menagerie/robotstudio_so101/so101.xml",
     ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"], [0.0, -0.5, 0.8, 0.8, 0.0], None, None),
]


def robot_facts():
    facts = []
    for title, key, xml, arm, home, tcp_body, tcp_off in ROBOTS:
        m = mujoco.MjModel.from_xml_path(str(ROOT / xml))
        d = mujoco.MjData(m)
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm]
        for j, q in zip(jids, home):
            d.qpos[m.jnt_qposadr[j]] = q
        mujoco.mj_forward(m, d)
        chain = [d.xanchor[j].copy() for j in jids]
        if tcp_body:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tcp_body)
            tcp = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array(tcp_off)
        else:
            s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
            tcp = d.site_xpos[s].copy()
        base = d.xpos[m.jnt_bodyid[jids[0]]].copy()
        base[2] = 0
        reach = max(np.linalg.norm(c - chain[0]) for c in chain[1:] + [tcp])
        geoms = [(d.geom_xpos[g].copy(), m.geom_rbound[g]) for g in range(m.ngeom) if m.geom_bodyid[g] > 0
                 and m.geom_contype[g] + m.geom_conaffinity[g] > 0]
        facts.append(dict(
            title=title, key=key, xml=xml, nq=int(m.nq), nu=int(m.nu), nbody=int(m.nbody), ngeom=int(m.ngeom),
            nmesh=int(m.nmesh), mass=float(sum(m.body_mass)), dt=float(m.opt.timestep),
            joints=[dict(name=j, lo=float(np.rad2deg(m.jnt_range[i, 0])), hi=float(np.rad2deg(m.jnt_range[i, 1])))
                    for j, i in zip(arm, jids)],
            acts=[dict(name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
                       force=float(m.actuator_forcerange[i, 1]), kind="position servo") for i in range(m.nu)],
            chain=[base.tolist()] + [c.tolist() for c in chain] + [tcp.tolist()],
            geoms=[(g.tolist(), float(r)) for g, r in geoms], reach=float(reach),
        ))
    return facts


def robot_svg(f, scale, W=330, H=300, ground=262):
    """Side view (x right, z up) at the home pose, drawn to the shared scale (px per metre)."""
    pts = np.array(f["chain"])
    x0 = 60 - pts[:, 0].min() * scale
    P = lambda p: (x0 + p[0] * scale, ground - p[2] * scale)
    parts = [f'<line x1="10" y1="{ground}" x2="{W-10}" y2="{ground}" class="rb-ground"/>']
    for k in range(0, int((W - 20) / (0.1 * scale)) + 1):        # 10 cm ticks along the floor
        x = 10 + k * 0.1 * scale
        parts.append(f'<line x1="{x:.1f}" y1="{ground}" x2="{x:.1f}" y2="{ground + (8 if k % 5 == 0 else 4)}" class="rb-tick"/>')
    for (g, r) in f["geoms"]:
        if r * scale < 1.2:
            continue
        cx, cy = P(g)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{min(r * scale, 22):.1f}" class="rb-geom"/>')
    path = " ".join(f"{P(p)[0]:.1f},{P(p)[1]:.1f}" for p in pts)
    parts.append(f'<polyline points="{path}" class="rb-link"/>')
    for i, p in enumerate(pts[1:-1]):
        cx, cy = P(p)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" class="rb-joint"/><text x="{cx + 7:.1f}" y="{cy - 6:.1f}" class="rb-num">{i + 1}</text>')
    tx, ty = P(pts[-1])
    parts.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="3.2" class="rb-tcp"/><text x="{tx + 7:.1f}" y="{ty + 4:.1f}" class="rb-num">TCP</text>')
    bx, by = P(pts[0])
    parts.append(f'<rect x="{bx - 16:.1f}" y="{by - 6:.1f}" width="32" height="6" class="rb-base"/>')
    return f'<svg viewBox="0 0 {W} {H}" class="rb-svg" role="img" aria-label="{esc(f["title"])} side view at its home pose, to scale">{"".join(parts)}<text x="12" y="{H - 8}" class="rb-cap">{esc(f["title"])} · home pose · ticks 10 cm</text></svg>'


# ---------------------------------------------------------------- 4. repository graph metrics
files = MAP["files"]
edges = [tuple(e) for e in MAP["edges"]]
indeg, outdeg = Counter(b for _, b in edges), Counter(a for a, _ in edges)
layer_of = {f["path"]: f["layer"] for f in files}
LAYERS = ["Core Benchmark Library", "Classical Robotics Stack", "Self-Modeling Experiments", "Evaluation and Analysis Scripts",
          "Experiment Grid Orchestration", "GPU Training Routes", "Protocol and Checkpoint Registry"]
SHORT = {"Core Benchmark Library": "Core", "Classical Robotics Stack": "Classical", "Self-Modeling Experiments": "Self-model",
         "Evaluation and Analysis Scripts": "Evaluation", "Experiment Grid Orchestration": "Grid", "GPU Training Routes": "GPU",
         "Protocol and Checkpoint Registry": "Protocol"}
matrix = Counter((layer_of[a], layer_of[b]) for a, b in edges if a in layer_of and b in layer_of)


def cycles():
    g = defaultdict(list)
    for a, b in edges:
        g[a].append(b)
    index, low, stack, on, out, n = {}, {}, [], set(), [], [0]

    def visit(v):
        index[v] = low[v] = n[0]; n[0] += 1; stack.append(v); on.add(v)
        for w in g[v]:
            if w not in index:
                visit(w); low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop(); on.discard(w); comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                out.append(comp)
    for v in list({x for e in edges for x in e}):
        if v not in index:
            visit(v)
    return out


CYCLES = cycles()
LIBRARY_UNUSED = [f["path"] for f in files if f["path"].startswith(("xbody/", "isaac/onehand_isaac/")) and indeg[f["path"]] == 0
                  and not f["path"].endswith("__init__.py")]


def layer_mermaid():
    out = ["flowchart TB"]
    for L in LAYERS:
        n = sum(1 for f in files if f["layer"] == L)
        out.append(f'  {nid(L)}["<b>{SHORT[L]}</b><br/><small>{n} files</small>"]')
    for (a, b), c in sorted(matrix.items(), key=lambda x: -x[1]):
        if a != b and a in SHORT and b in SHORT:
            out.append(f"  {nid(a)} -- {c} --> {nid(b)}")
    return "\n".join(out)


def file_mermaid():
    out = ["flowchart LR"]
    for L in LAYERS:
        out.append(f'  subgraph {nid("L_" + L)}["{SHORT[L]}"]')
        for f in files:
            if f["layer"] == L:
                out.append(f'    {nid(f["path"])}["{lab(f["path"].split("/")[-1])}"]')
        out.append("  end")
    for a, b in edges:
        if a in layer_of and b in layer_of:
            out.append(f"  {nid(a)} --> {nid(b)}")
    return "\n".join(out)


WRITERS = [  # script -> result files it writes (from each script's write calls)
    ("scripts/eval_classical.py", "results/classical/&lt;body&gt;__&lt;variant&gt;.json", "146 classical cells"),
    ("scripts/eval_ilc.py", "results/classical/&lt;body&gt;__control-ilc.json", "ILC cells"),
    ("scripts/summarize_classical.py", "results/classical/SUMMARY.md, summary.csv", "classical summary"),
    ("scripts/eval_s1.py", "results/phase0/s1_*.json", "early S1 variants"),
    ("scripts/eval_route_b_home.py", "results/phase0/route_b_home_*.json", "route B at home"),
    ("scripts/eval_route_b_transfer.py", "results/phase0/route_b_transfer_*.json", "route B on the SO100"),
    ("scripts/eval_route_b_r3.py", "results/phase0/route_b_r3_*.json", "route B on the xArm6"),
    ("scripts/selfmodel_readout_route_b.py", "results/phase0/selfmodel_readout_route_b.json", "latent readout"),
    ("scripts/summarize_phase0.py", "results/phase0/SUMMARY.md, summary.json", "Phase 0 summary"),
    ("scripts/selfmodel_teleop.py", "results/selfmodel_teleop*/cells.json", "teleop self-model"),
    ("scripts/selfmodel_sensors.py", "results/selfmodel_sensors*/cells.json", "sensor swap"),
    ("scripts/eval_selfmodel_task.py", "results/selfmodel_task/&lt;body&gt;.json", "self-model and task"),
]


# ---------------------------------------------------------------- 5. stored results, Franka vs SO-101
def classical_pairs():
    rows = defaultdict(dict)
    for p in sorted((ROOT / "results/classical").glob("*.json")):
        d = json.loads(p.read_text())
        body, _, tag = p.stem.partition("__")
        rows[tag][body] = (d["success"], d["episodes"], d.get("not_applicable", ""))
    groups = defaultdict(list)
    for tag, v in rows.items():
        layer = "default" if tag == "default" else tag.split("-")[0]
        groups[layer].append((tag, v))
    return groups


def bars_svg(groups):
    order = ["default", "input", "world", "endpoint", "planner", "timing", "tracking", "control", "actuator"]
    rows = []
    for g in order:
        for tag, v in sorted(groups.get(g, []), key=lambda x: x[0]):
            rows.append((g, tag, v))
    rh, W, left, barw = 15, 640, 230, 190
    H = len(rows) * rh + 60
    out = [f'<svg viewBox="0 0 {W} {H}" class="bars" role="img" aria-label="Classical success out of 20 per method, Franka and SO-101">']
    for k in range(0, 21, 5):
        for off in (0, barw + 20):
            x = left + off + k / 20 * barw
            out.append(f'<line x1="{x:.1f}" y1="34" x2="{x:.1f}" y2="{H - 18}" class="bar-grid"/><text x="{x:.1f}" y="{H - 4}" class="bar-ax">{k}</text>')
    out.append(f'<text x="{left}" y="20" class="bar-head">Franka (success of 20)</text><text x="{left + barw + 20}" y="20" class="bar-head">SO-101</text>')
    y, last = 40, None
    for g, tag, v in rows:
        if g != last:
            out.append(f'<line x1="0" y1="{y - 3}" x2="{W}" y2="{y - 3}" class="bar-sep"/>')
            last = g
        name = tag if g == "default" else tag[len(g) + 1:]
        out.append(f'<text x="{left - 8}" y="{y + 9}" class="bar-lab" text-anchor="end"><tspan class="bar-g">{g}</tspan> {esc(name)}</text>')
        for i, body in enumerate(("franka", "so101")):
            x0 = left + i * (barw + 20)
            if body not in v:
                continue
            s, n, na = v[body]
            if na:
                out.append(f'<text x="{x0 + 2}" y="{y + 9}" class="bar-na">not applicable</text>')
                continue
            w = s / max(n, 1) * barw
            out.append(f'<rect x="{x0}" y="{y}" width="{max(w, 0.8):.1f}" height="{rh - 4}" class="bar b-{body}"/><text x="{x0 + w + 4:.1f}" y="{y + 9}" class="bar-v">{s}</text>')
        y += rh
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- 6. reproduction (filled after the runs)
def repro_rows():
    rows = []
    for p in sorted((HERE / "repro").glob("*__planner-*.json")):
        new = json.loads(p.read_text())
        old = json.loads(   # the committed result, so a re-run that overwrote the working copy can't mask a change
subprocess.check_output(["git", "show", f"HEAD:results/classical/{p.name}"], cwd=ROOT))
        same = sum(a == b for a, b in zip(old["stages"], new["stages"]))
        rows.append(dict(cell=p.stem, old=old["success"], new=new["success"], n=new["episodes"], same=same,
                         old_stages=old["stages"], new_stages=new["stages"], sec=new["sec_per_episode"]))
    tele = []
    for body in ("franka", "so101"):
        f = HERE / "repro" / f"teleop_{body}.json"
        if f.exists():
            stored = json.loads((ROOT / ("results/selfmodel_teleop" if body == "franka" else "results/selfmodel_teleop_so101") / "cells.json").read_text())
            for r in json.loads(f.read_text()):
                ref = next((c for c in stored if c["seed"] == r["seed"] and c["mode"] == "cartesian" and c["steps"] == 400 and c["noise"] == "mid"), None)
                for k in ("nominal", "diff_only", "diff_abs", "calib"):
                    tele.append(dict(body=body, seed=r["seed"], model=k, new_mm=r[k][0], new_reach=r[k][1],
                                     old_mm=ref[k]["fk"]["pos_mm_median"] if ref else None, old_reach=ref[k]["reach"]["success"] if ref else None))
    return rows, tele


if __name__ == "__main__":
    data = dict(robots=robot_facts(), repro=repro_rows())
    (HERE / "analysis.json").write_text(json.dumps({"robots": data["robots"], "repro": data["repro"]}, indent=1, default=float))
    from render import render   # the page template lives next to this script
    page = render(dict(
        robots=data["robots"], robot_svg=robot_svg, repro=data["repro"], files=files, edges=edges, indeg=indeg,
        outdeg=outdeg, matrix=matrix, cycles=CYCLES, unused=LIBRARY_UNUSED, layer_mermaid=layer_mermaid(),
        file_mermaid=file_mermaid(), writers=WRITERS, groups=classical_pairs(), bars_svg=bars_svg, FILES=FILES,
        EDGES=EDGES, file_diagram=file_diagram, LAYERS=LAYERS, SHORT=SHORT, layer_of=layer_of, esc=esc,
    ))
    (HERE / "onehand-analysis.html").write_text(page, encoding="utf-8")
    print("wrote onehand-analysis.html", len(page))
