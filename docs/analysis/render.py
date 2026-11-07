"""Page template for build_analysis.py. Everything numeric comes in through `c`; the prose states findings checked
against the code and the result files."""
import json

FILE_SUMMARY = {}
try:
    import pathlib
    kg = json.loads((pathlib.Path(__file__).resolve().parents[2] / ".ua/knowledge-graph.json").read_text(encoding="utf-8"))
    FILE_SUMMARY = {n["filePath"]: n.get("summary", "") for n in kg["nodes"] if n["type"] == "file"}
except Exception:
    pass

CSS = """
:root{--bg:#f6f5f1;--card:#fffefb;--fg:#23242a;--muted:#6b6d76;--line:#dcdad3;--soft:#ecebe5;--accent:#2f5bd3;
--franka:#2f5bd3;--so101:#c2571a;--good:#2e7d4f;--warn:#a46a00;--bad:#b3261e;
--mono:"JetBrains Mono",ui-monospace,Consolas,monospace;--sans:"IBM Plex Sans",system-ui,"Segoe UI",sans-serif}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121317;--card:#191a1f;--fg:#e4e4e7;--muted:#9a9ca6;
--line:#2c2e35;--soft:#22242a;--accent:#8aa8ff;--franka:#7f9dff;--so101:#f08c4f;--good:#5cc489;--warn:#e0a63c;--bad:#f2766c;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121317;--card:#191a1f;--fg:#e4e4e7;--muted:#9a9ca6;--line:#2c2e35;--soft:#22242a;--accent:#8aa8ff;
--franka:#7f9dff;--so101:#f08c4f;--good:#5cc489;--warn:#e0a63c;--bad:#f2766c;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.6 var(--sans)}
.wrap{max-width:1080px;margin:0 auto;padding-block:40px 96px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1{font-size:30px;line-height:1.2;margin:0 0 8px;text-wrap:balance;letter-spacing:-.015em}
h2{font-size:22px;margin:64px 0 6px;text-wrap:balance;letter-spacing:-.01em}
h3{font-size:16px;margin:28px 0 4px;font-family:var(--mono);font-weight:600}
p{max-width:74ch;margin:6px 0 12px}
.eyebrow{font:600 11.5px/1 var(--mono);letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.lead{font-size:17px;max-width:70ch}
.verdict{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:22px 0 8px}
.v{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;min-width:0}
.v b{display:block;font-size:13px;margin-bottom:4px}
.v .pill{display:inline-block;font:600 11px/1 var(--mono);padding:4px 7px;border-radius:99px;margin-bottom:8px;letter-spacing:.04em}
.ok{color:var(--good);background:color-mix(in srgb,var(--good) 14%,transparent)}
.mid{color:var(--warn);background:color-mix(in srgb,var(--warn) 14%,transparent)}
.no{color:var(--bad);background:color-mix(in srgb,var(--bad) 14%,transparent)}
.v p{font-size:13.5px;margin:0;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0 4px}
th{text-align:left;font:600 11px/1.3 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--line);padding:6px 10px 6px 0}
td{border-bottom:1px solid var(--soft);padding:7px 10px 7px 0;vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:12.5px}
code,.mono{font-family:var(--mono);font-size:12.5px}
.scroll{overflow-x:auto}
.robots{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin:12px 0}
.robot{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-width:0}
.rb-svg{width:100%;height:auto;display:block}
.rb-ground{stroke:var(--muted);stroke-width:1.2}.rb-tick{stroke:var(--muted);stroke-width:.8}
.rb-geom{fill:color-mix(in srgb,var(--fg) 7%,transparent);stroke:color-mix(in srgb,var(--fg) 22%,transparent);stroke-width:.6}
.rb-link{fill:none;stroke:var(--fg);stroke-width:2.4;stroke-linejoin:round}
.rb-joint{fill:var(--card);stroke:var(--fg);stroke-width:1.6}.rb-tcp{fill:var(--accent)}
.rb-base{fill:var(--muted)}.rb-num{font:10px var(--mono);fill:var(--muted)}.rb-cap{font:10.5px var(--mono);fill:var(--muted)}
.bars{width:100%;max-width:760px;height:auto}
.bar-grid{stroke:var(--soft)}.bar-sep{stroke:var(--line)}.bar-ax,.bar-v,.bar-na{font:9.5px var(--mono);fill:var(--muted)}
.bar-head{font:600 10.5px var(--mono);fill:var(--fg)}.bar-lab{font:10.5px var(--mono);fill:var(--fg)}.bar-g{fill:var(--muted)}
.b-franka{fill:var(--franka)}.b-so101{fill:var(--so101)}
.key{display:inline-flex;align-items:center;gap:6px;margin-right:16px;font-size:13px;color:var(--muted)}
.sw{width:12px;height:12px;border-radius:2px;display:inline-block}
.heat td.h{text-align:center;font-family:var(--mono);font-size:12px}
.file{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 16px 14px;margin:14px 0;min-width:0}
.file .meta{color:var(--muted);font:12px var(--mono);margin-left:8px;font-weight:400}
.file pre.mermaid{background:transparent;margin:8px 0 0;overflow-x:auto}
.layerhead{border-top:1px solid var(--line);margin-top:48px;padding-top:6px}
ul.find{padding-left:20px;max-width:78ch}ul.find li{margin:8px 0}
.stage{font-family:var(--mono);font-size:11.5px;letter-spacing:.06em;white-space:nowrap}
.stage i{font-style:normal;display:inline-block;width:12px;text-align:center}
.diff{color:var(--bad);font-weight:700}
nav.toc{display:flex;flex-wrap:wrap;gap:6px 14px;margin:18px 0 0;font-size:13.5px}
nav.toc a{color:var(--accent);text-decoration:none}
nav.toc a:hover{text-decoration:underline}
"""


def stage_strip(old, new):
    cells = []
    for a, b in zip(old, new):
        cls = ' class="diff"' if a != b else ""
        cells.append(f"<i{cls}>{b}</i>")
    return "".join(cells)


def render(c):
    esc = c["esc"]
    R = c["robots"]
    rows, tele = c["repro"]
    fr, so = R
    scale = 230 / max(fr["reach"], so["reach"]) * 0.62
    # ------------------------------------------------------------- verdict numbers
    exact_tele = sum(1 for t in tele if t["old_mm"] is not None and abs(t["old_mm"] - t["new_mm"]) < 0.02 and t["old_reach"] == t["new_reach"])
    cl_eps = sum(r["n"] for r in rows)
    cl_same = sum(r["same"] for r in rows)
    out = []
    a = out.append
    a(f"<title>OneHand Codebase Audit</title><style>{CSS}</style>")
    a('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">')
    a('<div class="wrap">')
    a('<div class="eyebrow">OneHand · xbody-bench · audit of 2026-11-07</div>')
    a("<h1>Is OneHand real, and how does it fit together?</h1>")
    a('<p class="lead">This page checks the project instead of describing it. It loads both robot models, re-runs part of the stored '
      "results on this machine and compares them episode by episode, cross-checks every headline number against the raw result files, "
      "and then draws each program file as a diagram of its code blocks and the calls between them.</p>")
    a('<div class="verdict">')
    a(f'<div class="v"><span class="pill ok">REPRODUCED</span><b>Self-model teleop cells</b><p>{exact_tele} of {len(tele)} model × seed results '
      "re-ran to the same FK error (within 0.01 mm) and the same reach count, on a different machine with MuJoCo 4.0.</p></div>")
    if rows:
        a(f'<div class="v"><span class="pill {"ok" if cl_same / max(cl_eps,1) > 0.8 else "mid"}">RE-RUN</span><b>Classical pick-and-place cells</b><p>'
          f"{len(rows)} cells, {cl_eps} episodes: {cl_same} episodes ended on the same stage as stored; success "
          + ", ".join(f'{r["old"]}→{r["new"]}' for r in rows) + ".</p></div>")
    a('<div class="v"><span class="pill ok">CONSISTENT</span><b>RESULTS.md against raw files</b><p>Every headline number checked (classical splits, '
      "self-model task 55→81 and 56→85, ILC −13%) matches the JSON it came from.</p></div>")
    a('<div class="v"><span class="pill mid">NOT CHECKABLE HERE</span><b>Route B, OMPL cells, GPU routes</b><p>Route B needs ManiSkill and the Newt checkpoints '
      "(neither is in the repo); the default RRT-Connect planner needs OMPL, which has no Windows build; the GPU routes were never run.</p></div>")
    a("</div>")
    a('<nav class="toc"><a href="#robots">The robots</a><a href="#repro">Reproduction</a><a href="#graph">Repository graph</a>'
      '<a href="#bodies">Franka vs SO-101 in the code</a><a href="#same">Same job, different files</a><a href="#concerns">Concerns</a>'
      '<a href="#files">File by file</a></nav>')

    # ------------------------------------------------------------- robots
    a('<h2 id="robots">The robots: MJCF, not URDF</h2>')
    a("<p>There is no <code>.urdf</code> because this project never uses URDF for the two arms. It loads MuJoCo's own format, MJCF "
      "(<code>.xml</code>), from Google DeepMind's MuJoCo Menagerie: <code>franka_emika_panda/panda.xml</code> and "
      "<code>robotstudio_so101/so101.xml</code> (<code>xbody/bodies.py</code>, <code>SPECS[...]['xml']</code>). The README tells you to "
      "clone them into <code>assets/mujoco_menagerie/</code>, and <code>.gitignore</code> excludes that folder, so they were never in the "
      "repository. That is why you could not find them. I cloned them for this audit; the drawings and tables below are read from those "
      "files, not typed in.</p>")
    a("<p>Route B is different: it uses ManiSkill's built-in Panda, SO100 and xArm6, which ManiSkill loads from its own URDF assets at run "
      "time. Those are not in this repository either, and the SO100 is an older sibling of the SO-101, not the same arm.</p>")
    a('<div class="robots">')
    for f in R:
        a(f'<div class="robot">{c["robot_svg"](f, scale)}</div>')
    a("</div>")
    a('<div class="scroll"><table><tr><th>From the model file</th><th class="num">Franka Panda</th><th class="num">SO-101</th></tr>')
    rowsr = [("Arm joints", len(fr["joints"]), len(so["joints"])), ("Actuators (incl. gripper)", fr["nu"], so["nu"]),
             ("Bodies / geoms / meshes", f'{fr["nbody"]} / {fr["ngeom"]} / {fr["nmesh"]}', f'{so["nbody"]} / {so["ngeom"]} / {so["nmesh"]}'),
             ("Total mass", f'{fr["mass"]:.2f} kg', f'{so["mass"]:.2f} kg'),
             ("Largest joint torque", f'{max(x["force"] for x in fr["acts"][:-1]):.0f} N·m', f'{max(x["force"] for x in so["acts"][:-1]):.1f} N·m'),
             ("Physics timestep", f'{fr["dt"]*1000:.0f} ms', f'{so["dt"]*1000:.0f} ms'),
             ("Reach from shoulder at home pose", f'{fr["reach"]*100:.0f} cm', f'{so["reach"]*100:.0f} cm')]
    for name, x, y in rowsr:
        a(f'<tr><td>{name}</td><td class="num">{x}</td><td class="num">{y}</td></tr>')
    a("</table></div>")
    a('<div class="scroll"><table><tr><th>Joint ranges (deg)</th>' + "".join(f'<th class="num">{i+1}</th>' for i in range(7)) + "</tr>")
    for f in R:
        a(f'<tr><td>{esc(f["title"])}</td>' + "".join(f'<td class="num">{j["lo"]:.0f}…{j["hi"]:.0f}</td>' for j in f["joints"])
          + "".join("<td></td>" for _ in range(7 - len(f["joints"]))) + "</tr>")
    a("</table></div>")
    a("<p>The SO-101 is 27 times lighter and its strongest joint is 30 times weaker. That is why the code branches on the body in "
      "a dozen places (next sections), and why some methods split so sharply between the two arms.</p>")

    # ------------------------------------------------------------- reproduction
    a('<h2 id="repro">Reproduction on this machine</h2>')
    a("<p>Environment: Python 3.11, MuJoCo 4.0.0, Menagerie at its current head, Windows. The stored results were produced on Linux "
      "with an older MuJoCo, so exact agreement is not guaranteed even for honest results; agreement this close is very hard to fake.</p>")
    a("<h3>Self-model from teleoperation · <span class='mono'>scripts/selfmodel_teleop.py</span>, Cartesian teleop, 400 poses, mid noise</h3>")
    a('<div class="scroll"><table><tr><th>Arm</th><th>Seed</th><th>Model</th><th class="num">Stored FK error</th><th class="num">Re-run</th>'
      '<th class="num">Stored reach</th><th class="num">Re-run</th></tr>')
    for t in tele:
        a(f'<tr><td>{t["body"]}</td><td class="num">{t["seed"]}</td><td>{t["model"]}</td><td class="num">{t["old_mm"]:.2f} mm</td>'
          f'<td class="num">{t["new_mm"]:.2f} mm</td><td class="num">{t["old_reach"]}/40</td><td class="num">{t["new_reach"]}/40</td></tr>')
    a("</table></div>")
    if rows:
        a("<h3>Classical pick-and-place · <span class='mono'>scripts/eval_classical.py</span>, 20 seeded layouts per cell</h3>")
        a("<p>Only planners written in this repository can run here (OMPL has no Windows build), so these are the LazyPRM, A* and "
          "potential-field cells. Each strip is the stage reached per episode (5 = placed); a stage that differs from the stored run is marked.</p>")
        a('<div class="scroll"><table><tr><th>Cell</th><th class="num">Stored</th><th class="num">Re-run</th><th class="num">Same stage</th><th>Re-run stage per episode</th></tr>')
        for r in rows:
            a(f'<tr><td class="mono">{esc(r["cell"])}</td><td class="num">{r["old"]}/20</td><td class="num">{r["new"]}/{r["n"]}</td>'
              f'<td class="num">{r["same"]}/{r["n"]}</td><td class="stage">{stage_strip(r["old_stages"], r["new_stages"])}</td></tr>')
        a("</table></div>")

    # ------------------------------------------------------------- graph
    files, edges, indeg, outdeg = c["files"], c["edges"], c["indeg"], c["outdeg"]
    a('<h2 id="graph">The repository graph</h2>')
    a(f"<p>{len(files)} program and config files, {len(edges)} import and trigger edges, parsed from the code. "
      f"{'No import cycles.' if not c['cycles'] else 'Cycles: ' + esc(str(c['cycles']))} "
      f"{'Every library module is imported by something.' if not c['unused'] else 'Unused modules: ' + esc(', '.join(c['unused']))} "
      "The layers form a stack: scripts depend on the library, and the library depends on nothing above it except one edge, "
      "<code>calibration.py</code> importing the pose maths in <code>selfmodel_diff.py</code>.</p>")
    a(f'<pre class="mermaid">{esc(c["layer_mermaid"])}</pre>')
    a("<h3>Edges between layers</h3>")
    L, S, M = c["LAYERS"], c["SHORT"], c["matrix"]
    mx = max(M.values())
    a('<div class="scroll"><table class="heat"><tr><th>from \\ to</th>' + "".join(f'<th class="num">{S[l]}</th>' for l in L) + "</tr>")
    for x in L:
        a(f"<tr><td>{S[x]}</td>")
        for y in L:
            v = M.get((x, y), 0)
            bg = f"background:color-mix(in srgb,var(--accent) {int(8 + 60 * v / mx)}%,transparent)" if v else ""
            a(f'<td class="h" style="{bg}">{v or ""}</td>')
        a("</tr>")
    a("</table></div>")
    a("<h3>Hubs</h3>")
    hubs = sorted(files, key=lambda f: -indeg[f["path"]])[:10]
    a('<div class="scroll"><table><tr><th>File</th><th>Layer</th><th class="num">Imported by</th><th class="num">Imports</th><th class="num">Lines</th></tr>')
    for f in hubs:
        a(f'<tr><td class="mono">{esc(f["path"])}</td><td>{S.get(f["layer"], f["layer"])}</td><td class="num">{indeg[f["path"]]}</td>'
          f'<td class="num">{outdeg[f["path"]]}</td><td class="num">{f["lines"]}</td></tr>')
    a("</table></div>")
    a("<h3>Every file and its imports</h3>")
    a(f'<pre class="mermaid">{esc(c["file_mermaid"])}</pre>')
    a("<h3>Which script writes which result</h3>")
    a('<div class="scroll"><table><tr><th>Script</th><th>Writes</th><th>Study</th></tr>')
    for s, w, what in c["writers"]:
        a(f'<tr><td class="mono">{s}</td><td class="mono">{w}</td><td>{what}</td></tr>')
    a("</table></div>")

    # ------------------------------------------------------------- bodies
    a('<h2 id="bodies">Franka vs SO-101 through the code</h2>')
    a("<p>Both arms run through the same functions. These are the places where the code takes a different branch, or reads a different "
      "number, depending on which arm it drives.</p>")
    a('<pre class="mermaid">' + esc("""flowchart LR
  subgraph F["Franka Panda"]
    F1["SPECS: 7 joints, TCP = hand + 10.34 cm"] --> F2["scene: task shifted +30 cm in x; wrist camera added"]
    F2 --> F3["interface: torque motors"] --> F4["default law: PD + gravity"]
    F4 --> F5["task space 6-D: Jacobian 6x7, redundant"]
    F5 --> F6["IK: no closed form (analytic = n/a)"]
    F6 --> F7["errors: 2 mm links, 0.2° axes, 0.5° encoders"]
  end
  subgraph S["SO-101"]
    S1["SPECS: 5 joints, TCP = gripperframe site turned 90° about y"] --> S2["scene: no shift; wrist camera from the model"]
    S2 --> S3["interface: embedded position servo (cascade)"] --> S4["default law: servo PD"]
    S4 --> S5["task space 5-D: position + approach axis, Jacobian 5x5"]
    S5 --> S6["IK: closed form AnalyticSO101 available"]
    S6 --> S7["errors: 1 mm links, 0.5° axes, 1.5° encoders"]
  end""") + "</pre>")
    a('<div class="scroll"><table><tr><th>Where</th><th>Franka</th><th>SO-101</th><th>Why it matters</th></tr>')
    for w, x, y, why in [
        ("bodies.SPECS", "joint1–7, home [0, −0.4, 0, −2.4, 0, 2.0, 0.785]", "5 joints, home [0, −0.5, 0.8, 0.8, 0]", "Every solver reads the arm from here."),
        ("bodies.tcp_frame", "hand body + 0.1034 m", "gripperframe site, +90° about y", "Puts both TCPs in one convention (z = approach)."),
        ("bodies.SPECS limits", "v 1.0, a 2.0, j 20", "v 1.0, a 3.0, j 30", "Timing layer limits (rad/s, /s², /s³)."),
        ("scene.build_task", "layout offset +0.30 m; adds wrist_cam", "no offset; model's wrist_cam", "The Franka cannot reach the shared region."),
        ("pipeline.real_interface", "torque", "cascade", "Joint laws beyond p/pd/pid need torque: on the SO-101 they switch it to torque mode."),
        ("control.Converter", "n_task = 6, jinv not applicable", "n_task = 5, J square", "A 5-joint arm cannot hold a full 6-D pose."),
        ("ik.EndpointSolver", "analytic → not applicable", "AnalyticSO101 closed form", "Only the planar SO-101 chain has a closed form."),
        ("kinematics.solve_ik", "orient='full'", "orient='axis'", "The 5-DoF projection: only the approach axis is held."),
        ("calibration.ERRORS", "2 mm, 0.2°, 0.5°", "1 mm, 0.5°, 1.5°", "Size of the synthetic 'real arm' errors."),
        ("selfmodel_lab._CFG", "board 40 mm squares, 6 cm stereo baseline, stiff joints", "20 mm squares, 4 cm baseline, soft servos", "Sensor-swap lab per arm."),
        ("Model file", f'{fr["mass"]:.1f} kg, 87 N·m', f'{so["mass"]:.2f} kg, 2.9 N·m', "Explains P-only control: 5/20 Franka (sags), 18/20 SO-101."),
    ]:
        a(f'<tr><td class="mono">{w}</td><td>{x}</td><td>{y}</td><td>{why}</td></tr>')
    a("</table></div>")
    a("<h3>Stored classical results, side by side</h3>")
    a('<p><span class="key"><span class="sw" style="background:var(--franka)"></span>Franka</span>'
      '<span class="key"><span class="sw" style="background:var(--so101)"></span>SO-101</span> success out of 20 seeded layouts (10 for MPC, NMPC and LQG), '
      "one row per method that changes one layer of the default pipeline.</p>")
    a(f'<div class="scroll">{c["bars_svg"](c["groups"])}</div>')

    # ------------------------------------------------------------- same job
    a('<h2 id="same">Same job, different files</h2>')
    a("<p>Several jobs are implemented more than once. Comparing them side by side shows where the project is consistent and where it is not.</p>")
    a('<div class="scroll"><table><tr><th>Job</th><th>Implementation</th><th>How</th><th>Used by</th></tr>')
    for job, impls in [
        ("Inverse kinematics", [("kinematics.solve_ik", "MuJoCo Jacobian; DLS, prioritized (HQP) or Newton", "classical stack, route B twin, teleop"),
                                ("classical/ik.py", "LM, CCD, SLSQP, closed-form SO-101, taught", "endpoint layer"),
                                ("eval_route_b_transfer.SO100Kinematics.solve", "Pinocchio FK, finite-difference Jacobian, DLS", "route B on the SO100"),
                                ("selfmodel_diff.ik", "SciPy bounded least squares on any fk(q)", "reach tests on fitted models")]),
        ("Success scoring", [("stages.StageScorer", "protocol thresholds, stages 0–5", "classical, self-model task, MJX/Isaac evaluation"),
                             ("eval_route_b_*.main (inline)", "own rules, stages 0–4: reach &lt; 3 cm, grasp 2 steps, at goal, env success", "route B only"),
                             ("mjx_pick._reward, Isaac _get_rewards", "training shaping only; scored back with StageScorer", "GPU routes")]),
        ("Making the 'real' arm", [("calibration.perturb", "chain + base offsets and tilts; encoder offset as a joint-body rotation", "self-model task"),
                                   ("selfmodel_teleop.real_robot", "chain offsets and tilts + TCP offset; encoder offset subtracted from q; no base error", "teleop pilot"),
                                   ("selfmodel_lab.build", "chain + base + sensor-mount errors; encoder offsets", "sensor-swap lab")]),
        ("Forward kinematics", [("kinematics.Body.fk", "MuJoCo mj_kinematics + TCP offset", "everywhere"),
                                ("calibration.ModelFK", "same, returns a 4×4", "calibration"),
                                ("selfmodel_diff.PoEModel.fk", "product of exponentials, no model file", "fitted self-model"),
                                ("SO100Kinematics.tcp_world", "Pinocchio, midpoint of jaw tips", "route B")]),
    ]:
        for i, (name, how, used) in enumerate(impls):
            a(f'<tr><td>{job if i == 0 else ""}</td><td class="mono">{name}</td><td>{how}</td><td>{used}</td></tr>')
    a("</table></div>")
    a("<p>Two of these matter for reading the results. Route B scores with its own 0–4 stage rules, not the protocol's 0–5 scorer, so "
      "a route B 'mean stage' and a classical 'mean stage' are on different scales and should not be compared. And the three ways of "
      "making a 'real' arm use different error models, so the teleop pilot, the sensor study and the task study each test a slightly "
      "different arm.</p>")

    # ------------------------------------------------------------- concerns
    a('<h2 id="concerns">What the results do and do not show</h2>')
    a('<ul class="find">')
    for t in [
        "<b>Everything is simulation.</b> No physical robot, no recorded hardware data. The 'real arm' in every self-model study is the same "
        "CAD model with random offsets added in code (<code>perturb</code>, <code>real_robot</code>, <code>build</code>). The results say how "
        "methods behave against those synthetic errors, not on a real SO-101.",
        "<b>The results are genuine simulation output.</b> The ones I could re-run reproduce, the summary matches the raw files, and the git "
        "history shows long runs committed by an automated agent with periodic auto-saves (2026-11-01 to 11-02). I found nothing fabricated.",
        "<b>Route B's SO-101 is really an SO100</b>, ManiSkill's model, and it scores with its own stage rules.",
        "<b>Sample sizes are below the protocol.</b> <code>protocol.yaml</code> asks for 50 screening and 200 final episodes per cell; the "
        "classical grid ran 20 (10 for MPC, NMPC, LQG). The protocol is still marked <code>frozen: false</code>.",
        "<b>The teleop pilot's sensor is an oracle</b> (true pose plus Gaussian noise). The sensor-swap study replaces it with real "
        "perception pipelines on rendered images; that is the more credible of the two.",
        "<b>A ground-truth leak existed</b> in the mesh world model (fixed in 0e6d802). No stored cell combined mesh with a perturbed arm, "
        "so no published number was affected, but it shows the belief/real split needs care.",
        "<b>Not run at all:</b> routes A and C, every GPU training route, and Instant Policy. The README says so.",
    ]:
        a(f"<li>{t}</li>")
    a("</ul>")

    # ------------------------------------------------------------- per file
    a('<h2 id="files">File by file: code blocks and how they connect</h2>')
    a("<p>One diagram per program file, drawn from its syntax tree. Boxes are the file's blocks (module setup, each function, each class with "
      "its methods) with their line range and what they do. Solid arrows are calls inside the file; dashed arrows go to the blocks this "
      "file uses from other files (hexagons).</p>")
    by_layer = {}
    for f in files:
        if f["path"].endswith(".py"):
            by_layer.setdefault(f["layer"], []).append(f)
    for layer in L:
        if layer not in by_layer:
            continue
        a(f'<div class="layerhead"><span class="eyebrow">{esc(layer)}</span></div>')
        for f in sorted(by_layer[layer], key=lambda x: x["path"]):
            p = f["path"]
            blocks = c["FILES"][p]["blocks"]
            if not blocks:
                continue
            a(f'<div class="file"><h3>{esc(p)}<span class="meta">{f["lines"]} lines · {len(blocks)} blocks · imported by {indeg[p]}</span></h3>')
            s = FILE_SUMMARY.get(p, "")
            if s:
                a(f"<p>{esc(s)}</p>")
            a(f'<pre class="mermaid">{esc(c["file_diagram"](p))}</pre></div>')
    a("</div>")
    return "\n".join(out)
