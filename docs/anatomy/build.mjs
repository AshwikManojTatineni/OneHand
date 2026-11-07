// Fig 1 of the OneHand code anatomy: the whole codebase as a test bench.
// Every program file is one block; its height is its line count, the studs on its top are its functions and the
// rings its classes. Selecting a block lights the import wires that leave and enter it.
// Usage: node docs/anatomy/build.mjs [--light]   (writes docs/anatomy/onehand-anatomy[-light].html)
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { homedir } from "node:os";

const SKILL = join(homedir(), ".claude/skills/anatomy");
const k = await import(pathToFileURL(join(SKILL, "kit/iso-kit.mjs")).href);
const { kitScript } = await import(pathToFileURL(join(SKILL, "scripts/inline-kit.mjs")).href);

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const theme = process.argv.includes("--light") ? "light" : "dark";
const MAP = JSON.parse(readFileSync(join(HERE, "codemap.json"), "utf8"));

const WIDTH = 900;
const HEIGHT = 560;

// ------------------------------------------------------------------ the model: files, layers, edges
const LAYERS = {
  "Core Benchmark Library": "core",
  "Classical Robotics Stack": "classical",
  "Self-Modeling Experiments": "selfmodel",
  "GPU Training Routes": "gpu",
  "Evaluation and Analysis Scripts": "eval",
  "Experiment Grid Orchestration": "grid",
  "Protocol and Checkpoint Registry": "protocol",
  "Documentation and Results": "docs",
};
const ZONE_NAME = Object.fromEntries(Object.entries(LAYERS).map(([name, id]) => [id, name]));
const DOCS = ["README.md", "RESULTS.md", "isaac/README.md", "results/classical/SUMMARY.md", "results/phase0/SUMMARY.md",
  "results/selfmodel_sensors/SUMMARY.md", "results/selfmodel_sensors_so101/SUMMARY.md", "results/selfmodel_teleop/SUMMARY.md",
  "results/selfmodel_teleop_so101/SUMMARY.md"];
const files = MAP.files.map((f) => ({ ...f, zone: LAYERS[f.layer] }));
for (const path of DOCS) {
  const text = existsSync(join(ROOT, path)) ? readFileSync(join(ROOT, path), "utf8") : "";
  files.push({ path, lines: text.split("\n").length, functions: 0, classes: 0, layer: ZONE_NAME.docs, zone: "docs" });
}
const index = new Map(files.map((f, i) => [f.path, i]));
const EDGES = MAP.edges.filter(([a, b]) => index.has(a) && index.has(b)).map(([a, b]) => [index.get(a), index.get(b)]);
for (const f of files) {
  f.out = EDGES.filter(([a]) => files[a] === f).length;
  f.in = EDGES.filter(([, b]) => files[b] === f).length;
}
const byZone = (zone) => files.filter((f) => f.zone === zone).sort((a, b) => b.in - a.in || b.lines - a.lines);

// ------------------------------------------------------------------ world layout (units ~ px; x, y floor, z up)
const PLATE = { x: -280, y: -186, w: 560, d: 372, r: 10 };
const PLATE_H = 14;
const FOOT_H = 6;
const heightOf = (f) => (f.zone === "grid" ? 14 + f.lines * 0.9 : 5 + f.lines * 0.075);

// Zone fixtures: what each layer stands on.
const CORE_PAD = { x: 4, y: -88, w: 158, d: 92, r: 4 };
const CORE_PAD_H = 5;
const RAIL = { x: -262, y: -64, w: 238, d: 44 };
const SELF_PAD = { x: 178, y: -96, w: 92, d: 140, r: 4 };
const SELF_PAD_H = 3;
const SHELF = { x: -262, y: -176, w: 222, d: 46, r: 3 };
const SHELF_Z = 34;
const TRAY = { x: 154, y: -176, w: 116, d: 58, r: 3 };
const RACK = { x: -262, y: 34, w: 270, d: 84, r: 3 };
const HOPPER = { x: 30, y: 40, w: 104, d: 64, r: 2 };
const BOX = { x: 160, y: 62, w: 110, d: 76, r: 4 };
const BOX_H = 12;
const GAUGE = [222, 31];

const blocks = [];   // { file, plan, z, h, tone }
const place = (zone, origin, cols, cell, size, z = 0, tone = "hi") =>
  byZone(zone).forEach((f, i) => {
    const col = i % cols, row = Math.floor(i / cols);
    const big = zone === "core" && f.in >= 10 ? 1.25 : 1;
    const w = size[0] * big, d = size[1] * big;
    const plan = { x: origin[0] + col * cell[0] + (cell[0] - w) / 2, y: origin[1] + row * cell[1] + (cell[1] - d) / 2, w, d, r: 1.6 };
    blocks.push({ file: f, plan, z, h: heightOf(f), tone });
  });
place("core", [CORE_PAD.x + 2, CORE_PAD.y + 2], 4, [38.5, 44], [28, 30], CORE_PAD_H);
place("classical", [RAIL.x + 2, RAIL.y], 8, [29.25, 44], [22, 24], 7);
place("selfmodel", [SELF_PAD.x + 2, SELF_PAD.y + 2], 3, [29, 36], [22, 24], SELF_PAD_H);
place("gpu", [SHELF.x + 2, SHELF.y], 7, [31.2, 46], [22, 26], SHELF_Z + 3, "mid");
place("eval", [RACK.x + 3, RACK.y + 3], 7, [37.7, 39], [26, 26], 3);
place("grid", [HOPPER.x + 8, HOPPER.y + 10], 7, [13, 44], [6, 40], 2);
place("protocol", [BOX.x + 6, BOX.y + 8], 3, [33, 60], [26, 54], BOX_H);
// Docs: a stack of sheets in the tray, one sheet per document, offset like a pile of paper.
byZone("docs").forEach((f, i) => {
  const plan = { x: TRAY.x + 8 + (i % 3) * 1.6, y: TRAY.y + 7 + (i % 2) * 1.4, w: TRAY.w - 22, d: TRAY.d - 16, r: 0.8 };
  blocks.push({ file: f, plan, z: 3 + i * 1.6, h: 1.1, tone: "hi", sheet: true });
});
blocks.forEach((b) => { b.top = b.z + b.h; b.fi = index.get(b.file.path); });
const blockOf = new Map(blocks.map((b) => [b.fi, b]));

const tallest = Math.max(...blocks.map((b) => b.top));
const P = k.fitProjection(
  [...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0), ...blocks.map((b) => [b.plan.x, b.plan.y, b.top + 30]), [GAUGE[0], GAUGE[1], 60]],
  WIDTH, HEIGHT, { pad: 26, azimuth: 40 },
);
const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const centre = (plan) => [plan.x + plan.w / 2, plan.y + plan.d / 2];
const depthAt = (plan, z = 0) => k.depthOf([...centre(plan), z], P);

// ------------------------------------------------------------------ static structure
function plate() {
  const out = [`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`];
  const feet = [...k.corners(PLATE, 26), [PLATE.x + PLATE.w / 2, PLATE.y + 26], [PLATE.x + PLATE.w / 2, PLATE.y + PLATE.d - 26]];
  for (const [x, y] of feet.sort((a, b) => k.depthOf([...a, 0], P) - k.depthOf([...b, 0], P))) {
    out.push(S(k.cylinder(x, y, 9, -PLATE_H - FOOT_H, 2, P, 28), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 7, -PLATE_H - FOOT_H + 2, FOOT_H - 2, P, 28), { tone: "mid" }));
  }
  out.push(S(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 10, 1.6), { tone: "mid", inner: L(k.sideSeam(PLATE, -PLATE_H / 2, P, 10), { tone: "lo" }) }));
  out.push(L(k.planOutline(k.insetPlan(PLATE, 6), 0, P, 10), { tone: "lo" }));
  const screws = k.corners(PLATE, 12);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.6, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.55 }));
  // Ruler strip along the near edge: one major tick every 50 units, a minor every 10.
  const strip = { x: PLATE.x + 30, y: PLATE.y + PLATE.d - 18, w: PLATE.w - 60, d: 8, r: 1 };
  out.push(S(k.slabOf(strip, 0, 0.6, P, 3), { tone: "lo", crease: "none" }));
  const ticks = k.topTicks(strip.x + 5, strip.x + strip.w - 5, 10, 5, strip.y + 0.8, 0.6, [2.2, 4.2], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(D([at([strip.x + 2.5, strip.y + 4, 0.6]), at([strip.x + strip.w - 2.5, strip.y + 4, 0.6])], { size: 0.45 }));
  // Zone grooves on the plate: one inlaid frame per layer, so the plate reads as eight stations.
  for (const plan of [CORE_PAD, SELF_PAD, SHELF, TRAY, RACK, HOPPER, BOX, { ...RAIL, r: 3 }]) {
    out.push(L(k.planOutline(k.insetPlan(plan, -5), 0, P, 4), { tone: "faint" }));
  }
  return out.join("");
}

// The label plate at the front left: one engraved word, three rule lines for printed text.
function labelPlate() {
  const plan = { x: PLATE.x + 18, y: PLATE.y + PLATE.d - 50, w: 64, d: 24, r: 1.5 };
  const out = [S(k.slabOf(plan, 0, 1.2, P, 3, 0.4), { tone: "lo" })];
  out.push(k.faceTextSvg(k.topMatrix([plan.x + 6, plan.y + 8], 1.2, P, "x"), "ONEHAND", { size: 5, tone: "mid" }));
  for (const y of [13, 16.5, 20]) out.push(L(k.lineOnTop([plan.x + 6, plan.y + y], [plan.x + 40 - y, plan.y + y], 1.2, P), { tone: "faint", free: true }));
  out.push(D(k.corners(plan, 2.6).map(([x, y]) => at([x, y, 1.2])), { size: 0.4 }));
  return { key: depthAt(plan), svg: out.join("") };
}

function corePad() {
  const out = [S(k.slabOf(CORE_PAD, 0, CORE_PAD_H, P, 4, 0.8), { tone: "mid" })];
  out.push(L(k.planOutline(k.insetPlan(CORE_PAD, 3), CORE_PAD_H, P, 4), { tone: "faint" }));
  out.push(D(k.corners(CORE_PAD, 4).map(([x, y]) => at([x, y, CORE_PAD_H])), { size: 0.5 }));
  return { key: depthAt(CORE_PAD) - 200, svg: out.join("") };
}

function selfPad() {
  const out = [S(k.slabOf(SELF_PAD, 0, SELF_PAD_H, P, 4, 0.6), { tone: "mid" })];
  out.push(D(k.corners(SELF_PAD, 4).map(([x, y]) => at([x, y, SELF_PAD_H])), { size: 0.5 }));
  return { key: depthAt(SELF_PAD) - 200, svg: out.join("") };
}

// Classical stack: a guide rail with end stops; every module rides a carriage, in pipeline order.
function rail() {
  const out = [];
  for (const y of [RAIL.y + 6, RAIL.y + RAIL.d - 11]) {
    out.push(S(k.slabOf({ x: RAIL.x, y, w: RAIL.w, d: 5, r: 1 }, 0, 3, P, 3, 0.4), { tone: "mid" }));
    out.push(D([RAIL.x + 8, RAIL.x + RAIL.w / 2, RAIL.x + RAIL.w - 8].map((x) => at([x, y + 2.5, 3])), { size: 0.4 }));
  }
  for (const x of [RAIL.x - 6, RAIL.x + RAIL.w]) {
    out.push(S(k.slabOf({ x, y: RAIL.y + 2, w: 6, d: RAIL.d - 4, r: 1.2 }, 0, 9, P, 3, 0.5), { tone: "mid" }));
    out.push(D([at([x + 3, RAIL.y + 7, 9]), at([x + 3, RAIL.y + RAIL.d - 7, 9])], { size: 0.45 }));
  }
  return { key: depthAt(RAIL) - 300, svg: out.join("") };
}

function carriageUnder(b) {
  const plan = k.insetPlan(b.plan, -3);
  const out = [S(k.slabOf(plan, 3, 4, P, 3, 0.5), { tone: "mid" })];
  out.push(D(k.corners(plan, 2.2).map(([x, y]) => at([x, y, 7])), { size: 0.35, tone: "lo" }));
  return out.join("");
}

// GPU routes: a shelf on four posts, its modules in mid tone (written, never run on this machine).
function shelf() {
  const out = [];
  for (const [x, y] of k.corners(SHELF, 6).sort((a, b) => k.depthOf([...a, 0], P) - k.depthOf([...b, 0], P))) {
    out.push(S(k.cylinder(x, y, 4.2, 0, 1.4, P, 20), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 2.8, 1.4, SHELF_Z - 1.4, P, 20), { tone: "mid", crease: "none" }));
  }
  out.push(S(k.slabOf(SHELF, SHELF_Z, 3, P, 4, 0.6), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(SHELF, 3), SHELF_Z + 3, P, 4), { tone: "faint" }));
  out.push(D(k.corners(SHELF, 6).map(([x, y]) => at([x, y, SHELF_Z + 3])), { size: 0.5 }));
  return { key: depthAt(SHELF) - 300, svg: out.join("") };
}

function tray() {
  const out = [S(k.slabOf(TRAY, 0, 3, P, 4, 0.5), { tone: "mid" })];
  const lip = [
    { x: TRAY.x, y: TRAY.y, w: TRAY.w, d: 3, r: 1 },
    { x: TRAY.x, y: TRAY.y, w: 3, d: TRAY.d, r: 1 },
  ];
  for (const plan of lip) out.push(S(k.slabOf(plan, 3, 4, P, 2), { tone: "mid" }));
  return { key: depthAt(TRAY) - 300, svg: out.join("") };
}

function trayFront() {
  const lip = [
    { x: TRAY.x + TRAY.w - 3, y: TRAY.y, w: 3, d: TRAY.d, r: 1 },
    { x: TRAY.x, y: TRAY.y + TRAY.d - 3, w: TRAY.w, d: 3, r: 1 },
  ];
  return { key: depthAt(TRAY) + 300, svg: lip.map((plan) => S(k.slabOf(plan, 3, 4, P, 2), { tone: "mid" })).join("") };
}

// Evaluation scripts stand in a slotted rack.
function rack() {
  const out = [S(k.slabOf(RACK, 0, 3, P, 4, 0.6), { tone: "mid" })];
  for (let row = 0; row < 2; row++) {
    for (let col = 0; col < 7; col++) {
      const slot = { x: RACK.x + 3 + col * 37.7 + 4, y: RACK.y + 3 + row * 39 + 4, w: 29.7, d: 31, r: 1 };
      out.push(L(k.planOutline(slot, 3, P, 2), { tone: "faint" }));
    }
  }
  out.push(D(k.corners(RACK, 3).map(([x, y]) => at([x, y, 3])), { size: 0.45 }));
  return { key: depthAt(RACK) - 300, svg: out.join("") };
}

// Grid orchestration: punch cards standing in a hopper with low walls (back walls behind, front walls in front).
const HOP_WALL = 9;
function hopperBack() {
  const out = [S(k.slabOf(HOPPER, 0, 2, P, 2), { tone: "lo" })];
  out.push(S(k.slabOf({ x: HOPPER.x, y: HOPPER.y, w: HOPPER.w, d: 3, r: 0.8 }, 2, HOP_WALL, P, 2), { tone: "mid" }));
  out.push(S(k.slabOf({ x: HOPPER.x, y: HOPPER.y, w: 3, d: HOPPER.d, r: 0.8 }, 2, HOP_WALL, P, 2), { tone: "mid" }));
  return { key: depthAt(HOPPER) - 300, svg: out.join("") };
}
function hopperFront() {
  const out = [];
  out.push(S(k.slabOf({ x: HOPPER.x + HOPPER.w - 3, y: HOPPER.y, w: 3, d: HOPPER.d, r: 0.8 }, 2, HOP_WALL, P, 2), { tone: "mid" }));
  out.push(S(k.slabOf({ x: HOPPER.x, y: HOPPER.y + HOPPER.d - 3, w: HOPPER.w, d: 3, r: 0.8 }, 2, HOP_WALL, P, 2), { tone: "mid" }));
  out.push(D([at([HOPPER.x + 8, HOPPER.y + HOPPER.d, 2 + HOP_WALL]), at([HOPPER.x + HOPPER.w - 8, HOPPER.y + HOPPER.d, 2 + HOP_WALL])], { size: 0.4 }));
  return { key: depthAt(HOPPER) + 300, svg: out.join("") };
}

// Protocol and registry: a sealed box; its three files sit on the lid.
function box() {
  const out = [S(k.slabOf(BOX, 0, BOX_H, P, 6, 0.8), { tone: "mid", inner: L(k.sideSeam(BOX, BOX_H - 3, P, 6), { tone: "lo" }) })];
  out.push(L(k.planOutline(k.insetPlan(BOX, 3), BOX_H, P, 6), { tone: "faint" }));
  out.push(D(k.corners(BOX, 4).map(([x, y]) => at([x, y, BOX_H])), { size: 0.5 }));
  // hasp on the near face
  const hasp = { x: BOX.x + BOX.w / 2 - 5, y: BOX.y + BOX.d, w: 10, d: 2, r: 0.6 };
  out.push(S(k.slabOf(hasp, BOX_H - 8, 7, P, 2), { tone: "lo" }));
  out.push(L(k.sideRing([BOX.x + BOX.w / 2, BOX.y + BOX.d + 2, BOX_H - 4.5], 1.4, "y", P), { tone: "mid" }));
  return { key: depthAt(BOX) - 300, svg: out.join("") };
}

// The self-model station's gauge: a dial indicator whose needle reads the hand error the fit removes
// (SO-101: model file 12.2 mm, self-model 1.0 mm, RESULTS.md section 3). One turn = 20 mm.
const DIAL_R = 15, DIAL_Z = 40, DIAL_H = 7, DIAL_TOP = DIAL_Z + DIAL_H, PER_TURN = 20;
function gauge() {
  const [x, y] = GAUGE;
  const out = [S(k.slabOf({ x: x - 12, y: y - 10, w: 24, d: 20, r: 2 }, 0, 9, P, 3, 0.6), { tone: "mid" })];
  out.push(S(k.cylinder(x, y, 3, 9, DIAL_Z - 9, P, 20), { tone: "mid", crease: "none" }));
  out.push(L(k.sideArc(x, y, 3, 22, P, 12), { tone: "lo" }));
  out.push(S(k.cylinder(x, y, DIAL_R, DIAL_Z, DIAL_H, P, 64, 0.8), { tone: "hi", crease: "lo" }));
  out.push(L(k.knurl(x, y, DIAL_R, DIAL_TOP - 2.6, DIAL_TOP, 64, P, 0.3), { tone: "lo" }));
  out.push(L(k.ring(x, y, DIAL_R - 1.8, DIAL_TOP, P, 64), { tone: "lo" }));
  const t = k.radialTicks(x, y, DIAL_R - 2.4, 40, 5, DIAL_TOP, [1.1, 2.4], P);
  out.push(L(t.minor, { tone: "lo" }), L(t.major, { tone: "mid" }));
  out.push(L(k.ring(x, y, 1.3, DIAL_TOP, P, 16), { tone: "mid" }));
  out.push(k.faceTextSvg(k.topMatrix([x, y + 7], DIAL_TOP, P, "x"), "mm", { size: 2.2, tone: "lo", anchor: "middle" }));
  return { key: k.depthOf([x, y, 0], P), svg: out.join("") };
}
const needleAt = (mm) => {
  const a = (mm / PER_TURN) * Math.PI * 2;
  const [x, y] = GAUGE;
  return k.lineOnTop([x - Math.sin(a) * 2.6, y + Math.cos(a) * 2.6], [x + Math.sin(a) * (DIAL_R - 3.4), y - Math.cos(a) * (DIAL_R - 3.4)], DIAL_TOP + 0.2, P);
};

// ------------------------------------------------------------------ the file blocks
function blockSvg(b, i) {
  const { plan, z, h, file, tone } = b;
  const top = z + h;
  const parts = [];
  if (file.zone === "classical") parts.push(carriageUnder(b));
  const inner = b.sheet ? "" : L(k.sideSeam(plan, z + Math.min(2.2, h / 3), P, 3), { tone: "lo" });
  parts.push(`<g data-body>${S(k.slabOf(plan, z, h, P, 3, b.sheet ? 0 : 0.5), { tone, crease: b.sheet ? "none" : "lo", inner })}</g>`);
  if (!b.sheet) {
    const face = k.insetPlan(plan, 2.4);
    parts.push(L(k.planOutline(face, top, P, 3), { tone: "faint" }));
    // Studs: one per function, laid in rows over the top face.
    const n = file.functions;
    if (n) {
      const cols = Math.max(1, Math.ceil(Math.sqrt(n * face.w / face.d)));
      const rows = Math.ceil(n / cols);
      const gx = face.w / (cols + 1), gy = face.d / (rows + 1);
      const pts = [];
      for (let s = 0; s < n; s++) pts.push(at([face.x + gx * (1 + (s % cols)), face.y + gy * (1 + Math.floor(s / cols)), top]));
      parts.push(D(pts, { size: Math.min(0.62, 0.22 * Math.min(gx, gy)), tone: "mid" }));
    }
    // Rings: one per class, along the near edge of the top.
    for (let c = 0; c < file.classes; c++) {
      parts.push(L(k.ring(plan.x + plan.w - 3.2 - c * 3.6, plan.y + plan.d - 3.2, 1.2, top, P, 12), { tone: "mid" }));
    }
  } else {
    // A sheet: three printed rule lines.
    for (const f of [0.3, 0.5, 0.7]) {
      parts.push(L(k.lineOnTop([plan.x + 6, plan.y + plan.d * f], [plan.x + plan.w * (0.55 + 0.1 * (i % 3)), plan.y + plan.d * f], top, P), { tone: "faint", free: true }));
    }
  }
  return `<g data-file="${b.fi}">${parts.join("")}</g>`;
}

// Painter's order: every block and fixture by depth; fixtures carry offsets that keep them under their blocks.
const items = [plate()].map((svg) => ({ key: -1e9, svg }));
for (const f of [labelPlate(), corePad(), selfPad(), rail(), shelf(), tray(), trayFront(), rack(), hopperBack(), hopperFront(), box(), gauge()]) items.push(f);
const order = blocks.map((b, i) => ({ i, key: depthAt(b.plan, b.sheet ? b.z : 0) + (b.sheet ? b.z * 10 : 0) }));
for (const { i, key } of order) items.push({ key, svg: blockSvg(blocks[i], i) });
items.sort((a, b) => a.key - b.key);

// ------------------------------------------------------------------ import wires (one per edge, lit on selection)
const anchor = (b) => [...centre(b.plan), b.top];
function wireOf([a, b], e) {
  const A = anchor(blockOf.get(a)), B = anchor(blockOf.get(b));
  const span = Math.hypot(A[0] - B[0], A[1] - B[1]);
  const lift = 18 + span * 0.16;
  // A smooth hanging arc: the straight chord between the two tops, raised by a sine bump tall enough to clear
  // the higher end by `lift`.
  const H = lift + Math.abs(A[2] - B[2]) / 2;
  const via = [];
  for (let s = 1; s < 16; s++) {
    const t = s / 16;
    via.push(at([A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t, A[2] + (B[2] - A[2]) * t + H * Math.sin(Math.PI * t)]));
  }
  return `<g class="oh-wire" data-from="${a}" data-to="${b}">${k.wireSvg(at(A), at(B), { via, tone: "lo", size: 1.1 })}</g>`;
}
const wires = EDGES.map(wireOf).join("");

const svg = k.figureSvg({
  width: WIDTH, height: HEIGHT,
  label: "The OneHand codebase as a test bench. A surface plate carries eight stations, one per architectural layer: core library blocks on a pad, the classical stack on carriages along a rail, self-model files on a gauge station, GPU routes on a raised shelf, evaluation scripts in a rack, grid scripts as cards in a hopper, the protocol box, and a tray of documents. Each block's height is its line count, the studs on top are its functions and the rings its classes. Selecting a block lights the import wires to and from it.",
  body: [...items.map((it) => it.svg), `<path data-part="needle" class="iso-line" data-tone="lit" data-free="start" d="${needleAt(12.2)}" style="stroke-width:1"/>`, `<g class="oh-wires">${wires}</g>`],
});

// ------------------------------------------------------------------ live behaviour
const DATA = files.map((f) => ({ p: f.path, z: f.zone, n: f.lines, f: f.functions, c: f.classes, i: f.in, o: f.out }));
const TOUR = ["xbody/stats.py", "xbody/bodies.py", "xbody/kinematics.py", "xbody/scene.py", "xbody/classical/pipeline.py",
  "xbody/selfmodel_diff.py", "scripts/eval_selfmodel_task.py", "xbody/newt_agent.py", "scripts/eval_route_b_transfer.py",
  "xbody/learned/mjx_pick.py", "scripts/run_classical_grid.sh", "protocol.yaml"].map((p) => index.get(p));
const ZONES = { core: "core library", classical: "classical stack", selfmodel: "self-model", gpu: "GPU route (not run)", eval: "evaluation", grid: "grid script", protocol: "protocol", docs: "docs" };
const live = `
const P = ${JSON.stringify(P)};
const FILES = ${JSON.stringify(DATA)};
const TOUR = ${JSON.stringify(TOUR)};
const ZONES = ${JSON.stringify(ZONES)};
const LIFT = ${JSON.stringify(k.screenRise(5, P))};
const GAUGE = ${JSON.stringify(GAUGE)}, DIAL_R = ${DIAL_R}, DIAL_TOP = ${DIAL_TOP}, PER_TURN = ${PER_TURN};
const stage = document.querySelector(".iso-stage");
const svg = stage.querySelector("svg");
const readout = document.querySelector("[data-readout]");
const needle = svg.querySelector('[data-part="needle"]');
const groups = [];
for (const g of svg.querySelectorAll("[data-file]")) groups[+g.dataset.file] = g;
const wires = [...svg.querySelectorAll(".oh-wire")];
const still = matchMedia("(prefers-reduced-motion: reduce)");
const needleAt = ${needleAt.toString().replace("k.lineOnTop", "lineOnTop")};
let sel = -1, lift = groups.map(() => 0), want = groups.map(() => 0), frame = 0, last = 0, visible = false;
let touring = !still.matches, clock = 0, tourAt = 0, idle = 0, gaugeMm = 12.2, gaugeWant = 12.2;
function select(i) {
  if (i === sel) return;
  if (sel >= 0) { groups[sel].querySelector(".iso-solid")?.removeAttribute("data-lit"); want[sel] = 0; }
  sel = i;
  const f = FILES[i];
  groups[i].querySelector(".iso-solid")?.setAttribute("data-lit", "");
  want[i] = 1;
  for (const w of wires) {
    const on = +w.dataset.from === i || +w.dataset.to === i;
    w.toggleAttribute("data-on", on);
    w.querySelector(".iso-wire").setAttribute("data-tone", on ? "lit" : "lo");
    w.querySelector(".iso-line").setAttribute("data-tone", on ? "lit" : "lo");
  }
  gaugeWant = f.z === "selfmodel" ? 1.0 : 12.2;
  const text = f.p + " · " + ZONES[f.z] + " · " + f.n + " lines · " + f.f + " fn · in " + f.i + " / out " + f.o;
  readout.textContent = text;
  stage.setAttribute("aria-valuenow", String(i));
  stage.setAttribute("aria-valuetext", text);
  run();
}
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  const calm = still.matches;
  if (touring) { clock += dt; if (clock > tourAt) { tourAt = clock + 2.6; select(TOUR[Math.floor(clock / 2.6) % TOUR.length]); } }
  let moving = false;
  groups.forEach((g, i) => {
    const t = want[i];
    let v = calm ? t : lift[i] + (t - lift[i]) * (1 - Math.exp(-dt / 0.18));
    if (Math.abs(t - v) < 0.002) v = t; else moving = true;
    if (v !== lift[i] || t) { lift[i] = v; g.setAttribute("transform", "translate(0 " + (-LIFT * v).toFixed(2) + ")"); }
  });
  gaugeMm = calm ? gaugeWant : gaugeMm + (gaugeWant - gaugeMm) * (1 - Math.exp(-dt / 0.5));
  if (Math.abs(gaugeWant - gaugeMm) < 0.01) gaugeMm = gaugeWant; else moving = true;
  needle.setAttribute("d", needleAt(gaugeMm));
  frame = visible && (touring || moving) ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
const rest = () => { clearTimeout(idle); idle = setTimeout(() => { touring = !still.matches; clock = 0; tourAt = 0; run(); }, 5000); };
new IntersectionObserver((entries) => { visible = entries[entries.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px 0px" }).observe(stage);
svg.addEventListener("pointerover", (event) => {
  const g = event.target.closest("[data-file]");
  if (!g) return;
  touring = false; rest(); select(+g.dataset.file);
});
svg.addEventListener("click", (event) => { const g = event.target.closest("[data-file]"); if (g) { touring = false; rest(); select(+g.dataset.file); } });
stage.addEventListener("keydown", (event) => {
  let to = null;
  if (event.key === "ArrowRight" || event.key === "ArrowDown") to = (Math.max(sel, 0) + 1) % FILES.length;
  if (event.key === "ArrowLeft" || event.key === "ArrowUp") to = (Math.max(sel, 0) - 1 + FILES.length) % FILES.length;
  if (event.key === "Home") to = 0;
  if (event.key === "End") to = FILES.length - 1;
  if (to === null) return;
  event.preventDefault(); touring = false; rest(); select(to);
});
select(TOUR[0]);
`;

const css = `<style>
.oh-wire{opacity:0;transition:opacity .45s ease}
.oh-wire[data-on]{opacity:1}
.oh-wire .iso-line{stroke-width:.9}
[data-file]{cursor:pointer}
@media (prefers-reduced-motion: reduce){.oh-wire{transition:none}}
</style>`;

const fig = k.plateHtml({
  fig: "Fig 1",
  title: "OneHand, the bench",
  hint: "Hover a block, or use the arrow keys",
  readout: "",
  keys: [
    { mark: "lit", label: "The selected file and its import wires" },
    { mark: "raised", label: "One block per file: height is lines of code, studs are functions, rings are classes" },
    { mark: "dim", label: "GPU routes on the shelf: written, never run here" },
  ],
  caption: `${files.length} files, ${EDGES.length} import and trigger edges, eight layers. Back row: the GPU shelf and the document tray. Middle row: the classical stack on its rail, the core library on its pad (stats, bodies, kinematics and scene are the most imported, 10 to 14 importers each), and the self-model station, whose gauge reads the SO-101 hand error the fitted self-model removes (12.2 mm to 1.0 mm). Front row: the evaluation rack, the grid-script hopper and the protocol box.`,
  body: `${css}<div class="iso-stage" tabindex="0" role="slider" aria-label="OneHand codebase: arrow keys step through the files" aria-valuemin="0" aria-valuemax="${files.length - 1}" aria-valuenow="0">${svg}</div>`,
});

const report = existsSync(join(HERE, "report.html")) ? readFileSync(join(HERE, "report.html"), "utf8") : "";
const name = theme === "light" ? "onehand-anatomy-light.html" : "onehand-anatomy.html";
writeFileSync(join(HERE, name), k.pageHtml({ title: "OneHand code anatomy", theme, body: fig + report, script: kitScript() + live, width: 960 }));
if (process.argv.includes("--debug")) for (const b of blocks) console.log(b.file.zone, b.file.path, at([...centre(b.plan), b.top]).map(Math.round));
console.log("wrote", name, `${files.length} files, ${EDGES.length} edges, ${items.length} items`);
