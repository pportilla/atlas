"use strict";

const $ = (id) => document.getElementById(id);
const form = $("resolve-form");
let lastResult = null;
let lastRequest = null;
let busy = false;
let searchJob = null;
let searchStarting = false;
let searchTransition = 0;
let pollTimer = null;

// The exceptional component picked in the graph or the table, and the parts of
// the page that follow it. Both are reset for every new result.
let selected = null;
const selectionViews = [];
function selectComponent(id) {
  selected = id;
  for (const update of selectionViews) update(selected);
}

/* ---------- Small DOM helpers ---------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function svgElement(tag, attrs, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function recordJSON(value, space) {
  return JSON.stringify(value, (_key, field) => typeof field === "bigint" ? JSON.rawJSON(String(field)) : field, space);
}
function isNumeric(value) {
  return typeof value === "number" || typeof value === "bigint";
}
function stringify(value) {
  return typeof value === "string" ? value : recordJSON(value, 2);
}
function code(value) {
  return el("code", "", typeof value === "string" ? value : recordJSON(value));
}
function plural(count, single, multiple) {
  return count + " " + (count === 1 ? single : multiple);
}
function signed(number) {
  return String(number).replace(/^-/, "−");
}
function joined(nodes, separator) {
  const fragment = document.createDocumentFragment();
  nodes.forEach((node, index) => {
    if (index) fragment.append(separator);
    fragment.append(node);
  });
  return fragment;
}
function duration(ms) {
  if (!Number.isFinite(ms)) return "";
  if (ms < 1000) return Number(ms.toPrecision(ms < 10 ? 2 : 3)) + " ms";
  const seconds = ms / 1000;
  if (seconds < 60) return Number(seconds.toPrecision(3)) + " s";
  return Math.floor(seconds / 60) + " min " + Math.round(seconds % 60) + " s";
}

/* ---------- Typesetting ----------
   The engine and the user write polynomials as plain text: u^2*(v^2-u).
   typeset() only changes how that text looks. It never rearranges or
   simplifies, so what is shown is what was computed. */

const GREEK = { alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", zeta: "ζ", eta: "η",
  theta: "θ", iota: "ι", kappa: "κ", lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", rho: "ρ", sigma: "σ",
  tau: "τ", phi: "φ", chi: "χ", psi: "ψ", omega: "ω" };

function tokensOf(text) {
  return String(text).match(/\*\*|[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?|\S/g) || [];
}
function splitName(name) {
  const [, base, rest] = /^([A-Za-z]+)(.*)$/.exec(name);
  return { base, index: rest.replace(/^_/, "").replace(/_/g, ",") };
}
function isWord(name) {
  const { base } = splitName(name);
  return base.length > 1 && !GREEK[base];
}
// x → italic x, u1 → u with index 1, alpha12_1 → α with index 12,1.
function nameNode(name) {
  const { base, index } = splitName(name);
  const node = document.createDocumentFragment();
  if (GREEK[base]) node.append(el("i", "", GREEK[base]));
  else if (base.length === 1) node.append(el("i", "", base));
  else node.append(el("span", "word", base));
  if (index) node.append(el("sub", "", index));
  return node;
}
function fraction(numerator, denominator) {
  const node = el("span", "frac");
  node.append(el("span", "", numerator), el("span", "", denominator));
  return node;
}
function closing(tokens, open, end) {
  let depth = 0;
  for (let i = open; i < end; i++) {
    if (tokens[i] === "(") depth++;
    else if (tokens[i] === ")" && --depth === 0) return i;
  }
  return -1;
}
function renderTokens(tokens, start, end, out) {
  const at = (i) => (i < end ? tokens[i] : undefined);
  const integer = (t) => typeof t === "string" && /^\d+$/.test(t);
  const power = (t) => t === "^" || t === "**";
  let before = "start"; // start, open, op, sep, or operand
  let beforeWord = false;
  for (let i = start; i < end; i++) {
    const t = tokens[i];
    if (/^\d/.test(t)) {
      if (before !== "operand" && integer(t) && at(i + 1) === "/" && integer(at(i + 2)) && !power(at(i + 3))) {
        out.append(fraction(t, tokens[i + 2]));
        i += 2;
      } else out.append(t);
      before = "operand";
      beforeWord = false;
    } else if (/^[A-Za-z]/.test(t)) {
      out.append(nameNode(t));
      before = "operand";
      beforeWord = isWord(t);
    } else if (t === "+" || t === "-") {
      const sign = t === "-" ? "−" : "+";
      out.append(before === "operand" ? " " + sign + " " : sign);
      before = "op";
    } else if (t === "*") {
      const next = at(i + 1) || "";
      if (before !== "operand" || /^\d/.test(next)) out.append(" · ");
      else if (beforeWord || (/^[A-Za-z]/.test(next) && isWord(next))) out.append(" ");
      before = "op";
    } else if (power(t)) {
      const i0 = i;
      const next = at(i + 1);
      const sup = el("sup");
      if (integer(next)) {
        sup.textContent = next;
        i += 1;
      } else if (next === "-" && integer(at(i + 2))) {
        sup.textContent = "−" + tokens[i + 2];
        i += 2;
      } else if (next === "(" && closing(tokens, i + 1, end) > 0) {
        const close = closing(tokens, i + 1, end);
        renderTokens(tokens, i + 2, close, sup);
        i = close;
      } else {
        out.append("^");
        before = "op";
        continue;
      }
      // An index and an exponent on the same letter are stacked, as in print.
      const index = out.lastChild;
      if (index && index.nodeName === "SUB" && /^[A-Za-z]/.test(tokens[i0 - 1] || "")) {
        const stack = el("span", "scripts");
        stack.append(sup, index);
        out.append(stack);
      } else out.append(sup);
      before = "operand";
      beforeWord = false;
    } else if (t === "(") {
      // (1/2) is shown as the fraction alone.
      if (integer(at(i + 1)) && at(i + 2) === "/" && integer(at(i + 3)) && at(i + 4) === ")" && !power(at(i + 5))) {
        out.append(fraction(tokens[i + 1], tokens[i + 3]));
        i += 4;
        before = "operand";
        beforeWord = false;
      } else {
        out.append("(");
        before = "open";
      }
    } else if (t === ")") {
      out.append(")");
      before = "operand";
      beforeWord = false;
    } else if (t === ",") {
      out.append(", ");
      before = "sep";
    } else if (t === "=") {
      out.append(" = ");
      before = "sep";
    } else {
      out.append(t);
      before = "sep";
    }
  }
}
function typeset(source, suffix) {
  const text = String(source);
  const node = el("span", "m");
  try {
    if (text.length > 40000) throw new Error("too long to typeset");
    const tokens = tokensOf(text);
    renderTokens(tokens, 0, tokens.length, node);
  } catch {
    node.replaceChildren(text);
  }
  if (suffix) node.append(suffix);
  return node;
}
// E1 → E with index 1. Chart names such as B1.u stay as they are.
function idNode(id) {
  const match = /^([A-Z])(\d[A-Za-z0-9.]*)$/.exec(String(id));
  if (!match) return el("span", "", String(id));
  const node = el("span", "m");
  node.append(el("i", "", match[1]), el("sub", "", match[2]));
  return node;
}
function tuple(entries, open = "(", close = ")", separator = ", ") {
  const node = el("span", "m");
  node.append(open);
  entries.forEach((entry, index) => {
    if (index) node.append(separator);
    if (isNumeric(entry)) node.append(signed(entry));
    else {
      const tokens = tokensOf(entry);
      renderTokens(tokens, 0, tokens.length, node);
    }
  });
  node.append(close);
  return node;
}
// Long expressions are folded to a few lines.
function foldable(node, length) {
  if (length < 700) return node;
  const wrap = el("div");
  const box = el("div", "clamp");
  box.append(node);
  const toggle = el("button", "link", "Show all");
  toggle.type = "button";
  toggle.addEventListener("click", () => {
    toggle.textContent = box.classList.toggle("open") ? "Show less" : "Show all";
  });
  wrap.append(box, toggle);
  return wrap;
}

/* ---------- Generic rendering of engine records ----------
   Known fields get a readable label and typeset values. A field this page
   has never seen still appears, under its own name. */

const LABELS = {
  total_multiplicity: "Total multiplicity", multiplicity: "Multiplicity", ambient_order: "Ambient order",
  self_intersection: "Self-intersection", genus: "Genus", dimension: "Dimension", ray: "Ray",
  ray_index: "Ray index", ray_indices: "Rays of the cone", root: "Root", stratum_root: "Root",
  ambient_divisor: "Ambient divisor", ambient_components: "Meets", chart: "Chart", point: "Point",
  type: "Type", center: "Center", parent_chart: "In chart", exceptional_id: "Creates",
  strict_multiplicity: "Strict multiplicity", center_ideal: "Center ideal", center_charts: "Center, in charts",
  generators: "Generators", weight: "Weight", faces: "Faces", map_to_parent: "Map to the parent chart",
  final: "Final chart", tangent_cone: "Tangent cone", elapsed_ms: "Time", native_elapsed_ms: "Native time",
  euler_characteristic: "Euler characteristic", resolution_engine: "Resolution engine",
};
const MATH_KEYS = new Set(["equation", "strict_transform", "ambient_total_transform", "pullback_factor",
  "tangent_cone", "minimal_polynomial", "polynomial", "adapted_polynomial", "pivot_generator", "power_value",
  "unit", "generator", "root", "stratum_root", "ideal", "generators", "center_ideal", "ambient_ideal",
  "coordinates", "projective_coordinates", "face_coordinates"]);
const ID_KEYS = new Set(["exceptional_id", "ambient_divisor", "ambient_components", "vertices"]);

// Whether a string field holds an expression to typeset rather than prose.
function isMath(key, text) {
  return (MATH_KEYS.has(key) || /(equation|polynomial|transform|ideal|factor|generator)s?$/.test(key))
    && /^[\w\s+\-*/^().,=]*$/.test(text) && !/[A-Za-z]{2,}\s+[A-Za-z]{2,}/.test(text);
}
function labelFor(key) {
  return LABELS[key] || key.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
}
function centerCharts(list) {
  const wrap = el("span", "eqs");
  for (const entry of list) {
    const line = el("span");
    if (Array.isArray(entry.ideal)) line.append(tuple(entry.ideal));
    if (entry.chart !== undefined) line.append(" in chart " + entry.chart);
    wrap.append(line);
  }
  return wrap;
}
function valueNode(key, value) {
  if (value === null || value === undefined || value === "") return document.createTextNode("—");
  if (typeof value === "boolean") return document.createTextNode(value ? "yes" : "no");
  if (isNumeric(value)) {
    return document.createTextNode(/elapsed_ms$/.test(key) ? duration(value) : signed(value));
  }
  if (typeof value === "string") {
    if (ID_KEYS.has(key)) return idNode(value);
    if (isMath(key, value)) return foldable(typeset(value), value.length);
    return document.createTextNode(value);
  }
  if (Array.isArray(value)) {
    if (!value.length) return document.createTextNode("—");
    if (value.every(isNumeric)) return tuple(value);
    if (value.every((v) => typeof v === "string")) {
      if (ID_KEYS.has(key)) return joined(value.map(idNode), ", ");
      if (key === "point" || key === "center") return tuple(value);
      if (value.every((v) => isMath(key, v))) return joined(value.map((v) => typeset(v)), ", ");
      return document.createTextNode(value.join(", "));
    }
    if (key === "center_charts" && value.every((v) => v && typeof v === "object" && !Array.isArray(v))) {
      return centerCharts(value);
    }
    if (value.length <= 12 && value.every((v) => Array.isArray(v) && v.length && v.every(isNumeric))) {
      return joined(value.map((v) => tuple(v)), ", ");
    }
  }
  const text = recordJSON(value);
  return foldable(code(text), text.length);
}
function dataList(entries) {
  const list = el("dl", "facts");
  for (const [key, value] of entries) {
    const detail = el("dd");
    detail.append(value instanceof Node ? value : valueNode(key, value));
    list.append(el("dt", "", labelFor(key)), detail);
  }
  return list;
}
function sectionHead(title, hint) {
  const head = el("div", "section-head");
  const heading = el("h3", "", title);
  if (hint) heading.title = hint;
  head.append(heading);
  return head;
}
// A section that stays folded until its heading is clicked.
function foldedSection(panel, title, count) {
  const details = el("details", "fold");
  const summary = el("summary");
  const number = el("span", "count", count === undefined ? "" : String(count));
  const body = el("div", "fold-body");
  summary.append(el("h3", "", title), number);
  details.append(summary, body);
  panel.append(details);
  return { body, number };
}
function disclosure(label, content) {
  const details = el("details", "more");
  const body = el("div", "more-body");
  if (typeof content === "function") {
    function populate() {
      if (!details.open) return;
      body.append(content());
      details.removeEventListener("toggle", populate);
    }
    details.addEventListener("toggle", populate);
  } else {
    body.append(content);
  }
  details.append(el("summary", "", label), body);
  return details;
}
function rawRecord(label, value) {
  return disclosure(label, () => {
    const text = stringify(value);
    const maxPreview = 131072;
    if (text.length <= maxPreview) return el("pre", "", text);
    const wrapper = el("div");
    wrapper.append(el("p", "helper", "First 131,072 characters."),
      el("pre", "", text.slice(0, maxPreview)));
    return wrapper;
  });
}
// A table that adds rows a page at a time. makeRow returns the cells of a row
// and may mark the row itself through its third argument.
function table(headers, rows, makeRow = (row) => row, firstPage = 60) {
  const wrap = el("div", "table-wrap");
  const grid = el("table");
  const header = grid.createTHead().insertRow();
  for (const name of headers) {
    const th = el("th", "", name);
    th.scope = "col";
    header.append(th);
  }
  const body = grid.createTBody();
  const more = el("button", "link more-rows");
  more.type = "button";
  let displayed = 0;
  function appendPage() {
    const end = Math.min(displayed + (displayed ? 60 : firstPage), rows.length);
    for (; displayed < end; displayed++) {
      const tr = body.insertRow();
      for (const value of makeRow(rows[displayed], displayed, tr)) {
        const td = tr.insertCell();
        if (value instanceof Node) td.append(value);
        else td.textContent = value;
      }
    }
    more.hidden = displayed === rows.length;
    more.textContent = "Show more (" + displayed + " of " + rows.length + ")";
  }
  more.addEventListener("click", appendPage);
  appendPage();
  wrap.append(grid, more);
  return wrap;
}
// Columns for a list of records: the given keys first, then every other key
// in order of appearance.
function columnsOf(records, first, omit) {
  const keys = [];
  const present = new Set();
  for (const record of records) for (const key of Object.keys(record)) present.add(key);
  for (const key of first) if (present.has(key)) keys.push(key);
  for (const key of present) if (!keys.includes(key) && !omit.includes(key)) keys.push(key);
  return keys;
}

/* ---------- Dual graph ---------- */

const STEP_X = 84;
const STEP_Y = 60;
const SLOTS = {
  N: { angle: -90, x: 0, y: -12, anchor: "middle" }, S: { angle: 90, x: 0, y: 25, anchor: "middle" },
  E: { angle: 0, x: 12, y: 5, anchor: "start" }, W: { angle: 180, x: -12, y: 5, anchor: "end" },
  NE: { angle: -45, x: 9, y: -9, anchor: "start" }, NW: { angle: -135, x: -9, y: -9, anchor: "end" },
  SE: { angle: 45, x: 9, y: 21, anchor: "start" }, SW: { angle: 135, x: -9, y: 21, anchor: "end" },
};
function angleBetween(a, b) {
  const d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}

// A tree is drawn the way resolution graphs are drawn by hand: its longest
// chain runs left to right and the other chains hang from it. If that would
// make two vertices collide, the tree is drawn in layers instead.
function treeLayout(group, adj, order) {
  function pathFrom(start, banned) {
    const parent = new Map([[start, null]]);
    const depth = new Map([[start, 0]]);
    const queue = [start];
    for (const v of queue) {
      for (const w of adj.get(v)) {
        if (parent.has(w) || w === banned) continue;
        parent.set(w, v);
        depth.set(w, depth.get(v) + 1);
        queue.push(w);
      }
    }
    let end = start;
    for (const v of queue) {
      if (depth.get(v) > depth.get(end) || (depth.get(v) === depth.get(end) && order.get(v) < order.get(end))) end = v;
    }
    const path = [];
    for (let v = end; v !== null; v = parent.get(v)) path.push(v);
    return path.reverse();
  }
  const firstEnd = pathFrom(group[0]).pop();
  const spine = pathFrom(firstEnd);
  if (order.get(spine[spine.length - 1]) < order.get(spine[0])) spine.reverse();

  const points = new Map();
  function place(path, x, y, dx, dy) {
    path.forEach((v, k) => points.set(v, [x + k * dx * STEP_X, y + k * dy * STEP_Y]));
    for (const v of path) {
      const sides = [...adj.get(v)].filter((w) => !points.has(w)).sort((a, b) => order.get(a) - order.get(b));
      sides.forEach((w, n) => {
        const sign = n % 2 ? -1 : 1;
        const [vx, vy] = points.get(v);
        const sx = dx ? 0 : sign;
        const sy = dx ? sign : 0;
        place(pathFrom(w, v), vx + sx * STEP_X, vy + sy * STEP_Y, sx, sy);
      });
    }
  }
  place(spine, 0, 0, 1, 0);
  const placed = [...points.values()];
  const crowded = placed.some(([x, y], i) => placed.some(([u, v], j) => j > i && Math.hypot(x - u, y - v) < 40));
  if (!crowded) return points;

  points.clear();
  let leaf = 0;
  (function walk(v, from, depth) {
    const children = [...adj.get(v)].filter((w) => w !== from).sort((a, b) => order.get(a) - order.get(b));
    const heights = children.map((w) => walk(w, v, depth + 1));
    const y = children.length ? (heights[0] + heights[heights.length - 1]) / 2 : leaf++ * 54;
    points.set(v, [depth * STEP_X, y]);
    return y;
  })(spine[Math.floor(spine.length / 2)], null, 0);
  return points;
}
function ringLayout(group) {
  const radius = Math.max(58, group.length * 68 / (2 * Math.PI));
  return new Map(group.map((id, k) => {
    const angle = -Math.PI / 2 + 2 * Math.PI * k / group.length;
    return [id, [radius * Math.cos(angle), radius * Math.sin(angle)]];
  }));
}
// Returns the position of every vertex, or null when the complex is too
// large for a drawing to be of any use.
function graphLayout(ids, edges) {
  const order = new Map(ids.map((id, index) => [id, index]));
  const adj = new Map(ids.map((id) => [id, new Set()]));
  for (const [a, b] of edges) {
    if (a === b) continue;
    adj.get(a).add(b);
    adj.get(b).add(a);
  }
  const components = [];
  const seen = new Set();
  for (const id of ids) {
    if (seen.has(id)) continue;
    const group = [id];
    seen.add(id);
    for (const v of group) {
      for (const w of adj.get(v)) {
        if (seen.has(w)) continue;
        seen.add(w);
        group.push(w);
      }
    }
    components.push(group);
  }
  const isTree = (group) => group.reduce((sum, id) => sum + adj.get(id).size, 0) === 2 * (group.length - 1);
  const forest = components.every(isTree);
  if (ids.length > 64 || (!forest && (ids.length > 24 || edges.length > 72))) return null;

  // Lay out each connected piece, then set the pieces side by side in rows.
  const points = new Map();
  let row = [];
  let x = 0;
  let top = 0;
  function closeRow() {
    const height = Math.max(0, ...row.map((piece) => piece.height));
    for (const piece of row) {
      for (const [id, [px, py]] of piece.points) points.set(id, [piece.x + px, top + (height - piece.height) / 2 + py]);
    }
    top += height + STEP_Y + 16;
    row = [];
    x = 0;
  }
  for (const group of components) {
    const local = group.length === 1 ? new Map([[group[0], [0, 0]]]) : isTree(group) ? treeLayout(group, adj, order) : ringLayout(group);
    const xs = [...local.values()].map((p) => p[0]);
    const ys = [...local.values()].map((p) => p[1]);
    const width = Math.max(...xs) - Math.min(...xs);
    const shifted = new Map([...local].map(([id, [px, py]]) => [id, [px - Math.min(...xs), py - Math.min(...ys)]]));
    if (row.length && x + width > 840) closeRow();
    row.push({ points: shifted, x, height: Math.max(...ys) - Math.min(...ys) });
    x += width + STEP_X;
  }
  closeRow();
  return { points, adj };
}

function drawGraph(vertices, cells, branches) {
  const ids = vertices.map((v) => String(v.id));
  const record = new Map(vertices.map((v) => [String(v.id), v]));
  const valid = cells.map((cell) => (cell.vertices || []).map(String)).filter((list) => list.every((id) => record.has(id)));
  const edges = valid.filter((list) => list.length === 2);
  const faces = valid.filter((list) => list.length === 3);
  const layout = graphLayout(ids, edges);
  if (!layout) return null;
  const { points, adj } = layout;

  const everyVertexHas = (key) => ids.every((id) => isNumeric(record.get(id)[key]));
  const weightKey = everyVertexHas("self_intersection") ? "self_intersection" : null;
  const sizeKey = ["total_multiplicity", "multiplicity", "ambient_order"].find(everyVertexHas) || null;
  const arrows = new Map();
  for (const branch of branches) {
    const meets = Array.isArray(branch.ambient_components) ? branch.ambient_components.map(String) : [];
    if (meets.length !== 1 || !record.has(meets[0])) continue;
    if (!arrows.has(meets[0])) arrows.set(meets[0], []);
    arrows.get(meets[0]).push(String(branch.id));
  }

  const svg = svgElement("svg", { role: "group", "aria-label": "Dual graph with " + plural(ids.length, "vertex", "vertices") + " and " + plural(edges.length, "edge", "edges") });
  for (const face of faces) {
    svg.append(svgElement("polygon", { class: "face", points: face.map((id) => points.get(id).join(",")).join(" ") }));
  }
  const bundles = new Map();
  for (const [a, b] of edges) {
    const ends = [a, b].sort();
    const key = JSON.stringify(ends);
    if (!bundles.has(key)) bundles.set(key, { ends, count: 0 });
    bundles.get(key).count++;
  }
  for (const { ends: [a, b], count } of bundles.values()) {
    const [x1, y1] = points.get(a);
    const [x2, y2] = points.get(b);
    const length = Math.hypot(x2 - x1, y2 - y1);
    for (let i = 0; i < count; i++) {
      const bend = (i - (count - 1) / 2) * 34;
      let path;
      if (!length) {
        const radius = 28 + i * 11;
        path = "M " + x1 + " " + y1 + " c " + -radius + " " + -radius + " " + radius + " " + -radius + " 0 0";
      } else {
        const cx = (x1 + x2) / 2 - (y2 - y1) * bend / length;
        const cy = (y1 + y2) / 2 + (x2 - x1) * bend / length;
        path = "M " + x1 + " " + y1 + " Q " + cx + " " + cy + " " + x2 + " " + y2;
      }
      svg.append(svgElement("path", { class: "edge", d: path }));
    }
  }

  const groups = new Map();
  const extent = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity };
  for (const id of ids) {
    const [x, y] = points.get(id);
    extent.left = Math.min(extent.left, x);
    extent.right = Math.max(extent.right, x);
    extent.top = Math.min(extent.top, y);
    extent.bottom = Math.max(extent.bottom, y);
    const taken = [...adj.get(id)].map((other) => {
      const [ox, oy] = points.get(other);
      return Math.atan2(oy - y, ox - x) * 180 / Math.PI;
    });
    const v = record.get(id);
    const group = svgElement("g", { class: "vertex", transform: "translate(" + x + "," + y + ")", tabindex: "0", role: "button", "aria-pressed": "false" });
    const facts = [id];
    if (v.label && String(v.label) !== id) facts.push(String(v.label));
    if (weightKey) facts.push("self-intersection " + signed(v[weightKey]));
    if (sizeKey) facts.push(labelFor(sizeKey).toLowerCase() + " " + v[sizeKey]);
    group.append(svgElement("title", {}, facts.join(", ")));

    // Arrows for the branches of the strict transform fan out in the widest
    // sector left free by the edges: upwards on a chain, onwards at its end.
    const count = (arrows.get(id) || []).length;
    const fan = [];
    if (count) {
      let from = -270;
      let size = 360;
      if (taken.length) {
        const sorted = taken.map((a) => ((a % 360) + 360) % 360).sort((a, b) => a - b);
        let best = null;
        sorted.forEach((start, k) => {
          const span = sorted.length === 1 ? 360 : ((sorted[(k + 1) % sorted.length] - start) + 360) % 360;
          const lean = angleBetween(start + span / 2, -90) + angleBetween(start + span / 2, 0) / 1000;
          if (!best || span > best.span + 1 || (Math.abs(span - best.span) <= 1 && lean < best.lean)) best = { start, span, lean };
        });
        from = best.start;
        size = best.span;
      }
      const spread = Math.min(30 * (count - 1), 100, Math.max(0, size - 70));
      for (let k = 0; k < count; k++) fan.push(from + size / 2 + (count > 1 ? (k / (count - 1) - 0.5) * spread : 0));
      for (const degrees of fan) {
        const [ux, uy] = [Math.cos(degrees * Math.PI / 180), Math.sin(degrees * Math.PI / 180)];
        const tip = [34 * ux, 34 * uy];
        group.append(svgElement("line", { class: "arrow", x1: 7 * ux, y1: 7 * uy, x2: 30 * ux, y2: 30 * uy }),
          svgElement("polygon", { class: "arrow-head", points: [
            tip, [27 * ux - 3.4 * uy, 27 * uy + 3.4 * ux], [27 * ux + 3.4 * uy, 27 * uy - 3.4 * ux],
          ].map((p) => p.map((n) => n.toFixed(1)).join(",")).join(" ") }));
        extent.left = Math.min(extent.left, x + tip[0]);
        extent.right = Math.max(extent.right, x + tip[0]);
        extent.top = Math.min(extent.top, y + tip[1]);
        extent.bottom = Math.max(extent.bottom, y + tip[1]);
      }
    }

    // The name and the weight take the first place that is clear of the edges
    // and arrows at this vertex, and clear of each other.
    const occupied = [...taken, ...fan];
    const room = (slot) => Math.min(360, ...occupied.map((a) => angleBetween(a, SLOTS[slot].angle)));
    const place = (preference, apartFrom) => {
      const allowed = preference.filter((slot) => !apartFrom || angleBetween(SLOTS[slot].angle, SLOTS[apartFrom].angle) >= 90);
      return allowed.find((slot) => room(slot) >= 38) || allowed.reduce((best, slot) => (room(slot) > room(best) ? slot : best));
    };
    const along = (axis) => taken.every((a) => angleBetween(a, axis) < 20 || angleBetween(a, axis + 180) < 20);
    const upright = taken.length > 0 && along(90);
    const nameSlot = place(along(0) ? ["S", "SE", "SW", "N", "E", "W", "NE", "NW"]
      : upright ? ["E", "W", "SE", "NE", "S", "N", "SW", "NW"]
        : weightKey ? ["S", "SE", "SW", "E", "W", "N", "NE", "NW"] : ["S", "N", "E", "W", "SE", "SW", "NE", "NW"]);
    const weightSlot = weightKey ? place(upright ? ["W", "E", "NW", "SW", "N", "S", "NE", "SE"]
      : ["N", "NW", "NE", "W", "E", "S", "SW", "SE"], nameSlot) : null;

    group.append(svgElement("circle", { class: "hit", r: 15 }), svgElement("circle", { class: "ring", r: 9 }),
      svgElement("circle", { class: "dot", r: 4.3 }));
    const slot = SLOTS[nameSlot];
    const name = svgElement("text", { x: slot.x, y: slot.y, "text-anchor": slot.anchor });
    const shown = id.length > 12 ? id.slice(0, 11) + "…" : id;
    const parts = /^([A-Z])(\d[A-Za-z0-9.]*)$/.exec(shown);
    if (parts) {
      name.append(svgElement("tspan", { "font-style": "italic" }, parts[1]), svgElement("tspan", { class: "small", dy: 4 }, parts[2]));
      if (sizeKey) name.append(svgElement("tspan", { class: "paren", dy: -4 }, " (" + v[sizeKey] + ")"));
    } else {
      name.append(svgElement("tspan", {}, shown));
      if (sizeKey) name.append(svgElement("tspan", { class: "paren" }, " (" + v[sizeKey] + ")"));
    }
    group.append(name);
    if (weightSlot) {
      const w = SLOTS[weightSlot];
      group.append(svgElement("text", { class: "weight", x: w.x, y: w.y, "text-anchor": w.anchor }, signed(v[weightKey])));
    }
    const toggle = () => selectComponent(selected === id ? null : id);
    group.addEventListener("click", toggle);
    group.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      toggle();
    });
    groups.set(id, group);
    svg.append(group);
  }

  const box = [extent.left - 74, extent.top - 42, extent.right - extent.left + 148, extent.bottom - extent.top + 90];
  svg.setAttribute("viewBox", box.join(" "));
  svg.setAttribute("width", box[2]);
  svg.setAttribute("height", box[3]);

  const figure = el("figure", "graph");
  const scroll = el("div", "graph-scroll");
  scroll.append(svg);
  // A one-line key instead of a caption.
  const key = el("figcaption");
  const entry = (symbol, meaning) => {
    const item = el("span", "key");
    item.append(symbol, " " + meaning);
    key.append(item);
  };
  const n = () => el("i", "", "n");
  if (weightKey) {
    const symbol = el("span", "m");
    symbol.append(n());
    entry(symbol, "self-intersection");
  }
  if (sizeKey) {
    const symbol = el("span", "m");
    symbol.append("(", n(), ")");
    entry(symbol, labelFor(sizeKey).toLowerCase());
  }
  if (arrows.size) entry("↑", "strict transform");
  if (faces.length) entry("▲", "triple intersection");
  if (cells.some((cell) => (cell.vertices || []).length > 3)) entry("", "2-skeleton only");
  key.style.maxWidth = Math.max(240, box[2]) + "px";
  figure.append(scroll);
  if (key.childElementCount) figure.append(key);
  selectionViews.push((id) => {
    for (const [key, group] of groups) {
      group.classList.toggle("selected", key === id);
      group.setAttribute("aria-pressed", String(key === id));
    }
  });
  return figure;
}

/* ---------- Sections of the result ---------- */

// Which exceptional components each chart meets.
function componentsOf(chart) {
  const ids = new Set();
  for (const key of ["exceptional", "ambient_components"]) {
    if (!Array.isArray(chart[key])) continue;
    for (const entry of chart[key]) if (entry && entry.id !== undefined) ids.add(String(entry.id));
  }
  return ids;
}

function showComplex(data) {
  const dual = data.dual_complex;
  const panel = $("panel-complex");
  panel.replaceChildren();
  const vertices = dual?.vertices || [];
  const cells = dual?.cells || [];
  const complete = dual?.status === "complete";
  // Without a resolution there is no complex, and the header already says why.
  if (!complete && data.status !== "resolved") return;
  const isGraph = cells.every((cell) => (cell.vertices || []).length <= 2);
  panel.append(sectionHead(isGraph ? "Dual graph" : "Dual complex", dual?.convention));
  if (!complete) {
    panel.append(el("p", "notice", dual?.reason || "No dual complex was certified."));
    return;
  }
  if (!vertices.length) {
    panel.append(el("p", "empty", "The exceptional divisor is empty."));
    return;
  }

  const vertexIds = new Set(vertices.map((v) => String(v.id)));
  const branches = (Array.isArray(data.branches) ? data.branches : []).filter((b) => !vertexIds.has(String(b.id)));
  const chartCount = new Map();
  for (const chart of data.charts || []) {
    for (const id of componentsOf(chart)) chartCount.set(id, (chartCount.get(id) || 0) + 1);
  }

  const layout = el("div", "complex");
  const figure = drawGraph(vertices, cells, branches);
  if (figure) layout.append(figure);
  else panel.append(el("p", "helper", "Too large to draw."));

  const side = el("div", "components");
  const keys = columnsOf(vertices, ["total_multiplicity", "multiplicity", "ambient_order", "self_intersection", "genus", "dimension"], ["id", "label"]);
  const countCharts = chartCount.size > 0;
  const rows = [];
  side.append(table(["Component", ...keys.map(labelFor), ...(countCharts ? ["Charts"] : [])], vertices, (v, index, tr) => {
    const id = String(v.id);
    tr.dataset.id = id;
    tr.classList.toggle("selected", selected === id);
    rows.push(tr);
    const pick = el("button", "pick");
    pick.type = "button";
    pick.append(idNode(id));
    pick.setAttribute("aria-pressed", String(selected === id));
    pick.addEventListener("click", () => selectComponent(selected === id ? null : id));
    const first = document.createDocumentFragment();
    first.append(pick);
    if (v.label && String(v.label) !== id) first.append(" ", el("span", "helper", String(v.label)));
    return [first, ...keys.map((key) => valueNode(key, v[key])), ...(countCharts ? [String(chartCount.get(id) || "—")] : [])];
  }, vertices.length <= 24 ? 24 : 12));
  selectionViews.push((id) => {
    for (const tr of rows) {
      tr.classList.toggle("selected", tr.dataset.id === id);
      tr.querySelector(".pick").setAttribute("aria-pressed", String(tr.dataset.id === id));
    }
  });

  layout.append(side);
  panel.append(layout);

  if (branches.length) {
    const branchKeys = columnsOf(branches, ["ambient_components", "point", "chart"], ["id"]);
    panel.append(disclosure("Branches of the strict transform (" + branches.length + ")",
      () => table(["Branch", ...branchKeys.map((key) => key === "point" ? "At the point" : key === "chart" ? "Of chart" : labelFor(key))],
        branches, (b) => [idNode(b.id), ...branchKeys.map((key) => valueNode(key, b[key]))], 40)));
  }

  if (cells.length) {
    const cellKeys = columnsOf(cells, [], ["id", "vertices", "dimension"]);
    const details = disclosure(isGraph ? "Intersections (" + cells.length + ")" : "Cells (" + cells.length + ")",
      () => table(["Cell", "Components", "Dimension", ...cellKeys.map(labelFor)], cells, (cell, index) => {
        const ids = (cell.vertices || []).map(String);
        return [String(cell.id ?? index + 1), joined(ids.map(idNode), " ∩ "), String(ids.length - 1),
          ...cellKeys.map((key) => valueNode(key, cell[key]))];
      }));
    panel.append(details);
    if (!figure) details.open = true;
  }
  if (dual.root_definition) {
    panel.append(disclosure("Algebraic directions", dataList(Object.entries(dual.root_definition))));
  }
}

/* The zeta function of the local monodromy, as the engine computes it from
   the resolution by A'Campo's formula. */

const ZETA_CONVENTION = {
  "prod_q det(I - T*h^* | H^q(F_0; Q))^((-1)^(q+1))":
    "∏<sub><i>q</i></sub> det(<i>I</i> − <i>T</i> <i>h</i>* | <i>H</i><sup><i>q</i></sup>(<i>F</i><sub>0</sub>; ℚ))<sup>(−1)<sup><i>q</i>+1</sup></sup>",
  "prod_i (1-T^N_i)^(-chi_c(E_i^open intersect pi^(-1)(0)))":
    "∏<sub><i>i</i></sub> (1 − <i>T</i><sup><i>N</i><sub><i>i</i></sub></sup>)<sup>−χ<sub><i>c</i></sub>(<i>E</i><sub><i>i</i></sub>° ∩ π<sup>−1</sup>(0))</sup>",
};
function zetaName(variable) {
  const node = document.createDocumentFragment();
  const index = el("sub");
  index.append(el("i", "", "f"), ", 0");
  node.append("ζ", index, "(", el("i", "", variable), ")");
  return node;
}
// The quotient of products of (1 − T^n), built from the recorded factors.
function zetaQuotient(zeta, variable) {
  const factors = Array.isArray(zeta.factors) && zeta.factors.every((f) => Number.isInteger(f?.multiplicity) && Number.isInteger(f?.exponent))
    ? zeta.factors.filter((f) => f.exponent) : null;
  if (!factors) return typeset(zeta.expression ?? "");
  const product = (list) => {
    const node = el("span");
    if (!list.length) node.append("1");
    for (const { multiplicity, exponent } of list) {
      node.append("(1 − ", el("i", "", variable));
      if (multiplicity !== 1) node.append(el("sup", "", String(multiplicity)));
      node.append(")");
      if (Math.abs(exponent) !== 1) node.append(el("sup", "", String(Math.abs(exponent))));
    }
    return node;
  };
  const above = factors.filter((f) => f.exponent > 0);
  const below = factors.filter((f) => f.exponent < 0);
  if (!below.length) return product(above);
  const quotient = el("span", "quotient");
  quotient.append(product(above), product(below));
  return quotient;
}
function showZeta(data) {
  const panel = $("panel-zeta");
  panel.replaceChildren();
  const zeta = data.monodromy_zeta;
  if (!zeta || typeof zeta !== "object" || data.status !== "resolved") return;
  panel.append(sectionHead("Monodromy zeta function"));
  if (zeta.status !== "computed") {
    panel.append(el("p", "notice", zeta.reason || "Not computed for this resolution."));
    return;
  }
  const variable = typeof zeta.variable === "string" && /^[A-Za-z]$/.test(zeta.variable) ? zeta.variable : "T";
  const line = el("p", "zeta m");
  line.append(zetaName(variable), " = ", zetaQuotient(zeta, variable));
  panel.append(line);
  if (isNumeric(zeta.milnor_fiber_euler_characteristic)) {
    const euler = el("p", "zeta-euler m");
    euler.title = "Euler characteristic of the Milnor fiber";
    euler.append("χ(", el("i", "", "F"), el("sub", "", "0"), ") = " + signed(zeta.milnor_fiber_euler_characteristic));
    panel.append(euler);
  }

  const body = el("div", "stack");
  const convention = el("p", "zeta-convention m");
  convention.append(zetaName(variable));
  for (const text of [zeta.convention, zeta.formula]) {
    if (typeof text !== "string" || !text) continue;
    convention.append(" = ");
    if (ZETA_CONVENTION[text] && variable === "T") {
      const pretty = el("span");
      pretty.innerHTML = ZETA_CONVENTION[text];
      convention.append(pretty);
    } else convention.append(code(text));
  }
  if (convention.childNodes.length > 1) body.append(convention);
  const shown = ["status", "variable", "convention", "formula", "expression", "factors", "contributions", "milnor_fiber_euler_characteristic"];
  const rest = Object.entries(zeta).filter(([key]) => !shown.includes(key));
  if (rest.length) body.append(dataList(rest));
  const contributions = Array.isArray(zeta.contributions) ? zeta.contributions.filter((c) => c && typeof c === "object") : [];
  if (contributions.length) {
    const keys = columnsOf(contributions, ["multiplicity", "euler_characteristic"], ["component"]);
    body.append(table(["Component", ...keys.map(labelFor)],
      contributions, (c) => [idNode(c.component ?? "—"), ...keys.map((key) => valueNode(key, c[key]))], 12));
  }
  if (body.childElementCount) {
    panel.append(disclosure(contributions.length ? "Convention and contributions" : "Convention", body));
  }
}

/* The topological zeta function of Denef and Loeser, which the engine sums
   over the strata of the total transform. */

function minus(text) {
  return String(text).replace(/-/g, "−");
}
function showTopological(data) {
  const panel = $("panel-zeta");
  const zeta = data.topological_zeta;
  if (!zeta || typeof zeta !== "object" || data.status !== "resolved") return;
  panel.append(sectionHead("Topological zeta function"));
  if (zeta.status !== "computed") {
    panel.append(el("p", "notice", zeta.reason || "Not computed for this resolution."));
    return;
  }
  const variable = typeof zeta.variable === "string" && /^[A-Za-z]$/.test(zeta.variable) ? zeta.variable : "s";
  const name = () => {
    const node = document.createDocumentFragment();
    node.append(el("i", "", "Z"), el("sub", "", "top, 0"), "(", el("i", "", variable), ")");
    return node;
  };
  const line = el("p", "zeta m");
  line.append(name(), " = ");
  if (typeof zeta.numerator === "string" && typeof zeta.denominator === "string" && zeta.denominator !== "1") {
    const quotient = el("span", "quotient");
    const above = el("span");
    const below = el("span");
    above.append(typeset(zeta.numerator));
    below.append(typeset(zeta.denominator));
    quotient.append(above, below);
    line.append(quotient);
  } else line.append(typeset(zeta.expression ?? ""));
  panel.append(line);

  const poles = Array.isArray(zeta.poles) ? zeta.poles.filter((pole) => pole && typeof pole === "object") : [];
  if (poles.length) {
    const row = el("p", "zeta-euler m");
    row.title = "Poles of the topological zeta function";
    row.append("poles  ");
    poles.forEach((pole, index) => {
      if (index) row.append(",  ");
      row.append(minus(pole.value));
      if (Number.isInteger(pole.order) && pole.order > 1) row.append(el("sup", "", "(" + pole.order + ")"));
    });
    panel.append(row);
  }

  const body = el("div", "stack");
  if (typeof zeta.formula === "string" && zeta.formula) {
    const convention = el("p", "zeta-convention m");
    convention.append(name(), " = ");
    const pretty = el("span");
    pretty.innerHTML = "∑<sub><i>I</i></sub> χ(<i>E</i><sub><i>I</i></sub>° ∩ π<sup>−1</sup>(0)) ∏<sub><i>i</i> ∈ <i>I</i></sub> 1 / (<i>N</i><sub><i>i</i></sub> <i>s</i> + ν<sub><i>i</i></sub>)";
    convention.append(pretty);
    body.append(convention);
  }
  if (poles.length) {
    const known = poles.some((pole) => typeof pole.in_monodromy_zeta_at_origin === "boolean");
    const headers = ["Pole", "Order", "Components"];
    if (known) {
      const head = el("span", "m");
      head.title = "Whether exp(2πi·pole) is a zero or pole of the monodromy zeta function at the origin";
      head.append("e", el("sup", "", "2πi·pole"), " in ζ");
      headers.push(head);
    }
    const grid = table(headers.map((h) => (h instanceof Node ? "" : h)), poles, (pole) => {
      const cells = [minus(pole.value), String(pole.order ?? ""),
        (Array.isArray(pole.components) ? pole.components : []).join(", ") || "—"];
      if (known) cells.push(pole.in_monodromy_zeta_at_origin === true ? "yes" : pole.in_monodromy_zeta_at_origin === false ? "no" : "—");
      return cells;
    }, 12);
    if (known) grid.querySelector("thead th:last-child").replaceChildren(headers[3]);
    body.append(grid);
  }
  const numbers = Array.isArray(zeta.numerical_data) ? zeta.numerical_data.filter((row) => row && typeof row === "object") : [];
  if (numbers.length) {
    body.append(table(["Component", "N", "ν"], numbers,
      (row) => [idNode(row.component ?? "—"), String(row.multiplicity ?? ""), String(row.nu ?? "")], 12));
  }
  const strata = Array.isArray(zeta.strata) ? zeta.strata.filter((row) => row && typeof row === "object") : [];
  if (strata.length) {
    body.append(table(["Stratum", "χ"], strata,
      (row) => [(Array.isArray(row.components) ? row.components : []).join(" ∩ ") + "°", signed(row.euler_characteristic)], 12));
  }
  if (body.childElementCount) panel.append(disclosure("Poles, numerical data and strata", body));
}

function showSteps(data) {
  const panel = $("panel-steps");
  panel.replaceChildren();
  const steps = (Array.isArray(data.steps) ? data.steps : []).filter((step) => step && typeof step === "object");
  if (!steps.length) return;
  const keys = columnsOf(steps, ["type", "center", "parent_chart", "center_ideal", "center_charts", "generators", "weight",
    "exceptional_id", "strict_multiplicity", "total_multiplicity"], ["id"]);
  foldedSection(panel, "Construction", steps.length).body.append(table(["Step", ...keys.map(labelFor)], steps,
    (step, index) => [String(step.id ?? index + 1), ...keys.map((key) => valueNode(key, step[key]))]));
}

// "u = 0" for a divisor, "u = v = 0" for a locus cut out by several equations.
function locus(generators) {
  const node = el("span", "m");
  generators.forEach((g, index) => {
    if (index) node.append(" = ");
    node.append(typeset(g));
  });
  node.append(" = 0");
  return node;
}
function equationList(generators) {
  if (!generators.length) return el("span", "helper", "whole chart");
  if (generators.includes("1") || generators.includes("-1")) return el("span", "helper", "empty");
  const block = el("div", "eqs");
  for (const f of generators) block.append(typeset(f, " = 0"));
  return foldable(block, generators.join("").length);
}
function divisorList(divisors) {
  const block = el("div", "eqs");
  for (const d of divisors) {
    const line = el("span");
    line.append(idNode(d.id), el("span", "m", " : "), locus(d.ideal ?? [d.equation]));
    for (const [key, value] of Object.entries(d)) {
      if (["id", "equation", "ideal"].includes(key)) continue;
      const attribute = el("span", "attr", labelFor(key).toLowerCase() + " ");
      attribute.append(valueNode(key, value));
      line.append(attribute);
    }
    block.append(line);
  }
  return block;
}
function fieldName(name) {
  const match = /^Q(?:\(([A-Za-z][A-Za-z0-9_]*)\))?$/.exec(String(name));
  if (!match) return document.createTextNode(String(name));
  const node = el("span", "m", "ℚ");
  if (match[1]) node.append("(", nameNode(match[1]), ")");
  return node;
}
// FLINT writes an algebraic number as an indexed root of an integer polynomial,
// with the coefficients listed from the constant term up.
function exactNumber(exact) {
  const match = /^PolynomialRootIndexed\(List\(([-\d,\s]+)\),\s*(\d+)\)$/.exec(String(exact));
  if (!match) return code(exact);
  const terms = [];
  match[1].split(",").map((c) => c.trim()).forEach((c, degree) => {
    if (!c || /^-?0+$/.test(c)) return;
    const size = c.replace("-", "");
    const power = degree === 0 ? "" : degree === 1 ? "T" : "T^" + degree;
    terms.unshift((c.startsWith("-") ? "-" : "+") + (size === "1" && power ? "" : size + (power ? "*" : "")) + power);
  });
  const node = el("span");
  node.title = String(exact);
  node.append("root " + match[2] + " of ", typeset(terms.join("").replace(/^\+/, "")));
  return node;
}
function approximation(text) {
  return String(text).replace(/\s*\(deg \d+\)\s*$/, "").replace(/\*?\bI\b/g, " i").replace(/-/g, "−");
}
function fieldInfo(data) {
  const section = el("div");
  const field = data.coefficient_field;
  if (typeof field === "string") {
    section.append(el("h4", "sub", "Coefficient field"), dataList([["Name", fieldName(field)]]));
  } else if (field && typeof field === "object" && Object.keys(field).some((key) => key !== "name")) {
    section.append(el("h4", "sub", "Coefficient field"), dataList(Object.entries(field).map(
      ([key, value]) => key === "name" ? [key, fieldName(value)] : [key, value])));
  }
  if (Object.keys(data.algebraic_numbers || {}).length) {
    const rows = Object.entries(data.algebraic_numbers);
    section.append(el("h4", "sub", "Algebraic numbers"), table(["Symbol", "Value", "Definition"], rows,
      ([id, number]) => [typeset(id), approximation(number?.approximation ?? "—"), exactNumber(number?.exact ?? number)]));
  }
  if (Object.keys(data.root_families || {}).length) {
    const rows = Object.entries(data.root_families);
    section.append(el("h4", "sub", "Roots in the chart equations"), table(["Family", "Definition"], rows,
      ([id, family]) => [typeset(id), family?.definition ?? stringify(family)]));
  }
  return section.childElementCount ? section : null;
}

const CHART_KEYS = new Set(["id", "coordinates", "map", "map_to_parent", "parent", "parent_chart", "step", "final",
  "ambient_ideal", "ambient_total_transform", "pullback_factor", "strict_transform", "exceptional",
  "ambient_components", "base_domain", "exclude_points", "projective_coordinates", "pivot_generator"]);

function pointList(list) {
  if (!Array.isArray(list)) return valueNode("", list);
  return joined(list.map((p) => Array.isArray(p) ? tuple(p) : valueNode("point", p)), ", ");
}
function mapNode(map) {
  const node = el("div", "inline-eqs");
  for (const [name, image] of Object.entries(map)) node.append(typeset(name + " = " + image));
  return foldable(node, Object.values(map).join("").length);
}
// A chart is one line, its name, until it is opened.
function chartNode(chart, index, data) {
  const details = el("details", "chart");
  const body = el("div", "chart-body");
  details.append(el("summary", "", String(chart.id ?? "Chart " + (index + 1))), body);
  details.addEventListener("toggle", () => {
    if (details.open && !body.childElementCount) body.append(...chartContent(chart, data));
  });
  return details;
}
function chartContent(chart, data) {
  const rows = el("dl", "rows");
  const row = (label, content) => {
    const dt = el("dt", "", label);
    const dd = el("dd");
    dd.append(content);
    rows.append(dt, dd);
  };
  if (chart.coordinates?.length) row("Coordinates", tuple(chart.coordinates));
  if (chart.step !== undefined) row("Step", String(chart.step));
  const parent = chart.parent_chart ?? chart.parent;
  if (parent !== undefined) row("Parent chart", String(parent));
  if (chart.map && Object.keys(chart.map).length) row("Map", mapNode(chart.map));
  if (Array.isArray(chart.ambient_ideal) && chart.ambient_ideal.length) {
    row("Ambient chart", equationList(chart.ambient_ideal));
  }
  if (chart.ambient_total_transform !== undefined) {
    row("Total transform", foldable(typeset(chart.ambient_total_transform), String(chart.ambient_total_transform).length));
  }
  if (chart.pullback_factor !== undefined) row("Pullback factor", typeset(chart.pullback_factor));
  if (chart.strict_transform !== undefined) {
    const generators = Array.isArray(chart.strict_transform) ? chart.strict_transform : [chart.strict_transform];
    row("Strict transform", equationList(generators));
  }
  if (Array.isArray(chart.exceptional)) {
    row(data.mode === "intrinsic" ? "Exceptional locus" : "Exceptional divisor",
      chart.exceptional.length ? divisorList(chart.exceptional) : el("span", "helper", "none"));
  }
  const domain = [];
  if (chart.base_domain?.parent && String(chart.base_domain.parent) !== String(parent)) {
    domain.push(["Parent chart", String(chart.base_domain.parent)]);
  }
  if (chart.base_domain?.exclude_sibling_points?.length) domain.push(["Omit in parent", pointList(chart.base_domain.exclude_sibling_points)]);
  if (chart.base_domain?.invert_parent_coordinates?.length) domain.push(["Invert in parent", valueNode("coordinates", chart.base_domain.invert_parent_coordinates)]);
  if (chart.exclude_points?.length) domain.push(["Omit in this chart", pointList(chart.exclude_points)]);
  if (chart.projective_coordinates?.length) domain.push(["Homogeneous coordinates", tuple(chart.projective_coordinates, "[", "]", " : ")]);
  if (chart.pivot_generator) domain.push(["Center generator for this chart", typeset(chart.pivot_generator)]);
  if (domain.length) {
    const list = el("div", "domain");
    for (const [label, value] of domain) {
      const line = el("div");
      line.append(el("span", "k", label + ": "), value);
      list.append(line);
    }
    row("Domain", list);
  }

  const more = el("div", "stack");
  const extra = [];
  if (chart.map_to_parent && Object.keys(chart.map_to_parent).length) extra.push(["map_to_parent", mapNode(chart.map_to_parent)]);
  if (typeof chart.base_domain === "string") extra.push(["base_domain", chart.base_domain]);
  for (const [key, value] of Object.entries(chart)) {
    if (!CHART_KEYS.has(key)) extra.push([key, value]);
  }
  if (extra.length) more.append(dataList(extra));
  if (data.mode === "intrinsic" && chart.ambient_components?.length) {
    more.append(el("h4", "sub", "Ambient exceptional components"), divisorList(chart.ambient_components));
  }
  more.append(rawRecord("Chart record", chart));
  return [rows, disclosure("More", more)];
}

function showCharts(data) {
  const panel = $("panel-charts");
  panel.replaceChildren();
  const charts = Array.isArray(data.charts) ? data.charts : [];
  if (!charts.length) return;
  const { body, number } = foldedSection(panel, "Charts", charts.length);

  const meets = charts.map(componentsOf);
  const options = (data.dual_complex?.vertices || []).map((v) => String(v.id)).filter((id) => meets.some((set) => set.has(id)));
  const filter = el("select");
  filter.setAttribute("aria-label", "Show the charts that meet a component");
  filter.append(new Option("All charts", ""));
  for (const id of options) filter.append(new Option("Charts that meet " + id, id));
  const search = el("input");
  search.type = "search";
  search.placeholder = "Find a chart";
  search.setAttribute("aria-label", "Find a chart by name");
  // A short list needs no controls; picking a component in the graph still filters it.
  const controls = charts.length > 12;
  if (controls) {
    const tools = el("div", "tools");
    tools.append(search);
    if (options.length > 1) tools.append(filter);
    body.append(tools);
  }

  const symbols = fieldInfo(data);
  if (symbols) body.append(disclosure("Field and algebraic numbers", symbols));

  const note = el("p", "filter-note");
  const list = el("div", "chart-list");
  const buttons = el("div", "list-more");
  const next = el("button", "button");
  const all = el("button", "button");
  next.type = all.type = "button";
  buttons.append(next, all);
  body.append(note, list, buttons);

  const page = 40;
  let matching = [];
  let shown = 0;
  function extend(count) {
    const end = Math.min(shown + count, matching.length);
    for (; shown < end; shown++) list.append(chartNode(charts[matching[shown]], matching[shown], data));
    const rest = matching.length - shown;
    buttons.hidden = !rest;
    next.textContent = "Show " + Math.min(page, rest) + " more";
    all.textContent = "Show all " + matching.length;
    all.hidden = rest <= page;
  }
  function refresh() {
    const component = filter.value;
    const words = search.value.trim();
    matching = charts.map((chart, index) => index).filter((index) =>
      (!component || meets[index].has(component)) && (!words || String(charts[index].id ?? "").toLowerCase().includes(words.toLowerCase())));
    list.replaceChildren();
    shown = 0;
    note.replaceChildren();
    // With the controls on the page, they already say what is filtered.
    note.hidden = controls || (!component && !words);
    number.textContent = component || words ? matching.length + " of " + charts.length : String(charts.length);
    if (!note.hidden) {
      note.append("Meeting ", idNode(component), " · ");
      const clear = el("button", "link", "Show all");
      clear.type = "button";
      clear.addEventListener("click", () => {
        search.value = "";
        if (selected) selectComponent(null);
        else {
          filter.value = "";
          refresh();
        }
      });
      note.append(clear);
    }
    extend(page);
  }
  next.addEventListener("click", () => extend(page));
  all.addEventListener("click", () => extend(matching.length));
  search.addEventListener("input", refresh);
  filter.addEventListener("change", () => selectComponent(filter.value || null));
  selectionViews.push((id) => {
    filter.value = id && options.includes(id) ? id : "";
    refresh();
  });
  refresh();
}

function backendName(id) {
  return { opencl: "OpenCL GPU", cpu: "CPU", cpu_after_opencl: "CPU (OpenCL fallback)" }[id] || id;
}

function showComputation(data, request) {
  const panel = $("panel-record");
  panel.replaceChildren();
  const proof = data.certificate;
  const record = el("div", "record");
  foldedSection(panel, proof ? "Certificate" : "Computation").body.append(record);
  if (proof && typeof proof === "object" && proof.description) record.append(el("p", "prose", proof.description));
  else if (proof && typeof proof !== "object") record.append(el("p", "prose", String(proof)));

  record.append(dataList(Object.entries({
    Method: proof && typeof proof === "object" ? proof.method : undefined,
    Engine: data.engine,
    Device: data.execution?.device,
  }).filter(([, v]) => v)));

  if (data.diagnostic && Object.keys(data.diagnostic).length) {
    record.append(el("h4", "sub", "Quick search diagnostic"), dataList(Object.entries(data.diagnostic)));
  }
  if (proof?.rank_backend) {
    const rankInfo = [
      ["Rank backend", backendName(proof.rank_backend)],
      ["Device", proof.device?.name],
      ["Rank time", Number.isFinite(proof.rank_elapsed_ms) ? duration(proof.rank_elapsed_ms) : undefined],
      ["Rank", proof.rank],
      ["Matrix rows", proof.rows],
      ["Prime", proof.prime],
    ].filter(([, v]) => v !== undefined);
    record.append(el("h4", "sub", "Modular Jacobian rank"), dataList(rankInfo));
  }
  const batch = proof?.chart_batch;
  if (batch?.backend) {
    const chartInfo = [
      ["Transform backend", backendName(batch.backend)], ["Device", batch.device?.name],
      ["Chart–monomial pairs", batch.chart_term_pairs],
      ["Batch time", Number.isFinite(batch.elapsed_ms) ? duration(batch.elapsed_ms) : undefined],
      ["Native computation", Number.isFinite(batch.native_elapsed_ms) ? duration(batch.native_elapsed_ms) : undefined],
    ].filter(([, value]) => value !== undefined);
    record.append(el("h4", "sub", "Toric chart transforms"), dataList(chartInfo));
  }
  if (request.compute !== "cpu" && !proof?.rank_backend && !batch?.backend) {
    record.append(el("p", "helper", data.execution?.note || "The selected algorithm runs on the CPU."));
  }
  const notes = Array.isArray(data.notes) ? data.notes : [];
  // Without charts, the field and its algebraic numbers are described here.
  const fields = Array.isArray(data.charts) && data.charts.length ? null : fieldInfo(data);
  if (fields) record.append(fields);
  if (notes.length) {
    const list = el("ul", "notes");
    for (const note of notes) list.append(el("li", "", note));
    record.append(disclosure("Notes (" + notes.length + ")", list));
  }
  if (proof && typeof proof === "object") {
    const rest = Object.entries(proof).filter(([key]) => key !== "description" && key !== "method");
    if (rest.length) record.append(disclosure("Certificate details", () => dataList(rest)));
  }
  if (data.branches?.length) record.append(rawRecord("Branch records (" + data.branches.length + ")", data.branches));
  if (data.steps?.length) record.append(rawRecord("Step records (" + data.steps.length + ")", data.steps));
  record.append(rawRecord("Full record", data));
}

function showResult(data, request) {
  lastResult = data;
  lastRequest = request;
  previewRequested = false;
  selected = null;
  selectionViews.length = 0;
  $("placeholder").hidden = true;
  $("result").hidden = false;

  const equation = $("result-equation");
  equation.replaceChildren(typeset(request.polynomial, " = 0"));
  equation.classList.toggle("long", request.polynomial.length > 70);
  equation.classList.toggle("very-long", request.polynomial.length > 400);
  document.title = request.polynomial + " · Resolution Atlas";

  const status = { resolved: "Resolved", partial: "Partial result", unsupported: "Unsupported", inconclusive: "Search incomplete" }[data.status] || data.status || "Result";
  $("status").textContent = status;
  $("status").classList.toggle("incomplete", data.status !== "resolved");
  $("result-mode").textContent = request.mode === "embedded" ? "embedded" : "intrinsic";
  const facts = [];
  if (Number.isFinite(data.elapsed_ms)) facts.push(duration(data.elapsed_ms));
  if (data.certificate?.rank_backend) facts.push("rank on " + backendName(data.certificate.rank_backend));
  if (data.certificate?.chart_batch?.backend) facts.push("chart transforms on " + backendName(data.certificate.chart_batch.backend));
  $("metrics").textContent = facts.map((fact) => " · " + fact).join("");
  // A resolved germ needs no sentence; anything else says what went wrong.
  const summary = data.status === "resolved" ? "" : data.summary || data.reason || "";
  $("result-summary").hidden = !summary;
  $("result-summary").textContent = summary;
  const diagnostic = data.diagnostic?.description || data.diagnostic?.fast_path_limit;
  $("result-diagnostic").hidden = !diagnostic;
  $("result-diagnostic").textContent = diagnostic || "";

  showZeta(data);
  showTopological(data);
  showComplex(data);
  showSteps(data);
  showCharts(data);
  showComputation(data, request);
  $("export").disabled = false;
  $("announcements").textContent = status + ". " + (data.summary || "");
  updateInput();
  updateSearchOffer();
}

/* ---------- The input ---------- */

// These three reference curves have exact SVG paths in mathematical coordinates.
// For the cusp the cubic Bézier is (t², t³) for −3/2 ≤ t ≤ 3/2.
const planeExamples = {
  cusp: {
    polynomial: "y^2-x^3", equation: "y^2=x^3", title: "Plane cusp",
    path: "M2.25 -3.375 C-.75 3.375 -.75 -3.375 2.25 3.375",
    caption: () => ["The real cusp is parametrized by ", typeset("(t^2,t^3)"), "."],
  },
  node: {
    polynomial: "y^2-x^2", equation: "y^2=x^2", title: "Ordinary node",
    path: "M-3 -3L3 3 M-3 3L3 -3",
    caption: () => ["Two real branches, ", typeset("y=x"), " and ", typeset("y=-x"), ", cross transversely at the origin."],
  },
  tacnode: {
    polynomial: "y^2-x^4", equation: "y^2=x^4", title: "Tacnode",
    path: "M-2 4Q0 -4 2 4 M-2 -4Q0 4 2 -4",
    caption: () => ["The branches ", typeset("y=x^2"), " and ", typeset("y=-x^2"), " share a tangent at the origin."],
  },
};
const exampleButtons = form.querySelectorAll("[data-example]");
let drawnExample = null;
let previewRequested = false;

function showInputGerm(request) {
  const equation = $("preview-equation");
  const coordinates = $("preview-coordinates");
  const empty = !request.polynomial;
  $("preview-title").textContent = empty ? "Enter a germ" : "Input germ";
  equation.hidden = coordinates.hidden = empty;
  if (empty) {
    $("preview-note").textContent = "Enter a polynomial or choose an example.";
    return;
  }
  // Preview only a bounded prefix of the source. Exhaustive requests can be
  // large; truncation applies only here, never to the submitted polynomial.
  const limit = 400;
  const truncated = request.polynomial.length > limit;
  if (truncated) equation.replaceChildren(el("code", "", request.polynomial.slice(0, limit) + "…"));
  else equation.replaceChildren(typeset(request.polynomial, " = 0"));
  const names = request.variables.slice(0, 12).map((name) => name.length > 32 ? name.slice(0, 32) + "…" : name);
  if (request.variables.length > names.length) names.push("…");
  coordinates.textContent = names.length ? "Coordinates (" + names.join(", ") + ")" : "Enter the coordinate variables.";
  $("preview-note").textContent = (truncated ? "Showing the first " + limit + " characters. " : "")
    + "Resolve to inspect the dual complex and charts.";
}

function updatePlate() {
  const request = getRequest();
  const polynomial = request.polynomial.length <= 100 ? request.polynomial.replace(/\s+/g, "") : null;
  const plane = request.variables.length === 2 && request.variables.includes("x") && request.variables.includes("y");
  const key = plane && Object.keys(planeExamples).find((name) => planeExamples[name].polynomial === polynomial);
  for (const button of exampleButtons) button.setAttribute("aria-pressed", String(button.dataset.example === key));
  const available = !busy && !activeSearch();
  const show = Boolean(key) && available && (previewRequested || !lastRequest || lastRequest.polynomial !== request.polynomial);
  $("germ-plate").hidden = !show;
  $("plate-empty").hidden = !available || show || Boolean(lastResult) || !$("form-error").hidden;
  if (!$("plate-empty").hidden) showInputGerm(request);
  if (!show || key === drawnExample) return;
  const curve = planeExamples[key];
  $("plate-title").textContent = curve.title;
  $("plate-equation").replaceChildren(typeset(curve.equation));
  $("curve-title").textContent = curve.title + " in real coordinates";
  $("plate-curve").setAttribute("d", curve.path);
  $("plate-caption").replaceChildren(...curve.caption());
  drawnExample = key;
}

for (const button of exampleButtons) button.addEventListener("click", () => {
  previewRequested = true;
  $("polynomial").value = planeExamples[button.dataset.example].polynomial;
  $("variables").value = "x, y";
  variablesFollow = true;
  fitInput();
  updateInput();
});

// Section navigation should preserve the current report's shareable URL.
document.addEventListener("click", (event) => {
  const link = event.target.closest(".skip-link, .output-jump");
  if (!link || !link.hash || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  const target = $(link.hash.slice(1));
  if (!target) return;
  event.preventDefault();
  target.focus({ preventScroll: true });
  target.scrollIntoView({ block: "start" });
});

function getRequest() {
  return {
    polynomial: $("polynomial").value.trim(),
    variables: $("variables").value.split(/[\s,]+/).filter(Boolean),
    mode: form.elements.mode.value,
    compute: $("compute").value,
    monodromy: $("zeta").checked,
    ...($("exhaustive").checked ? { search: "exhaustive" } : {}),
  };
}
function sameGerm(a, b) {
  return Boolean(a && b) && a.polynomial === b.polynomial && (a.mode || "embedded") === (b.mode || "embedded")
    && (a.variables || []).join(",") === (b.variables || []).join(",");
}
// The variables are read off the polynomial until the list is edited by hand.
// A longer name such as xy is never declared this way: it is far more likely
// to be a product written without its *.
function detectVariables(source) {
  const names = [...new Set(source.match(/[A-Za-z][A-Za-z0-9_]*/g) || [])].filter((name) => /^[A-Za-z](?:[0-9_][A-Za-z0-9_]*)?$/.test(name));
  const fromX = names.some((name) => "xyz".includes(name[0]));
  const letters = fromX ? "xyzabcdefghijklmnopqrstuvw" : "abcdefghijklmnopqrstuvwxyz";
  const rank = (name) => [name[0] === name[0].toLowerCase() ? 0 : 1, letters.indexOf(name[0].toLowerCase()), Number(/\d+/.exec(name)?.[0] ?? -1)];
  return names.sort((a, b) => {
    const [ra, rb] = [rank(a), rank(b)];
    return ra[0] - rb[0] || ra[1] - rb[1] || ra[2] - rb[2] || a.localeCompare(b);
  });
}
let variablesFollow = true;
function followsPolynomial() {
  const declared = getRequest().variables;
  return !declared.length || declared.join(",") === detectVariables($("polynomial").value).join(",");
}
function fitInput() {
  const input = $("polynomial");
  input.style.height = "auto";
  input.style.height = input.scrollHeight + "px";
}
// The note about outdated results. It stays away while a search for exactly
// this input is running, because then the page is already working on it.
function updateStale() {
  const request = getRequest();
  $("stale").hidden = !lastResult || JSON.stringify(request) === JSON.stringify(lastRequest)
    || (activeSearch() && sameGerm(request, searchJob.request));
}
function updateInput() {
  const request = getRequest();
  const strangers = [...new Set(request.polynomial.match(/[A-Za-z][A-Za-z0-9_]*/g) || [])].filter((name) => !request.variables.includes(name));
  const hint = $("name-hint");
  hint.hidden = !strangers.length;
  if (strangers.length) {
    const name = strangers[0];
    hint.textContent = "“" + name + "” is not one of the variables."
      + (/^[A-Za-z]{2,4}$/.test(name) ? " If it is a product, write " + [...name].join("*") + "." : "");
  }
  $("form-error").hidden = true;
  updateStale();
  updateSearchOffer();
  updatePlate();
}

$("polynomial").addEventListener("input", () => {
  if (variablesFollow) {
    const detected = detectVariables($("polynomial").value);
    if (detected.length) $("variables").value = detected.join(", ");
  }
  fitInput();
  updateInput();
});
$("polynomial").addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
  event.preventDefault();
  form.requestSubmit();
});
$("variables").addEventListener("input", () => {
  variablesFollow = followsPolynomial();
  updateInput();
});
form.querySelectorAll('[name="mode"]').forEach((input) => input.addEventListener("change", updateInput));
$("compute").addEventListener("change", updateInput);
$("exhaustive").addEventListener("change", updateInput);
function setMode(mode) {
  const input = form.querySelector('input[name="mode"][value="' + mode + '"]');
  if (input) input.checked = true;
}
// Whether to compute the zeta function is remembered between visits.
try {
  if (localStorage.getItem("atlas-zeta") === "off") $("zeta").checked = false;
} catch { /* Private browsing. */ }
$("zeta").addEventListener("change", () => {
  try { localStorage.setItem("atlas-zeta", $("zeta").checked ? "on" : "off"); } catch { /* Private browsing. */ }
  updateInput();
});

// The address keeps the last resolved input, so a reload or a copied link
// comes back to the same germ.
let ownHash = "";
function writeAddress(request) {
  const params = new URLSearchParams({ f: request.polynomial, vars: request.variables.join(","), mode: request.mode });
  if (request.compute !== "cpu") params.set("compute", request.compute);
  if (!request.monodromy) params.set("zeta", "0");
  if (request.search === "exhaustive") params.set("search", request.search);
  ownHash = "#" + params;
  try { history.replaceState(null, "", ownHash); } catch { /* The page works without it. */ }
}
function readAddress() {
  const params = new URLSearchParams(location.hash.slice(1));
  const polynomial = params.get("f");
  if (!polynomial) return false;
  $("polynomial").value = polynomial;
  const variables = (params.get("vars") || "").split(",").filter(Boolean);
  $("variables").value = (variables.length ? variables : detectVariables(polynomial)).join(", ");
  if (["embedded", "intrinsic"].includes(params.get("mode"))) setMode(params.get("mode"));
  if (["cpu", "auto", "gpu"].includes(params.get("compute"))) $("compute").value = params.get("compute");
  $("zeta").checked = params.get("zeta") !== "0";
  $("exhaustive").checked = params.get("search") === "exhaustive";
  variablesFollow = followsPolynomial();
  return true;
}
window.addEventListener("hashchange", () => {
  if (location.hash === ownHash || !readAddress()) return;
  fitInput();
  updateInput();
  resolveGerm();
});

/* ---------- Resolving ---------- */

// The line shown below the form while there is no result yet.
function waiting(text) {
  $("placeholder-text").textContent = text;
  $("placeholder").hidden = Boolean(lastResult) || !text;
  updatePlate();
}

async function resolveGerm(followResult = false) {
  if (busy) return;
  const request = getRequest();
  $("form-error").hidden = true;
  if (!request.polynomial || !request.variables.length) {
    $("form-error").textContent = "Enter a polynomial and its variables.";
    $("form-error").hidden = false;
    return;
  }
  busy = true;
  $("run").disabled = true;
  $("run-label").textContent = "Resolving…";
  const started = Date.now();
  const clock = setInterval(() => {
    $("run-label").textContent = "Resolving… " + Math.round((Date.now() - started) / 1000) + " s";
  }, 1000);
  $("announcements").textContent = "Computing the resolution.";
  $("result").setAttribute("aria-busy", "true");
  waiting("Computing the resolution…");
  try {
    const response = await fetch(request.search === "exhaustive" ? "/api/searches" : "/api/resolve", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request),
    });
    let data;
    try { data = readRecord(await response.text()); } catch (error) {
      if (error instanceof SyntaxError) throw new Error("The local server returned an unreadable response.");
      throw error;
    }
    if (response.status === 409) {
      // The server is busy with a search, possibly the one for this very input.
      try {
        const current = await searchFetch("/api/searches/current");
        if (current.job) {
          ++searchTransition;
          renderSearch(current.job);
          pollSearch();
        }
      } catch { /* The error below still says what happened. */ }
      if (activeSearch() && sameGerm(request, searchJob.request)) {
        waiting("");
        return;
      }
    }
    if (!response.ok || data.error) throw new Error(data.error || "Resolution failed (HTTP " + response.status + ").");
    if (response.status === 202) {
      ++searchTransition;
      renderSearch(data);
      pollSearch();
      waiting("");
      $("announcements").textContent = "An exhaustive search is running.";
    } else {
      showResult(data, request);
      writeAddress(request);
      if (followResult && sameGerm(request, getRequest()) && matchMedia("(max-width: 820px)").matches) {
        $("result-equation").focus({ preventScroll: true });
        $("result").scrollIntoView({ block: "start" });
      }
    }
  } catch (error) {
    $("form-error").textContent = error instanceof TypeError ? "Could not reach the local resolution server." : error.message;
    $("form-error").hidden = false;
    $("announcements").textContent = "The computation did not finish.";
    waiting("");
  } finally {
    clearInterval(clock);
    busy = false;
    $("run").disabled = false;
    $("run-label").textContent = "Resolve";
    $("result").removeAttribute("aria-busy");
    updateStale();
    updateSearchOffer();
    updatePlate();
  }
}
form.addEventListener("submit", (event) => {
  event.preventDefault();
  resolveGerm(true);
});
$("export").addEventListener("click", () => {
  if (!lastResult) return;
  const blob = new Blob([recordJSON(lastResult, 2) + "\n"], { type: "application/json" });
  const address = URL.createObjectURL(blob);
  const link = el("a");
  link.href = address;
  link.download = "resolution-atlas-" + lastRequest.mode + ".json";
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(address), 1000);
});

/* ---------- Exhaustive search, run by the server in the background ---------- */

function activeSearch() {
  return searchJob && ["running", "cancelling"].includes(searchJob.state);
}

function updateSearchOffer() {
  const eligible = lastResult?.can_search === true && ["inconclusive", "unsupported"].includes(lastResult.status);
  $("search-offer").hidden = !eligible || !($("stale").hidden) || activeSearch();
  $("start-search").disabled = searchStarting || busy;
}

function elapsed(ms) {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return seconds + " s";
  return Math.floor(seconds / 60) + " min " + (seconds % 60) + " s";
}

// A finished search that was shown or dismissed does not come back when the
// page is reloaded in the same tab.
function forgetSearch() {
  $("search-job").hidden = true;
  try { sessionStorage.setItem("atlas-search-seen", searchJob.id); } catch { /* Private browsing. */ }
}
function searchSeen(job) {
  try { return sessionStorage.getItem("atlas-search-seen") === job.id; } catch { return false; }
}

// Put a finished search on the page, with the form set to what was searched.
function viewSearchResult() {
  if (!searchJob?.result) return;
  const request = { compute: "cpu", mode: "embedded", ...searchJob.request };
  $("polynomial").value = request.polynomial;
  $("variables").value = request.variables.join(", ");
  setMode(request.mode);
  $("compute").value = request.compute || "cpu";
  $("zeta").checked = request.monodromy === true;
  $("exhaustive").checked = true;
  variablesFollow = followsPolynomial();
  fitInput();
  showResult(searchJob.result, getRequest());
  writeAddress(getRequest());
  forgetSearch();
}

function renderSearch(job) {
  const wasActive = activeSearch();
  searchJob = job;
  $("search-job").hidden = !job;
  updateStale();
  updateSearchOffer();
  updatePlate();
  if (!job) return;
  const states = { running: "Searching for a verified resolution", cancelling: "Cancelling search",
    completed: "Search complete", failed: "Search stopped", cancelled: "Search cancelled" };
  const title = states[job.state] || job.state;
  if ($("search-job-state").textContent !== title) $("search-job-state").textContent = title;
  if (job.error) $("search-job-description").textContent = job.error;
  else {
    $("search-job-description").replaceChildren(typeset(job.request.polynomial),
      " · " + (job.request.mode || "embedded") + " · " + elapsed(job.elapsed_ms));
  }
  $("search-spinner").classList.toggle("idle", !activeSearch());
  $("cancel-search").hidden = !activeSearch();
  $("cancel-search").disabled = job.state === "cancelling";
  $("view-search").hidden = job.state !== "completed" || !job.result;
  $("dismiss-search").hidden = activeSearch();
  // A search that finishes while this page waits for it, for the input still
  // in the form, is shown at once. Otherwise the result waits behind a button.
  if (wasActive && job.state === "completed" && job.result && !busy && sameGerm(getRequest(), job.request)) {
    viewSearchResult();
    $("announcements").textContent = "Search complete. " + (job.result.summary || "");
  }
}

async function searchFetch(url, options) {
  const response = await fetch(url, options);
  const data = readRecord(await response.text());
  if (!response.ok) {
    const error = new Error(data.error || "Search request failed.");
    error.status = response.status;
    throw error;
  }
  return data;
}

function pollSearch(delay = 1200) {
  clearTimeout(pollTimer);
  if (!activeSearch()) return;
  const id = searchJob.id, token = searchTransition;
  pollTimer = setTimeout(async () => {
    try {
      const job = await searchFetch("/api/searches/" + id);
      if (token !== searchTransition) return;
      renderSearch(job);
      pollSearch();
    } catch (error) {
      if (token !== searchTransition) return;
      if (error.status === 404 || error.name === "ExactIntegerError") {
        renderSearch({ ...searchJob, state: "failed", error: error.message });
        return;
      }
      $("search-job-state").textContent = "Waiting for the local search server";
      pollSearch(3500);
    }
  }, delay);
}

$("start-search").addEventListener("click", async () => {
  if (searchStarting || busy || activeSearch() || !lastResult?.can_search) return;
  searchStarting = true;
  const token = ++searchTransition;
  updateSearchOffer();
  try {
    const job = await searchFetch("/api/searches", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(lastRequest),
    });
    if (token !== searchTransition) return;
    renderSearch(job);
    pollSearch();
  } catch (error) {
    if (error.status === 409) {
      try {
        const data = await searchFetch("/api/searches/current");
        if (data.job) { renderSearch(data.job); pollSearch(); }
      } catch { /* Keep the original error from the start request. */ }
    }
    $("form-error").textContent = error.message;
    $("form-error").hidden = false;
  } finally {
    searchStarting = false;
    updateSearchOffer();
  }
});

$("cancel-search").addEventListener("click", async () => {
  if (!activeSearch()) return;
  ++searchTransition;
  clearTimeout(pollTimer);
  $("cancel-search").disabled = true;
  try {
    renderSearch(await searchFetch("/api/searches/" + searchJob.id, { method: "DELETE" }));
  } catch (error) {
    $("search-job-description").textContent = error.message;
    $("cancel-search").disabled = false;
  }
  pollSearch();
});

$("view-search").addEventListener("click", () => {
  if (!searchJob?.result) return;
  viewSearchResult();
  $("result").scrollIntoView({ block: "start" });
});

$("dismiss-search").addEventListener("click", () => {
  if (searchJob && !activeSearch()) forgetSearch();
});

async function restoreSearch() {
  const token = searchTransition;
  try {
    const data = await searchFetch("/api/searches/current");
    // A search cancelled before this page was opened is not worth a banner.
    if (token === searchTransition && data.job && !searchSeen(data.job) && data.job.state !== "cancelled") {
      renderSearch(data.job);
      pollSearch();
    }
  } catch { /* Quick local calculations remain available. */ }
}

const addressed = readAddress();
fitInput();
updateInput();
if (addressed) resolveGerm();
restoreSearch();
