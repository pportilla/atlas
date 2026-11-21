"use strict";

const $ = (id) => document.getElementById(id);
const form = $("resolve-form");
const runs = [];
let opened = null;
let serial = 0;
const report = $("result");

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
  if (!vertices.length || vertices.length > 64 || cells.length > 256) return null;
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

// Some older records store the incident divisors without an explicit cell dimension.
function cellDimension(cell) {
  return cell.dimension ?? (cell.vertices.length - 1);
}

function complexData(vertices, cells, charts, branches, dual) {
  const side = el("div", "complex-data");
  const chartCount = new Map();
  for (const chart of charts) {
    for (const id of componentsOf(chart)) chartCount.set(id, (chartCount.get(id) || 0) + 1);
  }

  side.append(el("h4", "sub", "0-dimensional cells · vertices"));
  const keys = columnsOf(vertices, ["total_multiplicity", "multiplicity", "ambient_order", "self_intersection", "genus", "dimension"], ["id", "label"]);
  const countCharts = chartCount.size > 0;
  const rows = [];
  side.append(table(["Divisor", ...keys.map((key) => key === "dimension" ? "Divisor dimension" : labelFor(key)), ...(countCharts ? ["Charts"] : [])], vertices, (v, index, tr) => {
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

  // A geometric intersection can have several components with the same
  // incident divisors. Preserve each cell and its own recorded face IDs.
  const dimensions = new Map();
  for (const cell of cells) {
    const dimension = cellDimension(cell);
    if (!dimensions.has(dimension)) dimensions.set(dimension, []);
    dimensions.get(dimension).push(cell);
  }
  for (const [dimension, group] of [...dimensions].sort(([a], [b]) => a - b)) {
    const extras = columnsOf(group, [], ["id", "vertices", "faces", "dimension"]);
    const hasFaces = group.some((cell) => Array.isArray(cell.faces));
    side.append(el("h4", "sub", dimension + "-dimensional cells"));
    side.append(table(["Cell ID", "Intersecting divisors", ...(hasFaces ? ["Boundary faces"] : []), ...extras.map(labelFor)],
      group, (cell) => [
        cell.id ?? "—",
        joined(cell.vertices.map(idNode), " ∩ "),
        ...(hasFaces ? [cell.faces?.length ? joined(cell.faces.map(idNode), ", ") : "—"] : []),
        ...extras.map((key) => valueNode(key, cell[key])),
      ]));
  }

  if (branches.length) {
    const keys = columnsOf(branches, ["ambient_components", "point", "chart"], ["id"]);
    side.append(disclosure("Branches of the strict transform (" + branches.length + ")",
      () => table(["Branch", ...keys.map((key) => key === "point" ? "At the point" : key === "chart" ? "Of chart" : labelFor(key))],
        branches, (b) => [idNode(b.id), ...keys.map((key) => valueNode(key, b[key]))], 40)));
  }
  if (dual.root_definition) {
    side.append(disclosure("Algebraic directions", dataList(Object.entries(dual.root_definition))));
  }
  return side;
}

function showComplex(data) {
  const dual = data.dual_complex;
  const panel = $("panel-complex");
  panel.replaceChildren();
  if (!dual || (dual.status !== "complete" && dual.status !== "partial" && data.status !== "resolved")) return;

  const vertices = dual.vertices || [];
  const cells = dual.cells || [];
  const isGraph = cells.every((cell) => cellDimension(cell) <= 1);
  const head = sectionHead(isGraph ? "Dual graph" : "Dual complex", dual.convention);
  head.classList.add("complex-heading");
  panel.append(head);
  if (dual.status !== "complete") {
    const state = dual.status === "partial" ? "Partial complex" : "Dual complex unavailable";
    panel.append(el("p", "notice", dual.reason ? state + ". " + dual.reason : state + "."));
  }
  if (!vertices.length) {
    if (dual.status === "complete") panel.append(el("p", "empty", "The exceptional divisor is empty."));
    return;
  }

  const vertexIds = new Set(vertices.map((v) => String(v.id)));
  const branches = (data.branches || []).filter((b) => !vertexIds.has(String(b.id)));
  const figure = drawGraph(vertices, cells, branches);
  const controls = el("div", "complex-switch");
  controls.setAttribute("role", "group");
  controls.setAttribute("aria-label", "Dual complex view");
  const picture = el("button", "", "Picture");
  const records = el("button", "", "Data");
  picture.type = records.type = "button";
  controls.append(picture, records);
  head.append(controls);

  if (figure) panel.append(figure);
  else {
    picture.disabled = true;
    panel.append(el("p", "helper", "The complex is too large to draw."));
  }
  let details;
  function showData(open) {
    if (open && !details) {
      details = complexData(vertices, cells, data.charts || [], branches, dual);
      panel.append(details);
    }
    picture.setAttribute("aria-pressed", String(!open));
    records.setAttribute("aria-pressed", String(open));
    if (figure) figure.hidden = open;
    if (details) details.hidden = !open;
  }
  picture.addEventListener("click", () => showData(false));
  records.addEventListener("click", () => showData(true));
  showData(!figure);
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
  if (!zeta || typeof zeta !== "object") return;
  panel.append(sectionHead("Monodromy zeta function at the origin"));
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
  const contributions = Array.isArray(zeta.contributions) ? zeta.contributions.filter((c) => c && typeof c === "object") : [];
  if (contributions.length) {
    const keys = columnsOf(contributions, ["multiplicity", "euler_characteristic"], ["component"]);
    body.append(table(["Component", ...keys.map(labelFor)],
      contributions, (c) => [idNode(c.component ?? "—"), ...keys.map((key) => valueNode(key, c[key]))], 12));
  }
  if (body.childElementCount) {
    panel.append(disclosure("Formula", body));
  }
  const strata = zeta.nearby_strata;
  if (Array.isArray(strata) && strata.length) {
    panel.append(sectionHead("Monodromy zeta functions on certified nearby strata"));
    panel.append(table(["Stratum", "Local zeta function"], strata, stratum => {
      const locus = el("div");
      locus.append(stratumLocus(stratum));
      const model = el("div", "helper");
      if (stratum.model === "X^a + Y^b") {
        model.append("Transverse model ", typeset(`X^${stratum.powers[0]} + Y^${stratum.powers[1]}`));
      } else if (stratum.model === "X*Y + Z^m") {
        model.append("Transverse model ", typeset(`X*Y + Z^${stratum.powers[2]}`));
      } else if (stratum.model === "sum of squares") model.append("Transverse Morse singularity");
      else if (stratum.model === "nearby transverse node") model.append("Transverse node");
      else if (stratum.model === "constant transverse type") model.append("Constant transverse type");
      else if (stratum.model === "transverse composition") {
        const names = stratum.transverse_coordinates;
        model.append("Transverse germ ", typeset(stratum.transverse_germ), " at the origin, with ",
          typeset(names.map((name, i) => `${name} = ${stratum.coordinates[i]}`).join(", ")));
      }
      if (model.hasChildNodes()) locus.append(model);
      const formula = el("span", "m local-zeta");
      formula.setAttribute("role", "math");
      formula.setAttribute("aria-label", stratum.expression);
      formula.append(zetaQuotient(stratum, variable));
      return [locus, formula];
    }));
  }
}

function stratumLocus(stratum) {
  const locus = el("span");
  if (stratum.model === "smooth component") {
    if (stratum.component === "strict") locus.append(stratum.multiplicity > 1
      ? "Generic smooth points of the reduced hypersurface" : "Generic smooth points of the hypersurface");
    else locus.append("Generic smooth points of branch ", idNode(stratum.component));
    const equation = stratum.equation || stratum.coordinates?.[0];
    if (equation) locus.append(" (", typeset(equation + " = 0"), ")");
    return locus;
  }
  if (Array.isArray(stratum.equations)) {
    stratum.equations.forEach((equation, index) => {
      if (index) locus.append(", ");
      locus.append(typeset(equation + " = 0"));
    });
    return locus;
  }
  if (stratum.model === "sum of squares" || stratum.model === "transverse composition") locus.append("Generic points of ");
  const equations = stratum.coordinates.join(" = ") + " = 0";
  locus.append(typeset(equations));
  if (stratum.model === "constant transverse type") locus.append(" away from the origin");
  if (stratum.model === "X^a + Y^b" || stratum.model === "X*Y + Z^m") {
    for (const name of stratum.parameters) locus.append(", ", typeset(name), " ≠ 0");
  }
  return locus;
}

/* The topological zeta function of Denef and Loeser, which the engine sums
   over the strata of the total transform. */

function minus(text) {
  return String(text).replace(/-/g, "−");
}
function showTopological(data) {
  const panel = $("panel-zeta");
  const zeta = data.topological_zeta;
  if (!zeta || typeof zeta !== "object") return;
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

  if (Array.isArray(zeta.poles)) {
    const row = el("p", "zeta-euler m");
    const values = zeta.poles.map(pole => minus(pole.value));
    row.append("Poles: " + (values.join(", ") || "none"));
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
  if (body.childElementCount) panel.append(disclosure("Formula", body));
  showMonodromyCheck(panel, data);
}

function showMonodromyCheck(panel, data) {
  const check = data.monodromy_check;
  if (data.monodromy_zeta?.status !== "computed" || !check) return;
  panel.append(sectionHead("Topological monodromy conjecture"));
  const verified = check.status === "verified";
  const verdict = el("p", "monodromy-verdict");
  verdict.append(el("strong", "", verified ? "Verified for this germ." :
    check.status === "unresolved" ? "Unresolved for this germ." : "Check unavailable."));
  if (verified) verdict.append(" Each pole gives a monodromy eigenvalue at or near the origin.");
  else if (check.reason) verdict.append(" " + check.reason);
  panel.append(verdict);

  const witnesses = {
    "origin zeta": "Monodromy zeta at the origin",
    "origin H^0": "Degree-zero cohomology at the origin",
    "nearby smooth component": "Nearby smooth component",
    "nearby transverse binomial": "Nearby transverse binomial singularity",
    "nearby transverse Morse": "Nearby transverse Morse singularity",
    "nearby transverse composition": "Nearby transverse germ",
  };
  if (check.poles?.length) {
    const strata = data.monodromy_zeta.nearby_strata || [];
    const zetaRole = e => e > 0 ? " (zeta zero)" : e < 0 ? " (zeta pole)" : "";
    const grid = table(["Pole", "Predicted eigenvalue", "Witness"], check.poles, pole => {
      const eigenvalue = el("div", "pole-eigenvalue");
      const phase = "exp(2πi · (" + minus(pole.value) + "))";
      const value = el("span", "m", ({1: "1", 2: "−1"})[pole.eigenvalue_order] || phase);
      value.title = phase;
      eigenvalue.append(value, el("div", "helper", "Order " + pole.eigenvalue_order));
      const locations = el("ul", "witness-loci");
      const matches = (pole.nearby_matches || []).filter(match => strata[match.stratum_index]);
      if (pole.witness === "origin zeta" || pole.witness === "origin H^0" || !matches.length) {
        const witness = witnesses[pole.witness] || pole.witness || "Undetermined";
        const component = pole.witness_component ? " (" + pole.witness_component + ")" : "";
        locations.append(el("li", "", witness + component + zetaRole(pole.cyclotomic_exponent)));
      }
      for (const match of matches) {
        const location = el("li");
        location.append(stratumLocus(strata[match.stratum_index]), zetaRole(match.cyclotomic_exponent));
        locations.append(location);
      }
      return [minus(pole.value), eigenvalue, locations];
    });
    grid.classList.add("pole-witnesses");
    panel.append(grid);
  }
}

function stepMethod(step, data) {
  let method = step.method || step.type;
  if (!method && Array.isArray(step.center) && step.parent_chart && step.exceptional_id) method = "point blowup";
  if (method === "blowup" && Array.isArray(step.center_ray_indices)) method = "toric coordinate blowup";
  if (method === "blowup" && (step.kind || step.center) && data.engine === "canonical-blowups") {
    method = "smooth-center blowup";
  }
  return method ? method.replace(/blowup/g, "blow-up")
    .replace(/^./, c => c.toUpperCase()) : "Unrecorded";
}

function stepChartCenter(entry) {
  const line = el("div");
  line.append(locus(entry.ideal));
  if (entry.chart) line.append(" in " + entry.chart);
  else if (entry.history?.length) {
    const path = entry.history.map(([stage, pivot]) => "step " + stage + ", pivot " + (pivot + 1));
    line.append(" in " + path.join(" → "));
  } else line.append(" in the input chart");
  if (entry.ambient_ideal?.length) line.append(" subject to ", locus(entry.ambient_ideal));
  return line;
}

function stepCenter(step, data, chartCoordinates, repair) {
  const block = el("div", "eqs");
  if (repair) {
    const above = el("span", "", "Over ");
    above.append(locus(repair.ideal), " in " + repair.chart);
    block.append(above);
  }
  if (Array.isArray(step.center)) {
    const input = step.parent_chart === "origin";
    const coordinates = input ? data.variables : chartCoordinates.get(step.parent_chart);
    const point = el("span");
    if (coordinates?.length) point.append(tuple(coordinates), " = ");
    point.append(tuple(step.center));
    if (step.parent_chart) point.append(input ? " in the input chart" : " in " + step.parent_chart);
    block.append(point);
  } else {
    const kind = step.center || step.kind;
    if (kind) {
      const label = kind === "subvariety" && step.dimension != null ? step.dimension + "-dimensional center" : kind;
      block.append(el("span", "", label.replace(/^./, c => c.toUpperCase())));
    }
  }
  const ideal = step.center_ideal || step.generators || step.ideal;
  if (ideal?.length) block.append(locus(ideal));
  if (step.center_charts?.length) {
    const [first, ...rest] = step.center_charts;
    block.append(stepChartCenter(first));
    if (rest.length) block.append(disclosure(plural(rest.length, "other chart", "other charts"),
      () => joined(rest.map(stepChartCenter), "")));
  }
  if (Array.isArray(step.center_ray_indices)) {
    block.append("Orbit closure for rays " + step.center_ray_indices.join(", "));
  } else if (step.type?.includes("toric modification")) {
    // A fan modification can represent many blow-ups without specifying their factorization.
    const images = data.certificate?.boundary_image_coordinate_ideals;
    if (images?.length) {
      block.append("Exceptional images");
      for (const ideal of images) block.append(locus(ideal));
    } else if (data.fan?.isomorphism_off_origin || data.engine === "native-toric") {
      block.append("Over the origin");
    } else block.append("Specified by the fan");
  }
  return block.hasChildNodes() ? block : "—";
}

function stepDetails(step, data) {
  const block = el("div", "eqs");
  function line(label, content) {
    const item = el("span", "", label + " ");
    item.append(content);
    block.append(item);
  }
  const divisor = step.exceptional_id || step.exceptional;
  if (divisor) line(data.mode === "intrinsic" ? "Ambient divisor" : "Creates", idNode(divisor));
  for (const key of ["strict_multiplicity", "total_multiplicity", "multiplicity", "reduced_support_strict_multiplicity",
                     "reduced_support_multiplicity", "weight", "rule"]) {
    if (step[key] != null) line(labelFor(key), valueNode(key, step[key]));
  }
  if (step.components?.length) line("On", joined(step.components.map(id => id === "strict" ? "strict transform" : idNode(id)), ", "));
  if (step.image_zero_coordinates?.length) line("Image", locus(step.image_zero_coordinates));
  if (step.new_ray_index != null) line("New ray", String(step.new_ray_index));
  if (step.type?.includes("toric modification") && data.fan?.cones) {
    line("Fan with", plural(data.fan.cones.length, "maximal cone", "maximal cones"));
  }
  if (step.type === "polynomial graph isomorphism" && data.graph?.equation) {
    line("Graph equation", typeset(data.graph.equation, " = 0"));
  }
  return block.hasChildNodes() ? block : "—";
}

function showSteps(data) {
  const panel = $("panel-steps");
  panel.replaceChildren();
  const steps = (Array.isArray(data.steps) ? data.steps : []).filter((step) => step && typeof step === "object");
  if (!steps.length) {
    if (data.status === "resolved" && Array.isArray(data.steps)) {
      foldedSection(panel, "Resolution steps", 0).body.append(el("p", "", "The identity map resolves this germ."));
    }
    return;
  }
  const chartCoordinates = new Map((data.charts || []).map(chart => [chart.id, chart.coordinates]));
  const rows = steps.flatMap(step => step.centers?.length
    ? step.centers.map(center => ({step: {...center, type: "smooth-center blowup"}, repair: step}))
    : [{step}]);
  foldedSection(panel, "Resolution steps", rows.length).body.append(table(["Step", "Method", "Center", "Details"], rows,
    ({step, repair}, index) => [String(step.id ?? index + 1), stepMethod(step, data),
      stepCenter(step, data, chartCoordinates, repair), stepDetails(step, data)]));
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
  const roots = {...data.dual_complex?.root_families, ...data.root_families};
  if (Object.keys(roots).length) {
    const rows = Object.entries(roots);
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

function showResult(run) {
  const {result} = run;
  selected = null;
  selectionViews.length = 0;
  run.body.append(report);
  report.hidden = false;
  const summary = result.status === "resolved" ? "" : result.summary || result.reason;
  const diagnostic = result.diagnostic?.description || result.diagnostic?.fast_path_limit;
  $("result-summary").textContent = summary || "";
  $("result-summary").hidden = !summary;
  $("result-diagnostic").textContent = diagnostic || "";
  $("result-diagnostic").hidden = !diagnostic;
  report.querySelector(".result-head").hidden = !summary && !diagnostic;
  showComplex(result);
  showZeta(result);
  showTopological(result);
  showSteps(result);
  showCharts(result);
  $("export").disabled = false;
}

/* ---------- The input ---------- */

const examples = { cusp: "y^2-x^3", node: "y^2-x^2", tacnode: "y^2-x^4",
  "singular-line": "x^2+y^2+t*z^2+z^3" };
const exampleButtons = form.querySelectorAll("[data-example]");

function updateExampleButtons(request) {
  const polynomial = request.polynomial.length <= 100 ? request.polynomial.replace(/\s+/g, "") : null;
  for (const button of exampleButtons) {
    const example = examples[button.dataset.example];
    const names = detectVariables(example);
    button.setAttribute("aria-pressed", String(example === polynomial && request.variables.length === names.length
      && names.every(name => request.variables.includes(name))));
  }
}

for (const button of exampleButtons) button.addEventListener("click", () => {
  $("polynomial").value = examples[button.dataset.example];
  $("variables").value = detectVariables($("polynomial").value).join(", ");
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
    zetas: zetaChoices.filter(({id}) => $(id).checked).map(({key}) => key),
    search: $("exhaustive").checked ? "exhaustive" : "auto",
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
  updateExampleButtons(request);
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
const zetaChoices = [
  { id: "zeta", key: "monodromy", label: "monodromy zeta function" },
  { id: "topological", key: "topological", label: "topological zeta function" },
];
for (const {id} of zetaChoices) {
  const input = $(id);
  try {
    const saved = localStorage.getItem("atlas-" + id);
    if (saved !== null) input.checked = saved === "on";
  } catch { /* Preferences are optional. */ }
  input.addEventListener("change", () => {
    try { localStorage.setItem("atlas-" + id, input.checked ? "on" : "off"); } catch { /* Optional. */ }
  });
}

function writeAddress(request) {
  const params = new URLSearchParams({ f: request.polynomial, vars: request.variables.join(","), mode: request.mode });
  if (request.compute && request.compute !== "cpu") params.set("compute", request.compute);
  params.set("zetas", request.zetas.join(","));
  if (request.search === "exhaustive") params.set("search", request.search);
  try { history.replaceState(null, "", "#" + params); } catch { /* Very large URLs can exceed browser limits. */ }
}

function readAddress() {
  const params = new URLSearchParams(location.hash.slice(1));
  const polynomial = params.get("f");
  if (!polynomial) return null;
  $("polynomial").value = polynomial;
  const names = (params.get("vars") || "").split(",").filter(Boolean);
  $("variables").value = (names.length ? names : detectVariables(polynomial)).join(", ");
  setMode(params.get("mode") === "intrinsic" ? "intrinsic" : "embedded");
  $("compute").value = ["auto", "gpu"].includes(params.get("compute")) ? params.get("compute") : "cpu";
  $("exhaustive").checked = params.get("search") === "exhaustive";
  const zetas = params.get("zetas");
  for (const {id, key} of zetaChoices) {
    $(id).checked = zetas === null ? params.get("zeta") !== "0" : zetas.split(",").includes(key);
  }
  variablesFollow = followsPolynomial();
  return getRequest();
}

/* ---------- Search history ---------- */

const activeStates = new Set(["submitting", "queued", "running", "cancelling"]);

function normalizeRequest(request) {
  return {
    ...request, mode: request.mode || "embedded", compute: request.compute || "cpu",
    zetas: request.zetas || (request.monodromy ? ["monodromy", "topological"] : []),
    search: request.search || "exhaustive",
  };
}

async function requestJSON(url, method = "GET", body) {
  const options = { method };
  if (body) {
    options.headers = { "Content-Type": "application/json" };
    options.body = recordJSON(body);
  }
  const response = await fetch(url, options);
  const data = readRecord(await response.text());
  if (!response.ok) {
    const error = new Error(data.error || "Request failed (HTTP " + response.status + ").");
    error.status = response.status;
    throw error;
  }
  return data;
}

function action(label, callback) {
  const button = el("button", "button", label);
  button.type = "button";
  button.addEventListener("click", callback);
  return button;
}

function stateLabel(run) {
  if (run.pendingZeta) return "Computing zeta";
  if (run.result) return ({
    resolved: "Resolved", zetas_computed: "Zeta functions computed", inconclusive: "Inconclusive",
    partial: "Partial", unsupported: "Unsupported",
  })[run.result.status] || "Finished";
  return ({
    submitting: "Starting", queued: "Queued", running: "Searching", cancelling: "Cancelling",
    cancelled: "Cancelled", failed: "Failed",
  })[run.state] || run.state;
}

function makeRun(request, restored = false) {
  const node = el("article", "resolution-run");
  const heading = el("h2", "run-heading");
  const toggle = el("button", "run-toggle");
  toggle.type = "button";
  const equation = el("span", "run-equation");
  const source = request.polynomial;
  equation.append(source.length <= 180 ? typeset(source, " = 0") : source.slice(0, 180) + "…");
  const mode = el("span", "run-mode", request.mode);
  const status = el("span", "run-status");
  const body = el("div", "run-body");
  body.id = "run-body-" + (++serial);
  body.hidden = true;
  toggle.setAttribute("aria-controls", body.id);
  toggle.setAttribute("aria-expanded", "false");
  const arrow = el("span", "run-chevron");
  arrow.setAttribute("aria-hidden", "true");
  toggle.append(arrow, equation, mode, status);
  heading.append(toggle);
  const message = el("p", "run-message");
  message.setAttribute("role", "status");
  const controls = el("div", "run-actions");
  const progress = el("div", "run-progress");
  const stage = el("p", "run-stage");
  const blowups = el("span", "run-blowups");
  const elapsed = el("span", "run-elapsed");
  progress.append(stage, blowups, elapsed);
  body.append(message, progress, controls);
  node.append(heading, body);
  const run = {request, node, toggle, body, message, controls, status, progress, stage, blowups, elapsed, state: "submitting",
               pendingZeta: null, warning: "", connectionWarning: "", timer: null};
  if (restored) {
    $("runs").append(node);
    runs.push(run);
  } else {
    $("runs").prepend(node);
    runs.unshift(run);
  }
  toggle.addEventListener("click", () => openRun(opened === run ? null : run));
  refreshRun(run);
  return run;
}

function openRun(run) {
  if (opened) {
    opened.toggle.setAttribute("aria-expanded", "false");
    opened.body.hidden = true;
    report.hidden = true;
  }
  opened = run;
  if (!run) return;
  run.toggle.setAttribute("aria-expanded", "true");
  run.body.hidden = false;
  writeAddress(run.request);
  refreshRun(run);
  if (run.result) showResult(run);
}

function refreshRun(run) {
  run.node.dataset.state = run.state;
  run.status.textContent = stateLabel(run);
  const active = activeStates.has(run.state);
  const wasFocused = document.activeElement?.parentNode === run.controls ? document.activeElement.textContent : null;
  run.controls.replaceChildren();
  let message = run.error || run.warning || run.connectionWarning;
  if (!message && active) message = run.state === "queued" ? "Waiting for a computation slot."
    : run.state === "cancelling" ? "Cancelling this search…" : "Searching for a verified resolution…";
  if (!message && run.pendingZeta) message = "Computing " + run.pendingZeta + " zeta from the saved resolution…";
  run.message.textContent = message || "";
  run.message.hidden = !message;
  updateProgress(run);
  if (active && run.id) {
    const cancel = action("Cancel search", () => cancelRun(run));
    cancel.disabled = run.state === "cancelling";
    run.controls.append(cancel);
  }
  if (["resolved", "inconclusive", "zetas_computed"].includes(run.result?.status)) {
    for (const {key, label} of zetaChoices) {
      const zeta = run.result[key + "_zeta"];
      const legacyMonodromy = key === "monodromy" && zeta?.status === "computed" && !Array.isArray(zeta.nearby_strata);
      if (zeta?.status === "computed" && !legacyMonodromy) continue;
      const button = action(legacyMonodromy ? "Compute nearby monodromy zeta functions" : "Compute " + label,
        () => computeZeta(run, key));
      button.disabled = Boolean(run.pendingZeta);
      run.controls.append(button);
    }
  }
  if (run.result?.can_search && run.request.search !== "exhaustive") {
    run.controls.append(action("Start exhaustive search", () => submitRun({...run.request, search: "exhaustive"})));
  } else if (["failed", "cancelled"].includes(run.state)) {
    run.controls.append(action("Retry resolution", () => submitRun(run.request)));
  }
  run.controls.hidden = !run.controls.childElementCount;
  if (wasFocused) {
    const current = [...run.controls.children].find(button => button.textContent === wasFocused && !button.disabled);
    (current || run.toggle).focus({preventScroll: true});
  }
}

function updateProgress(run) {
  run.progress.hidden = !run.id || !activeStates.has(run.state);
  if (run.progress.hidden) return;
  const stage = run.progressData?.stage;
  const method = {origin: "Smooth-centre search", certificates: "Certificate search", newton: "Newton search"}[run.progressData?.method];
  run.stage.textContent = stage && method ? method + " · " + stage : stage || "";
  run.stage.hidden = !stage || run.state === "queued";
  const count = run.progressData?.blowups;
  run.blowups.hidden = run.state === "queued";
  run.blowups.textContent = count == null ? "Blow-up count pending" : plural(count, "blowup", "blowups") + (method ? " in this attempt" : " so far");
  run.elapsed.hidden = !Number.isFinite(run.elapsedMs);
  const seconds = Math.floor(run.elapsedMs / 1000);
  const elapsed = seconds >= 60 ? Math.floor(seconds / 60) + " min " + seconds % 60 + " s" : seconds + " s";
  run.elapsed.textContent = run.elapsed.hidden ? "" : "Elapsed " + elapsed;
}

function updateJob(run, job) {
  clearTimeout(run.timer);
  run.id = job.id;
  const previous = run.state;
  run.state = job.state;
  run.error = job.error || "";
  run.progressData = job.progress;
  run.elapsedMs = job.elapsed_ms;
  run.connectionWarning = "";
  if (job.result) {
    run.result = job.result;
    $("workspace-logo").hidden = true;
  }
  if (job.request) run.request = normalizeRequest(job.request);
  // Avoid replacing a focused cancel control on every poll.
  if (run.state !== previous || run.error || !run.controls.childElementCount || run.message.textContent.startsWith("Reconnecting")) {
    refreshRun(run);
  } else updateProgress(run);
  if (run === opened && run.result) showResult(run);
  if (activeStates.has(run.state)) schedulePoll(run);
  else if (previous !== run.state) {
    $("announcements").textContent = stateLabel(run) + ": " + run.request.polynomial.slice(0, 100);
  }
}

function schedulePoll(run, delay = 850) {
  clearTimeout(run.timer);
  run.timer = setTimeout(async () => {
    const version = run.version;
    try {
      const job = await requestJSON("/api/searches/" + run.id);
      if (version === run.version) updateJob(run, job);
    } catch (error) {
      if (version !== run.version) return;
      if (error.status === 404 || error.name === "ExactIntegerError") {
        run.state = "failed";
        run.error = error.message;
        refreshRun(run);
      } else {
        run.connectionWarning = "Reconnecting to the search server…";
        refreshRun(run);
        schedulePoll(run, 3000);
      }
    }
  }, delay);
}

async function cancelRun(run) {
  clearTimeout(run.timer);
  run.version = (run.version || 0) + 1;
  run.state = "cancelling";
  refreshRun(run);
  try {
    updateJob(run, await requestJSON("/api/searches/" + run.id, "DELETE"));
  } catch (error) {
    run.warning = error.message;
    schedulePoll(run);
    refreshRun(run);
  }
}

async function computeZeta(run, key) {
  if (run.pendingZeta || !run.result) return;
  run.pendingZeta = key;
  run.warning = "";
  refreshRun(run);
  const zetas = [key];
  try {
    let fields;
    try {
      const job = await requestJSON("/api/searches/" + run.id + "/zetas", "POST", {zetas});
      fields = job.result;
    } catch (error) {
      if (error.status !== 404) throw error;
      fields = await requestJSON("/api/zetas", "POST", {record: run.result, zetas});
    }
    Object.assign(run.result, fields);
    if (opened === run) {
      showZeta(run.result);
      showTopological(run.result);
    }
  } catch (error) { run.warning = error.message; }
  finally {
    run.pendingZeta = null;
    refreshRun(run);
  }
}

async function submitRun(request) {
  if (!request.polynomial || !request.variables.length) {
    $("form-error").textContent = "Enter a polynomial and its variables.";
    $("form-error").hidden = false;
    return;
  }
  const run = makeRun(request);
  openRun(run);
  $("announcements").textContent = "Searching for a verified resolution.";
  try {
    const job = await requestJSON("/api/searches", "POST", request);
    // A list response can already contain this submission before its POST
    // reaches the browser. Keep the original row, its order and open state.
    const duplicate = runs.find(other => other !== run && other.id === job.id);
    if (duplicate) {
      clearTimeout(duplicate.timer);
      duplicate.version = (duplicate.version || 0) + 1;
      runs.splice(runs.indexOf(duplicate), 1);
      duplicate.node.remove();
      if (opened === duplicate) openRun(run);
    }
    updateJob(run, job);
  } catch (error) {
    run.state = "failed";
    run.error = error.message;
    refreshRun(run);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitRun(getRequest());
  if (matchMedia("(max-width: 820px)").matches && opened) {
    opened.toggle.focus({preventScroll: true});
    opened.node.scrollIntoView({block: "start"});
  }
});

$("export").addEventListener("click", () => {
  if (!opened?.result) return;
  const blob = new Blob([recordJSON(opened.result, 2) + "\n"], {type: "application/json"});
  const address = URL.createObjectURL(blob);
  const link = el("a");
  link.href = address;
  link.download = "resolution-atlas-" + opened.request.mode + ".json";
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(address), 1000);
});

function addressedRun(request) {
  return runs.find(run => sameGerm(run.request, request) && run.request.compute === request.compute
    && run.request.search === request.search && run.request.zetas.join() === request.zetas.join());
}
function followAddress(request) {
  const found = addressedRun(request);
  if (found) openRun(found);
  else submitRun(request);
}
window.addEventListener("hashchange", () => {
  const request = readAddress();
  if (request) {
    fitInput();
    updateInput();
    followAddress(request);
  }
});

async function restoreRuns() {
  const addressed = readAddress();
  fitInput();
  updateInput();
  const originalHash = location.hash;
  try {
    const {jobs} = await requestJSON("/api/searches");
    // A submit can race the initial list request. Rows submitted in this tab stay first.
    for (const job of jobs) {
      if (runs.some(run => run.id === job.id)) continue;
      updateJob(makeRun(normalizeRequest(job.request), true), job);
    }
  } catch { /* A local server can still accept a new submission. */ }
  if (opened || location.hash !== originalHash) return;
  if (addressed) followAddress(addressed);
  else if (runs.length) openRun(runs[0]);
}
restoreRuns();
