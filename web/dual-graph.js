"use strict";

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

function drawGraph(vertices, cells, branches, toggleVertex, registerHighlight) {
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
    if (sizeKey) facts.push(sizeKey.replaceAll("_", " ") + " " + v[sizeKey]);
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
    const toggle = () => toggleVertex(id);
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
    entry(symbol, sizeKey.replaceAll("_", " "));
  }
  if (arrows.size) entry("↑", "strict transform");
  if (faces.length) entry("▲", "triple intersection");
  if (cells.some((cell) => (cell.vertices || []).length > 3)) entry("", "2-skeleton only");
  key.style.maxWidth = Math.max(240, box[2]) + "px";
  figure.append(scroll);
  if (key.childElementCount) figure.append(key);
  registerHighlight((id) => {
    for (const [key, group] of groups) {
      group.classList.toggle("selected", key === id);
      group.setAttribute("aria-pressed", String(key === id));
    }
  });
  return figure;
}
