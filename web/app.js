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

function recordJSON(value, space) {
  return JSON.stringify(value, (_key, field) => typeof field === "bigint" ? JSON.rawJSON(String(field)) : field, space);
}
function stringify(value) {
  return typeof value === "string" ? value : recordJSON(value, 2);
}
function code(value) {
  return el("code", "", typeof value === "string" ? value : recordJSON(value));
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
// Remember which parts of a resolution the reader wants visible, including
// when a follow-up computation redraws the report or another run is opened.
const foldPreferences = new Map();
function foldedSection(panel, title, count, initiallyOpen = false, key = title, heading = "h3") {
  const details = el("details", "fold");
  const summary = el("summary");
  const number = el("span", "count", count === undefined ? "" : String(count));
  const body = el("div", "fold-body");
  summary.append(el(heading, "", title), number);
  details.append(summary, body);
  details.open = foldPreferences.get(key) ?? initiallyOpen;
  details.addEventListener("toggle", () => {
    if (details.isConnected) foldPreferences.set(key, details.open);
  });
  panel.append(details);
  return { body, number, summary };
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

  const vertexTable = foldedSection(side, "0-dimensional cells · vertices", vertices.length, true, "vertices", "h4");
  const keys = columnsOf(vertices, ["total_multiplicity", "multiplicity", "ambient_order", "self_intersection", "genus", "dimension"], ["id", "label"]);
  const countCharts = chartCount.size > 0;
  const rows = [];
  vertexTable.body.append(table(["Divisor", ...keys.map((key) => key === "dimension" ? "Divisor dimension" : labelFor(key)), ...(countCharts ? ["Charts"] : [])], vertices, (v, index, tr) => {
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
    const title = dimension + "-dimensional cells";
    const cellTable = foldedSection(side, title, group.length, true, title, "h4");
    cellTable.body.append(table(["Cell ID", "Intersecting divisors", ...(hasFaces ? ["Boundary faces"] : []), ...extras.map(labelFor)],
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
  const section = $("panel-complex");
  section.replaceChildren();
  if (!dual || (dual.status !== "complete" && dual.status !== "partial" && data.status !== "resolved")) return;

  const vertices = dual.vertices || [];
  const cells = dual.cells || [];
  const isGraph = cells.every((cell) => cellDimension(cell) <= 1);
  const { body: panel, summary } = foldedSection(section, isGraph ? "Dual graph" : "Dual complex",
    undefined, true, "dual complex");
  if (dual.convention) summary.title = dual.convention;
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
  const figure = drawGraph(vertices, cells, branches,
    id => selectComponent(selected === id ? null : id), update => selectionViews.push(update));
  const controls = el("div", "complex-switch");
  controls.setAttribute("role", "group");
  controls.setAttribute("aria-label", "Dual complex view");
  const picture = el("button", "", "Picture");
  const records = el("button", "", "Data");
  picture.type = records.type = "button";
  controls.append(picture, records);
  panel.append(controls);

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
function showZeta(data) {
  const section = $("panel-zeta");
  section.replaceChildren();
  const zeta = data.monodromy_zeta;
  if (!zeta || typeof zeta !== "object") return;
  const { body: panel } = foldedSection(section, "Monodromy zeta function at the origin", undefined, true);
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
    const nearby = foldedSection(section, "Monodromy zeta functions on certified nearby strata", strata.length, true);
    nearby.body.append(table(["Stratum", "Local zeta function"], strata, stratum => {
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

/* The topological zeta function of Denef and Loeser, which the engine sums
   over the strata of the total transform. */

function minus(text) {
  return String(text).replace(/-/g, "−");
}
function showTopological(data) {
  const section = $("panel-zeta");
  const zeta = data.topological_zeta;
  if (!zeta || typeof zeta !== "object") return;
  const { body: panel } = foldedSection(section, "Topological zeta function", undefined, true);
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
  showMonodromyCheck(section, data);
}

function showMonodromyCheck(section, data) {
  const check = data.monodromy_check;
  if (data.monodromy_zeta?.status !== "computed" || !check) return;
  const { body: panel } = foldedSection(section, "Topological monodromy conjecture", undefined, true);
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
