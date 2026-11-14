"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("algebra-form");
const operation = byId("operation");
const polynomial = byId("polynomial");
const polynomialLabels = {
  gcd: "First polynomial",
  factor: "Polynomial to factor",
  "normal-form": "Polynomial to reduce",
  "local-normal-form": "Polynomial to reduce",
  membership: "Polynomial to test",
  saturation: "Saturating polynomial f",
  "local-jacobian-algebra": "Hypersurface polynomial f with f(p) = 0",
  "milnor-number": "Hypersurface polynomial f with f(p) = 0",
  "tjurina-number": "Hypersurface polynomial f with f(p) = 0",
};
const polynomialInputs = { gcd: "(x+y)*(x+1)", factor: "-3/2*(x-y)^2*(x+y)", "normal-form": polynomial.value, "local-normal-form": polynomial.value, membership: "y^2-1", saturation: "x", hypersurface: "x^2+y^3" };
const quotientDescriptions = {
  "quotient-basis": "Find a monomial basis of the quotient by I over ℚ, including points away from the origin.",
  "local-quotient-basis": "Find a basis of the quotient by I in coordinates centered at the chosen point p.",
  "local-jacobian-algebra": "In the local ring at p, take the quotient by the partial derivatives of f.",
  "milnor-number": "Compute the dimension over ℚ of the local quotient by the partial derivatives of f.",
  "tjurina-number": "Compute the dimension over ℚ of the local quotient by f and its partial derivatives.",
  "local-standard-basis": "Compute a standard basis of the ideal in the local ring at the chosen point.",
  "local-normal-form": "Reduce f modulo the ideal in the local ring at the chosen point.",
  "hilbert-samuel-function": "For the local quotient A at p, compute lengths through each chosen degree and dimensions of its graded pieces. Here m is the maximal ideal of A.",
  "hilbert-samuel-multiplicity": "Compute the Hilbert–Samuel multiplicity of the local quotient A at p with respect to its maximal ideal.",
};
const session = new AlgebraSession();
let activeOperation = operation.value;
let running = false;
let submitted = null;
let responseText = null;

function addRow(group, addButton, value = "") {
  const row = document.createElement("div");
  row.className = "generator-row";
  const input = document.createElement("input");
  input.type = "text";
  input.required = true;
  input.autocomplete = "off";
  input.spellcheck = false;
  input.setAttribute("autocapitalize", "off");
  input.value = value;
  const remove = document.createElement("button");
  remove.className = "button";
  remove.type = "button";
  remove.textContent = "×";
  row.append(input, remove);
  group.append(row);
  labelRows(group);
  remove.addEventListener("click", () => {
    const next = row.nextElementSibling || row.previousElementSibling;
    row.remove();
    labelRows(group);
    if (next) next.querySelector("input").focus();
    else addButton.focus();
    markChanges();
  });
  return input;
}

function labelRows(group) {
  const name = group.id === "generators" ? "Generator" : "Second ideal generator";
  Array.from(group.children).forEach((row, index) => {
    const label = name + " " + (index + 1);
    row.querySelector("input").setAttribute("aria-label", label);
    row.querySelector("button").setAttribute("aria-label", "Remove " + label.toLowerCase());
  });
}

function variables() {
  const names = byId("variables").value.trim();
  return names ? names.split(",").map((name) => name.trim()) : [];
}

function updateEliminationChoices() {
  const group = byId("eliminate");
  const previous = new Map(Array.from(group.querySelectorAll("input"), (input) => [input.value, input.checked]));
  group.replaceChildren();
  for (const name of new Set(variables().filter(Boolean))) {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = name;
    checkbox.checked = previous.get(name) ?? (!previous.size && !group.children.length);
    label.append(checkbox, " " + name);
    group.append(label);
  }
  if (!group.children.length) group.textContent = "Enter variables above to choose which to eliminate.";
}

function updateOperation() {
  const polynomialInput = (op) => AlgebraSession.hypersurfaceOperations.has(op) ? "hypersurface" : op;
  if (activeOperation in polynomialLabels) polynomialInputs[polynomialInput(activeOperation)] = polynomial.value;
  activeOperation = operation.value;
  const needsPolynomial = activeOperation in polynomialLabels;
  showOperandGroup("polynomial-field", needsPolynomial);
  if (needsPolynomial) {
    byId("polynomial-label").textContent = polynomialLabels[activeOperation];
    polynomial.value = polynomialInputs[polynomialInput(activeOperation)];
  }
  byId("saturation-help").hidden = activeOperation !== "saturation";
  const needsTwoPolynomials = activeOperation === "gcd";
  const needsFirstIdeal = !needsTwoPolynomials && activeOperation !== "factor" && !AlgebraSession.hypersurfaceOperations.has(activeOperation);
  showOperandGroup("first-ideal", needsFirstIdeal);
  showOperandGroup("other-polynomial-field", needsTwoPolynomials);
  byId("gcd-help").hidden = !needsTwoPolynomials;
  byId("zero-ideal-help").hidden = !needsFirstIdeal;
  const description = byId("quotient-description");
  description.textContent = quotientDescriptions[activeOperation] || "";
  description.hidden = !description.textContent;
  if (description.hidden) operation.removeAttribute("aria-describedby");
  else operation.setAttribute("aria-describedby", description.id);
  byId("order-label").textContent = AlgebraSession.pointOperations.has(activeOperation) ? "Local order tie-breaker" : "Monomial order";
  showOperandGroup("local-point-field", AlgebraSession.pointOperations.has(activeOperation));
  showOperandGroup("degrees-field", activeOperation === "hilbert-samuel-function");
  const needsIdeal = ["intersection", "quotient"].includes(activeOperation);
  showOperandGroup("second-ideal", needsIdeal);
  if (needsIdeal) {
    byId("second-ideal-help").textContent = activeOperation === "quotient"
      ? "Computes I : J, the ideal quotient." : "Computes I ∩ J, the intersection.";
  }
  showOperandGroup("elimination-field", activeOperation === "eliminate");
  byId("result-name").required = activeOperation === "ideal";
  byId("result-name").placeholder = activeOperation === "ideal" ? "Ideal name" : "Optional name";
  updateSources();
  markChanges();
}

function inputRing() {
  return { field: "QQ", variables: variables(), order: byId("order").value };
}

function ringText(ring) {
  const field = ring.field === "QQ" ? "ℚ" : ring.field;
  return field + (ring.variables.length ? "[" + ring.variables.join(", ") + "]" : "") + " · " + ring.order;
}

function updateSources() {
  const ring = inputRing();
  byId("point-label").textContent = ring.variables.length ? "Coordinates (" + ring.variables.join(", ") + ")" : "Point in zero coordinates";
  byId("point-help").textContent = ring.variables.length
    ? "Enter comma-separated integers or fractions. Leave blank for the origin."
    : "The unique point has no coordinates. Enter () to specify it explicitly, or leave blank for the origin.";
  const ideals = ["ideal", "basis", "local-jacobian-algebra"];
  for (const [source, group, kinds] of [
    ["ideal-source", "first-ideal-inputs", AlgebraSession.localBasisInputs.has(operation.value) ? [...ideals, "local-standard-basis"] : ideals],
    ["second-ideal-source", "second-ideal-inputs", ideals],
    ["polynomial-source", "polynomial", ["polynomial"]],
    ["other-polynomial-source", "other-polynomial", ["polynomial"]],
  ]) {
    const select = byId(source);
    const previous = select.value;
    select.replaceChildren(select.options[0]);
    for (const [name, { result, owner }] of session.objects) {
      if (!kinds.includes(result.type)) continue;
      let kind = result.type;
      if (owner) kind = result.polynomial;
      if (kind === "local-jacobian-algebra") kind = "Jacobian ideal (partial derivatives)";
      if (kind === "local-standard-basis") kind = "local standard basis " + scopeText(result);
      const option = new Option(name + " · " + kind + " · " + ringText(result.ring), name);
      option.disabled = !AlgebraSession.sameRing(result.ring, ring);
      select.add(option);
      // Preserve the selection on a ring edit. Submission then explains the
      // mismatch rather than silently reverting to different literal operands.
      if (name === previous) option.selected = true;
    }
    byId(source + "-field").hidden = select.options.length === 1;
    showOperandGroup(group, !select.value);
  }
}

function showOperandGroup(id, show) {
  const control = byId(id);
  control.hidden = !show;
  control.disabled = !show;
}

function rows(group) {
  return Array.from(byId(group).querySelectorAll("input"), (input) => input.value.trim());
}

function readRequest() {
  const operands = {};
  const sources = {};
  function operand(key, select, literal) {
    const name = byId(select).value;
    if (name) sources[key] = name;
    else operands[key] = literal();
  }
  if (!byId("first-ideal").disabled) operand("ideal", "ideal-source", () => rows("generators"));
  if (!byId("second-ideal").disabled) operand("other_ideal", "second-ideal-source", () => rows("second-generators"));
  if (operation.value in polynomialLabels) operand("polynomial", "polynomial-source", () => polynomial.value.trim());
  if (!byId("other-polynomial-field").disabled) {
    operand("other_polynomial", "other-polynomial-source", () => byId("other-polynomial").value.trim());
  }
  if (operation.value === "eliminate") {
    operands.variables = Array.from(byId("eliminate").querySelectorAll("input:checked"), (input) => input.value);
  }
  if (AlgebraSession.pointOperations.has(operation.value)) {
    const point = byId("local-point").value.trim();
    if (point) operands.point = point === "()" ? [] : point.split(",").map(value => value.trim());
  }
  if (operation.value === "hilbert-samuel-function") {
    const degrees = byId("degrees").value.trim();
    if (degrees) operands.degrees = degrees === "[]" ? [] : degrees.split(",").map(value => value.trim());
  }
  return {
    operation: operation.value,
    ring: inputRing(),
    operands,
    sources,
    name: byId("result-name").value.trim(),
    compute: byId("compute").value,
  };
}

function markChanges() {
  const current = readRequest();
  byId("stale").hidden = !submitted || running || JSON.stringify(current) === JSON.stringify(submitted);
  byId("algebra-problem").classList.toggle("compact", Boolean(submitted) && !running && byId("stale").hidden);
  showProblem(current);
}

for (const [group, button, defaults] of [
  ["generators", "add-generator", ["x^2-1", "y-x"]],
  ["second-generators", "add-second-generator", ["x"]],
]) {
  const container = byId(group);
  const addButton = byId(button);
  for (const value of defaults) addRow(container, addButton, value);
  addButton.addEventListener("click", () => {
    addRow(container, addButton).focus();
    markChanges();
  });
}
operation.addEventListener("change", updateOperation);
byId("variables").addEventListener("input", updateEliminationChoices);
byId("variables").addEventListener("input", updateSources);
byId("ideal-source").addEventListener("change", () => {
  const saved = session.objects.get(byId("ideal-source").value);
  if (!AlgebraSession.pointOperations.has(operation.value) || !saved) return;
  if (["local-standard-basis", "local-jacobian-algebra"].includes(saved.result.type)) {
    byId("local-point").value = saved.result.point ? saved.result.point.join(", ") || "()" : "";
  }
});
for (const id of ["order", "ideal-source", "second-ideal-source", "polynomial-source", "other-polynomial-source"]) {
  byId(id).addEventListener("change", updateSources);
}
form.addEventListener("input", markChanges);
form.addEventListener("change", markChanges);
const resultTitles = {
  factor: "Polynomial factorization",
  gcd: "Monic polynomial GCD",
  ideal: "Ideal",
  groebner: "Reduced Gröbner basis",
  "normal-form": "Normal form",
  membership: "Ideal membership",
  eliminate: "Elimination ideal",
  intersection: "Intersection",
  quotient: "Ideal quotient",
  saturation: "Saturation",
  dimension: "Krull dimension",
  "quotient-basis": "Quotient basis",
  "local-quotient-basis": "Local quotient basis",
  "local-jacobian-algebra": "Local Jacobian algebra",
  "milnor-number": "Milnor number",
  "tjurina-number": "Tjurina number",
  "local-standard-basis": "Local standard basis",
  "local-normal-form": "Local normal form",
  "hilbert-samuel-function": "Hilbert–Samuel function",
  "hilbert-samuel-multiplicity": "Hilbert–Samuel multiplicity",
};

// Preserve every source token in the DOM while setting powers, fractions and
// variables as mathematical text. Long source/result strings stay selectable.
function mathSpan(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function expressionText(source) {
  const math = mathSpan("span", "math-expression");
  if (source.length > 720) {
    math.textContent = source;
    return math;
  }
  const tokens = source.match(/[A-Za-z][A-Za-z_0-9]*|[0-9]+|\*\*|\s+|./gu) || [];
  function atom(token) {
    return /^[A-Za-z]/.test(token) ? mathSpan("i", "", token) : token;
  }
  for (let i = 0; i < tokens.length; ++i) {
    const token = tokens[i];
    const next = tokens[i + 1];
    if ((token === "^" || token === "**") && next && /^\d+$/.test(next)) {
      math.append(mathSpan("span", "power-mark", token), mathSpan("sup", "", next));
      ++i;
    } else if (/^[A-Za-z0-9]/.test(token) && next === "/" && /^\d+$/.test(tokens[i + 2] || "")
        && !["^", "**"].includes(tokens[i + 3])) {
      const fraction = mathSpan("span", "math-fraction");
      const numerator = mathSpan("span");
      numerator.append(atom(token));
      fraction.append(numerator, mathSpan("span", "fraction-bar", "/"), mathSpan("span", "", tokens[i + 2]));
      math.append(fraction);
      i += 2;
    } else if (["+", "-", "*"].includes(token)) {
      const sign = { "+": "plus", "-": "minus", "*": "product" }[token];
      math.append(mathSpan("span", "math-operator math-" + sign, token));
    } else math.append(atom(token));
  }
  return math;
}

const problemNotation = {
  factor: "factor(<i>f</i>)",
  "normal-form": "NF<sub>I</sub>(<i>f</i>)",
  groebner: "GB(<i>I</i>)",
  membership: "<i>f</i> ∈ <i>I</i> ?",
  ideal: "⟨<i>g</i><sub>1</sub>, …, <i>g</i><sub>m</sub>⟩",
  quotient: "<i>I</i> : <i>J</i>",
  saturation: "<i>I</i> : <i>f</i><sup>∞</sup>",
  intersection: "<i>I</i> ∩ <i>J</i>",
  dimension: "dim(<i>R</i>/<i>I</i>)",
  gcd: "gcd(<i>f</i>, <i>g</i>)",
  "quotient-basis": "<i>R</i>/<i>I</i>",
  "local-quotient-basis": "<i>R</i><sub><i>p</i></sub>/<i>I R</i><sub><i>p</i></sub>",
  "local-jacobian-algebra": "<i>R</i><sub><i>p</i></sub>/J(<i>f</i>)",
  "milnor-number": "μ<sub><i>p</i></sub>(<i>f</i>)",
  "tjurina-number": "τ<sub><i>p</i></sub>(<i>f</i>)",
  "local-standard-basis": "SB<sub><i>p</i></sub>(<i>I</i>)",
  "local-normal-form": "NF<sub><i>I,p</i></sub>(<i>f</i>)",
  "hilbert-samuel-function": "H<sub><i>A,p</i></sub>(<i>n</i>)",
  "hilbert-samuel-multiplicity": "e<sub><i>p</i></sub>(<i>A</i>)",
};

function showProblem(request) {
  const { ring, operation, operands, sources } = request;
  const rows = byId("problem-equations");
  function operand(label, key) {
    const row = mathSpan("div", "equation-row");
    const rhs = mathSpan("div", "equation-rhs");
    if (sources[key]) {
      const name = sources[key];
      rhs.append(mathSpan("i", "", name));
      if (session.objects.get(name).result.type === "local-jacobian-algebra") rhs.append(".generators");
    }
    else if (Array.isArray(operands[key])) {
      const polynomials = operands[key];
      const preview = polynomials.slice(0, 4);
      rhs.append("⟨");
      preview.forEach((f, i) => {
        if (i) rhs.append(", ");
        rhs.append(expressionText(f || "□"));
      });
      if (!polynomials.length) rhs.append("0");
      if (preview.length < polynomials.length) {
        rhs.append(", …", mathSpan("span", "generator-count", " (" + polynomials.length + " generators)"));
      }
      rhs.append("⟩");
    } else rhs.append(expressionText(operands[key] || "□"));
    row.append(mathSpan("i", "equation-label", label), mathSpan("span", "equals", "="), rhs);
    rows.append(row);
  }
  const title = byId("problem-title");
  byId("problem-operation").textContent = byId("operation").selectedOptions[0].textContent;
  if (operation === "eliminate") {
    const survivors = ring.variables.filter(v => !operands.variables.includes(v));
    title.replaceChildren(mathSpan("i", "", "I"), " ∩ ℚ", survivors.length ? "[" + survivors.join(", ") + "]" : "");
  } else title.innerHTML = problemNotation[operation]; // Static notation; operands use text nodes.
  rows.replaceChildren();
  if ("polynomial" in operands || sources.polynomial) operand("f", "polynomial");
  if ("other_polynomial" in operands || sources.other_polynomial) operand("g", "other_polynomial");
  if ("ideal" in operands || sources.ideal) operand("I", "ideal");
  if ("other_ideal" in operands || sources.other_ideal) operand("J", "other_ideal");
  if (AlgebraSession.pointOperations.has(operation)) {
    const point = operands.point ?? ring.variables.map(() => "0");
    const row = mathSpan("div", "equation-row");
    row.append(mathSpan("i", "equation-label", "p"), mathSpan("span", "equals", "="),
      mathSpan("span", "equation-rhs", "(" + point.join(", ") + ")"));
    rows.append(row);
  }
  if (operation === "hilbert-samuel-function") {
    const degrees = operands.degrees ?? ["0", "1", "2", "3", "4", "5"];
    const row = mathSpan("div", "equation-row");
    row.append(mathSpan("i", "equation-label", "n"), mathSpan("span", "equals", "="),
      mathSpan("span", "equation-rhs", "[" + degrees.join(", ") + "]"));
    rows.append(row);
  }
  byId("problem-ring").textContent = "R = " + ringText(ring).split(" · ")[0];
  const selected = byId("order").selectedOptions[0].textContent;
  const order = AlgebraSession.pointOperations.has(operation) ? "Lowest total degree first · " + selected.toLowerCase() + " tie-breaker" : selected + " order";
  byId("problem-order").textContent = ring.variables.length > 1 ? order + " · " + ring.variables.join(" > ") : order;
}

updateEliminationChoices();
updateOperation();

function showQuotient(value, result) {
  value.replaceChildren();
  function entry(label, content) {
    value.append(mathSpan("h3", "quotient-label", label), content);
  }
  entry("Dimension over ℚ", result.finite ? String(result.dimension) : "Infinite");
  if (result.finite) {
    const label = result.scope === "point" ? "Basis in coordinates centered at p" : "Monomial basis";
    entry(label, result.basis.length ? expressionText(result.basis.join(", ")) : "∅ (empty basis)");
  }
  if (result.type === "local-jacobian-algebra") {
    entry("Jacobian ideal generators", result.generators.length ? expressionText(result.generators.join(", ")) : "(0)");
  }
}

function showFactorization(value, result) {
  value.replaceChildren(mathSpan("h3", "quotient-label", "Rational unit"), expressionText(result.unit));
  if (!result.factors.length) {
    value.append(mathSpan("p", "helper", "This constant has no irreducible factors."));
    return;
  }
  const table = resultTable(["Irreducible factor", "Multiplicity"]);
  const rows = table.createTBody();
  for (const { polynomial, multiplicity } of result.factors) {
    const row = rows.insertRow();
    row.insertCell().append(expressionText(polynomial));
    row.insertCell().textContent = String(multiplicity);
  }
  value.append(table);
}

function resultTable(headings) {
  const table = document.createElement("table");
  table.className = "algebra-table";
  const header = table.createTHead().insertRow();
  for (const heading of headings) {
    const column = mathSpan("th", "", heading);
    column.scope = "col";
    header.append(column);
  }
  return table;
}

function showHilbertSamuel(value, result) {
  const multiplicity = result.type === "integer" ? result.value : result.multiplicity;
  value.replaceChildren(mathSpan("h3", "quotient-label", "Hilbert–Samuel multiplicity"), String(multiplicity),
    mathSpan("h3", "quotient-label", "Krull dimension"), String(result.dimension));
  if (result.type !== "hilbert-samuel-function") return;
  if (!result.degrees.length) {
    value.append(mathSpan("p", "helper", "No degrees requested."));
    return;
  }
  const table = resultTable(["Degree n", "Cumulative length ℓ(A/mⁿ⁺¹)", "Graded dimension dim(mⁿ/mⁿ⁺¹)"]);
  const body = table.createTBody();
  result.degrees.forEach((degree, i) => {
    const row = body.insertRow();
    for (const entry of [degree, result.values[i], result.graded_values[i]]) row.insertCell().textContent = String(entry);
  });
  value.append(table);
}

function scopeText(result) {
  if (result.scope === "origin") return "at the origin";
  if (result.scope === "global") return "global quotient";
  if (result.scope === "point") {
    const coordinates = result.ring.variables.map((name, i) => name + " = " + result.point[i]);
    return "at (" + coordinates.join(", ") + ")";
  }
  return "";
}

function showLocalNormalForm(value, result) {
  const kind = result.reduction === "weak" ? "Weak normal form" : "Standard normal form";
  value.replaceChildren(mathSpan("h3", "quotient-label", kind));
  if (result.unit === "1") {
    value.append(expressionText(result.remainder));
  } else {
    const fraction = mathSpan("span", "math-fraction local-representative");
    fraction.append(expressionText(result.remainder), mathSpan("span", "fraction-bar", " / "), expressionText(result.unit));
    value.append(fraction, mathSpan("p", "helper", "This fraction represents f modulo I. Its numerator is the remainder and its denominator is a unit with value 1 at this point."));
  }
  if (result.reduction === "weak") {
    value.append(mathSpan("p", "helper", "The leading monomial of a nonzero remainder is irreducible. Later terms can still be reducible."));
  }
}

function showResult(record) {
  const { result } = record;
  const value = byId("result-value");
  switch (result.type) {
    case "basis":
    case "local-standard-basis":
    case "ideal":
      if (!result.generators.length) {
        value.textContent = result.type === "ideal" ? "(0)" : "∅ (empty basis)";
      } else {
        const list = document.createElement("ul");
        list.setAttribute("aria-label", "Generators");
        for (const generator of result.generators) {
          const item = document.createElement("li");
          item.append(expressionText(generator));
          list.append(item);
        }
        value.replaceChildren(list);
      }
      break;
    case "polynomial":
      value.replaceChildren(expressionText(result.polynomial));
      break;
    case "factorization":
      showFactorization(value, result);
      break;
    case "local-normal-form":
      showLocalNormalForm(value, result);
      break;
    case "quotient-basis":
    case "local-jacobian-algebra":
      showQuotient(value, result);
      break;
    case "hilbert-samuel-function":
      showHilbertSamuel(value, result);
      break;
    case "boolean":
    case "integer":
      if (record.operation === "hilbert-samuel-multiplicity") showHilbertSamuel(value, result);
      else value.textContent = result.finite === false ? "Infinite" : String(result.value);
      break;
    default:
      throw new Error("The service returned an unknown result type.");
  }

  const ringName = [ringText(result.ring), scopeText(result)].filter(Boolean).join(" · ");
  byId("result-ring").textContent = ringName;
  byId("result-title").textContent = [submitted.name, resultTitles[record.operation]].filter(Boolean).join(" · ");
  byId("result").hidden = false;
  byId("completion").textContent = resultTitles[record.operation] + " complete. Result in " + ringName + ".";
}

function addDetail(list, label, value) {
  const term = document.createElement("dt");
  term.textContent = label;
  const definition = document.createElement("dd");
  definition.textContent = value;
  list.append(term, definition);
}

function showExecution(execution) {
  const summary = byId("execution");
  addDetail(summary, "Requested", execution.requested);
  addDetail(summary, "Executed on", execution.device);
  if (execution.backend) addDetail(summary, "Backend", execution.backend);
  const saved = session.objects.get(submitted.sources.ideal);
  if (saved?.result.type === "basis" && ["normal-form", "membership"].includes(submitted.operation)) {
    addDetail(summary, "Saved basis", submitted.sources.ideal + " reused");
  }
  const reasons = execution.fallback_reasons || [];
  for (const reason of reasons) addDetail(summary, "Fallback", reason);
  if (execution.note && execution.note !== reasons.join(" ")) {
    addDetail(summary, "Execution note", execution.note);
  }
  summary.hidden = false;

  const stages = byId("stages");
  stages.replaceChildren();
  for (const stage of execution.stages) {
    const details = [stage.backend, stage.device?.name, stage.algorithm];
    for (const key of ["batches", "primes", "pairs", "terms", "bases", "generators", "known", "divisors"]) {
      if (stage[key] !== undefined) details.push(key + " " + stage[key]);
    }
    const name = [stage.label, stage.operation].filter(Boolean).join(" · ").replaceAll("_", " ");
    addDetail(stages, name, details.filter(Boolean).join(" · "));
  }
  byId("stage-summary").textContent = "Execution stages (" + execution.stages.length + ")";
  byId("stage-details").hidden = !execution.stages.length;
}

function showReplay(replay) {
  byId("command").textContent = replay.command;
  byId("script").textContent = replay.script;
  byId("script-invocation").textContent = "./atlas run " + submitted.operation + ".atlas";
  byId("replay").hidden = false;
}

function sessionScript() {
  const preview = byId("session-script");
  // Keep one copy in the preview. Read the text only when needed for export.
  if (!preview.hasChildNodes()) preview.textContent = session.script();
  return preview;
}

byId("session-replay").addEventListener("toggle", () => {
  if (byId("session-replay").open) sessionScript();
});

function showSession(name, result) {
  if (name) {
    const item = document.createElement("li");
    const label = document.createElement("strong");
    label.textContent = name;
    item.append(label, " · " + [result.type, ringText(result.ring), scopeText(result)].filter(Boolean).join(" · "));
    byId("session-items").append(item);
    updateSources();
  }
  byId("session-script").replaceChildren();
  if (byId("session-replay").open) sessionScript();
  byId("session-history").hidden = false;
  byId("session-status").textContent = "";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (running) return;
  submitted = readRequest();
  running = true;
  responseText = null;
  const run = byId("run");
  const fromSubmitButton = event.submitter === run
    && (document.activeElement === run || document.activeElement === document.body);
  byId("run").disabled = true;
  for (const id of ["result", "error", "stale", "replay", "export", "execution", "stage-details"]) byId(id).hidden = true;
  byId("execution").replaceChildren();
  byId("copy-status").textContent = "";
  byId("completion").textContent = "";
  byId("pending").textContent = "Computing " + resultTitles[submitted.operation].toLowerCase() + "…";
  byId("pending").hidden = false;

  try {
    session.validateName(submitted.name);
    const endpoint = byId("exhaustive").checked ? "/api/algebra/exhaustive" : "/api/algebra";
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(session.request(submitted)),
    });
    const body = await response.text();
    let record;
    try { record = readRecord(body); } catch (error) {
      if (error.name === "ExactIntegerError") {
        responseText = body;
        byId("export").hidden = false;
        throw error;
      }
      throw new Error("The algebra service returned an unreadable response (HTTP " + response.status + ").");
    }
    if (record.status === "ok" || record.status === "error") {
      responseText = body;
      byId("export").hidden = false;
      if (record.execution) showExecution(record.execution);
      if (record.replay) showReplay(record.replay);
    }
    if (!response.ok || record.status !== "ok") {
      const reason = typeof record.error === "string" ? record.error : record.error?.message;
      throw new Error(reason || "The algebra service returned HTTP " + response.status + ".");
    }
    showResult(record);
    session.record(submitted, record);
    showSession(submitted.name, record.result);
  } catch (error) {
    const message = error instanceof TypeError && !responseText ? "Could not read a response from the local algebra service." : error.message;
    byId("error").textContent = "Calculation failed. " + message;
    byId("error").hidden = false;
  } finally {
    running = false;
    byId("pending").hidden = true;
    byId("run").disabled = false;
    markChanges();
    if (fromSubmitButton && window.matchMedia("(max-width: 820px)").matches
        && byId("stale").hidden && (document.activeElement === run || document.activeElement === document.body)) {
      const bounds = run.getBoundingClientRect();
      if (bounds.bottom > 0 && bounds.top < window.innerHeight) {
        (byId("error").hidden ? byId("result") : byId("error")).scrollIntoView({ block: "start" });
      }
    }
  }
});

function download(text, filename, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

byId("export").addEventListener("click", () => {
  if (responseText !== null) download(responseText, submitted.operation + ".json", "application/json");
});
byId("download-script").addEventListener("click", () => {
  download(byId("script").textContent, submitted.operation + ".atlas", "text/plain;charset=utf-8");
});
byId("download-session").addEventListener("click", () => {
  download(sessionScript().textContent, "session.atlas", "text/plain;charset=utf-8");
});

for (const [button, id, status] of [
  ["copy-command", "command", "copy-status"],
  ["copy-script", "script", "copy-status"],
  ["copy-session", "session-script", "session-status"],
]) {
  byId(button).addEventListener("click", async () => {
    const source = id === "session-script" ? sessionScript() : byId(id);
    try {
      await navigator.clipboard.writeText(source.textContent);
      byId(status).textContent = id === "command" ? "Command copied." : "Script copied.";
    } catch {
      const range = document.createRange();
      range.selectNodeContents(source);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      byId(status).textContent = "Text selected. Use your browser’s Copy command.";
    }
  });
}
