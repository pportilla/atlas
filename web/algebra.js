"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("algebra-form");
const operation = byId("operation");
const polynomial = byId("polynomial");
const polynomialLabels = {
  gcd: "First polynomial",
  "normal-form": "Polynomial to reduce",
  membership: "Polynomial to test",
  saturation: "Saturating polynomial f",
};
const polynomialInputs = { gcd: "(x+y)*(x+1)", "normal-form": polynomial.value, membership: "y^2-1", saturation: "x" };
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
  if (activeOperation in polynomialLabels) polynomialInputs[activeOperation] = polynomial.value;
  activeOperation = operation.value;
  const needsPolynomial = activeOperation in polynomialLabels;
  showOperandGroup("polynomial-field", needsPolynomial);
  if (needsPolynomial) {
    byId("polynomial-label").textContent = polynomialLabels[activeOperation];
    polynomial.value = polynomialInputs[activeOperation];
  }
  byId("saturation-help").hidden = activeOperation !== "saturation";
  const needsTwoPolynomials = activeOperation === "gcd";
  showOperandGroup("first-ideal", !needsTwoPolynomials);
  showOperandGroup("other-polynomial-field", needsTwoPolynomials);
  byId("gcd-help").hidden = !needsTwoPolynomials;
  byId("zero-ideal-help").hidden = needsTwoPolynomials;
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
  for (const [source, group, kinds] of [
    ["ideal-source", "first-ideal-inputs", ["ideal", "basis"]],
    ["second-ideal-source", "second-ideal-inputs", ["ideal", "basis"]],
    ["polynomial-source", "polynomial", ["polynomial"]],
    ["other-polynomial-source", "other-polynomial", ["polynomial"]],
  ]) {
    const select = byId(source);
    const previous = select.value;
    select.replaceChildren(select.options[0]);
    for (const [name, { result }] of session.objects) {
      if (!kinds.includes(result.type)) continue;
      const option = new Option(name + " · " + result.type + " · " + ringText(result.ring), name);
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
for (const id of ["order", "ideal-source", "second-ideal-source", "polynomial-source", "other-polynomial-source"]) {
  byId(id).addEventListener("change", updateSources);
}
form.addEventListener("input", markChanges);
form.addEventListener("change", markChanges);
const resultTitles = {
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
  "normal-form": "NF<sub>I</sub>(<i>f</i>)",
  groebner: "GB(<i>I</i>)",
  membership: "<i>f</i> ∈ <i>I</i> ?",
  ideal: "⟨<i>g</i><sub>1</sub>, …, <i>g</i><sub>m</sub>⟩",
  quotient: "<i>I</i> : <i>J</i>",
  saturation: "<i>I</i> : <i>f</i><sup>∞</sup>",
  intersection: "<i>I</i> ∩ <i>J</i>",
  dimension: "dim(<i>R</i>/<i>I</i>)",
  gcd: "gcd(<i>f</i>, <i>g</i>)",
};

function showProblem(request) {
  const { ring, operation, operands, sources } = request;
  const rows = byId("problem-equations");
  function operand(label, key) {
    const row = mathSpan("div", "equation-row");
    const rhs = mathSpan("div", "equation-rhs");
    if (sources[key]) rhs.append(mathSpan("i", "", sources[key]));
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
  byId("problem-ring").textContent = "R = " + ringText(ring).split(" · ")[0];
  const order = byId("order").selectedOptions[0].textContent + " order";
  byId("problem-order").textContent = ring.variables.length > 1 ? order + " · " + ring.variables.join(" > ") : order;
}

updateEliminationChoices();
updateOperation();

function showResult(record) {
  const { result } = record;
  const value = byId("result-value");
  switch (result.type) {
    case "basis":
    case "ideal":
      if (!result.generators.length) {
        value.textContent = result.type === "basis" ? "∅ (empty basis)" : "(0)";
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
    case "boolean":
    case "integer":
      value.textContent = String(result.value);
      break;
    default:
      throw new Error("The service returned an unknown result type.");
  }

  const ringName = ringText(result.ring);
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
    item.append(label, " · " + result.type + " · " + ringText(result.ring));
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
    try { record = JSON.parse(body); } catch {
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
