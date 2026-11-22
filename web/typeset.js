"use strict";

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
function isNumeric(value) {
  return typeof value === "number" || typeof value === "bigint";
}
function plural(count, single, multiple) {
  return count + " " + (count === 1 ? single : multiple);
}
function signed(number) {
  return String(number).replace(/^-/, "−");
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
  if (["sum of squares", "transverse composition", "two smooth branches"].includes(stratum.model)) locus.append("Generic points of ");
  if (!Array.isArray(stratum.coordinates)) return el("span", "", stratum.model || "Recorded stratum");
  const equations = stratum.coordinates.join(" = ") + " = 0";
  locus.append(typeset(equations));
  if (stratum.model === "constant transverse type") locus.append(" away from the origin");
  if (stratum.model === "X^a + Y^b" || stratum.model === "X*Y + Z^m") {
    for (const name of stratum.parameters || []) locus.append(", ", typeset(name), " ≠ 0");
  }
  return locus;
}
