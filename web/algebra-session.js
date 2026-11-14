"use strict";

// A successful step keeps the names it used, even after the form changes.
// Mathematical values and basis references always come from the service.
class AlgebraSession {
  constructor() {
    this.steps = [];
    this.objects = new Map();
  }

  static sameRing(a, b) {
    return a.field === b.field && a.order === b.order &&
      a.variables.length === b.variables.length && a.variables.every((v, i) => v === b.variables[i]);
  }

  validateName(name) {
    if (!name) return;
    if (!/^[a-zA-Z][a-zA-Z0-9_]*$/.test(name)) throw new Error("Use a name starting with a letter, followed by letters, digits or underscores.");
    if (this.objects.has(name)) throw new Error("A session object already has the name “" + name + "”. Choose a new name.");
  }

  request(form) {
    const operands = { ...form.operands };
    let owner;
    for (const [key, name] of Object.entries(form.sources)) {
      const saved = this.objects.get(name);
      if (!saved || !AlgebraSession.sameRing(saved.result.ring, form.ring)) {
        throw new Error("“" + name + "” belongs to a different ring. Check the ordered variables and monomial order.");
      }
      if (owner && owner !== saved.space) {
        throw new Error("These objects have different coordinate spaces. Choose objects from the same declaration or elimination.");
      }
      owner = saved.space;
      if (saved.result.type === "local-standard-basis") {
        if (key !== "ideal" || form.operation !== "local-normal-form") {
          throw new Error("Choose a local standard basis as the ideal for a local normal form.");
        }
        const point = operands.point ?? form.ring.variables.map(() => "0");
        if (!AlgebraSession.samePoint(point, saved.result.point)) {
          throw new Error("“" + name + "” was computed at (" + saved.result.point.join(", ") +
            "). Enter that point, or choose generators for a different point.");
        }
        operands.point = saved.result.point;
      }
      if (key === "ideal" && saved.result.type === "basis" && ["normal-form", "membership"].includes(form.operation)) {
        if (!saved.basisRef) throw new Error("Compute and save this basis again to obtain a reference from the server.");
        delete operands.ideal;
        operands.basis_ref = saved.basisRef;
      } else {
        operands[key] = saved.result.type === "polynomial" ? saved.result.polynomial : saved.result.generators;
      }
    }
    return { operation: form.operation, ring: form.ring, operands, compute: form.compute };
  }

  // Compare an entered rational to the server's normalized coordinates exactly.
  // The wire format accepts a signed numerator and an optional signed denominator.
  static samePoint(input, normalized) {
    return input.length === normalized.length && input.every((coordinate, i) => {
      const match = /^([+-]?[0-9]+)(?:\/([+-]?[0-9]+))?$/.exec(coordinate);
      if (!match) return false;
      const denominator = BigInt(match[2] ?? "1");
      const [numerator, divisor = "1"] = normalized[i].split("/");
      return denominator !== 0n && BigInt(match[1]) * BigInt(divisor) === BigInt(numerator) * denominator;
    });
  }

  record(form, answer) {
    if (answer.result.scope === "point") {
      form = { ...form, operands: { ...form.operands, point: answer.result.point } };
    }
    this.steps.push(form);
    if (!form.name) return;

    const firstSource = Object.values(form.sources)[0];
    // Eliminations of different subsets can reach the same surviving
    // coordinates. Keep the original declaration, independent of that path.
    const origin = firstSource ? this.objects.get(firstSource).origin : form.ring;
    const survivors = answer.result.ring.variables;
    const space = survivors.length ? JSON.stringify([origin, survivors]) : "QQ";
    this.objects.set(form.name, { result: answer.result, basisRef: answer.basis_ref, origin, space });
    if (answer.result.type === "factorization") {
      answer.result.factors.forEach(({ polynomial }, index) => {
        const field = ".factors[" + index + "].polynomial";
        this.objects.set(form.name + field, {
          result: { type: "polynomial", ring: answer.result.ring, polynomial },
          owner: form.name, field, origin, space,
        });
      });
    }
  }

  script() {
    // Allocate locals across the whole transcript. A later user-chosen name
    // can otherwise shadow an earlier coordinate or a generated result name.
    const occupied = new Set([...AlgebraSession.reserved, ...this.objects.keys()]);
    const nextSuffix = new Map();
    function local(preferred, prefix) {
      let name = preferred;
      if (!name || occupied.has(name)) {
        let suffix = nextSuffix.get(prefix) || 1;
        do { name = prefix + suffix++; } while (occupied.has(name));
        nextSuffix.set(prefix, suffix);
      }
      occupied.add(name);
      return name;
    }
    const names = new Map();
    for (const [name, saved] of this.objects) {
      if (saved.owner) names.set(name, names.get(saved.owner) + saved.field);
      else names.set(name, AlgebraSession.reserved.has(name) ? local("", "saved") : name);
    }
    const rings = new Map();
    const lines = [];

    function ringKey(ring) {
      return JSON.stringify([ring.field, ring.variables, ring.order]);
    }
    function ringCoordinates(ring) {
      const key = ringKey(ring);
      if (rings.has(key)) return rings.get(key);
      const args = ring.variables.map(v => JSON.stringify(v));
      const aliases = ring.variables.map(v => local(v, "coord"));
      if (aliases.length) lines.push(aliases.join(", ") + " = coordinates(" + [...args, "coeffs=QQ"].join(", ") + ")");
      const coords = "[" + aliases.join(", ") + "]";
      rings.set(key, coords);
      return coords;
    }

    for (const form of this.steps) {
      const { operation, operands, sources, ring } = form;
      const firstSource = Object.values(sources)[0];
      const owner = firstSource && (this.objects.get(firstSource).owner || firstSource);
      const coords = owner ? names.get(owner) + ".coordinates" : ringCoordinates(ring);
      const variableArgs = "variables=" + coords;
      const quote = JSON.stringify;
      const literal = (expr) => "polynomial(" + quote(expr) + ", " + variableArgs + ")";
      const rawIdeal = (rows) => "ideal(" + [...rows.map(literal), variableArgs, "order=" + quote(ring.order)].join(", ") + ")";
      const ideal = (key) => {
        if (!sources[key]) return rawIdeal(operands[key]);
        const saved = names.get(sources[key]);
        if (operation === "ideal" || this.objects.get(sources[key]).result.type === "local-jacobian-algebra") {
          return "ideal(" + saved + ".generators, " + variableArgs + ", order=" + quote(ring.order) + ")";
        }
        return saved;
      };
      const poly = (key) => names.get(sources[key]) || literal(operands[key]);
      const resultName = form.name ? names.get(form.name) : local("", "answer");
      if (form.name && resultName !== form.name) lines.push("# " + form.name + " is stored as " + resultName);
      let expression;
      if (operation === "ideal") {
        expression = ideal("ideal");
      } else {
        let args;
        if (operation === "gcd") {
          args = [poly("polynomial"), poly("other_polynomial"), "order=" + quote(ring.order)];
        } else if (operation === "factor" || AlgebraSession.hypersurfaceOperations.has(operation)) {
          args = [poly("polynomial"), "order=" + quote(ring.order)];
        } else {
          args = [ideal("ideal")];
          if (["normal-form", "local-normal-form", "membership"].includes(operation)) args.unshift(poly("polynomial"));
          else if (operation === "saturation") args.push(poly("polynomial"));
          else if (operation === "quotient" || operation === "intersection") args.push(ideal("other_ideal"));
          else if (operation === "eliminate") args.push(...operands.variables.map(v => quote(v)));
          if (AlgebraSession.pointOperations.has(operation)) args.push("point=" + quote(operands.point));
        }
        args.push("compute=" + quote(form.compute));
        expression = operation.replaceAll("-", "_") + "(" + args.join(", ") + ")";
      }
      lines.push(resultName + " = " + expression);
      if (operation !== "ideal") lines.push("print(" + resultName + ".json)");
    }
    return lines.join("\n") + "\n";
  }
}

AlgebraSession.hypersurfaceOperations = new Set(["local-jacobian-algebra", "milnor-number", "tjurina-number"]);
AlgebraSession.pointOperations = new Set(["local-standard-basis", "local-normal-form"]);
AlgebraSession.reserved = new Set((
  "False None True and as assert async await break class continue def del elif else except finally for from " +
  "global if import in is lambda nonlocal not or pass raise return try while with yield " +
  "QQ coordinates polynomial resolve len print ideal groebner normal_form membership eliminate intersection " +
  "quotient saturation saturate dimension gcd factor algebra_request help quit exit " +
  "quotient_basis local_quotient_basis local_jacobian_algebra milnor_number tjurina_number local_standard_basis local_normal_form"
).split(" "));
