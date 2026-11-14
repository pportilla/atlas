"""The total transform of an embedded resolution: its components and its strata over the origin.

Near the fibre over the origin the total transform of f is a sum of smooth
divisors with normal crossings. Each is given a label: an exceptional
divisor, or the strict transform of a factor of f. A label may stand for
several disjoint irreducible divisors with the same numbers (conjugate
ones, for instance); this changes none of the formulas below.

The data computed here are, for every label, the order N of f along the
divisor and the number nu, one more than the order of the pulled-back volume
form, and, for every set I of labels, the Euler characteristic of the set of
points of the fibre over the origin lying on the divisors of I and on no
other. They give A'Campo's formula for the monodromy zeta function and the
formula of Denef and Loeser for the topological zeta function.

How they are read off depends on the kind of resolution:

  - a toric modification (a fan): the strata are unions of torus orbits, and
    a divisor of ray a has N = min <a, exponents> and nu = sum of the a_i
    (engine/newton_repair.py, which also covers blowups made afterwards);
  - point blowups of a plane curve: the dual graph. An exceptional curve is a
    projective line; nu comes from the adjunction formula
        sum_j (nu_j - 1) E_j.E_i = -2 - E_i^2,
    whose matrix is negative definite, and the projection formula
        sum_j N_j E_j.E_i + (orders of the branches through E_i) = 0
    is checked;
  - the blowup of an ordinary point of multiplicity d in n variables: the
    exceptional divisor is P^(n-1) with N = d and nu = n, and the strict
    transform cuts it in a smooth hypersurface of degree d;
  - the point and line blowups of the surfaces ab + c^k: all centres are
    coordinate subspaces, so the charts are those of a fan, which is read
    off from the monomial chart maps;
  - the blowup of an ideal, in graph charts: the strata are counted on the
    pieces "first nonzero homogeneous coordinate", and nu - 1 is the order of
    the Jacobian of the projection, computed as the order of a Fitting ideal
    (engine/monodromy_graph.py);
  - a divisor with normal crossings at the origin: the identity, with the
    origin as the only stratum.
"""
import re

import sympy as sp

from .polynomial import InputError, parse

TORIC = ("binomial-coordinate-blowups", "newton-polyhedron", "native-toric")


def _records(data, strata):
    from .newton_repair import records
    return records({"data": data, "strata": strata})


def _sparse(expr, symbols):
    from fractions import Fraction
    poly = sp.Poly(expr, *symbols, domain=sp.QQ)
    return {e: Fraction(int(c.p), int(c.q)) for e, c in poly.terms()}


def _parts(record, variables, matrix=None):
    """The factors of a divisor with multiplicities as (label, N, polynomial),
    in the coordinates v of the fan when x = matrix * v. None for a reduced one."""
    info = record.get("input_divisor")
    if not info:
        return None
    factors = info["factors"]
    symbols = sp.symbols(" ".join(f"strata_{i}" for i in range(len(variables))), seq=True)
    images = symbols if matrix is None else [sum(c*v for c, v in zip(row, symbols)) for row in matrix]
    parts = []
    for k, factor in enumerate(factors, 1):
        terms = parse(factor["equation"], list(variables), bounded=False)
        expr = sp.Add(*(sp.Rational(c.numerator, c.denominator) * sp.Mul(*(x**a for x, a in zip(images, e)))
                        for e, c in terms.items()))
        parts.append(("strict" if len(factors) == 1 else f"S{k}", factor["multiplicity"], _sparse(expr, symbols)))
    return parts


def _toric(record, polynomial):
    from .newton_repair import strata
    variables = record["variables"]
    proof = record.get("certificate", {})
    source = record.get("input_divisor", {}).get("reduced_equation")
    matrix = None
    if record["engine"] == "newton-polyhedron":
        source = proof["adapted_polynomial"]
        matrix = proof.get("linear_map_to_input")
    support = parse(source, list(variables), bounded=False) if source else polynomial
    answer = strata(support, variables, record["fan"], _parts(record, variables, matrix))
    if answer is None:
        return None
    method = "Torus orbits of the regular fan"
    if answer["repairs"]:
        method += ", and the divisors of the blowups made where a face is degenerate"
    return answer["data"], answer["strata"], method


def _power_factors(text):
    """[(factor, exponent)] of a product written (f1)^m1*(f2)*..., or None."""
    out, i = [], 0
    while i < len(text):
        if text[i] != "(":
            return None
        depth, j = 0, i
        while j < len(text):
            depth += (text[j] == "(") - (text[j] == ")")
            if depth == 0:
                break
            j += 1
        if depth:
            return None
        inner, j, power = text[i + 1:j], j + 1, 1
        if j < len(text) and text[j] == "^":
            k = j + 1
            while k < len(text) and text[k].isdigit():
                k += 1
            if k == j + 1:
                return None
            power, j = int(text[j + 1:k]), k
        out.append((inner, power))
        if j < len(text):
            if text[j] != "*":
                return None
            j += 1
        i = j
    return out


def _is_zero(value):
    value = sp.sympify(value)
    if value.is_Rational:
        return value == 0
    t = sp.Dummy("t")
    return sp.minimal_polynomial(value, t) == t


def _branches(record):
    """(exceptional curve, label, N) for every point of the strict transform over the origin."""
    branches = record["branches"]
    info = record.get("input_divisor")
    if not info:
        return [(b["ambient_components"][0], "strict", 1) for b in branches]
    factors = info["factors"]
    if len({f["multiplicity"] for f in factors}) == 1:
        return [(b["ambient_components"][0], "strict", factors[0]["multiplicity"]) for b in branches]
    # Factors of different multiplicities: the one through each point is the
    # factor of the strict transform of its chart that vanishes there.
    from .nonreduced import _algebraics, _read
    charts = {chart["id"]: chart for chart in record["charts"]}
    offset = 0 if info.get("unit_at_origin") == "1" else 1
    out = []
    for b in branches:
        chart = charts[b["chart"]]
        listed = _power_factors(chart["strict_transform"])
        if listed is None or len(listed) != len(factors) + offset:
            return None
        listed = listed[offset:]
        words = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", " ".join([*b["point"], *(text for text, _ in listed)])))
        names = _algebraics(record.get("algebraic_numbers", {}), words)
        field = chart.get("weight_field")
        if field:
            names[field["generator"]] = sp.sympify(field["exact"])
        symbols = {name: sp.Symbol(name) for name in chart["coordinates"]}
        point = {symbols[name]: _read(text, names) for name, text in zip(chart["coordinates"], b["point"])}
        through = [k for k, (text, _) in enumerate(listed)
                   if _is_zero(_read(text, {**names, **symbols}).xreplace(point))]
        if len(through) != 1:
            return None
        k = through[0]
        out.append((b["ambient_components"][0], f"S{k + 1}", factors[k]["multiplicity"]))
    return out


def _plane(record):
    from .monodromy_blowups import _plane as open_curves
    rows = open_curves(record)
    if rows is None:
        return None
    dual = record["dual_complex"]
    vertices = dual["vertices"]
    position = {v["id"]: k for k, v in enumerate(vertices)}
    meet = sp.zeros(len(vertices), len(vertices))
    for k, v in enumerate(vertices):
        meet[k, k] = v["self_intersection"]
    for cell in dual["cells"]:
        a, b = (position[name] for name in cell["vertices"])
        meet[a, b] += 1
        meet[b, a] += 1
    if meet.det() == 0:
        return None
    canonical = meet.LUsolve(sp.Matrix([-2 - v["self_intersection"] for v in vertices]))
    if any(not k.is_Integer or k < 1 for k in canonical):
        return None
    branches = _branches(record)
    if branches is None:
        return None
    orders = sp.Matrix([v["total_multiplicity"] for v in vertices])
    through = [0]*len(vertices)
    for curve, _, order in branches:
        through[position[curve]] += order
    if list(meet*orders + sp.Matrix(through)) != [0]*len(vertices):
        return None  # the projection formula fails: the data are inconsistent
    data = {v["id"]: (v["total_multiplicity"], int(k) + 1) for v, k in zip(vertices, canonical)}
    strata = {frozenset({row["component"]}): row["euler_characteristic"] for row in rows}
    for cell in dual["cells"]:
        key = frozenset(cell["vertices"])
        strata[key] = strata.get(key, 0) + 1
    for curve, label, order in branches:
        if data.setdefault(label, (order, 1)) != (order, 1):
            return None
        key = frozenset({curve, label})
        strata[key] = strata.get(key, 0) + 1
    return data, strata, "Dual graph of the point blowups, with the adjunction formula"


def _ordinary(record):
    from .monodromy_blowups import _ordinary as open_part
    rows = open_part(record)
    if rows is None:
        return None
    row = rows[0]
    n, degree, order, chi = row["projective_dimension"] + 1, row["tangent_degree"], row["multiplicity"], row["euler_characteristic"]
    if order % degree:
        return None
    # The strict transform meets P^(n-1) in the smooth tangent hypersurface,
    # of Euler characteristic n - chi(complement).
    label = row["component"]
    data = {label: (order, n), "strict": (order//degree, 1)}
    strata = {frozenset({label}): chi, frozenset({label, "strict"}): n - chi}
    return data, strata, "Blowup of an ordinary point: a smooth hypersurface in projective space"


def _constants(record):
    """The exact value of the generator of the coefficient field of a chain record."""
    field = record.get("coefficient_field") or {}
    name = field.get("generator")
    if not name:
        return {}
    match = re.fullmatch(rf"{re.escape(name)}\^2-\(([-\d/]+)\)", field.get("minimal_polynomial", ""))
    if not match:
        raise InputError("Unrecognized coefficient field in the chart maps.")
    return {name: sp.sqrt(sp.Rational(match[1]))}


def _chain(record):
    """The fan of the point and line blowups of ab + c^k, from the chart maps."""
    from .newton_repair import strata
    from .nonreduced import _read
    variables = record["variables"]
    root = record["certificate"]["normalized_chart"]
    names = root["coordinates"]
    n = len(names)
    normal = parse(root["equation"], list(names), bounded=False)
    weight = 1
    if record.get("input_divisor"):
        factors = record["input_divisor"]["factors"]
        if len(factors) != 1:
            return None
        weight = factors[0]["multiplicity"]
    constants = _constants(record)
    symbols = [sp.Symbol(name) for name in names]
    forms = [sp.expand(_read(root["map"][v], {**dict(zip(names, symbols)), **constants})) for v in variables]
    linear = sp.Matrix([[form.coeff(s) for s in symbols] for form in forms])
    if any(sp.expand(form - sum(c*s for c, s in zip(row, symbols))) != 0
           for form, row in zip(forms, linear.tolist())) or linear.det() == 0:
        return None
    inverse = linear.inv()
    rays = [[int(i == j) for j in range(n)] for i in range(n)]
    labels, cones = {}, []
    for chart in record["charts"]:
        if not chart.get("final", True):
            continue
        coordinates = [sp.Symbol(name) for name in chart["coordinates"]]
        local = {**dict(zip(chart["coordinates"], coordinates)), **constants}
        image = sp.Matrix([_read(chart["map"][v], local) for v in variables])
        powers = []
        for value in inverse*image:
            terms = [(e, c) for e, c in sp.Poly(sp.expand(value), *coordinates).terms() if sp.simplify(c) != 0]
            if len(terms) != 1:
                return None  # a root coordinate is not a monomial in this chart
            powers.append(terms[0][0])
        divisors = {e["equation"]: e["id"] for e in chart["exceptional"]}
        cone = []
        for i, name in enumerate(chart["coordinates"]):
            ray = [powers[j][i] for j in range(n)]
            label = divisors.get(name)
            if (label is None) != (sum(ray) == 1 and ray in rays[:n]):
                return None
            if ray not in rays:
                rays.append(ray)
            index = rays.index(ray)
            if label is not None and labels.setdefault(index, label) != label:
                return None
            cone.append(index)
        cones.append(cone)
    answer = strata(normal, names, {"rays": rays, "cones": cones}, [("strict", weight, normal)], labels)
    if answer is None or answer["repairs"]:
        return None
    return answer["data"], answer["strata"], "Torus orbits of the fan of the point and line blowups"


def _crossings(record):
    from .monodromy_special import crossing_components
    named = crossing_components(record)
    if named is None:
        return None
    labels = ["strict"] if len(named) == 1 else [f"S{k}" for k in range(1, len(named) + 1)]
    data = {label: (order, 1) for label, (_, order) in zip(labels, named)}
    return data, {frozenset(labels): 1}, "Normal crossings at the origin: the identity is a resolution"


def resolution_strata(record, polynomial=None):
    """(numerical data, strata, method) of a certified resolution record, as
    JSON rows, or None when the record does not determine them.

    The rows are {"component", "multiplicity", "nu"} and {"components",
    "euler_characteristic"}; strata of Euler characteristic zero are left out.
    """
    if not isinstance(record, dict) or record.get("status") != "resolved":
        return None
    engine, variables = record.get("engine"), record.get("variables")
    proof = record.get("certificate") or {}
    if proof.get("strata") and proof.get("numerical_data") and not record.get("input_divisor"):
        # Computed with the resolution (blowups from the origin, or a Newton
        # modification completed by blowups).
        return (proof["numerical_data"], proof["strata"],
                "Divisors of the blowups, with exact Euler characteristics of their strata"
                if engine == "canonical-blowups" else
                "Torus orbits of the regular fan, and the divisors of the blowups made where a face is degenerate")
    if polynomial is None:
        polynomial = parse(record.get("polynomial"), list(variables), bounded=False)
    found = None
    try:
        if engine in TORIC:
            # Valid in either mode: the fan gives the ambient modification.
            found = _toric(record, polynomial)
        elif record.get("mode") == "embedded":
            if engine == "C++ / FLINT plane point blowups":
                found = _plane(record)
            elif engine == "native-ordinary":
                found = _ordinary(record)
            elif engine == "native-a-series":
                found = _chain(record)
            elif engine == "verified-ideal-blowup":
                from .monodromy_graph import graph_strata
                found = graph_strata(record, polynomial)
        if found is None:
            found = _crossings(record)
    except (KeyError, IndexError, TypeError) as exc:
        raise InputError("Incomplete or inconsistent chart or divisor data in the resolution record.") from exc
    if found is None:
        return None
    data, strata, method = found
    return (*_records(data, strata), method)
