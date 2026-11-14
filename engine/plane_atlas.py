"""Verify a cover obtained by successive blowups of marked points of a smooth plane.

Each pair of affine opens has one parent and one centre. The first owns the
affine line of finite slopes on its exceptional P^1, and the second owns the
point at infinity. A later blowup replaces its centre on that P^1, including
the reciprocal direction in the other open. Composing the two standard maps
recovers the projection; propagating the coordinate axes recovers the boundary.
Only the new exceptional line can contain a later centre in this presentation.

All strict equations, orders, crossings and constructible Euler integrals are
computed from the original polynomial. On a surface, intersections of distinct
exceptional curves are points, so their Euler characteristics count the edges.
On the smooth strict curve the fibre is also finite, with one vertex per point.

Algebraic centres can be transported through the whole cover when one of the
marked constants generates a number field containing all chart coefficients.
QQbar verifies every value at the same indexed root; maps and reciprocal
directions are compared modulo its irreducible polynomial. Thus the cover of
each chart over Q consists of conjugates of its marked chart. Each conjugate
has the same counts and Euler integrals, and dividing by the field degree
selects one sheet. Distinct conjugate centres on this sheet are still distinct
blowups. If no given coefficient generates a common field, the chart-local
number-field calculation handles the larger cover.
"""
from collections import defaultdict
from functools import lru_cache
import re

from . import native_library
from .chart_kernel import InconsistentSheets, crossings_rows, strata_rows, strata_totals
from .incidence import dual_cells
from .monomial_atlas import _row, _same_locus
from .point_atlas import _same_ideal
from .polynomial import InputError, format_poly, integer_from_decimal, integer_text, parse_expression, rational_text, sparse_row
from .rational_point import coordinate_fraction
from .strata import InvalidChart, records


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_RATIONAL = re.compile(r"-?\d+(?:/\d+)?")


@lru_cache(maxsize=256)
def _root(exact):
    match = re.fullmatch(r"PolynomialRootIndexed\(List\((-?\d+(?:, -?\d+)*)\), (\d+)\)", exact)
    if match is None:
        raise InvalidChart("A plane coefficient needs an exact indexed root.")
    coeffs = tuple(map(integer_from_decimal, match[1].split(", ")))
    index = int(match[2])
    degree = len(coeffs) - 1
    if degree <= 1 or not coeffs[-1] or not 1 <= index <= degree:
        raise InvalidChart("Invalid defining polynomial or marked root index.")
    polynomial = format_poly({(j,): a for j, a in enumerate(coeffs)}, ("T",))
    factor = native_library.affine_request(["factor", "T", polynomial])
    if len(factor) != 3 or factor[1] != "1":
        raise InvalidChart("The marked coefficient must have an irreducible defining polynomial.")
    return degree, polynomial, f"flintroot {degree} {' '.join(map(integer_text, coeffs))} {index-1}"


@lru_cache(maxsize=2048)
def _field_value(generator, expr, theta):
    values = native_library.field_request("atlas_field_coefficient", (generator, expr))
    if values is None:
        raise native_library.Unavailable("The native kernel lacks marked coefficient arithmetic.")
    if values == "outside":
        return None
    return format_poly({(i,): coordinate_fraction(q) for i, q in enumerate(values.split())}, (theta,))


def _field(record, items, witnesses, saved):
    if saved:
        data = items[0].get("coefficient_field")
        if any(c.get("coefficient_field") != data for c in items):
            raise InvalidChart("All plane transitions must use the same marked coefficient field.")
        if not data:
            return None, {}
        theta = data["generator"]
        if (not isinstance(theta, str) or not _NAME.fullmatch(theta) or theta in record["variables"] or
                any(c["coordinates"][-1] != theta for c in items)):
            raise InvalidChart("The marked coefficient parameter must be a separate coordinate.")
        degree, polynomial, _ = _root(data["embedding"])
        if (type(data["degree"]) != int or degree != data["degree"] or
                not _same_ideal((theta,), (), [data["minimal_polynomial"]], polynomial.replace("T", theta))):
            raise InvalidChart("The marked root differs from its saved field relation.")
        return data, {}

    marked = record.get("algebraic_numbers", {})
    if not any(a in marked for w in witnesses for a in w["center"]):
        return None, {}
    words = set()
    for c in items:
        for eq in (*c["map"].values(), *(a for pt in c["exclude_points"] for a in pt)):
            words.update(_NAME.findall(eq))
    used = sorted(words.intersection(marked))
    if not used or any(set(c["coordinates"]).intersection(used) for c in items):
        return None  # ambiguous name in an older chart
    theta = "atlas_field"
    while theta in {*record["variables"], *used} or any(theta in c["coordinates"] for c in items):
        theta += "_"
    roots = {k: _root(marked[k]["exact"]) for k in used}
    name = max(used, key=lambda k: roots[k][0])
    degree, polynomial, generator = roots[name]
    values = {k: _field_value(generator, root[2], theta) for k, root in roots.items()}
    if any(value is None for value in values.values()):
        from .marked_field import common_fields
        exact, relation, common_degree, images = common_fields(tuple(marked[k]["exact"] for k in used), theta)
        return dict(generator=theta, degree=common_degree, minimal_polynomial=relation, embedding=exact), dict(zip(used, images))
    data = dict(generator=theta, degree=degree, minimal_polynomial=polynomial.replace("T", theta),
                embedding=marked[name]["exact"])
    return data, values


def _point(data, field, coefficients):
    if not isinstance(data, (list, tuple)) or len(data) != 2:
        raise InvalidChart("A blowup centre needs two coordinates.")
    result = []
    for value in data:
        if isinstance(value, str) and _RATIONAL.fullmatch(value):
            try:
                result.append(rational_text(coordinate_fraction(value)))
            except InputError as exc:
                raise InvalidChart("Invalid rational coordinate.") from exc
        elif field:
            t = field["generator"]
            poly = coefficients.get(value, value)
            parse_expression(poly, (t,))  # enforce the chart grammar before algebraic reduction
            row, = native_library.affine_request(["pullbacks", t, t, t, field["minimal_polynomial"], poly])
            result.append(format_poly(sparse_row(row), (t,)))
        else:
            raise InvalidChart("The point requires a marked coefficient field.")
    return tuple(result)


@lru_cache(maxsize=512)
def _inverse(value, theta, embedding):
    if _RATIONAL.fullmatch(value):
        rational = coordinate_fraction(value)
        if not rational:
            raise ZeroDivisionError("A zero slope has no inverse.")
        row, = native_library.affine_request(["function", "u", f"1/({value})"])
        return format_poly(sparse_row(row), ("u",))
    root = _root(embedding)[2]
    # Evaluate the same marked polynomial and invert its value in QQbar.
    terms = parse_expression(value, (theta,))
    code = "q 0"
    for (power,), coefficient in terms.items():
        code += f" {root} pow {integer_text(power)} q {rational_text(coefficient)} * +"
    return _field_value(root, code + " pow -1", theta)


def _cover(record, saved):
    source = record["atlas"] if saved else record.get("charts", ())
    if saved:
        if not any("point_blowup" in c for c in source):
            return None
        if any("point_blowup" not in c for c in source):
            raise InvalidChart("A plane blowup chart is missing its transition.")
        witnesses = [c["point_blowup"] for c in source]
    else:
        if not source or any("exclude_points" not in c or "parent_chart" not in c for c in source):
            return None
        steps = {s["id"]: s for s in record["steps"]}
        positions = {"origin": -1}
        witnesses = []
        for c in source:
            step = steps[c["step"]]
            if c["parent_chart"] != step["parent_chart"] or c["id"] in positions:
                raise InvalidChart("Inconsistent plane chart parent.")
            witnesses.append({"parent": positions[c["parent_chart"]], "center": step["center"]})
            positions[c["id"]] = len(witnesses)-1
    if not source or len(source) % 2:
        raise InvalidChart("Each plane blowup needs both affine opens.")
    common = _field(record, source, witnesses, saved)
    if common is None:
        return None
    field, coefficients = common
    centres = []
    removed = defaultdict(set)
    seen = set()
    for i in range(0, len(source), 2):
        witness = witnesses[i]
        parent, center = witness["parent"], _point(witness["center"], field, coefficients)
        if witness != witnesses[i+1] or type(parent) != int:
            raise InvalidChart("The two blowup opens disagree about their centre.")
        if i == 0:
            if parent != -1 or center != ("0", "0"):
                raise InvalidChart("The first blowup must be at the origin.")
        else:
            if not 0 <= parent < i:
                raise InvalidChart("A blowup parent must be an earlier chart.")
            pivot = parent % 2
            if center[pivot] != "0" or pivot == 1 and center != ("0", "0"):
                raise InvalidChart("The blowup centre lies outside its parent's owned exceptional line.")
            if (parent, center) in seen:
                raise InvalidChart("Two blowups replace the same point.")
            seen.add((parent, center))
            removed[parent].add(center)
            if center[1-pivot] != "0":
                # The nonzero finite slope is present in both affine opens.
                inverse = _inverse(center[1], field["generator"] if field else "", field["embedding"] if field else "")
                removed[parent + 1].add((inverse, "0"))
        centres.extend([(parent, center)]*2)
    return source, centres, removed, field, coefficients


def _plane_strata(n, base, ideals, removed):
    if len(removed) <= 1:
        return strata_rows(n, base, ideals, removed)
    # The cover removes distinct points. Over a marked field each point is
    # a section, and distinct sections stay disjoint on every conjugate sheet.
    # Subtract each point in its own divisor stratum on the specified base.
    rows = [(subset, 1, count) for subset, count in strata_rows(n, base, ideals, ()).items()]
    for point in removed:
        rows.extend((subset, 1, count, -1) for subset, count in strata_rows(n, [*base, *point], ideals, ()).items())
    return strata_totals(rows)


def _crosses(n, ambient, ideals, owned, removed):
    for _, _, bad in crossings_rows(n, 2, ambient, ideals, owned, ()):
        if not removed:
            return True
        # Deleting finitely many points cannot remove a positive-dimensional
        # bad locus. A zero-dimensional locus is counted by its Euler number.
        if int(native_library.ideal_request(f"dimension {n}", bad)[0]) > 0:
            return True
        if _plane_strata(n, bad, [()], removed).get((0,), 0):
            return True
    return False


def _chart(record, item, i, parent, center, removed, frames, f, saved, field, coefficients):
    base = tuple(record["variables"])
    marked = (field["generator"],) if field else ()
    coords = tuple(item["coordinates"]) + (() if saved else marked)
    modulus = field["minimal_polynomial"] if field else "0"
    ambient = [modulus] if field else []
    if (len(coords) != 2 + len(marked) or any(not isinstance(v, str) or not _NAME.fullmatch(v) for v in coords)
            or len(set(coords)) != len(coords)):
        raise InvalidChart("A plane chart needs two distinct coordinates and its separate field parameter.")

    def native_equation(eq):
        eq = _NAME.sub(lambda m: "(" + coefficients[m[0]] + ")" if m[0] in coefficients else m[0], eq)
        parsed = parse_expression(eq, coords)
        if field:
            row, = native_library.affine_request(["pullbacks", " ".join(coords), " ".join(coords),
                                                  *coords, modulus, format_poly(parsed, coords)])
            parsed = sparse_row(row)
        return format_poly(parsed, coords)

    pivot = i % 2
    axes = [None, None] if parent < 0 else list(frames[parent][2])
    for j in range(2):
        if center[j] != "0":
            axes[j] = None
    axes[pivot] = f"E{i // 2 + 1}"
    parent_coords, parent_maps = (base+marked, base) if parent < 0 else frames[parent][:2]
    local = [f"({center[j]})+({coords[j] if j == pivot else '*'.join(coords[:2])})" for j in range(2)]
    mapped = native_library.affine_request(["pullbacks", " ".join(coords), " ".join(parent_coords),
                                            *local, *marked, modulus, *parent_maps])
    projection = [format_poly(sparse_row(row), coords) for row in mapped]
    if (set(item["map"]) != set(base) or
            any(native_equation(item["map"][v]) != g for v, g in zip(base, projection))):
        raise InvalidChart("The plane chart projection differs from the composed point blowups.")
    frames.append((coords, projection, axes))
    exceptional = [j for j in range(2) if axes[j]]
    header = ["chart", " ".join(coords), " ".join(base+marked), *projection, *marked, modulus]
    axis_list = " ".join(map(str, exceptional))
    rows = native_library.affine_request([*header, f, axis_list])
    comps = []
    for j, row in zip(exceptional, rows):
        order, nu = map(int, row.split())
        comps.append(dict(id=axes[j], ideal=[coords[j]], multiplicity=order, nu=nu))
    factors = record.get("input_divisor", {}).get("factors", [])
    for k, factor in enumerate(factors or [dict(equation=f,multiplicity=1)], 1):
        part = native_library.affine_request([*header, factor["equation"], axis_list]) if factors else rows
        strict = sparse_row(part[-1])
        if any(any(e[:2]) for e in strict):
            label = "strict" if len(factors) <= 1 else f"S{k}"
            comps.append(dict(id=label, ideal=[format_poly(strict, coords)], multiplicity=factor["multiplicity"], nu=1))
    region = [coords[pivot]] if not pivot else [coords[1], coords[0]]
    points = sorted(removed)
    deleted = [[native_equation(f"{v}-({a})") for v, a in zip(coords[:2], point)] for point in points]
    if saved:
        if (item.get("torus") or set(_row(g, coords) for g in item["ambient_ideal"]) !=
                set(_row(g, coords) for g in ambient)):
            raise InvalidChart("The plane chart has a different ambient space from its blowup.")
        given = {c["id"]: c for c in item["components"]}
        if set(given) != {c["id"] for c in comps} or len(given) != len(item["components"]):
            raise InvalidChart("The saved plane boundary differs from its blowup axes or strict transform.")
        for c in comps:
            old = given[c["id"]]
            if (any(type(old[key]) != int or old[key] != c[key] for key in ("multiplicity", "nu"))
                    or not _same_ideal(coords, ambient, old["ideal"], c["ideal"][0])):
                raise InvalidChart("Incorrect orders or divisor ideal on a saved plane chart.")
        def equal(a, b):
            return _same_locus(len(coords), [_row(g, coords) for g in (*ambient, *a)],
                               [_row(g, coords) for g in (*ambient, *b)])
        if not equal(item["fibre"], projection) or not equal(projection+item["region"], projection+region):
            raise InvalidChart("The saved plane fibre or chart partition differs from its map.")
        if (any(not any(equal(z, w) for w in deleted) for z in item["removed"]) or
                any(not any(equal(z, w) for w in item["removed"]) for z in deleted)):
            raise InvalidChart("Removed centres differ from the points replaced by later blowups.")
    else:
        if ({_point(z, field, coefficients) for z in item["exclude_points"]} != set(points) or
                {(c["id"], c["equation"]) for c in item["ambient_components"]} !=
                {(axes[j],coords[j]) for j in exceptional}):
            raise InvalidChart("The plane chart boundary or removed centres disagree with the blowups.")
    return dict(coordinates=list(coords), ambient_ideal=ambient, map=dict(zip(base, projection)), fibre=projection,
                region=region, removed=deleted, components=comps,
                point_blowup=dict(parent=parent, center=list(center)), **({"coefficient_field": field} if field else {}))


def _calculate(record, polynomial, cover, saved, complex, intrinsic, integrals):
    items, centres, removed, field, coefficients = cover
    if "input_divisor" in record:
        from .weighted_record import verify_factors
        verify_factors(polynomial, record["variables"], record["input_divisor"]["factors"])
    degree = field["degree"] if field else 1
    f = format_poly(polynomial, record["variables"])
    charts, frames = [], []
    numbers, contributions = {}, []
    embedded = True
    for i, (item, (parent, center)) in enumerate(zip(items, centres)):
        c = _chart(record, item, i, parent, center, removed[i], frames, f, saved, field, coefficients)
        charts.append(c)
        coords = tuple(c["coordinates"])
        ambient = [_row(g, coords) for g in c["ambient_ideal"]]
        n = len(coords)
        labels = [part["id"] for part in c["components"]]
        for part in c["components"]:
            value = part["multiplicity"], part["nu"]
            if numbers.setdefault(part["id"], value) != value:
                raise InvalidChart("Plane charts disagree on divisor orders.")
        ideals = [(_row(part["ideal"][0], coords),) for part in c["components"]]
        owned = list(dict.fromkeys(_row(g, coords) for g in (*c["fibre"], *c["region"])))
        cuts = [[_row(g, coords) for g in point] for point in c["removed"]]
        if intrinsic and _crosses(n, ambient, [ideal for label, ideal in zip(labels, ideals) if label == "strict"], owned, cuts):
            raise InvalidChart("The strict plane transform is singular over the origin.")
        if (not intrinsic or integrals) and _crosses(n, ambient, ideals, owned, cuts):
            if not intrinsic:
                raise InvalidChart("The embedded plane transform fails normal crossings.")
            embedded = False
        if integrals or complex:
            for subset, total in _plane_strata(n, [*ambient, *owned], ideals, cuts).items():
                key = tuple(labels[k] for k in subset)
                exceptional = tuple(label for label in key if label.startswith("E"))
                contributions.append((("stratum", frozenset(key)), degree, total))
                if complex and intrinsic and "strict" in key:
                    if len(exceptional) != 1 and total:
                        return None  # the general verifier handles coincident divisor labels on the curve
                    if exceptional:
                        contributions.append((("point", exceptional[0]), degree, total))
                elif complex and not intrinsic and len(exceptional) == 2:
                    contributions.append((("edge", frozenset(exceptional)), degree, total))
    try:
        counts = strata_totals(contributions)
    except InconsistentSheets as exc:
        raise ArithmeticError("Conjugate plane strata disagree about the Euler characteristic.") from exc
    out = {"cells": None, "reason": None}
    if integrals and embedded:
        rows, listed = records(dict(data=numbers, strata={key: chi for (kind, key), chi in counts.items() if kind == "stratum"}))
        out.update(numerical_data=rows, strata=listed)
    if complex:
        vertices, cells = [], []
        labels = [name for name in numbers if name.startswith("E")]
        for name in labels:
            count = counts.get(("point", name), 0) if intrinsic else 1
            vertices.extend(dict(id=f"{name}.{j+1}" if count > 1 else name, dimension=0 if intrinsic else 1)
                            for j in range(count))
        if not intrinsic:
            edges = {key: chi for (kind, key), chi in counts.items() if kind == "edge"}
            vertices_by_label = {frozenset((name,)): 1 for name in labels}
            cells, reason = dual_cells({**vertices_by_label, **edges}, labels.index)
            if reason:
                raise ArithmeticError(reason)
        convention = ("Exceptional divisor on the smooth strict transform" if intrinsic else
                      "Ambient exceptional divisor") + " in the germ over the origin over Qbar."
        out.update(cells=cells, dual_complex=dict(status="complete", convention=convention, vertices=vertices, cells=cells))
    return (items if saved else charts), out


def plane_outputs(record, polynomial, *, complex=True, intrinsic=False, integrals=True):
    if len(record["variables"]) != 2 or native_library._MODE == "sympy":
        return None
    if intrinsic and record.get("input_divisor"):
        return None
    saved = "atlas" in record
    try:
        cover = _cover(record, saved)
        if cover is None:
            return None
        result = _calculate(record, polynomial, cover, saved, complex, intrinsic, integrals)
    except native_library.Unavailable:
        return None
    if result and native_library.CHECK:
        from . import atlas
        charts = atlas.from_record(record, polynomial)
        reference = atlas.outputs(charts, 2, complex=complex, intrinsic=intrinsic,
                                  integrals=integrals and "strata" in result[1])
        if result[1] != reference:
            raise AssertionError("The sparse plane atlas and reference outputs disagree.")
    return result
