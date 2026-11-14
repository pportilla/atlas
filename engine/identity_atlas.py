"""Verify an affine identity chart at its point fibre from rational sparse equations.

The projection identifies coordinates in order. Its volume form has order zero,
and the origin belongs to exactly the listed branches when their weighted
product agrees with the input up to a unit at that point. Independent linear
parts certify smooth transverse branches. This is a special fibre of the
general atlas computation: a single owned point, with Euler characteristic one.
"""
from fractions import Fraction
from math import lcm
import re

from . import native_library
from .polynomial import InputError, checked_native, format_poly, integer_from_decimal, integer_text, parse_expression, sparse_row
from .strata import InvalidChart


def independent(parts, dimension):
    if not parts or len(parts) > dimension:
        return False
    units = [tuple(int(i == j) for i in range(dimension)) for j in range(dimension)]
    rows = [[f.get(e, 0) for e in units] for f in parts]
    if len(rows) == 1:
        return any(rows[0])

    def rank():
        inputs = ["lattice_dimension", str(dimension), " ".join(["0"] * dimension)]
        for row in rows:
            scalar = lcm(*(c.denominator for c in row))
            inputs.append(" ".join(integer_text(c * scalar) for c in row))
        answer, = native_library.affine_request(inputs)
        return int(answer)

    def reference():
        from sympy import Matrix
        return Matrix(rows).rank()

    return checked_native(rank, reference) == len(rows)


def rational_factors(poly, names):
    """Unordered primitive rational factors and multiplicities, excluding the scalar."""
    def ordered(pairs):
        return sorted(pairs, key=lambda pair: tuple(sorted(pair[0].items())))

    def native():
        rows = native_library.affine_request(["factor", " ".join(names), format_poly(poly, names)])
        return ordered((sparse_row(row), integer_from_decimal(m)) for m, row in zip(rows[1::2], rows[2::2]))

    def reference():
        from sympy import Poly, QQ, Symbol
        f = Poly.from_dict(poly.copy(), *map(Symbol, names), domain=QQ)
        return ordered(({e: Fraction(c) for e, c in g.terms()}, m) for g, m in f.factor_list()[1])

    return checked_native(native, reference)


def branch_order(poly, length):
    """The nested coefficient ordering of archived factors in the bounded SNC recognizer."""
    if length == 0:
        return poly.get((), 0)
    coefficients = {}
    for (power, *rest), value in poly.items():
        coefficients.setdefault(power, {})[tuple(rest)] = value
    degree = max(coefficients, default=-1 if length == 1 else 0)
    return [branch_order(coefficients.get(i, {}), length - 1) for i in range(degree, -1, -1)]


def _active_factors(f, names):
    """A local divisor at the origin: retain the rational primes with zero constant term."""
    from .monodromy_special import _simple_coordinates
    coordinate = _simple_coordinates(f, names)
    if coordinate:
        return {((tuple(int(v == name) for v in names), 1),): m for name, m in coordinate}
    zero = (0,) * len(names)
    return {tuple(sorted(g.items())): m for g, m in rational_factors(f, names) if zero not in g}


def _certify_divisor(f, parts, names):
    if len(parts) == 1 and parts[0] == (f, 1):
        return True
    # Independent smooth branches have order one at the origin. This also
    # rules out malformed large recorded powers before any factorization.
    if min(map(sum, f)) != sum(m for _, m in parts):
        return False
    active = _active_factors(f, names)
    for g, m in parts:
        prime = tuple(sorted(g.items()))
        if prime not in active:
            # Each smooth principal generator can include a nonvanishing unit.
            factors = _active_factors(g, names)
            if len(factors) != 1 or next(iter(factors.values())) != 1:
                return False
            prime, = factors
        if active.pop(prime, None) != m:
            return False
    return not active


def _serialized(record, polynomial):
    if "atlas" in record:
        return record["atlas"]
    raw = record.get("charts")
    if not isinstance(raw, list) or len(raw) != 1 or record.get("steps"):
        return None
    item = raw[0]
    names = record["variables"]
    coords = item.get("coordinates")
    if (not isinstance(coords, list) or len(coords) != len(names) or
            item.get("map") != dict(zip(names, coords)) or
            any(item.get(k) for k in ("ambient_ideal", "exceptional", "ambient_components", "blown_up",
                                      "coefficient_field", "ray_indices", "region", "removed")) or
            record.get("coefficient_field")):
        return None
    factors = record.get("input_divisor", {}).get("factors") or record.get("certificate", {}).get("components")
    if factors:
        # Archived branch equations are written in the original coordinates.
        coords = names
        components = [{"id": "strict" if len(factors) == 1 else f"S{i+1}",
                       "multiplicity": row["multiplicity"], "nu": 1,
                       "ideal": [format_poly(parse_expression(row["equation"], names), names)]}
                      for i, row in enumerate(factors)]
    else:
        components = [{"id": "strict", "multiplicity": 1, "nu": 1,
                       "ideal": [format_poly(polynomial, coords)]}]
    return [{"coordinates": coords, "ambient_ideal": [], "components": components, "fibre": coords,
             "map": dict(zip(names, coords)), "region": [], "removed": []}]


def identity_outputs(record, polynomial, *, complex=True, intrinsic=False, integrals=True):
    """Return (serialized atlas, outputs), or None when the map needs other chart operations."""
    charts = _serialized(record, polynomial)
    if not isinstance(charts, list) or len(charts) != 1:
        return None
    chart = charts[0]
    if "split_cone" in chart:
        return None  # its regular fan certificate is checked against the projection
    names = record["variables"]
    coords = chart["coordinates"]
    if (len(coords) != len(names) or chart.get("map") != dict(zip(names, coords)) or
            any(chart.get(k) for k in ("ambient_ideal", "region", "removed", "torus", "coefficient_field"))):
        return None
    if (len(set(coords)) != len(coords) or any(not isinstance(c, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*", c) for c in coords)):
        raise InputError("Each chart needs distinct coordinate names.")
    if set(chart["fibre"]) != set(coords):
        # Other generators can cut out the same point set. Check the origin
        # belongs and every principal open D(x_i) in the fibre is empty.
        if set(chart["fibre"]).issubset(coords):
            raise InvalidChart("The claimed identity fibre omits a coordinate.")
        fibre = [parse_expression(g, coords) for g in chart["fibre"]]
        if any((0,) * len(coords) in g for g in fibre):
            raise InvalidChart("The projection fibre misses the origin.")
        def native():
            rows = ["frame_empty", " ".join(coords), str(len(fibre)), *(format_poly(f, coords) for f in fibre)]
            return all(native_library.affine_request([*rows, v]) == ["1"] for v in coords)

        def reference():
            from sympy import Dummy, Poly, QQ, S, Symbol, groebner
            variables = tuple(map(Symbol, coords))
            t = Dummy("t")
            equations = [Poly.from_dict(f.copy(), *variables, domain=QQ).as_expr() for f in fibre]
            return all(groebner([*equations, 1-t*v], t, *variables, domain=QQ).contains(S.One)
                       for v in variables)

        if not checked_native(native, reference):
            raise InvalidChart("The claimed identity fibre contains points outside the origin.")
    data, parts, present = {}, [], set()
    zero = (0,) * len(coords)
    for row in chart["components"]:
        label, power = row["id"], row["multiplicity"]
        if not isinstance(label, str) or not re.fullmatch(r"strict|S[0-9]+", label):
            return None
        if (type(power) is not int or power <= 0 or type(row["nu"]) is not int or row["nu"] != 1 or
                label in data):
            raise InvalidChart("Invalid divisor data on an identity chart.")
        if len(row["ideal"]) != 1:
            return None
        equation = parse_expression(row["ideal"][0], coords)
        if not equation:
            raise InvalidChart("The zero equation does not define a divisor.")
        data[label] = (power, 1)
        if zero not in equation:
            present.add(label)
            parts.append((equation, power))
    if (not independent([g for g, _ in parts], len(coords)) or
            not _certify_divisor(polynomial, parts, coords) or intrinsic and len(parts) != 1):
        raise InvalidChart("The listed equations do not give transverse branches of the input at the origin.")

    output = {"cells": [] if complex else None, "reason": None}
    if integrals:
        from .strata import records
        rows, strata = records({"data": data, "strata": {frozenset(present): 1}})
        output.update(numerical_data=rows, strata=strata)
    if complex:
        convention = ("Exceptional divisor on the smooth strict transform" if intrinsic else
                      "Ambient exceptional divisor") + " in the germ over the origin over Qbar."
        output["dual_complex"] = {"status": "complete", "convention": convention, "vertices": [], "cells": []}
    if native_library.CHECK:
        from . import atlas
        expected = atlas.outputs(atlas.from_json(charts), len(coords), complex=complex,
                                 intrinsic=intrinsic, integrals=integrals)
        if expected != output:
            raise AssertionError("The point-fibre strata differ from the general atlas computation.")
    return charts, output
