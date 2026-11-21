"""Verify an affine identity chart at its point fibre from rational sparse equations.

The projection identifies coordinates in order. Its volume form has order zero,
and the origin belongs to exactly the listed branches when their weighted
product agrees with the input up to a unit at that point. Independent linear
parts certify smooth transverse branches. This is a special fibre of the
general atlas computation: a single owned point, with Euler characteristic one.
"""
import re

from . import native_library
from .kernel_json import kernel_json
from .point_geometry import encode_terms, request as point_request
from .polynomial import InputError, format_poly, integer_from_decimal, integer_text, parse_expression, sparse_row
from .strata import InvalidChart


def independent(parts, dimension):
    return point_request("independent", point=["0"] * dimension, ideal=list(map(encode_terms, parts)))


def rational_factors(poly, names):
    """Unordered primitive rational factors and multiplicities, excluding the scalar."""
    def ordered(pairs):
        return sorted(pairs, key=lambda pair: tuple(sorted(pair[0].items())))

    rows = native_library.affine_request(["factor", " ".join(names), format_poly(poly, names)])
    return ordered((sparse_row(row), integer_from_decimal(m)) for m, row in zip(rows[1::2], rows[2::2]))


def reduced_at_origin(f, names):
    return kernel_json("atlas_identity_divisor", "reduced_at_origin", width=len(names), terms=encode_terms(f))


def _certify_divisor(f, parts, names, intrinsic):
    return kernel_json("atlas_identity_divisor", "certify", width=len(names), terms=encode_terms(f),
                       parts=[[encode_terms(g), integer_text(m)] for g, m in parts], intrinsic=intrinsic)


def origin_indices(equations, width):
    return kernel_json("atlas_identity_divisor", "origin_indices", width=width,
                       equations=list(map(encode_terms, equations)))


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
        if len(origin_indices(fibre, len(coords))) != len(fibre):
            raise InvalidChart("The projection fibre misses the origin.")
        rows = ["frame_empty", " ".join(coords), str(len(fibre)), *(format_poly(f, coords) for f in fibre)]
        if not all(native_library.affine_request([*rows, v]) == ["1"] for v in coords):
            raise InvalidChart("The claimed identity fibre contains points outside the origin.")
    data, equations = {}, []
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
        equations.append(equation)
    labels = list(data)
    active = origin_indices(equations, len(coords))
    present = {labels[i] for i in active}
    parts = [(equations[i], data[labels[i]][0]) for i in active]
    if (not independent([g for g, _ in parts], len(coords)) or
            not _certify_divisor(polynomial, parts, coords, intrinsic)):
        raise InvalidChart("The listed equations do not give transverse branches of the input at the origin.")

    output = {"cells": [] if complex else None, "reason": None}
    if integrals:
        from .strata import records
        rows, strata = records({"data": data, "strata": {frozenset(present): 1}})
        # The certified projection identifies these coordinate columns in order.
        # Print the checked component generators in the original ring.
        original_equations = {labels[i]: format_poly(equations[i], names) for i in active}
        for row in rows:
            row["equation"] = original_equations[row["component"]]
        output.update(numerical_data=rows, strata=strata)
    if complex:
        convention = ("Exceptional divisor on the smooth strict transform" if intrinsic else
                      "Ambient exceptional divisor") + " in the germ over the origin over Qbar."
        output["dual_complex"] = {"status": "complete", "convention": convention, "vertices": [], "cells": []}
    return charts, output
