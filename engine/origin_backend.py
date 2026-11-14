"""Embedded resolution of a hypersurface germ by blowups with smooth centres,
started at the origin itself.

The centres are those of the complete rule of engine/local_blowups.py: the
maximum locus of the invariant of Bierstone and Milman while the strict
transform is singular at a point where the total transform fails normal
crossings, and afterwards, in dimension three, the centres of the procedure
of engine/surface.py. Every centre is checked to be smooth, and the run ends
only when the total transform has normal crossings at every point over the
origin, which is verified in every final chart by the Jacobian criterion.
Failing normal crossings is a closed condition and the fibre over the origin
is compact, so the total transform then has normal crossings over a
neighbourhood of the origin: the composite of the blowups is an embedded
resolution of the germ.

This engine needs no hypothesis on the Newton polyhedron. It is used when
the engines that read a resolution off the polyhedron give no answer.
"""
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from time import monotonic

import sympy as sp

from .local_blowups import GiveUp, acampo_over, cells, dual_cells
from .newton_repair import _text, records


def resolve_origin(poly, variables, mode, seconds=120):
    """The record of the resolution, or None when the rule gives no answer in
    the time allowed. `poly` maps exponents to rational coefficients of the
    reduced equation.

    The computation runs in a process of its own, which is stopped when the
    time is over: a single step of it can take long, and the time is only
    looked at between steps."""
    request = json.dumps({"terms": [[list(exponents), str(c)] for exponents, c in poly.items()],
                          "variables": list(variables), "mode": mode, "seconds": seconds})
    try:
        done = subprocess.run([sys.executable, "-m", "engine.origin_backend"], input=request, text=True,
                              capture_output=True, cwd=Path(__file__).resolve().parents[1], timeout=seconds + 2)
    except subprocess.TimeoutExpired:
        return None
    if done.returncode != 0 or not done.stdout.strip():
        return None
    return json.loads(done.stdout)


def _resolve(poly, variables, mode, seconds):
    """The computation of `resolve_origin`, in this process."""
    names = list(variables)
    symbols = tuple(sp.Symbol(name) for name in names)
    equation = sp.Add(*(sp.Rational(c.numerator, c.denominator) * sp.Mul(*(s**e for s, e in zip(symbols, exponents)))
                        for exponents, c in poly.items()))
    # The rule is for a reduced equation: a repeated factor is left to the
    # treatment of divisors with multiplicities.
    repeated = equation
    for s in symbols:
        repeated = sp.gcd(repeated, sp.diff(equation, s))
    if repeated.free_symbols:
        return None
    try:
        data, strata, charts, centres = acampo_over(symbols, [("strict", 1, 1, (equation,))], symbols,
                                                    dimension=len(symbols), deadline=monotonic() + seconds)
    except GiveUp:
        return None
    if not centres:
        return None  # normal crossings at the origin already: other engines describe that case
    rows, listed = records({"data": data, "strata": strata})
    births = [centre["id"] for centre in centres]
    try:
        found, reason = dual_cells(cells(charts, len(symbols)), births.index)
    except GiveUp as error:
        found, reason = None, str(error)
    final = []
    for k, chart in enumerate(charts, 1):
        divisors = [(label, m, [_text(g) for g in ideal]) for label, m, _, ideal in chart.components]
        strict = [ideal for label, _, ideal in divisors if label == "strict"]
        final.append({
            "id": f"C{k}", "coordinates": [v.name for v in chart.variables],
            "ambient_ideal": [_text(g) for g in chart.ambient],
            "map": {name: _text(expr) for name, expr in chart.coordinates if name in names},
            "exceptional": [{"id": label, "ideal": ideal, "equation": ", ".join(ideal), "total_multiplicity": m}
                            for label, m, ideal in divisors if label != "strict"],
            "strict_transform": strict[0] if strict else ["1"],
            # The chart is used off these centres, whose blowups other charts describe.
            **({"blown_up": [{"ideal": [_text(g) for g in z]} for z in chart.removed]} if chart.removed else {}),
        })
    steps = [{"id": centre["id"], "type": "blowup", "center": centre["kind"], "rule": centre["rule"],
              **({"invariant": centre["invariant"]} if "invariant" in centre else {}),
              "components": centre["components"]} for centre in centres]
    created = [row for row in rows if row["component"] != "strict"]
    points = sum(1 for centre in centres if centre["dimension"] == 0)
    curves = sum(1 for centre in centres if centre["dimension"] == 1)
    kinds = ", ".join(text for text in (f"{points} at points" if points else "", f"{curves} along curves" if curves else "",
                                        f"{len(centres) - points - curves} along larger centres"
                                        if len(centres) > points + curves else "") if text)
    return {
        "status": "resolved", "engine": "canonical-blowups",
        "summary": f"{len(centres)} blowup{'s' if len(centres) != 1 else ''} with smooth centres ({kinds}); "
                   f"{len(final)} final charts.",
        "steps": steps, "charts": final,
        "dual_complex": {
            "status": "complete" if found is not None else "partial",
            "convention": "Ambient exceptional divisor over the origin over Qbar.",
            **({} if found is not None else
               {"reason": reason + " The divisors created by the blowups are listed as vertices without their incidences."}),
            "vertices": [{"id": row["component"], "ambient_order": row["multiplicity"]} for row in created],
            "cells": found or [],
        },
        "certificate": {
            "method": "blowups with smooth centres chosen by the invariant of Bierstone and Milman and, once the strict "
                      "transform is smooth, by the traces of the divisors on it; certified by ideal calculations",
            "description": "Every centre is smooth and lies over the origin or meets its fibre. In every final chart the "
                           "total transform has normal crossings at all points over the origin, by the Jacobian criterion; "
                           "hence it has normal crossings over a neighbourhood of the origin.",
            "numerical_data": rows, "strata": listed,
        },
    }



if __name__ == "__main__":
    asked = json.load(sys.stdin)
    answer = _resolve({tuple(exponents): Fraction(c) for exponents, c in asked["terms"]}, asked["variables"],
                      asked["mode"], asked["seconds"])
    print(json.dumps(answer))
