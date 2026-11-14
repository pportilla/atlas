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

def _text(expr):
    return str(expr).replace("**", "^")


def resolve_origin(poly, variables, mode, seconds=120, *, defer_outputs=False, tangent_only=False):
    """The record of the resolution, or None when the rule gives no answer in
    the time allowed. `poly` maps exponents to rational coefficients of the
    reduced equation.

    The computation runs in a process of its own, which is stopped when the
    time is over: a single step of it can take long, and the time is only
    looked at between steps."""
    from .polynomial import rational_text
    request = json.dumps({"terms": [[list(exponents), rational_text(c)] for exponents, c in poly.items()],
                          "variables": list(variables), "mode": mode, "seconds": seconds,
                          "defer_outputs": defer_outputs, "tangent_only": tangent_only})
    try:
        done = subprocess.run([sys.executable, "-m", "engine.origin_backend"], input=request, text=True,
                              capture_output=True, cwd=Path(__file__).resolve().parents[1],
                              timeout=None if seconds is None else seconds + 2)
    except subprocess.TimeoutExpired:
        return None
    if done.returncode != 0 or not done.stdout.strip():
        return None
    return json.loads(done.stdout)


def _resolve(poly, variables, mode, seconds, *, defer_outputs=False, tangent_only=False):
    """The computation of `resolve_origin`, in this process."""
    names = list(variables)
    from . import native_library
    from .local_blowups import GiveUp, acampo_over
    if native_library._MODE == "native":
        from . import native_polynomial
        symbols = tuple(map(native_polynomial.Symbol, names))
        polynomial = native_polynomial.from_terms
    else:
        from .reference import sp
        from .atlas import _polynomial
        symbols = tuple(map(sp.Symbol, names))
        polynomial = lambda ring, terms: _polynomial(terms, ring)
    equation = polynomial(symbols, poly)
    if tangent_only:
        from .repeated_tangent import tangent_power_support
        try:
            tangent_power_support(symbols, equation)
        except GiveUp:
            return None
    # The rule is for a reduced equation: a repeated factor is left to the
    # treatment of divisors with multiplicities.
    from .affine_kernel import rational_factor_rows
    if any(power > 1 for _, power, _ in rational_factor_rows(symbols, equation)[1]):
        return None
    rules = ("complete", "direct")
    if len(symbols) >= 4:
        rules = ("tangent",) if tangent_only else ("tangent", *rules)
    try:
        data, strata, charts, centres = acampo_over(symbols, [("strict", 1, 1, (equation,))], symbols,
                                                    dimension=len(symbols), rules=rules,
                                                    deadline=None if seconds is None else monotonic() + seconds,
                                                    budget=None if seconds is None else 400,
                                                    count_strata=not defer_outputs)
    except GiveUp:
        return None
    return _record(names, charts, centres, data, strata, defer_outputs=defer_outputs)


def _record(names, charts, centres, data, strata, *, defer_outputs):
    """Serialize certified charts from any of the successive smooth-centre rules."""
    from .atlas import to_json
    from .local_blowups import GiveUp, cells, dual_cells
    from .strata import records

    if not centres:
        return None  # normal crossings at the origin already: other engines describe that case
    rows, listed = records({"data": data, "strata": strata})
    births = [centre["id"] for centre in centres]
    found, reason = None, "Incidence is computed from the final charts."
    if not defer_outputs:
        try:
            found, reason = dual_cells(cells(charts, len(names)), births.index)
        except GiveUp as error:
            reason = str(error)
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
    if defer_outputs:
        orders = {label: order for chart in charts for label, order, _, _ in chart.components}
        created = [{"id": label, "ambient_order": orders[label]} for label in births if label in orders]
    else:
        created = [{"id": row["component"], "ambient_order": row["multiplicity"]}
                   for row in rows if row["component"] != "strict"]
    points = sum(1 for centre in centres if centre["dimension"] == 0)
    curves = sum(1 for centre in centres if centre["dimension"] == 1)
    kinds = ", ".join(text for text in (f"{points} at points" if points else "", f"{curves} along curves" if curves else "",
                                        f"{len(centres) - points - curves} along larger centres"
                                        if len(centres) > points + curves else "") if text)
    return {
        "status": "resolved", "engine": "canonical-blowups",
        "summary": f"{len(centres)} blowup{'s' if len(centres) != 1 else ''} with smooth centres ({kinds}); "
                   f"{len(final)} final charts.",
        "steps": steps, "charts": final, "atlas": to_json(charts),
        "dual_complex": {
            "status": "complete" if found is not None else "partial",
            "convention": "Ambient exceptional divisor over the origin over Qbar.",
            **({} if found is not None else
               {"reason": reason + " The divisors created by the blowups are listed as vertices without their incidences."}),
            "vertices": created,
            "cells": found or [],
        },
        "certificate": {
            "method": "Blowups with smooth centres; certified by ideal calculations. Centre rule: " +
                      ", ".join(dict.fromkeys(step["rule"] for step in steps)),
            "description": "Every centre is smooth and lies over the origin or meets its fibre. In every final chart the "
                           "total transform has normal crossings at all points over the origin, by the Jacobian criterion; "
                           "hence it has normal crossings over a neighbourhood of the origin.",
            **({} if defer_outputs else {"numerical_data": rows, "strata": listed}),
        },
    }



if __name__ == "__main__":
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    asked = json.load(sys.stdin)
    answer = _resolve({tuple(exponents): Fraction(c) for exponents, c in asked["terms"]}, asked["variables"],
                      asked["mode"], asked["seconds"], defer_outputs=asked.get("defer_outputs", False),
                      tangent_only=asked.get("tangent_only", False))
    print(json.dumps(answer))
