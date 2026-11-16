"""Embedded resolution by weighted blowups: transport to the native engine.

The native helper ``native/build/weighted`` computes everything: the centres of
Abramovich, Temkin and Wlodarczyk in their logarithmic form (Abramovich and
Quek), along points and along bad loci of positive dimension, their toric
weighted blowups, the bad points and loci of each chart, and the certificate
that the strict transform and the exceptional divisors cross normally over the
origin. This module sends the parsed polynomial, runs the helper in a forked
process with a deadline, and turns its answer into a resolution record.
"""
from fractions import Fraction
import json
import os
import subprocess

from .polynomial import InputError, LimitError, format_poly, rational_text

SECONDS = 40
METHOD = "weighted blowups (Abramovich-Temkin-Wlodarczyk, logarithmic)"


def _rows(terms):
    return ";".join(" ".join([rational_text(c), *map(str, e)]) for e, c in terms.items() if c)


def _terms(row, width):
    terms = {}
    for term in row.split(";"):
        words = term.split()
        if not words:
            continue
        exponents = tuple(int(w) for w in words[1:])
        if len(exponents) != width:
            raise RuntimeError("The weighted engine returned a polynomial of another width.")
        terms[exponents] = terms.get(exponents, 0) + Fraction(words[0])
    return {e: c for e, c in terms.items() if c}


def native_resolution(terms, variables, *, seconds=SECONDS):
    """The native engine's answer for the polynomial ``terms`` (exponent tuple -> rational) at the origin."""
    payload = {"operation": "resolve", "variables": len(variables), "polynomial": _rows(terms),
               "logarithmic": True, "certify": True, "seconds": seconds}
    from . import native_process
    # The resolution and its certificate each have `seconds`; the margin covers
    # a native step that does not look at the clock.
    timeout = 2 * seconds + 10
    try:
        done = native_process.run("weighted", input=json.dumps(payload), timeout=timeout,
                                  env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    except FileNotFoundError:
        raise RuntimeError("The native weighted resolution is not built. Run make.") from None
    except subprocess.TimeoutExpired:
        raise LimitError(f"The weighted resolution did not finish within {timeout} seconds.") from None
    if done.returncode:
        raise LimitError(f"The native weighted resolution stopped with status {done.returncode}.")
    try:
        answer = json.loads(done.stdout)
    except ValueError:
        raise RuntimeError("The native weighted resolution did not return JSON.") from None
    if "error" in answer:
        raise InputError(answer["error"])
    native = answer["result"]
    if native["status"] != "resolved":
        reason = native.get("reason", "")
        if reason == "The time allowed is over.":
            raise LimitError(f"The weighted resolution did not finish within {seconds} seconds.")
        raise InputError(f"The weighted resolution does not cover this input yet: {reason}")
    certificate = native.get("certificate", {})
    if not certificate.get("certified"):
        problem = certificate.get("problem") or json.dumps(certificate.get("failures", []))[:600]
        raise LimitError(f"The weighted resolution could not certify normal crossings: {problem}")
    return native


def _fresh(base, count, taken):
    for letter in (base, "s", "u", "z", "q"):
        names = [f"{letter}{i}" for i in range(1, count + 1)]
        if not taken.intersection(names):
            return names
    raise RuntimeError("No free names for the chart coordinates.")


def resolution_record(native, terms, variables):
    """A resolution record (engine "weighted") from the native engine's answer."""
    variables = list(variables)
    width = max((c["variables"] for c in native["charts"]), default=len(variables))
    names = _fresh("t", width, set(variables))
    generator = "theta"
    while generator in variables or generator in names:
        generator += "_"
    labels = {d["label"]: f"E{k}" for k, d in enumerate(native["divisors"], start=1)}
    charts, orbifold = [], False
    for chart in native["charts"]:
        k = chart["variables"]
        local = names[:k]
        field = chart.get("field")
        written = local + ([generator] if field else [])

        def text(row):
            return format_poly(_terms(row, len(written)), written)

        record = {"id": chart["name"], "coordinates": local,
                  "map": {v: text(row) for v, row in zip(variables, chart["map"])},
                  "strict_transform": text(chart["strict"]),
                  "exceptional": [{"id": labels[b["divisor"]], "equation": local[b["coordinate"]]}
                                  for b in chart["boundary"]]}
        if chart.get("ambient"):
            record["ambient_ideal"] = [text(row) for row in chart["ambient"]]
        if field:
            coefficients = {(j,): Fraction(c) for j, c in enumerate(field) if Fraction(c)}
            record["coefficient_field"] = {"generator": generator, "degree": len(field) - 1,
                                           "minimal_polynomial": format_poly(coefficients, [generator]),
                                           "embedding": "root 1 (FLINT order)"}
        order = int(chart.get("orbifold_order", "1"))
        if order > 1:
            orbifold = True
            record["orbifold_order"] = order
        record["over"] = [text(row) for row in chart.get("fibre", []) if row]
        record["owned_faces"] = [[local[i] for i in face] for face in chart.get("owned_faces", [])]
        charts.append(record)
    if not native["blowups"]:
        # A germ that is smooth at the origin: the identity resolves it.
        local = names[:len(variables)]
        charts.append({"id": "identity", "coordinates": local,
                       "map": {v: t for v, t in zip(variables, local)},
                       "strict_transform": format_poly(terms, local),
                       "exceptional": []})
    steps = []
    for blowup in native["blowups"]:
        steps.append({"id": blowup["label"], "type": "weighted blowup", "at": blowup["at"],
                      "locus": blowup["locus"], "invariant": blowup["invariant"],
                      "monomial_part": blowup["monomial_part"], "field_degree": blowup["field_degree"],
                      "charts": blowup["charts"], "centre_from": blowup.get("centre_from", "")})
    vertices = [{"id": labels[d["label"]], "blowup": d["label"].split(".")[0], "total_multiplicity": int(d["N"]),
                 "nu": int(d["nu"]), "dimension": len(variables) - 1} for d in native["divisors"]]
    blowups = len(steps)
    summary = (f"Weighted resolution with {blowups} weighted blowup{'s' if blowups != 1 else ''} and "
               f"{len(charts)} charts; normal crossings over the origin are certified." if blowups else
               "The germ is smooth at the origin, so the identity resolves it.")
    notes = ["Each centre is the logarithmic centre of Abramovich, Temkin and Wlodarczyk at a bad point or "
             "along a bad locus of positive dimension; each chart is the monomial map of one cone of the "
             "normal fan of the centre's Newton polyhedron."]
    if orbifold:
        notes.append("A chart with orbifold_order r > 1 is the finite cover, of degree r, of a chart of the "
                     "weighted blowup, which has a cyclic quotient singularity there.")
    certificate = native["certificate"]
    return {
        "status": "resolved", "engine": "weighted", "mode": "embedded", "schema_version": 1,
        "summary": summary, "notes": notes,
        "polynomial": format_poly(terms, variables), "variables": variables,
        "base_field": "Q", "geometric_field": "algebraic closure of Q", "origin": [0] * len(variables),
        "charts": charts, "steps": steps,
        "dual_complex": {"status": "unavailable", "convention": "Exceptional divisors over the origin.",
                         "reason": "The weighted resolution records each exceptional divisor with its numerical "
                                   "data (N, nu); it does not compute their intersections.",
                         "vertices": vertices, "cells": []},
        "certificate": {"method": METHOD, "certified": True,
                        "check": "normal crossings of the strict transform and the exceptional divisors over the "
                                 "origin on every face that each chart owns",
                        "charts": certificate["charts"], "faces": certificate["faces"]},
        "weighted": {"algebraic_orbits": native.get("algebraic_orbits", 0),
                     "native_seconds": native.get("seconds")},
    }


def weighted_resolution(terms, variables, *, seconds=SECONDS):
    """Resolve the hypersurface germ terms = 0 at the origin by weighted blowups and return its record."""
    if not 1 <= len(variables) <= 16:
        raise InputError("The weighted resolution takes between 1 and 16 variables.")
    return resolution_record(native_resolution(terms, variables, seconds=seconds), terms, variables)
