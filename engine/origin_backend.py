"""Resolution of a hypersurface germ by blowups with smooth centres,
started at the origin itself.

At an isolated singular point the centres are first those of the complete
rule of engine/local_blowups.py: the maximum locus of the invariant of
Bierstone and Milman while the strict transform is singular at a point
where the total transform fails normal crossings, and afterwards, in
dimension three, the centres of the procedure of engine/surface.py. When
that rule gives up or uses its share of the time, the maximal order rule
starts again: while the strict transform is singular at such a point, the
locus near the fibre where its order is largest, or the singular locus of
that locus. Along a curve or surface of singular points the two rules are
tried in the other order. Every centre is checked to be smooth. The Jacobian criterion certifies smoothness of the strict
transform for intrinsic resolution and normal crossings of the total
transform for embedded resolution. The relevant failure locus is closed.
By properness of the blowup map, its image misses a neighbourhood of the
origin when all charts pass over the origin fibre.

This engine needs no hypothesis on the Newton polyhedron. It is used when
the engines that read a resolution off the polyhedron give no answer.
"""
import json
import subprocess
import sys
from .polynomial import kernel_fraction
from pathlib import Path
from time import monotonic

def _text(expr):
    return str(expr).replace("**", "^")


def certified_outputs(record, names, mode, integrals):
    """Verify the owned outputs of the charts just constructed by the native blowups."""
    from .general_atlas import native_general_outputs
    charts = record["atlas"]
    repeated = any(step.get("rule") == "repeated tangent" for step in record["steps"])
    output = native_general_outputs({"variables": names}, charts, intrinsic=mode == "intrinsic",
                                    integrals=integrals, factor_fibre=repeated, canonical_cover=True)
    return charts, output


# Seconds of the first turn of each rule when the rules take turns (see `acampo_over`).
SAFE_TURN_SECONDS = 10


def resolve_origin(poly, variables, mode, seconds=120, *, defer_outputs=False, tangent_only=False,
                   verified=None, integrals=False, skip_tangent=False, safe=False, diagnostics=None, turn=None):
    """The record of the resolution, or None when the rule gives no answer in
    the time allowed. `poly` maps exponents to rational coefficients of the
    reduced equation.

    The computation runs in a process of its own, which is stopped when the
    time is over: a single step of it can take long, and the time is only
    looked at between steps.

    With `safe` the method is the last resort of its caller: no budget of
    charts or stages and no share of the time ends it. The rules take turns
    and continue where they were, so with `seconds=None` it works until it
    has an answer, the process is interrupted, or every rule has stopped for
    a reason of its own. `diagnostics` receives that reason under "reason".
    `turn` is the length of the first turn in seconds (SAFE_TURN_SECONDS
    when it is not given): a search with a time budget makes it a share of
    the budget.
    """
    from .polynomial import rational_text
    deadline = None if seconds is None else monotonic() + seconds
    request = json.dumps({"terms": [[list(exponents), rational_text(c)] for exponents, c in poly.items()],
                          "variables": list(variables), "mode": mode, "seconds": seconds,
                          "defer_outputs": defer_outputs, "tangent_only": tangent_only,
                          "skip_tangent": skip_tangent, "safe": safe, "turn": turn})

    def declined(reason):
        if diagnostics is not None:
            diagnostics["reason"] = reason
        return None

    try:
        from . import fork_server
        done = fork_server.run([sys.executable, "-m", "engine.origin_backend"], input=request, text=True,
                              capture_output=True, cwd=Path(__file__).resolve().parents[1],
                              timeout=None if seconds is None else seconds + 2)
    except subprocess.TimeoutExpired:
        return declined("The time allowed for the blowups is over.")
    if done.returncode != 0 or not done.stdout.strip():
        lines = (done.stderr or "").strip().splitlines()
        return declined("The process of the smooth-centre rules stopped: " + (lines[-1] if lines else "no output."))
    record = json.loads(done.stdout)
    if isinstance(record, dict) and "gave_up" in record:
        return declined(record["gave_up"])
    if record is not None and (verified is not None or mode == "intrinsic" and not defer_outputs):
        checked = (certified_outputs(record, variables, mode, integrals) if verified is not None else
                   (record["atlas"], {"dual_complex": record["dual_complex"]}))
        if mode == "intrinsic":
            from .exhaustive_resolution import refine_intrinsic_boundary
            record, checked = refine_intrinsic_boundary(poly, variables, record, checked, deadline, integrals,
                                                        safe=safe, turn=turn)
            if not defer_outputs:
                record["dual_complex"] = checked[1]["dual_complex"]
        if verified is not None:
            verified.append((record, checked))
    return record


RATIONAL_FACTORS = "rational factors"


def strict_components(symbols, equation, factors, *, separate):
    """The components of the reduced hypersurface that the blowups carry.

    An intrinsic resolution asks for a smooth strict transform, so the whole
    hypersurface is one component "strict": its branches have to come apart.
    An embedded resolution asks for normal crossings of the total transform.
    Smooth branches that cross normally already have them, so with
    `separate` each factor over Q is a component S1, S2, ... of its own, in
    the order of `rational_factor_rows`. A factor that splits over Qbar is
    one component, which the Jacobian criterion then requires to be smooth:
    its conjugate branches have to come apart as before.
    """
    if not separate or len(factors) < 2:
        return [("strict", 1, 1, (equation,))]
    from .affine_kernel import _sparse_expression
    return [(f"S{i}", 1, 1, (_sparse_expression(symbols, row),)) for i, (row, _, _) in enumerate(factors, 1)]


def _resolve(poly, variables, mode, seconds, *, defer_outputs=False, tangent_only=False, skip_tangent=False,
             full_boundary=False, order_only=False, safe=False, reasons=None, turn=None):
    """The computation of `resolve_origin`, in this process.

    With `order_only`, only the maximal order rule runs, and only at an
    isolated singular point, where the other rules come first. An
    exhaustive search runs it in a lane of its own, because there the
    complete rule has no deadline that would pass the time on.

    With `safe` the rules take turns without a budget of charts (see
    `resolve_origin`). `reasons` receives why no record is returned.
    """
    names = list(variables)
    from .local_blowups import GiveUp, acampo_over
    from . import native_polynomial
    symbols = tuple(map(native_polynomial.Symbol, names))
    polynomial = native_polynomial.from_terms
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
    factors = rational_factor_rows(symbols, equation)[1]
    if any(power > 1 for _, power, _ in factors):
        return None
    # At an isolated singular point the complete rule comes first. Along a
    # curve or surface of singular points the maximal order rule comes
    # first: it blows up that locus whole and needs no hypersurface of
    # maximal contact. Each keeps a share of the time for the other.
    isolated = _isolated(symbols, equation)
    if order_only:
        if not isolated:
            return None
        rules = ("order",)
    else:
        rules = ("complete", "order", "direct") if isolated else ("order", "complete", "direct")
    if len(symbols) >= 4 and not (skip_tangent or order_only):
        rules = ("tangent",) if tangent_only else ("tangent", *rules)
    intrinsic = mode == "intrinsic"
    whole = strict_components(symbols, equation, factors, separate=False)
    parts = strict_components(symbols, equation, factors, separate=not intrinsic)
    deadline = None if seconds is None else monotonic() + seconds

    def blowups(components, rules):
        # The repeated-tangent rule has five prescribed centres: it needs no turns.
        turns = (turn or SAFE_TURN_SECONDS) if safe and rules[0] != "tangent" else None
        return acampo_over(symbols, components, symbols, dimension=len(symbols), rules=rules, deadline=deadline,
                           budget=None if seconds is None or safe else 400,
                           count_strata=not (defer_outputs or intrinsic),
                           intrinsic=intrinsic and not full_boundary, rounds=turns)

    try:
        if rules[0] == "tangent" and (len(parts) > 1 or safe and len(rules) > 1):
            # The repeated-tangent rule reads the tangent cone of the whole hypersurface.
            try:
                data, strata, charts, centres = blowups(whole, rules[:1])
                parts = whole
            except GiveUp:
                if len(rules) == 1 or deadline is not None and monotonic() > deadline:
                    raise
                data, strata, charts, centres = blowups(parts, rules[1:])
        else:
            data, strata, charts, centres = blowups(parts, rules)
    except GiveUp as error:
        if reasons is not None:
            reasons.append(str(error))
        return None
    record = _record(names, charts, centres, data, strata, defer_outputs=defer_outputs or intrinsic,
                     intrinsic=intrinsic)
    if record is not None and len(parts) > 1:
        record["certificate"]["strict_components"] = RATIONAL_FACTORS
    if record is not None and intrinsic and not defer_outputs:
        _, output = certified_outputs(record, names, mode, False)
        record["dual_complex"] = output["dual_complex"]
    return record


def _isolated(symbols, equation):
    """Whether the singular point is isolated: its Tjurina number is finite (native local algebra)."""
    from .local_kernel import local_request
    try:
        length, *_ = local_request("local_tjurina_number", symbols, "grevlex", (equation,),
                                   point=(0,) * len(symbols))
    except (ArithmeticError, ValueError):
        return True  # The established order when the length is not available.
    return length != "infinite"


def _record(names, charts, centres, data, strata, *, defer_outputs, intrinsic=False):
    """Serialize certified charts from any of the successive smooth-centre rules."""
    from .atlas import to_json
    from .local_blowups import GiveUp, _hypersurface, _is_strict, cells, dual_cells
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
        strict = [ideal for label, _, ideal in divisors if _is_strict(label)]
        if len(strict) > 1:
            # The hypersurface is the union of its separate components.
            strict = [[_text(g) for g in _hypersurface(chart)]]
        final.append({
            "id": f"C{k}", "coordinates": [v.name for v in chart.variables],
            "ambient_ideal": [_text(g) for g in chart.ambient],
            "map": {name: _text(expr) for name, expr in chart.coordinates if name in names},
            "exceptional": [{"id": label, "ideal": ideal, "equation": ", ".join(ideal), "total_multiplicity": m}
                            for label, m, ideal in divisors if not _is_strict(label)],
            "strict_transform": strict[0] if strict else ["1"],
            # The chart is used off these centres, whose blowups other charts describe.
            **({"blown_up": [{"ideal": [_text(g) for g in z]} for z in chart.removed]} if chart.removed else {}),
        })
    steps = [{"id": centre["id"], "type": "blowup", "center": centre["kind"], "rule": centre["rule"],
              "dimension": centre["dimension"], "center_charts": centre["center_charts"],
              **({"invariant": centre["invariant"]} if "invariant" in centre else {}),
              "components": centre["components"]} for centre in centres]
    if defer_outputs:
        orders = {label: order for chart in charts for label, order, _, _ in chart.components}
        created = [{"id": label, "ambient_order": orders[label]} for label in births if label in orders]
    else:
        created = [{"id": row["component"], "ambient_order": row["multiplicity"]}
                   for row in rows if not _is_strict(row["component"])]
    points = sum(1 for centre in centres if centre["dimension"] == 0)
    curves = sum(1 for centre in centres if centre["dimension"] == 1)
    kinds = ", ".join(text for text in (f"{points} at points" if points else "", f"{curves} along curves" if curves else "",
                                        f"{len(centres) - points - curves} along larger centres"
                                        if len(centres) > points + curves else "") if text)
    conclusion = ("the strict transform is smooth" if intrinsic else "the total transform has normal crossings")
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
            "description": f"Every centre is smooth. In every final chart {conclusion} on its part of the origin "
                           "fibre by the Jacobian criterion. Properness gives the same conclusion over a "
                           "neighbourhood of the origin.",
            **({} if defer_outputs else {"numerical_data": rows, "strata": listed}),
        },
    }



if __name__ == "__main__":
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    asked = json.load(sys.stdin)
    why = []
    answer = _resolve({tuple(exponents): kernel_fraction(c) for exponents, c in asked["terms"]}, asked["variables"],
                      asked["mode"], asked["seconds"], defer_outputs=asked.get("defer_outputs", False),
                      tangent_only=asked.get("tangent_only", False), skip_tangent=asked.get("skip_tangent", False),
                      safe=asked.get("safe", False), reasons=why, turn=asked.get("turn"))
    print(json.dumps({"gave_up": why[0]} if answer is None and why else answer))
