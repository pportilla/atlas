"""Compute the exceptional dual complex of a saved Atlas resolution."""
from .polynomial import InputError, parse, saved_polynomial_input


@saved_polynomial_input()
def dual_complex(record):
    if not isinstance(record, dict) or record.get("status") != "resolved":
        raise InputError("Supply a resolved Atlas record with its charts.")
    if record.get("mode") not in ("embedded", "intrinsic"):
        raise InputError("The resolution needs its embedded or intrinsic mode.")
    names = record.get("variables")
    if not isinstance(names, list) or not names or record.get("origin", [0]*len(names)) != [0]*len(names):
        raise InputError("The resolution needs its coordinates and must be at the origin.")
    if not (record.get("atlas") or record.get("charts")):
        raise InputError("This record lacks the chart cover required for geometric incidence.")
    if record.get("engine") == "toric-graph":
        from .toric_graph import graph_complex
        return graph_complex(record)

    from .strata import rational_chart_outputs
    polynomial = parse(record.get("polynomial"), names, bounded=False)
    intrinsic = record["mode"] == "intrinsic"
    if intrinsic:
        from .identity_atlas import reduced_at_origin
        from .strata import InvalidChart
        if not reduced_at_origin(polynomial, names):
            raise InvalidChart("An intrinsic resolution requires a reduced hypersurface at the origin.")
    computed = rational_chart_outputs(record, polynomial, intrinsic=intrinsic, integrals=False, replay_canonical=True)
    if computed is not None:
        return computed[1]["dual_complex"]
    if record.get("engine") == "canonical-blowups":
        return dict(status="partial", vertices=[], cells=[], convention="Exceptional divisor over the origin.",
                    reason="The saved record lacks the canonical centre rule needed to reconstruct its chart cover.")
    from . import atlas
    from .centres import GiveUp
    charts, reason = atlas.verified_modification_charts(record, polynomial)
    if not charts:
        return dict(status="partial", vertices=[], cells=[],
                    convention="Exceptional divisor in the germ over the origin over Qbar.", reason=reason)
    factored = any(step.get("rule") == "repeated tangent" for step in record.get("steps", ()))
    try:
        computed = atlas.outputs(charts, len(names), intrinsic=intrinsic, integrals=False, factor_fibre=factored)
    except GiveUp as error:
        raise InputError(str(error)) from error
    return computed["dual_complex"]
