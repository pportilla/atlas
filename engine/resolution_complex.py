"""Compute the exceptional dual complex of a saved Atlas resolution."""
from .polynomial import InputError, parse


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
    computed = rational_chart_outputs(record, polynomial, intrinsic=intrinsic, integrals=False)
    if computed is not None:
        return computed[1]["dual_complex"]
    from . import atlas
    from .centres import GiveUp
    charts = atlas.from_record(record, polynomial)
    if not charts:
        raise InputError("This record lacks the chart cover required for geometric incidence.")
    factored = any(step.get("rule") == "repeated tangent" for step in record.get("steps", ()))
    try:
        computed = atlas.outputs(charts, len(names), intrinsic=intrinsic, integrals=False,
                                 factor_fibre=factored)
    except GiveUp as error:
        raise InputError(str(error)) from error
    return computed["dual_complex"]
