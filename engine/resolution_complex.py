"""Compute the exceptional dual complex of a saved Atlas resolution."""
from .polynomial import InputError, LimitError, parse, saved_polynomial_input


def _named_intrinsic_boundary(record, polynomial, verified):
    """Recover the native branch names on a verified plane or A-series atlas."""
    from .strata import annotate_complex, rational_chart_outputs
    boundary = verified[1]["dual_complex"]
    engine = record.get("engine")
    names = record["variables"]
    charts = record.get("atlas") or record.get("charts", ())
    if engine == "native-a-series" or (
        not engine and len(names) == 3 and all("split_cone" in c for c in charts)
    ):
        from .chain_backend import resolve_chain as construct
    elif engine == "C++ / FLINT plane point blowups" or (
        not engine and len(names) == 2 and all("point_blowup" in c or "parent_chart" in c for c in charts)
    ):
        from .plane_backend import resolve_plane as construct
    else:
        return boundary

    # The generic verifier labels components by their ambient exceptional
    # divisor. Recover the original branch markings from the input only when
    # the native model gives the same chart cover as the verified archive.
    try:
        model = construct(polynomial, names, "intrinsic")
    except (LimitError, RuntimeError):
        return boundary
    if not model or model.get("status") != "resolved":
        return boundary
    if "charts" in record and record["charts"] != model["charts"]:
        return boundary
    model.update(mode="intrinsic", variables=names, polynomial=record["polynomial"])
    original = rational_chart_outputs(model, polynomial, intrinsic=True, integrals=False)
    if original is None or original[0] != verified[0]:
        return boundary
    return annotate_complex(model["dual_complex"], boundary)


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
        if intrinsic:
            return _named_intrinsic_boundary(record, polynomial, computed)
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
