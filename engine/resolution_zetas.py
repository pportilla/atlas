"""Select local zeta functions and evaluate them on the same resolution strata."""
from .polynomial import InputError

KINDS = ("monodromy", "topological")


def selected_zetas(request):
    legacy = request.get("monodromy", False)
    if not isinstance(legacy, bool):
        raise InputError("monodromy must be true or false.")
    if "zetas" not in request:
        return KINDS if legacy else ()
    kinds = request["zetas"]
    if not isinstance(kinds, list) or any(kind not in KINDS for kind in kinds):
        raise InputError("zetas must be a list containing monodromy, topological, or both.")
    return tuple(kind for kind in KINDS if kind in kinds)


def zetas_from_strata(record, kinds, strata):
    if strata is None:
        reason = ("The resolution has not been certified." if record.get("status") != "resolved" else
                  "These charts do not certify the embedded strata over the origin. "
                  "Compute an embedded resolution to obtain the missing data.")
        return {kind + "_zeta": {"status": "unavailable", "reason": reason} for kind in kinds}
    results = {}
    if "monodromy" in kinds:
        from .monodromy import monodromy_zeta
        results["monodromy_zeta"] = monodromy_zeta(record, strata=strata)
    if "topological" in kinds:
        from .topological import topological_zeta
        results["topological_zeta"] = topological_zeta(
            record, strata=strata, monodromy=results.get("monodromy_zeta"))
    return results


def replay_zetas(request):
    """Read only saved charts, including any embedded companion on the record."""
    from .strata import resolution_strata
    record = request.get("record")
    if not isinstance(record, dict) or record.get("status") != "resolved":
        raise InputError("Supply a resolved Atlas record.")
    kinds = selected_zetas(request)
    if not kinds:
        raise InputError("Select a zeta function.")
    found = resolution_strata(record)
    results = zetas_from_strata(record, kinds, found)
    if record.get("mode") == "intrinsic" and record.get("embedded_companion"):
        for zeta in results.values():
            zeta["source"] = "companion embedded resolution"
    return results
