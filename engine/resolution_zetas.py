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


def zetas_from_strata(record, kinds, strata, *, companion=False, polynomial=None):
    if strata is None:
        reason = ("The resolution has not been certified." if record.get("status") != "resolved" else
                  "These charts do not certify the embedded strata over the origin. "
                  "Compute an embedded resolution to obtain the missing data.")
        results = {kind + "_zeta": {"status": "unavailable", "reason": reason} for kind in kinds}
    else:
        results = {}
        if "monodromy" in kinds:
            from .monodromy import monodromy_zeta
            results["monodromy_zeta"] = monodromy_zeta(record, strata=strata, polynomial=polynomial)
        if "topological" in kinds:
            from .topological import topological_zeta
            results["topological_zeta"] = topological_zeta(
                record, strata=strata, monodromy=results.get("monodromy_zeta"))
    if companion:
        for zeta in results.values():
            zeta["source"] = "companion embedded resolution"
    if len(results) == len(KINDS) and all(zeta["status"] == "computed" for zeta in results.values()):
        from .exact_json import sorted_record
        from .session_monodromy import check_zetas
        check = check_zetas(results["topological_zeta"], results["monodromy_zeta"])
        results["monodromy_check"] = sorted_record(check)
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
    # Imported zeta fields may be stale. Replay any computed member of the
    # pair alongside the newly requested one on the very same saved strata.
    kinds = tuple(kind for kind in KINDS if kind in kinds or
                  (isinstance(saved := record.get(kind + "_zeta"), dict) and saved.get("status") == "computed"))
    companion = record.get("mode") == "intrinsic" and bool(record.get("embedded_companion"))
    return zetas_from_strata(record, kinds, resolution_strata(record), companion=companion)
