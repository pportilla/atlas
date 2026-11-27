"""Select local zeta functions and evaluate them on the same resolution strata."""
from .resolution_errors import computation_limits

from .polynomial import InputError, LimitError, parse, parse_expression, saved_polynomial_input

KINDS = ("monodromy", "topological")


def selected_zetas(request):
    legacy = request.get("monodromy", False)
    if not isinstance(legacy, bool):
        raise InputError("monodromy must be true or false.")
    only = request.get("zetas_only", False)
    if not isinstance(only, bool):
        raise InputError("zetas_only must be true or false.")
    if only and (request.get("mode", "embedded") != "embedded" or request.get("zetas") == []):
        raise InputError("zetas_only requires embedded mode and a selected zeta function.")
    if "zetas" not in request:
        return KINDS if legacy or only else ()
    kinds = request["zetas"]
    if not isinstance(kinds, list) or any(kind not in KINDS for kind in kinds):
        raise InputError("zetas must be a list containing monodromy, topological, or both.")
    return tuple(kind for kind in KINDS if kind in kinds)


def zetas_from_strata(record, kinds, strata, *, companion=False, polynomial=None):
    if strata is None:
        reason = "The bounded search has not certified the embedded strata over the origin."
        if record.get("status") == "inconclusive":
            reason = (record.get("diagnostic") or {}).get("description") or record.get("summary") or reason
        results = {kind + "_zeta": {"status": "inconclusive", "reason": reason} for kind in kinds}
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
    return with_pole_check(results)


def with_pole_check(results):
    if len(results) == len(KINDS) and all(zeta["status"] == "computed" for zeta in results.values()):
        from .exact_json import sorted_record
        from .session_monodromy import check_zetas
        check = check_zetas(results["topological_zeta"], results["monodromy_zeta"])
        results["monodromy_check"] = sorted_record(check)
    return results


def finish_requested_check(results, record, search):
    check = results.get("monodromy_check")
    if search == "exhaustive" and check and check["status"] == "unresolved":
        from .exact_json import sorted_record
        from .session_monodromy import finish_pole_check
        results["monodromy_check"] = sorted_record(finish_pole_check(check, record))
    return results


def _selected_outputs(record, kinds, *, companion=False):
    results = {kind + "_zeta": record[kind + "_zeta"] for kind in kinds}
    if companion:
        results = {key: dict(value, source="companion embedded resolution") for key, value in results.items()}
    if "monodromy_check" in record and len(kinds) == len(KINDS):
        results["monodromy_check"] = record["monodromy_check"]
    return results


def attach_companion(record, embedded, kinds):
    """Keep the companion's geometric evidence for replay and copy its selected results."""
    if all(embedded[k + "_zeta"]["status"] == "computed" for k in kinds):
        record["embedded_companion"] = {k: v for k, v in embedded.items()
                                        if k not in ("monodromy_zeta", "topological_zeta", "monodromy_check")}
    record.update(_selected_outputs(embedded, kinds, companion=True))


@saved_polynomial_input()
def replay_zetas(request):
    """Verify saved evidence or recompute missing embedded data from the defining germ."""
    results = _replay_zetas(request)
    # A budget in seconds makes the search a quick one.
    policy = "quick" if request.get("seconds") is not None else request.get("search", "exhaustive")
    return finish_requested_check(results, request["record"], policy)


def _replay_zetas(request):
    from .strata import resolution_strata
    record = request.get("record")
    if not isinstance(record, dict) or record.get("status") not in ("resolved", "inconclusive", "zetas_computed"):
        raise InputError("Supply an Atlas resolution record with its defining germ.")
    from .quick_budget import request_seconds, with_budget
    try:
        # A budget in seconds makes the search a quick one.
        request = with_budget(request)
        policy = request.get("search", "exhaustive")
        if policy not in ("quick", "exhaustive"):
            raise InputError("Search must be quick or exhaustive.")
        request_seconds(request)
    except ValueError as error:
        raise InputError(str(error)) from None
    if request.get("compute") not in (None, "auto", "cpu", "gpu"):
        raise InputError("Compute must be cpu, auto, or gpu.")
    names = record.get("variables")
    source = record.get("polynomial")
    poly = parse(source, names, bounded=False)
    if record.get("mode") not in ("embedded", "intrinsic") or record.get("origin", [0]*len(names)) != [0]*len(names):
        raise InputError("The resolution must specify an embedded or intrinsic germ at the coordinate origin.")
    if record["status"] == "resolved" and record["mode"] == "intrinsic":
        from .identity_atlas import reduced_at_origin
        from .strata import InvalidChart
        if not reduced_at_origin(poly, names):
            raise InvalidChart("An intrinsic resolution requires a reduced hypersurface at the origin.")
    kinds = selected_zetas(request)
    if not kinds:
        raise InputError("Select a zeta function.")
    # Imported zeta fields may be stale. Replay any computed member of the
    # pair alongside the newly requested one on the very same saved strata.
    kinds = tuple(kind for kind in KINDS if kind in kinds or
                  (isinstance(saved := record.get(kind + "_zeta"), dict) and saved.get("status") == "computed"))
    embedded = record.get("embedded_companion")
    if embedded is not None:
        if record["mode"] != "intrinsic" and record.get("engine") != "weighted":
            raise InputError("An ordinary embedded record must carry its own certificate.")
        if not isinstance(embedded, dict):
            raise InputError("Invalid embedded companion.")
        if "polynomial" in embedded:
            if (embedded.get("mode") != "embedded" or embedded.get("variables") != names or
                    "embedded_companion" in embedded or embedded.get("engine") == "weighted"):
                raise InputError("The embedded companion must certify the same germ in its original coordinates.")
            if (source != embedded["polynomial"] and
                    parse_expression(source, names, as_row=True) != parse_expression(embedded["polynomial"], names, as_row=True)):
                raise InputError("The embedded companion disagrees with the original polynomial.")
            result = replay_zetas(dict(request, record=embedded, zetas=list(kinds)))
            return _selected_outputs(result, kinds, companion=True)
        if not isinstance(embedded.get("atlas"), list) or not embedded["atlas"]:
            raise InputError("The embedded companion needs its resolution atlas.")
    if "contact_certificate" in record:
        from .contact_zetas import replay_contact
        return replay_contact(record, kinds)
    if record.get("status") == "zetas_computed":
        raise InputError("The saved zeta calculation is missing its contact certificate.")
    if record["status"] == "resolved" and record.get("engine") != "weighted":
        strata = resolution_strata(record, poly, strict=True)
        if strata is not None:
            return zetas_from_strata(record, kinds, strata, companion=bool(embedded), polynomial=poly)
    from .resolve import resolve
    calculation = dict(polynomial=source, variables=names, mode="embedded", method="auto",
                       search=policy, compute=request.get("compute"), zetas=list(kinds), zetas_only=True)
    companion = record["mode"] == "intrinsic" or record.get("engine") == "weighted"
    if policy == "exhaustive" and record["status"] == "resolved":
        # An older producer may have found a short resolution using its
        # bounded choice of centres. Recover it before enumerating ideals.
        try:
            fresh = resolve(dict(calculation, search="quick", bounded=True))
        except computation_limits():
            fresh = {}
        if all(fresh.get(k + "_zeta", {}).get("status") == "computed" for k in kinds):
            return _selected_outputs(fresh, kinds, companion=companion)
    if policy == "quick":
        fresh = _budgeted(dict(calculation, **({} if request.get("seconds") is None else {"seconds": request["seconds"]})))
    else:
        fresh = resolve(calculation)
    return _selected_outputs(fresh, kinds, companion=companion)


def _budgeted(calculation):
    """A quick solve of the germ in a worker of its own, which is stopped when its time budget ends."""
    from .resolve_cli import run_job
    status, fresh = run_job(calculation)
    if status == 200:
        return fresh
    reason = fresh.get("error", "The quick search did not return a record.")
    if status == 400:
        raise InputError(reason)
    if status == 422:
        raise LimitError(reason)
    raise RuntimeError(reason)
