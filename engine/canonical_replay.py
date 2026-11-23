"""Replay stored smooth-centre resolutions from their original hypersurface."""
from . import atlas, native_polynomial as np
from .exact_json import sorted_record
from .local_blowups import GiveUp, _Canonical, _Direct, _Order, _run
from .origin_backend import _record
from .repeated_tangent import RepeatedTangent
from .strata import InvalidChart


def canonical_outputs(record, polynomial, *, complex, intrinsic, integrals):
    steps = record.get("steps")
    if not steps or any(not isinstance(s, dict) or "rule" not in s for s in steps):
        return None
    rules = {s["rule"] for s in steps}
    smooth_endpoint = record["mode"] == "intrinsic"
    names = record["variables"]
    axes = tuple(map(np.Symbol, names))

    def reconstruct(support):
        components = [("strict", 1, 1, (np.from_terms(axes, support),))]
        order_rules = {"maximal order", "maximal order with boundary"}
        if rules & order_rules and rules <= order_rules | {"invariant", "surface", "complete", "shared boundary curve"}:
            rule = _Order(components, boundary_centres="maximal order with boundary" in rules)
        elif rules <= {"invariant", "surface", "complete", "shared boundary curve"}:
            rule = _Canonical(components, shared_curves="shared boundary curve" in rules)
        elif rules == {"direct"}:
            rule = _Direct()
        elif rules == {"repeated tangent"}:
            rule = RepeatedTangent()
        else:
            raise InvalidChart("Unknown sequence of canonical blowup rules.")
        try:
            data, strata, charts, centres = _run(rule, axes, components, axes, None, None,
                                                  max_blowups=len(steps), count_strata=False, intrinsic=smooth_endpoint)
        except GiveUp as exc:
            raise InvalidChart(f"The saved canonical blowup sequence failed replay. {exc}") from exc
        if len(centres) != len(steps):
            raise InvalidChart("The saved canonical blowup sequence differs from its defining germ.")
        return _record(names, charts, centres, data, strata, defer_outputs=True, intrinsic=smooth_endpoint)

    if "input_divisor" in record:
        from .weighted_record import nonreduced_resolution
        rebuilt = nonreduced_resolution(polynomial, names, record["mode"], reconstruct)
        if rebuilt is None or rebuilt["input_divisor"] != record["input_divisor"]:
            raise InvalidChart("The original divisor differs from the canonical record's factors.")
    else:
        rebuilt = reconstruct(polynomial)
    for given, actual in zip(steps, rebuilt["steps"]):
        if any(sorted_record(given[k]) != sorted_record(v) for k, v in actual.items() if k in given):
            raise InvalidChart("The saved canonical centre differs from the reconstructed blowup.")

    expected = rebuilt["atlas"]
    if "atlas" in record and record["atlas"] != expected:
        # Canonical sparse polynomial transport also accepts equivalent
        # expressions in records from older writers.
        if atlas.to_json(atlas.from_json(record["atlas"], native=True)) != expected:
            raise InvalidChart("The saved canonical charts differ from the complete reconstructed cover.")
    if "atlas" not in record and not record.get("charts"):
        return None
    raw = record.get("charts")
    sheets = raw is not None and any("rational_exceptional" in c for c in raw)
    from .general_atlas import native_general_outputs
    output = native_general_outputs(record, expected, complex=complex or sheets, intrinsic=intrinsic,
                                    integrals=integrals, factor_fibre=rules == {"repeated tangent"},
                                    canonical_cover=True)
    if raw is not None:
        actual = rebuilt["charts"]
        if sheets:
            from .atlas_finite import geometric_chart_records
            actual = geometric_chart_records(dict(charts=actual, dual_complex=output["dual_complex"]))
        if raw != actual:
            raise InvalidChart("The displayed canonical charts differ from the complete reconstructed cover.")
    return expected, output
