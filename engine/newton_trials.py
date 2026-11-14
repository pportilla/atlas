"""Try regular monomial ideal blowups and verify the original equation in every chart."""
from .newton_fan import bounded_fan, build_fan, trial_supports
from .polynomial import LimitError
from .strata import InvalidChart


def _record(names, support, fan, centres, charts):
    proof = {"method": "regular monomial modification and chart Jacobian criteria",
             "fan_ideal_exponents": [list(e) for e in sorted(support)],
             "boundary_image_coordinate_ideals": centres,
             "description": "The fan is regular and covers the positive orthant. The exceptional "
                            "images lie in the singular locus. Each chart uses the original polynomial."}
    return dict(status="resolved", engine="verified-monomial-modification", variables=names, mode="embedded",
                  charts=charts, steps=[{"id":"Toric", "type":"regular toric modification"}],
                  summary=f"Certified toric modification with {len(charts)} charts.", fan=fan, certificate=proof)


def _try(poly, names, support, *, fan=None, bounded=False, verified=None, integrals=False):
    from .newton_backend import _boundary_centres, monomial
    from .newton_faces import exhaustive_faces
    if fan is None:
        fan = build_fan(support, exhaustive=exhaustive_faces())
    if len(fan["cones"]) > 32768:
        return None
    if verified is not None:
        from .monomial_construction import construct
        value = construct(poly, names, fan, integrals)
        if value is not None:
            centres, charts, atlas, output = value
            result = _record(names, support, fan, centres, charts)
            result["dual_complex"] = output["dual_complex"]
            verified.append((result, (atlas, output)))
            return result
    centres = _boundary_centres(poly, names, fan) if fan["boundary_rays"] else []
    if centres is None:
        return None
    coordinates = [f"u{i+1}" for i in range(len(names))]
    charts = []
    for k, cone in enumerate(fan["cones"], 1):
        rays = [fan["rays"][i] for i in cone]
        mapping = dict(zip(names, (monomial(coordinates, e) for e in zip(*rays))))
        charts.append(dict(id=f"N{k}", coordinates=coordinates, ray_indices=cone, map=mapping))
    result = _record(names, support, fan, centres, charts)
    from .monomial_atlas import fan_outputs
    try:
        checked = fan_outputs(result, poly, integrals=False, bounded=bounded)
    except InvalidChart:
        return None
    if checked is None:
        return None
    atlas, outputs = checked
    # The public charts and serialized atlas describe the same pullbacks.
    # Keep the compact fan chart format so the shared verifier will stream
    # the original equation on the requested fibre when called by the solver.
    for chart, owned in zip(charts, atlas):
        strict = [c for c in owned["components"] if c["id"] == "strict"]
        chart["strict_transform"] = strict[0]["ideal"][0] if strict else "1"
        chart["exceptional"] = [{"id":c["id"], "equation":c["ideal"][0], "total_multiplicity":c["multiplicity"]}
                                for c in owned["components"] if c["id"] != "strict"]
    result["dual_complex"] = outputs["dual_complex"]
    return result


def resolve_trials(poly, names, *, verified=None, integrals=False):
    for support in trial_supports(poly):
        try:
            result = _try(poly, names, support, verified=verified, integrals=integrals)
        except LimitError:
            continue
        if result is not None:
            return result
    return None


def bounded_first_trial(poly, names, *, verified=None, integrals=False):
    """Try only the first Newton monomial candidate with fixed native work."""
    from . import native_library
    if (native_library._MODE != "native" or not 3 <= len(names) <= 4 or len(poly) > 3
            or any(c.numerator.bit_length() > 64 or c.denominator.bit_length() > 64 for c in poly.values())
            or not all(hasattr(native_library._library(), name) for name in
                       ("atlas_bounded_crossings_abi", "atlas_fan_charts_abi"))):
        return None
    from .newton_backend import _boundary_centres
    support = list(poly)
    source_fan = bounded_fan(support)
    # Newton tries monomial ideals only after the source fan's boundary
    # modification fails the singular-centre check. Any positive or uncertain
    # preceding case is left to the original exhaustive race.
    if not source_fan or not source_fan["boundary_rays"] or _boundary_centres(poly, names, source_fan) is not None:
        return None
    proposals = bounded_fan(support, trials=True)
    if not proposals:
        return None
    first = proposals[0]
    fan = bounded_fan(first)
    if (not fan or len(fan["cones"]) > 4
            or any(value > 3 for ray in fan["rays"] for value in ray)):
        return None
    return _try(poly, names, first, fan=fan, bounded=True, verified=verified, integrals=integrals)
