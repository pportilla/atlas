"""Try regular monomial ideal blowups and verify the original equation in every chart."""
from .newton_fan import bounded_fan, build_fan, trial_supports
from .polynomial import LimitError, parse_expression
from .strata import InvalidChart


def _record(names, support, fan, centres, charts, mode):
    proof = {"method": "regular monomial modification and chart Jacobian criteria",
             "fan_ideal_exponents": [list(e) for e in sorted(support)],
             "boundary_image_coordinate_ideals": centres,
             "description": "The fan is regular and covers the positive orthant. The exceptional "
                            "images lie in the singular locus. Each chart uses the original polynomial."}
    return dict(status="resolved", engine="verified-monomial-modification", variables=names, mode=mode,
                  charts=charts, steps=[{"id":"Toric", "type":"regular toric modification"}],
                  summary=f"Certified toric modification with {len(charts)} charts.", fan=fan, certificate=proof)


def _chart_divisors(owned, sections):
    strict = "1"
    ambient = []
    for component in owned["components"]:
        equation, = component["ideal"]
        if component["id"] == "strict":
            strict = equation
        else:
            ambient.append(dict(id=component["id"], equation=equation, total_multiplicity=component["multiplicity"]))
    if sections is None:
        return dict(strict_transform=strict, exceptional=ambient)
    local = [dict(section, equation=", ".join(section["ideal"])) for section in sections]
    return dict(strict_transform=strict, ambient_components=ambient, exceptional=local)


def verify_intrinsic_presentation(record, atlas, output):
    """Cross-check public fan charts against rederived pullbacks and local sections."""
    charts = record["charts"]
    if (len(charts) != len(atlas) or len({chart.get("id") for chart in charts}) != len(atlas) or
            record.get("root_families", {}) != output.get("root_families", {})):
        raise InvalidChart("The displayed chart cover or root labels differ from the verified intrinsic fan.")
    fan = record.get("fan", {})
    rays = fan.get("rays", ())
    cones = [tuple(chart.get("ray_indices", ())) for chart in charts]
    if (not rays or any(len(ray) != len(record["variables"]) or any(type(k) is not int for k in ray) for ray in rays)
            or any(type(i) is not int for cone in cones for i in cone)
            or {i for cone in cones for i in cone} != set(range(len(rays)))
            or sorted(cones) != sorted(map(tuple, fan.get("cones", ())))):
        raise InvalidChart("The displayed cones differ from the intrinsic fan chart cover.")
    for chart, owned, sections in zip(charts, atlas, output["chart_sections"], strict=True):
        coordinates = owned["coordinates"]
        indices = chart.get("ray_indices", ())
        if (chart.get("coordinates") != coordinates or
                [tuple(rays[i]) for i in indices] != list(map(tuple, owned["torus"]["rays"]))):
            raise InvalidChart("The displayed intrinsic chart does not match its verified cone.")
        def agrees(actual, expected):
            return isinstance(actual, str) and (actual == expected or
                parse_expression(actual, coordinates, as_row=True) == parse_expression(expected, coordinates, as_row=True))
        mapping = chart.get("map", {})
        if set(mapping) != set(owned["map"]) or any(not agrees(mapping[k], v) for k, v in owned["map"].items()):
            raise InvalidChart("The displayed intrinsic chart projection differs from its verified map.")
        shown = _chart_divisors(owned, sections)
        if not agrees(chart.get("strict_transform"), shown["strict_transform"]):
            raise InvalidChart("The displayed strict transform differs from its verified pullback.")
        for kind in ("ambient_components", "exceptional"):
            expected = {c["id"]: c for c in shown[kind]}
            reported = chart.get(kind, ())
            if len(reported) != len(expected) or {c.get("id"): c for c in reported} != expected:
                raise InvalidChart("The displayed exceptional ideals differ from the verified intrinsic sections.")


def _try(poly, names, support, *, mode="embedded", fan=None, bounded=False, verified=None, integrals=False,
         construct=True):
    from .newton_backend import _boundary_centres, monomial
    from .newton_faces import exhaustive_faces
    if fan is None:
        fan = build_fan(support, exhaustive=exhaustive_faces())
    if len(fan["cones"]) > 32768:
        return None
    if verified is not None and construct and mode == "embedded":
        from .monomial_construction import construct as native_construct
        value = native_construct(poly, names, fan, integrals)
        if value is not None:
            centres, charts, atlas, output = value
            result = _record(names, support, fan, centres, charts, mode)
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
    result = _record(names, support, fan, centres, charts, mode)
    from .monomial_atlas import fan_outputs
    try:
        intrinsic = mode == "intrinsic"
        checked = fan_outputs(result, poly, intrinsic=intrinsic, integrals=False, bounded=bounded,
                              local_sections=intrinsic)
    except InvalidChart:
        return None
    if checked is None:
        return None
    atlas, outputs = checked
    # The public charts and serialized atlas describe the same pullbacks.
    # Keep the compact fan chart format so the shared verifier will stream
    # the original equation on the requested fibre when called by the solver.
    for index, (chart, owned) in enumerate(zip(charts, atlas, strict=True)):
        chart.update(_chart_divisors(owned, outputs["chart_sections"][index] if intrinsic else None))
    if "root_families" in outputs:
        result["root_families"] = outputs["root_families"]
    result["dual_complex"] = outputs["dual_complex"]
    if verified is not None and not bounded and not integrals:
        # The caller attaches this certified atlas instead of verifying the
        # same fan charts again. Zeta requests still verify with integrals.
        verified.append((result, (atlas, outputs)))
    return result


def resolve_trials(poly, names, *, mode="embedded", verified=None, integrals=False, construct=True):
    def supports():
        if mode == "intrinsic":
            yield poly  # The original fan has not yet been tested on the strict transform.
        yield from trial_supports(poly)

    for support in supports():
        try:
            result = _try(poly, names, support, mode=mode, verified=verified, integrals=integrals, construct=construct)
        except LimitError:
            continue
        if result is not None:
            return result
    return None


def bounded_first_trial(poly, names, *, verified=None, integrals=False):
    """Try only the first Newton monomial candidate with fixed native work."""
    from . import native_library
    if (not 3 <= len(names) <= 4 or len(poly) > 3
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


def bounded_coordinate_graph(poly, names):
    """Test the exhaustive solver’s ordered graph prefix after a certified Newton obstruction."""
    from . import native_library
    if (not all(hasattr(native_library._library(), marker) for marker in
                    ("atlas_transverse_pencil_abi", "atlas_coordinate_exceptional_abi"))
            or any(c.numerator.bit_length() > 64 or c.denominator.bit_length() > 64 for c in poly.values())):
        return None
    from .point_geometry import encode_terms
    from .kernel_json import kernel_json
    centre = kernel_json("atlas_resolution_candidate", "transverse_pencil", width=len(names), terms=encode_terms(poly))
    if centre is None:
        return None
    from .general_resolution import search
    return search(poly, names, "embedded", exhaustive=True, transverse_centre=centre)
