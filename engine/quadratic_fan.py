"""Transport verified fan charts through an origin-preserving completion of the square."""
from .kernel_json import kernel_json
from .native_library import affine_request
from .polynomial import encode_sparse_row, format_poly, parse_expression, sparse_row
from .strata import InvalidChart


def translations(polynomial, names, *, ramification=False):
    operation = "ramification_translations" if ramification else "quadratic_translations"
    choices = kernel_json("atlas_resolution_candidate", operation,
                          width=len(names), row=encode_sparse_row(polynomial))
    for choice in choices:
        shift = format_poly(sparse_row(choice["shift"]), names)
        change = {"coordinate": names[choice["axis"]], "offset": shift}
        if ramification:
            change["method"] = "ramification graph"
        yield change, sparse_row(choice["adapted"])


def translate_map(chart, names, translation, *, inverse=False):
    """Compose one chart projection with the indicated triangular automorphism."""
    axis = translation["coordinate"]
    projection = chart["map"]
    if set(projection) != set(names):
        raise InvalidChart("The translated fan chart needs one image per input coordinate.")
    sign = "-" if inverse else "+"
    image = f"{axis}{sign}({translation['offset']})"
    coords = chart["coordinates"]
    row, = affine_request(["pullbacks", " ".join(coords), " ".join(names),
                           *(projection[name] for name in names), "0", image])
    return {**projection, axis: format_poly(sparse_row(row), coords)}


def _adapted_fan(adapted, names, mode, compute, *, ramification=False):
    if len(names) == 3 and not ramification:
        from .toric_backend import _diagonal
        diagonal = _diagonal(adapted, names, mode, compute)
        if diagonal is not None:
            return diagonal
    from .newton_trials import _try
    return _try(adapted, names, adapted, mode=mode, construct=False)


def _translated_certificate(proof, names, translation):
    proof = dict(proof, fan_coordinate_frame="adapted")
    if "boundary_image_coordinate_ideals" in proof:
        centres = proof.pop("boundary_image_coordinate_ideals")
        proof["adapted_boundary_image_coordinate_ideals"] = centres
        identity = dict(zip(names, names))
        images = translate_map({"coordinates": names, "map": identity}, names, translation, inverse=True)
        proof["boundary_image_ideals"] = [[images[coordinate] for coordinate in centre] for centre in centres]
    return proof


def resolve_translated(polynomial, names, mode, compute=None, *, ramification=False, expected_translation=None):
    for translation, adapted in translations(polynomial, names, ramification=ramification):
        if expected_translation is not None and translation != expected_translation:
            continue
        record = _adapted_fan(adapted, names, mode, compute, ramification=ramification)
        if record is None:
            continue
        for chart in record["charts"]:
            chart["map"] = translate_map(chart, names, translation)
        proof = record["certificate"] = _translated_certificate(record["certificate"], names, translation)
        proof.update(coordinate_translation=translation, adapted_polynomial=format_poly(adapted, names))
        method = "Straightening a repeated ramification branch" if ramification else "Completing the square"
        proof["description"] = (method + " gives an origin-preserving polynomial change of coordinates "
                                "with Jacobian determinant one. " + proof["description"])
        return record
    return None


def _verify_display(record, model, translation):
    """Check the displayed fan charts, including split intrinsic sections."""
    from .exact_json import result_json, sorted_record
    names = record["variables"]
    def same(a, b):
        return result_json(sorted_record(a)) == result_json(sorted_record(b))
    if "fan" in record and (not isinstance(record["fan"], dict) or
                            any(k not in model["fan"] or not same(v, model["fan"][k])
                                for k, v in record["fan"].items())):
        raise InvalidChart("The declared fan differs from the translated charts.")
    if "root_families" in record and not same(record["root_families"], model.get("root_families", {})):
        raise InvalidChart("The root families differ from the translated fan sections.")
    proof = record.get("certificate", {})
    expected = _translated_certificate(model["certificate"], names, translation)
    # Older translated records kept the image ideals in adapted coordinates.
    legacy = "boundary_image_coordinate_ideals"
    if legacy in proof and ("fan_coordinate_frame" in proof or legacy not in model["certificate"]
                            or not same(proof[legacy], model["certificate"][legacy])):
        raise InvalidChart("The declared boundary images differ from the adapted fan.")
    fields = ("weights", "weighted_degree", "principal_exponents", "fan_ideal_exponents", "fan_coordinate_frame",
              "adapted_boundary_image_coordinate_ideals", "boundary_image_ideals")
    if any(k in proof and (k not in expected or not same(proof[k], expected[k])) for k in fields):
        raise InvalidChart("The declared fan certificate differs from the translated polynomial.")
    if "charts" not in record:
        return
    if len(record["charts"]) != len(model["charts"]):
        raise InvalidChart("The displayed charts differ from the translated fan.")
    for chart, source in zip(record["charts"], model["charts"]):
        if any(chart.get(k) != source.get(k) for k in ("id", "coordinates", "ray_indices", "exceptional", "ambient_components")):
            raise InvalidChart("The displayed fan chart has inconsistent rays or exceptional ideals.")
        coords = chart["coordinates"]
        def agrees(actual, wanted):
            return isinstance(actual, str) and (
                actual == wanted or parse_expression(actual, coords) == parse_expression(wanted, coords))
        if ("pullback_factor" in chart) != ("pullback_factor" in source) or any(
                not agrees(chart.get(k), source[k]) for k in ("strict_transform", "pullback_factor") if k in source):
            raise InvalidChart("The displayed chart equations differ from their computed pullback.")
        projection = translate_map(source, names, translation)
        if set(chart["map"]) != set(names) or any(not agrees(chart["map"][v], projection[v]) for v in names):
            raise InvalidChart("The displayed projection differs from the translated fan.")


def _verify_owned_fan(charts, fan, n):
    """Match the labelled geometric divisors in the owned cover with the fan."""
    from .monomial_atlas import _same_principal
    rays = list(map(tuple, fan["rays"]))
    identity = tuple(rays[:n])
    labels = {ray: f"E{i}" for i, ray in enumerate(rays[n:], 1)}
    cones = {frozenset(rays[i] for i in cone) for cone in fan["cones"]}
    for chart in charts:
        torus = chart.get("torus", {})
        cone = tuple(map(tuple, torus.get("rays", ())))
        key = frozenset(cone)
        if key not in cones or tuple(map(tuple, torus.get("linear_map", ()))) != identity:
            raise InvalidChart("The owned cones differ from the translated fan.")
        cones.remove(key)
        coords = chart["coordinates"]
        if len(cone) != len(coords):
            raise InvalidChart("The owned coordinates differ from the translated fan.")
        expected = {labels[ray]: axis for ray, axis in zip(cone, coords) if ray in labels}
        components = [c for c in chart["components"] if c["id"] != "strict"]
        if ({c["id"] for c in components} != set(expected)
                or any(not _same_principal(coords, c["ideal"], expected[c["id"]]) for c in components)):
            raise InvalidChart("The owned divisor labels differ from the translated fan.")
    if cones:
        raise InvalidChart("The owned charts omit cones of the translated fan.")


def _weighted_outputs(record, polynomial, translation, options):
    """Lift and independently check the original divisor on the translated support charts."""
    from . import atlas
    from .exact_json import result_json, sorted_record
    from .general_atlas import native_general_outputs
    from .weighted_record import nonreduced_resolution

    names = record["variables"]
    if record["mode"] != "embedded" or not (record.get("atlas") or record.get("charts")):
        raise InvalidChart("A translated effective divisor requires an embedded chart cover.")

    def reconstruct(support):
        model = resolve_translated(support, names, "embedded", "cpu", expected_translation=translation,
                                   ramification=translation.get("method") == "ramification graph")
        if model is None:
            raise InvalidChart("The reduced divisor has no such verified coordinate translation.")
        return dict(model, variables=names, mode="embedded")

    model = nonreduced_resolution(polynomial, names, "embedded", reconstruct)
    if model is None:
        raise InvalidChart("The coordinate translation disagrees with the defining polynomial.")
    owned = atlas.to_json(atlas.from_record(model, polynomial, native=True))
    model["atlas"] = owned
    def same(a, b):
        return result_json(sorted_record(a)) == result_json(sorted_record(b))
    for key in ("atlas", "charts", "input_divisor", "fan", "root_families", "steps"):
        if key in record and not same(record[key], model.get(key)):
            raise InvalidChart(f"The saved translated divisor {key} differs from its reconstruction.")
    proof = record.get("certificate", {})
    for key in ("weights", "weighted_degree", "principal_exponents", "fan_ideal_exponents", "fan_coordinate_frame",
                "boundary_image_ideals", "adapted_boundary_image_coordinate_ideals", "adapted_polynomial"):
        if key in proof and not same(proof[key], model["certificate"].get(key)):
            raise InvalidChart("The saved translated divisor certificate differs from its reconstruction.")
    return owned, native_general_outputs(model, owned, complex=options["complex"], integrals=options["integrals"])


def translated_outputs(record, polynomial, translation, verify, options):
    """Verify the fan after undoing the coordinate translation on every chart map."""
    names = record["variables"]
    if not isinstance(translation, dict):
        raise InvalidChart("A coordinate translation needs its coordinate and offset.")
    ramification = translation.get("method") == "ramification graph"
    adapted = next((p for change, p in translations(polynomial, names, ramification=ramification)
                    if change == translation), None)
    if adapted is None or record.get("input_divisor"):
        return _weighted_outputs(record, polynomial, translation, options)
    model = _adapted_fan(adapted, names, record["mode"], "cpu", ramification=ramification)
    if model is None:
        raise InvalidChart("The adapted polynomial did not verify the declared fan.")
    _verify_display(record, model, translation)
    key = "atlas" if "atlas" in record else "charts"
    charts = record[key]
    if key == "atlas":
        _verify_owned_fan(charts, model["fan"], len(names))
    normalized = []
    for chart in charts:
        torus = chart.get("torus", {})
        if key == "atlas" and torus.get("coordinate_translation") != translation:
            raise InvalidChart("The charts disagree on the translation to the input coordinates.")
        item = dict(chart, map=translate_map(chart, names, translation, inverse=True))
        if torus:
            item["torus"] = {k: v for k, v in torus.items() if k != "coordinate_translation"}
        normalized.append(item)
    certificate = {k: v for k, v in record.get("certificate", {}).items() if k != "coordinate_translation"}
    if "adapted_polynomial" in certificate and parse_expression(certificate["adapted_polynomial"], names) != adapted:
        raise InvalidChart("The adapted polynomial differs from the translated source.")
    # A minimal archive may omit the producer name. Its verified coordinate
    # translation still identifies the chart and root conventions.
    normalized_record = {**record, "engine": model["engine"], key: normalized, "fan": model["fan"],
                         "certificate": model["certificate"]}
    if "root_families" in model:
        normalized_record["root_families"] = model["root_families"]
    if "polynomial" in normalized_record:
        normalized_record["polynomial"] = format_poly(adapted, names)
    if key == "atlas" and "charts" in record:
        normalized_record["charts"] = model["charts"]
    verified = verify(normalized_record, adapted, **options)
    if verified is None:
        return None
    atlas, outputs = verified
    if key == "atlas":
        return charts, outputs
    # The fibre ideal generated by the adapted coordinates equals the ideal
    # generated by the translated ones because the automorphism fixes zero.
    return [dict(item, map=chart["map"], torus={**item["torus"], "coordinate_translation": translation})
            for item, chart in zip(atlas, charts, strict=True)], outputs
