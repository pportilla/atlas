"""Transport verified fan charts through an origin-preserving completion of the square."""
from .kernel_json import kernel_json
from .native_library import affine_request
from .polynomial import encode_sparse_row, format_poly, parse_expression, sparse_row
from .strata import InvalidChart


def translations(polynomial, names):
    choices = kernel_json("atlas_resolution_candidate", "quadratic_translations",
                          width=len(names), row=encode_sparse_row(polynomial))
    for choice in choices:
        shift = format_poly(sparse_row(choice["shift"]), names)
        yield {"coordinate": names[choice["axis"]], "offset": shift}, sparse_row(choice["adapted"])


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


def _adapted_fan(adapted, names, mode, compute):
    if len(names) == 3:
        from .toric_backend import _diagonal
        return _diagonal(adapted, names, mode, compute)
    from .newton_trials import _try
    return _try(adapted, names, adapted, mode=mode, construct=False)


def resolve_translated(polynomial, names, mode, compute=None):
    for translation, adapted in translations(polynomial, names):
        record = _adapted_fan(adapted, names, mode, compute)
        if record is None:
            continue
        for chart in record["charts"]:
            chart["map"] = translate_map(chart, names, translation)
        proof = record["certificate"]
        proof.update(coordinate_translation=translation, adapted_polynomial=format_poly(adapted, names))
        proof["description"] = ("Completing the square gives an origin-preserving polynomial change "
                                "of coordinates with Jacobian determinant one. " + proof["description"])
        return record
    return None


def _verify_display(record, adapted, translation):
    """Check the displayed fan charts, including split intrinsic sections."""
    from .exact_json import sorted_record
    names = record["variables"]
    expected = _adapted_fan(adapted, names, record["mode"], "cpu")
    if expected is None or len(record["charts"]) != len(expected["charts"]):
        raise InvalidChart("The displayed charts differ from the translated fan.")
    if "fan" in record and any(record["fan"].get(k) != expected["fan"][k] for k in ("rays", "cones")):
        raise InvalidChart("The declared fan differs from the translated charts.")
    if "root_families" in record and record["root_families"] != expected.get("root_families", {}):
        raise InvalidChart("The root families differ from the translated fan sections.")
    proof = record.get("certificate", {})
    fields = ("weights", "weighted_degree", "principal_exponents", "fan_ideal_exponents", "boundary_image_coordinate_ideals")
    if any(k in proof and sorted_record(proof[k]) != sorted_record(expected["certificate"].get(k)) for k in fields):
        raise InvalidChart("The declared fan certificate differs from the translated polynomial.")
    for chart, source in zip(record["charts"], expected["charts"]):
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
    return expected["charts"]


def translated_outputs(record, polynomial, translation, verify, options):
    """Verify the fan after undoing the coordinate translation on every chart map."""
    names = record["variables"]
    adapted = next((p for change, p in translations(polynomial, names) if change == translation), None)
    if adapted is None:
        raise InvalidChart("The coordinate translation disagrees with the defining polynomial.")
    displayed = _verify_display(record, adapted, translation) if "charts" in record else None
    key = "atlas" if "atlas" in record else "charts"
    charts = record[key]
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
    producer = "native-toric" if len(names) == 3 else "verified-monomial-modification"
    normalized_record = {"engine": producer, **record, key: normalized, "certificate": certificate}
    if "polynomial" in normalized_record:
        normalized_record["polynomial"] = format_poly(adapted, names)
    if key == "atlas" and displayed is not None:
        normalized_record["charts"] = displayed
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
