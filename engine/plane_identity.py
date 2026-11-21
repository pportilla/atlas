"""Certify the marked identity chart for two transverse conjugate plane branches."""
from . import native_library
from .kernel_json import kernel_json
from .monomial_atlas import fan_outputs
from .polynomial import InputError, LimitError, encode_sparse_row, format_poly, integer_from_decimal, sparse_row


def conjugate_plane_identity(poly, names, *, degree, integrals, verified):
    if (degree > 24 or len(poly) > 32 or
            any(c.numerator.bit_length() > 128 or c.denominator.bit_length() > 128 for c in poly.values()) or
            not hasattr(native_library._kernel(), "atlas_conjugate_plane_crossing")):
        return None
    try:
        found = kernel_json("atlas_conjugate_plane_crossing", "recognize", width=len(names),
                            polynomial=encode_sparse_row(poly))
        if found is None:
            return None
        generator = "a"
        while generator in names:
            generator += "_"
        coords = [*names, generator]
        field = dict(generator=generator, degree=2, embedding="root 1 (FLINT order)",
                     minimal_polynomial=format_poly(sparse_row(found["modulus"]), [generator]))
        power = integer_from_decimal(found["multiplicity"])
        components = [dict(id=f"S{i}", multiplicity=power, nu=1, ideal=[format_poly(sparse_row(row), coords)])
                      for i, row in enumerate(found["branches"], 1)]
        source = format_poly(poly, names)
        mapping = dict(zip(names, names))
        chart = dict(id="U0", coordinates=list(names), map=mapping, coefficient_field=field,
                     strict_transform=source, strict_transform_components=components, exceptional=[])
        owned = dict(coordinates=coords, ambient_ideal=[field["minimal_polynomial"]],
                     coefficient_field=field, map=mapping, components=components, fibre=list(names),
                     region=[], removed=[], split_cone=found["cone"])
        record = dict(status="resolved", engine="conjugate-plane-crossing", mode="embedded",
                      variables=names, polynomial=source, charts=[chart], atlas=[owned], steps=[],
                      summary="Two conjugate branches meet transversely at the origin. The identity resolves the divisor.",
                      certificate={"method": "marked branch equations and chart Jacobian criterion"})
        checked = fan_outputs(record, poly, integrals=integrals)
    except (InputError, LimitError, native_library.Unavailable):
        return None
    if checked is None:
        return None
    record["dual_complex"] = checked[1]["dual_complex"]
    if verified is not None:
        verified.append((record, checked))
    return record
