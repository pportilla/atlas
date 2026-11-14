"""Recognize rational coordinate and smooth transverse factors through the origin."""
from .monodromy import _request
from .kernel_json import kernel_json
from .point_geometry import decode_terms, encode_terms, support_profile
from .polynomial import format_poly, integer_from_decimal


def _simple_coordinates(p, names):
    # If f=x^e*h with h(0) nonzero, the coordinate powers give the entire
    # divisor of the germ. This includes monomials multiplied by a general unit.
    powers = _request("coordinate_divisor", terms=encode_terms(p), width=len(names))
    if powers is None:
        return None
    return [(name, integer_from_decimal(m)) for name, m in zip(names, powers) if m != "0"]


def _factor_snc(p, names):
    # A smooth rational factor through the origin gives a single geometric
    # branch there. Independent differentials give normal crossings.
    if len(p) > 192 or len(names) > 8 or support_profile(p)[1] > 24:
        return None
    parts = kernel_json("atlas_identity_divisor", "transverse_germ_factors",
                        width=len(names), terms=encode_terms(p))
    if parts is None:
        return None
    return [(format_poly(decode_terms(row), names), integer_from_decimal(m)) for row, m in parts]
