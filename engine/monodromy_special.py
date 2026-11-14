"""Recognize rational coordinate and smooth transverse factors through the origin."""
from .monodromy import _request
from .point_geometry import encode_terms, support_profile
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
    from .identity_atlas import independent, rational_factors
    zero = (0,) * len(names)
    active = [(f, m) for f, m in rational_factors(p, names) if zero not in f]
    if independent([f for f, _ in active], len(names)):
        from .affine_kernel import archived_factor_order
        active = [active[i] for i in archived_factor_order(active, len(names))]
        return [(format_poly(f, names), m) for f, m in active]
    return None
