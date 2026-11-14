"""Transport sparse point-local operations to the exact kernel."""

from .kernel_json import kernel_json
from .polynomial import integer_from_decimal, integer_text, kernel_fraction, rational_text


def request(operation, **operands):
    return kernel_json("atlas_point_geometry", operation, **operands)


def encode_terms(terms):
    return [[rational_text(c), [integer_text(k) for k in powers]] for powers, c in terms.items() if c]


def decode_terms(data):
    return {tuple(map(integer_from_decimal, powers)): kernel_fraction(c) for c, powers in data}


def point_operation(operation, point, terms, **options):
    return request(operation, point=list(map(rational_text, point)), terms=encode_terms(terms), **options)


def weighted_orders(width, valuations):
    rows = [[integer_text(weight), list(map(integer_text, orders))] for weight, orders in valuations]
    return list(map(integer_from_decimal, kernel_json("atlas_identity_divisor", "weighted_orders",
                                                      width=width, rows=rows)))


def support_profile(powers):
    """Coordinate degrees, total degree, and homogeneity of a sparse support."""
    support = [list(map(integer_text, row)) for row in powers]
    if not support:
        return (), -1, True
    result = kernel_json("atlas_component_geometry", "profile", support=support)
    return (tuple(map(integer_from_decimal, result["degrees"])), integer_from_decimal(result["degree"]),
            result["homogeneous"])
