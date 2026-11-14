"""Factor and substitute polynomials over a marked number field."""
from functools import lru_cache

from .reference import QQ
from .chart_expressions import chart_expressions
from .affine_kernel import pullbacks, rational_factors
from .groebner import CHECK, _MODE
from .native_library import affine_request
from .polynomial import integer_from_decimal


@lru_cache(maxsize=256)
def factors(poly):
    if poly.domain == QQ:
        return rational_factors(poly)[1]
    if _MODE in ("sympy", "check"):
        from tests.field_polynomial_oracle import factors as reference
    if _MODE == "sympy":
        return reference(poly)[1]

    from .groebner import _kernel
    if not hasattr(_kernel(), "atlas_field_factorization_abi"):
        if _MODE == "check":
            return reference(poly)[1]
        raise ArithmeticError("Rebuild the native number-field kernel with make groebner.")
    from .candidate_field import NativeField
    from .field_polynomial import _line_polynomial, _parameter, lift, modulus
    from .field_wide import _expanded_row
    variables, field = tuple(poly.gens), poly.domain
    theta = _parameter(field)
    extended = (*variables, theta)
    # A compatibility Poly already stores exact power-basis coefficients.
    value = poly.as_expr() if isinstance(field, NativeField) else poly
    rows = affine_request(["field_factorization", str(len(extended)),
                           _expanded_row(extended, modulus(field, theta)),
                           _expanded_row(extended, lift(value, variables, field, theta))])
    result = [(_line_polynomial(row, variables, field), integer_from_decimal(power))
              for power, row in zip(rows[::2], rows[1::2])]
    if CHECK and result != reference(poly)[1]:
        raise AssertionError("The number-field factors differ from the reference.")
    return result


def compose(variables, mapping, base, functions, field):
    """Substitute polynomials over K in the common presentation Q[theta]/m."""
    if field == QQ:
        return pullbacks(variables, mapping, base, functions)
    from .field_polynomial import lift, lower, modulus
    theta = chart_expressions(variables).Dummy("field_map")
    equations = tuple(lift(f, base, field, theta) for f in functions)
    images = tuple(lift(f, variables, field, theta) for f in mapping)
    values = pullbacks((*variables, theta), (*images, theta), (*base, theta), equations, modulus(field, theta))
    return tuple(lower(f, variables, field, theta) for f in values)


def linear(variables, expression, x, field):
    """The two coefficients of a polynomial of degree one in x."""
    from .affine_kernel import frame_linear
    if field == QQ:
        return frame_linear(variables, expression, x)
    from .field_polynomial import lift, lower
    theta = chart_expressions(variables).Dummy("field_linear")
    pair = frame_linear((*variables, theta), lift(expression, variables, field, theta), x)
    return None if pair is None else tuple(lower(f, variables, field, theta) for f in pair)
