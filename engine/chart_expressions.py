"""Keep auxiliary coordinates and constants in the chart caller's expression type."""

from . import native_polynomial
from .reference import sp


def chart_expressions(variables):
    return native_polynomial if not variables or native_polynomial.native_ring(variables) else sp


def runtime_expressions():
    """Select the independent symbolic representation only in diagnostic modes."""
    from .native_library import _MODE
    return sp if _MODE in ("sympy", "check") else native_polynomial
