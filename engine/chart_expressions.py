"""Keep auxiliary coordinates and constants in the chart caller's expression type."""

from . import native_polynomial
from .reference import sp


def chart_expressions(variables):
    return native_polynomial if not variables or native_polynomial.native_ring(variables) else sp


def runtime_expressions():
    """Runtime chart coordinates always use the native polynomial kernel."""
    return native_polynomial
