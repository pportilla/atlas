"""Order of a sparse hypersurface at a rational point, without binomial expansion."""

from . import native_polynomial as np
from .point_geometry import point_operation
from .polynomial import integer_from_decimal


def order_at_rational_point(variables, shifts, f):
    """Compute the finite order of a nonzero polynomial with rational coefficients."""
    return integer_from_decimal(point_operation("order", shifts, np.terms(variables, f)))
