"""Order of a sparse hypersurface at a rational point, without binomial expansion."""

from math import gcd

from . import native_polynomial as np
from .point_germs import nonzero_terms_at_point


def _new_row(row, pivots):
    """Retain a new independent moment row using integer elimination."""
    for i in range(len(row)):
        value = row[i]
        if not value:
            continue
        pivot = pivots.get(i)
        if pivot is None:
            content = gcd(*row)
            if content > 1:
                row = [v // content for v in row]
            pivots[i] = row
            return row
        common = gcd(value, pivot[i])
        left, right = pivot[i] // common, value // common
        row = [left * v - right * w for v, w in zip(row, pivot)]


def _torus_order(point, terms):
    """The first degree of an Euler derivative with nonzero value at the point."""
    exponents, coefficients = zip(*terms.items())
    frontier = [[1] * len(exponents)]
    pivots = {}
    order = 0
    # On x^a, x_i*d/dx_i acts by a_i. Constant columns add no new rows.
    axes = [axis for axis in zip(*exponents) if min(axis) != max(axis)]
    while frontier:
        new = []
        for row in frontier:
            row = _new_row(row, pivots)
            if row is None:
                continue
            weighted = {e: c*v for e, c, v in zip(exponents, coefficients, row) if v}
            if nonzero_terms_at_point(point, weighted):
                return order
            new.append(row)
        frontier = [[v*k for v, k in zip(row, axis)] for row in new for axis in axes]
        order += 1
    # Distinct exponent vectors are separated by polynomials in the axes.
    # Their evaluation rows span all functions on the finite support.
    raise ArithmeticError("The nonzero polynomial has no nonvanishing derivative.")


def order_at_rational_point(variables, shifts, f):
    """Compute the finite order of a nonzero polynomial with rational coefficients."""
    zero_axes = [i for i, q in enumerate(shifts) if not q]
    active = [i for i, q in enumerate(shifts) if q]
    point = tuple(shifts[i] for i in active)
    groups = {}
    for e, coefficient in np.terms(variables, f).items():
        zero_powers = tuple(e[i] for i in zero_axes)
        powers = tuple(e[i] for i in active)
        groups.setdefault(zero_powers, {})[powers] = coefficient
    if not groups:
        raise ValueError("The zero polynomial has no finite order.")
    # Different powers of the coordinates that vanish at p cannot cancel.
    best = None
    for zeros in sorted(groups, key=sum):
        lower = sum(zeros)
        if best is not None and lower >= best:
            break
        candidate = lower + _torus_order(point, groups[zeros])
        best = candidate if best is None else min(best, candidate)
    return best
