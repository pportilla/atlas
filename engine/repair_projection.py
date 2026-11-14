"""Recover a repair chart's polynomial projection from two maps to the input."""
from .affine_kernel import pullbacks
from .chart_expressions import chart_expressions
from .groebner import groebner
from .repair_gluing import _birational_parameters


def recover_projection(child, parent):
    """Return the unique regular lift to an affine parent, when certified.

    A birational inverse on the parent gives each coordinate as a fraction
    on the original base. In the child ring A, a fraction n/d extends over
    the entire chart precisely when its image in A[1/d] lies in A, provided
    A -> A[1/d] is injective. Lex elimination with u*d=1 tests both facts.
    The regular lift must compose to the saved child map on all of Spec(A).
    """
    if parent.ambient or parent.coefficient_field:
        return None
    names = tuple(name for name, _ in parent.coordinates)
    images = dict(child.coordinates)
    if (not names or len(names) != len(set(names)) or len(images) != len(child.coordinates)
            or images.keys() != set(names) or len(parent.variables) != len(names)):
        return None
    inverse = _birational_parameters(parent, tuple(f for _, f in parent.coordinates))
    if inverse is None:
        return None
    original, fractions = inverse
    coords = child.variables
    child_map = tuple(images[name] for name in names)
    expressions = chart_expressions(coords)
    ambient = groebner(child.ambient, coords)
    if ambient.contains(expressions.S.One):
        return None
    words = pullbacks(coords, child_map, original, (f for pair in fractions for f in pair))
    inverse_axis = expressions.Dummy("repair_inverse")
    localizations = {}
    projection = []
    for numerator, denominator in zip(words[::2], words[1::2]):
        a, b = ambient.reduce(numerator)[1], ambient.reduce(denominator)[1]
        if not b:
            return None
        if b == 1:
            projection.append(a)
            continue
        if b not in localizations:
            basis = groebner((*child.ambient, inverse_axis*b - 1), (inverse_axis, *coords), order="lex")
            # The contraction is I : b^infinity. If it is larger than I,
            # multiplication by b can lose an entire component or a nilpotent.
            if any(not ambient.contains(g) for g in basis.exprs if inverse_axis not in g.free_symbols):
                return None
            localizations[b] = basis
        quotient = localizations[b].reduce(inverse_axis*a)[1]
        if inverse_axis in quotient.free_symbols:
            return None
        projection.append(quotient)

    composed = pullbacks(coords, projection, parent.variables, (f for _, f in parent.coordinates))
    if not all(ambient.contains(f - g) for f, g in zip(composed, child_map)):
        return None
    return tuple(projection)
