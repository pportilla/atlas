"""Sparse FLINT bases over Q when packed monomials cannot hold the ideal.

The basis consists of monic primitive autoreduced Buchberger rows. Lex
and grevlex use the FLINT orders directly. Normal forms use multivariate
division by those same rows. Keep the sparse rows intact: a single power
of degree a billion takes one term throughout this computation.
"""
from functools import cached_property

from . import native_polynomial as np
from .chart_expressions import chart_expressions
from .reference import sp

from .affine_kernel import _library, _request, _ring, _sparse_expression, _sparse_terms


def _order_key(monomial, order):
    if order == "lex":
        return monomial
    return sum(monomial), tuple(-e for e in reversed(monomial))


class WideBasis:
    def __init__(self, gens, order, rows):
        self.gens = gens
        self.order = order
        self.rows = sorted(rows, key=lambda row: _order_key(_sparse_terms(row)[0][0], order), reverse=True)
        self.exprs = [_sparse_expression(gens, row) for row in self.rows]
        self._ring = _ring(gens)

    @cached_property
    def polys(self):
        # Compatibility view for callers that explicitly request Poly objects.
        if np.native_ring(self.gens):
            return self.exprs
        return [sp.Poly.from_dict(dict(_sparse_terms(row)), *self.gens, domain=sp.QQ) for row in self.rows]

    def reduce(self, value):
        rows = _request(["wide_division", self._ring.header, self.order, self._ring.text(value),
                         *map(self._ring.text, self.exprs)])
        # Public rational bases expose the remainder, as on the packed path.
        return None, _sparse_expression(self.gens, rows[-1])

    def contains(self, value):
        return self.reduce(value)[1] == 0


def wide_basis(ideal, gens, order):
    if order == "first":
        from .field_wide import wide_block_basis
        return wide_block_basis(ideal, gens, order)
    if order not in ("lex", "grevlex") or not gens or not hasattr(_library(), "atlas_wide_abi"):
        return None
    ring = _ring(gens)
    rows = _request(["wide_basis", ring.header, order, *(ring.text(f) for f in ideal)])
    return WideBasis(tuple(gens), order, rows)


def wide_colon(ideal, by, variables):
    """Compute I:J by elimination and exact polynomial divisions over Q.

    The contraction of (tI,(1-t)J) is I intersect J. For a nonzero
    principal J=(b), divide those intersection generators by b to obtain
    I:b. Intersect the results for the generators of the denominator ideal.
    """
    from .groebner import groebner
    if not hasattr(_library(), "atlas_wide_block"):
        return None
    variables = tuple(variables)
    t = chart_expressions(variables).Dummy("intersection")

    def meet(left, right):
        lifted = (*(t*f for f in left), *((1-t)*f for f in right))
        result = groebner(lifted, (t, *variables), order="first")
        return tuple(f for f in result.exprs if t not in f.free_symbols)

    result = (chart_expressions(variables).S.One,)
    ring = _ring(variables)
    for b in by:
        if b == 0:
            continue
        shared = meet(ideal, (b,))
        rows = []
        for f in shared:
            q, r = _request(["wide_division", ring.header, "grevlex", ring.text(f), ring.text(b)])
            if r:
                raise ArithmeticError("An intersection generator was not divisible by its principal generator.")
            rows.append(_sparse_expression(variables, q))
        result = tuple(rows) if result == (1,) else meet(result, rows)
    return tuple(groebner(result, variables).exprs)
