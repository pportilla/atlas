"""Independent symbolic comparison of exact rational bases, term orders and remainders."""
import sympy as sp

from engine import native_polynomial as np


def reference_order(order):
    if order != "first":
        return order
    from sympy.polys.orderings import ProductOrder, grevlex, lex
    return ProductOrder((lex, lambda monomial: monomial[:1]), (grevlex, lambda monomial: monomial[1:]))


def check_expanded(gens, expr, row):
    from engine.affine_kernel import _sparse_terms
    from sympy.polys.rings import ring
    sparse, *_ = ring(gens, sp.QQ)
    if dict(_sparse_terms(row)) != dict(sparse.from_expr(expr)):
        raise AssertionError("Native polynomial expansion differs from the QQ term list.")


class _SymbolicBasis:
    """Independent sparse QQ basis and division in optional diagnostic modes."""
    def __init__(self, generators, variables, order, trusted):
        from sympy.polys.groebnertools import groebner
        from sympy.polys.rings import ring
        self.variables = variables
        symbols = tuple(sp.Dummy(variable.name) for variable in variables)
        self.ring, *_ = ring(symbols, sp.QQ, order=reference_order(order))
        rows = [self._symbolic(f) for f in generators if f]
        self.divisors = rows if trusted else groebner(rows, self.ring)
        self.exprs = [self._native(f) for f in self.divisors]

    def _symbolic(self, polynomial):
        return self.ring.from_dict(np.terms(self.variables, polynomial))

    def _native(self, polynomial):
        return np.from_terms(self.variables, {e: sp.QQ.to_sympy(c) for e, c in polynomial.items()})

    def reduce(self, polynomial):
        quotients, remainder = self._symbolic(polynomial).div(self.divisors)
        return [self._native(q) for q in quotients], self._native(remainder)

    def remainder(self, polynomial):
        return self._native(self._symbolic(polynomial).rem(self.divisors))

    def contains(self, polynomial):
        return not self._symbolic(polynomial).rem(self.divisors)
