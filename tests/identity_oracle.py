"""Independent rational calculations for identity-chart comparison mode."""
from fractions import Fraction


def independent(parts, dimension):
    from sympy import Matrix
    if not parts or len(parts) > dimension:
        return False
    units = [tuple(int(i == j) for i in range(dimension)) for j in range(dimension)]
    return Matrix([[f.get(e, 0) for e in units] for f in parts]).rank() == len(parts)


def rational_factors(poly, names):
    from sympy import Poly, QQ, Symbol
    f = Poly.from_dict(poly.copy(), *map(Symbol, names), domain=QQ)
    return [({e: Fraction(c) for e, c in g.terms()}, m) for g, m in f.factor_list()[1]]


def fibre_is_origin(fibre, coords):
    from sympy import Dummy, Poly, QQ, S, Symbol, groebner
    variables = tuple(map(Symbol, coords))
    t = Dummy("t")
    equations = [Poly.from_dict(f.copy(), *variables, domain=QQ).as_expr() for f in fibre]
    return all(groebner([*equations, 1-t*v], t, *variables, domain=QQ).contains(S.One)
               for v in variables)
