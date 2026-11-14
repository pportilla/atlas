"""Independent symbolic conversion and division for marked polynomial comparisons."""
from engine import groebner as gb
from engine.reference import sp


def coefficient(value, domain):
    try:
        return domain.convert(value).to_list()
    except sp.polys.polyerrors.CoercionFailed:
        # Historical coercion sometimes needs the real and imaginary parts
        # of a principal complex radical separated.
        return domain.convert(sp.expand_complex(value)).to_list()


def scalar(value, domain):
    generator = domain.ext.as_expr()
    scale, root = generator.as_coeff_Mul()
    if isinstance(root, sp.CRootOf) and scale.is_Rational:
        return sp.Add(*(sp.Rational(c)*scale**j*root**j for j,c in enumerate(reversed(value.to_list())) if c))
    return domain.to_sympy(value)


def polynomial(expression, variables, domain):
    return sp.Poly(expression, *variables, domain=domain)


def division(basis, expr, quotients):
    reference = sp.groebner(basis.exprs, *basis.gens, order=gb._reference_order(basis.order), domain=basis.domain)
    qs, r = reference.reduce(expr)
    return [*qs, r] if quotients else [r]


def factors(poly):
    return poly.factor_list()


def rational_sqrt(value, scale, offset):
    return offset + scale * sp.sqrt(value)


def compare_radical(result, value, scale, offset, same_spelling):
    expected = rational_sqrt(value, scale, offset)
    if result == expected:
        return
    if same_spelling or not scale:
        raise AssertionError("The marked quadratic radical differs from its historical spelling.")
    # Both displays use the principal root. Independent squaring handles
    # large square factors left in the historical spelling.
    for radical in (result, expected):
        if sp.simplify(((radical - offset) / scale)**2 - value) != 0:
            raise AssertionError("Quadratic radicals differ after normalization.")


def indexed_field(poly, index):
    root = sp.CRootOf(poly.as_expr(), index)
    return sp.QQ.algebraic_field(root), root


def root_position(poly, primitive):
    if poly.degree() == 1 and primitive == -poly.TC()/poly.LC():
        return 0
    return next(i for i in range(poly.degree())
                if poly.same_root(primitive, sp.CRootOf(poly.as_expr(), i)))
