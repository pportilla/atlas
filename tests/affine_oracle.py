"""Independent SymPy verification of the affine chart and frame operations."""
from fractions import Fraction
from math import gcd

from engine import groebner as gb
from engine.affine_kernel import _polynomial
from engine.polynomial import parse_expression
from engine.reference import sp


def _ref_image(f, variables, mapping, base, modulus):
    image = sp.expand(sp.sympify(f).xreplace(dict(zip(base, mapping))))
    return image if modulus is None else sp.expand(sp.rem(image, modulus, variables[-1]))


def _primitive_support(support):
    """Reference for the native divisibility test on exposed Newton vertices."""
    n = len(support[0])
    divisor = 0
    for j in range(n):
        def rotated(powers):
            return powers[j:] + powers[:j]
        divisor = gcd(divisor, *min(support, key=rotated), *max(support, key=rotated))
        if divisor == 1:
            return True
    if len(support) <= n + 1 and gcd(*(e for m in support for e in m)) == 1:
        return lattice_dimension(support) == len(support) - 1
    return False


def lattice_dimension(rows):
    return sp.Matrix([[x-y for x,y in zip(row, rows[0])] for row in rows[1:]]).rank()


def sparse_terms(variables, value):
    from sympy.polys.rings import ring
    sparse, *_ = ring(variables, sp.QQ)
    terms = sparse.from_expr(sp.sympify(value))
    return tuple(sorted(((powers, Fraction(int(c.numerator), int(c.denominator))) for powers, c in terms.items()),
                        reverse=True))


def decode(variables, names, text):
    terms = parse_expression(text, list(names), _reference=True)
    return _polynomial(terms, variables)


def chart_data(variables, mapping, divisors, equation, base, modulus, indices):
    strict = _ref_image(equation, variables, mapping, base, modulus)
    if len(mapping) != len(variables):
        raise ArithmeticError("Expected a square affine chart.")
    jac = sp.S.One if tuple(mapping) == variables else sp.Matrix(mapping).jacobian(variables).det()
    if modulus is not None:
        jac = sp.rem(jac, modulus, variables[-1])
    if not strict or not jac:
        raise ArithmeticError("Expected a dominant chart and a nonzero pullback.")
    if not indices:
        return (), strict
    def orders_in(value):
        terms = [t.as_powers_dict() for t in sp.Add.make_args(sp.expand(value))]
        return [min(t.get(v, 0) for t in terms) for v in divisors]

    orders = tuple((int(n), int(j) + 1) for n, j in zip(orders_in(strict), orders_in(jac)))
    return orders, sp.expand(strict / sp.prod(v ** n for v, (n, _) in zip(divisors, orders)))


def inverse_map(variables, mapping, base):
    if len(variables) != len(base):
        raise ArithmeticError("Expected a square chart map.")
    if tuple(mapping) == variables:
        return tuple((x, sp.S.One) for x in base)
    # Coordinates may have the same printed names on the two sides.
    u = tuple(sp.Dummy() for _ in variables)
    graph = [sp.sympify(f).xreplace(dict(zip(variables, u))) - x for f, x in zip(mapping, base)]
    fractions = []
    for v in u:
        other = tuple(w for w in u if w != v)
        basis = sp.groebner(graph, *other, v, *base, order="lex", domain=sp.QQ)
        for relation in basis:
            if relation.has(*other) or sp.degree(relation, v) != 1:
                continue
            num, den = -relation.subs(v, 0), relation.coeff(v)
            leading = sp.Poly(den, *base).LC()
            fractions.append((sp.expand(num / leading), sp.expand(den / leading)))
            break
        else:
            raise ArithmeticError("The chart map has no rational inverse.")
    return tuple(fractions)


def transition_pole(variables, mapping, base, inverse):
    if len(inverse) != len(base):
        raise ArithmeticError("Expected one fraction per coordinate.")
    denominators = []
    for num, den in inverse:
        a, b = (_ref_image(f, variables, mapping, base, None) for f in (num, den))
        if not b:
            raise ArithmeticError("A transition denominator vanished identically.")
        denominators.append(sp.cancel(a/b).as_numer_denom()[1])
    return sp.Poly(sp.prod(denominators), *variables, domain=sp.QQ).sqf_part().monic().as_expr()


def frame_locus(variables, equations, units, closure):
    basis = sp.groebner(equations, *variables, order="grevlex", domain=sp.QQ)
    empty = list(basis) == [1]
    product = sp.S.One
    for u in units:
        if empty:
            break
        u = basis.reduce(sp.sympify(u))[1]
        product = basis.reduce(sp.expand(product*u))[1]
        empty = product == 0
    if empty:
        return (sp.S.One,) if closure else True
    if product.is_number:
        return tuple(basis) if closure else False
    if closure and len(basis) == 1:
        # In a UFD the contraction of (f) discards exactly the prime
        # factors of f dividing the unit, with their full multiplicities.
        f, u = basis.polys[0], sp.Poly(product, *variables, domain=sp.QQ)
        while True:
            common = f.gcd(u)
            if common.total_degree() == 0:
                return (f.mul_ground(1/f.LC(order="grevlex")).as_expr(),)
            f = f.exquo(common)
    if closure and len(equations) == 2:
        # Over a finite residue field the other row generates a
        # principal ideal in a polynomial UFD. Coprimality with the
        # unit certifies that the ideal is already saturated.
        for relation, other in (equations, equations[::-1]):
            names = tuple(sp.sympify(relation).free_symbols)
            if len(names) != 1 or len(variables) < 2:
                continue
            theta, = names
            modulus = sp.Poly(relation, theta, domain=sp.QQ)
            if modulus.degree() < 2 or not modulus.is_irreducible:
                continue
            field = sp.QQ.alg_field_from_poly(modulus)
            coords = tuple(v for v in variables if v != theta)

            def over_field(f):
                poly = sp.Poly(f, *coords, domain=sp.QQ[theta])
                terms = {m: field(sp.Poly(c, theta, domain=sp.QQ).rem(modulus).all_coeffs())
                         for m, c in poly.terms()}
                return sp.Poly.from_dict(terms, *coords, domain=field)

            if over_field(other).gcd(over_field(product)).total_degree() == 0:
                return tuple(basis)
            break
    t = sp.Dummy()
    extended = (*basis, t*product - 1)
    if not closure:
        return list(sp.groebner(extended, t, *variables, order="grevlex", domain=sp.QQ)) == [1]
    # A block order eliminates the inverse without lex-converting the
    # frame coordinates; the surviving rows already have grevlex order.
    order = gb._reference_order("first")
    elimination = sp.groebner(extended, t, *variables, order=order, domain=sp.QQ)
    return tuple(f for f in elimination if not f.has(t))


def frame_field(rows, columns):
    jac = sp.Matrix(rows).jacobian(columns)
    return tuple(sp.expand((-1)**(len(rows) + j)*jac[:, [i for i in range(len(columns)) if i != j]].det())
                 for j in range(len(columns)))


def rational_factor_rows(variables, expression):
    poly = sp.Poly(expression, *variables, domain=sp.QQ)
    scalar, factors = poly.factor_list()
    rows = []
    for p, m in factors:
        terms = p.terms()
        row = ";".join(f"{c} " + " ".join(map(str, e)) for e, c in terms)
        rows.append((row, m, _primitive_support(tuple(e for e, _ in terms))))
    return Fraction(scalar), tuple(rows)


def frame_linear(f, k):
    p = sp.Poly(f, k)
    if p.degree() == 1:
        return sp.expand(p.nth(1)), sp.expand(p.nth(0))
    return None


def frame_restrict(variables, f, k, a, b):
    if a == 0:
        raise ArithmeticError("The linear coefficient is zero.")
    p = sp.Poly(f, k)
    degree = p.degree()
    if degree <= 0:
        return sp.expand(f)
    result = sp.expand(sum(p.nth(j)*(-b)**j*a**(degree-j) for j in range(degree+1)))
    while result != 0 and not sp.sympify(a).is_number:
        q, r = sp.div(result, a, *variables)
        if r != 0:
            break
        result = q
    return sp.expand(result)


def frame_quotient(variables, f, k):
    quotient, rem = sp.div(f, k, *variables, domain=sp.QQ)
    return quotient if rem == 0 else None


def pullbacks(functions, variables, mapping, base, modulus):
    return tuple(_ref_image(f, variables, mapping, base, modulus) for f in functions)


def frame_function(value):
    return sp.expand(value)


def frame_minor(rows, columns):
    return sp.expand(sp.Matrix(rows).jacobian(columns).det(method="berkowitz"))


def frame_derivative(f, field):
    return sp.expand(sum(c*sp.diff(f, v) for v, c in field.items()))
