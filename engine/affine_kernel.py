"""Rational affine maps, their divisorial data and transition poles (KERNEL.md §9.5).

FLINT receives the polynomial expressions, including products and powers. The
portable formulas also handle rings or monomials outside the packed native ABI.
"""
import ctypes
import re
from fractions import Fraction
from functools import cmp_to_key, lru_cache
from math import gcd
from operator import index

from . import native_polynomial as np
from .chart_expressions import chart_expressions
from .reference import sp

from . import groebner as gb
from .fan_geometry import canonical_laurent_terms, lattice_characters, lattice_pivots


from .native_library import Unavailable as _AffineUnavailable, affine_request as _request, _library


class _RingText:
    """Assign FLINT names by generator position when printed names are ambiguous."""
    def __init__(self, variables):
        if len(set(variables)) != len(variables):
            raise ValueError("The polynomial ring needs distinct coordinates.")
        names = tuple(v.name for v in variables)
        valid = len(set(names)) == len(names) and all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n) for n in names)
        if not valid:
            names = tuple(f"atlasVar{i}" for i in range(len(variables)))
        self.names = dict(zip(variables, names))
        self.header = " ".join(names)

    @lru_cache(maxsize=16384)
    def text(self, value):
        # FLINT accepts the expression in construction order. The presentation
        # printer sorts every sum and can cost more than the computation.
        if isinstance(value, np.Expr):
            return value.text(self.names)
        if isinstance(value, (int, Fraction)):
            return str(value)
        if not isinstance(value, sp.Basic):
            raise TypeError("Supply a parsed rational polynomial.")
        if value.is_Symbol:
            if value not in self.names:
                raise ArithmeticError("A polynomial uses an undeclared coordinate.")
            return self.names[value]
        if value.is_Rational:
            return str(value.p) if value.q == 1 else f"({value.p}/{value.q})"
        if value.is_Add or value.is_Mul:
            sep = "+" if value.is_Add else "*"
            return "(" + sep.join(self.text(a) for a in value.args) + ")"
        if value.is_Pow and value.exp.is_Integer:
            exponent = int(value.exp)
            power = f"({self.text(value.base)})^{abs(exponent)}"
            return power if exponent >= 0 else f"(1/{power})"
        raise ArithmeticError("Expected a rational polynomial in the specified coordinates.")


@lru_cache(maxsize=256)
def _ring(variables):
    return _RingText(variables)


def _ask(op, variables, mapping, base, functions=(), *, modulus=None, indices=None):
    if len(mapping) != len(base):
        raise ValueError("Each base coordinate needs one image.")
    chart, source = _ring(tuple(variables)), _ring(tuple(base))
    lines = [op, chart.header, source.header, *map(chart.text, mapping)]
    if modulus is not None:
        lines.append(chart.text(modulus))
    lines.extend(map(source.text, functions))
    if indices is not None:
        lines.append(" ".join(map(str, indices)))
    return _request(lines)


def _checked(native, reference, variables=()):
    if gb._MODE == "sympy":
        return reference()
    try:
        answer = native()
    except _AffineUnavailable:
        if gb._MODE != "check" or np.native_ring(variables) or not variables:
            raise
        return reference()
    if gb.CHECK and answer != reference():
        raise AssertionError("The native affine-map operation differs from its SymPy reference.")
    return answer


def _integer_rows(vectors):
    return tuple(tuple(map(index, v)) for v in vectors)


def _lattice_support(exponents):
    rows = _integer_rows(exponents)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Expected nonempty Laurent support with a common vector length.")

    return rows


def lattice_dimension(exponents):
    """The affine dimension over Q of a finite, nonempty integer support."""
    rows = _lattice_support(exponents)

    def native():
        rank, = _request(["lattice_dimension", str(len(rows[0])), *(" ".join(map(str,r)) for r in rows)])
        return int(rank)

    def reference():
        return sp.Matrix([[x-y for x,y in zip(row, rows[0])] for row in rows[1:]]).rank()

    return _checked(native, reference)


@lru_cache(maxsize=16384)
def _sparse_terms(row):
    """Decode an expanded native polynomial without a dense coefficient array."""
    from .polynomial import sparse_row
    return tuple(sparse_row(row).items())


@lru_cache(maxsize=16384)
def _sparse_expression(variables, row):
    from .atlas import _polynomial
    value = _polynomial(dict(_sparse_terms(row)), variables)
    if len(gb._lines) >= gb._CACHE_LIMIT:
        gb._lines.clear()
    gb._lines[(variables, value)] = row
    return value


@lru_cache(maxsize=16384)
def sparse_terms(variables, value):
    """Sparse rational coefficients of a chart expression in its declared ring.

    Affine maps, pullbacks and chart equations already arrive as term lists.
    FLINT expands any other expression before returning its coefficients.
    """
    variables = tuple(variables)
    if np.native_ring(variables):
        return tuple(sorted(np.terms(variables, value).items(), reverse=True))

    def native():
        row = gb._lines.get((variables, value))
        if row is None or "*" in row:
            ring = _ring(variables)
            row, = _request(["decode", ring.header, ring.text(value)])
        return tuple(sorted(_sparse_terms(row), reverse=True))

    def reference():
        from sympy.polys.rings import ring
        sparse, *_ = ring(variables, sp.QQ)
        terms = sparse.from_expr(sp.sympify(value))
        return tuple(sorted(((powers, Fraction(int(c.numerator), int(c.denominator))) for powers, c in terms.items()),
                            reverse=True))

    return _checked(native, reference, variables)


def decode(variables, text):
    """Read one record expression with the polynomial grammar over Q."""
    from .polynomial import InputError, parse_expression
    names = tuple(v.name for v in variables)
    if (any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in names)
            or len(set(names)) != len(names) or not text.isascii()
            or any(c in text for c in "\n\r\0")):
        raise InputError("Invalid chart expression or coordinate names.")

    def native():
        try:
            row, = _request(["decode", " ".join(names), text])
        except _AffineUnavailable:
            raise
        except ArithmeticError as error:
            raise InputError(str(error)) from error
        return _sparse_expression(variables, row)

    def reference():
        from .atlas import _polynomial
        terms = parse_expression(text, list(names), _reference=True)
        return _polynomial(terms, variables)

    return _checked(native, reference, variables)


def _ref_image(f, variables, mapping, base, modulus):
    image = sp.expand(sp.sympify(f).xreplace(dict(zip(base, mapping))))
    return image if modulus is None else sp.expand(sp.rem(image, modulus, variables[-1]))


def pullbacks(variables, mapping, base, functions, modulus=None):
    """Compose each function on the base with the chart map, optionally over Q[theta]/m."""
    variables, functions = tuple(variables), tuple(functions)

    def native():
        rows = _ask("pullbacks", variables, mapping, base, functions, modulus=modulus or 0)
        return tuple(_sparse_expression(variables, row) for row in rows)

    return _checked(native, lambda: tuple(_ref_image(f, variables, mapping, base, modulus) for f in functions), variables)


def chart_data(variables, mapping, divisors, equation, base, modulus=None):
    """Return the (N, nu) for each coordinate divisor and the strict transform.

    On a finite coefficient cover append its parameter to both coordinate lists
    and append its identity map. The modulus is its minimal polynomial.
    """
    variables = tuple(variables)
    indices = [variables.index(v) for v in divisors]
    if len(set(indices)) != len(indices):
        raise ValueError("A chart coordinate divisor was specified twice.")

    def native():
        rows = _ask("chart", variables, mapping, base, (equation,), modulus=modulus or 0, indices=indices)
        return tuple(tuple(map(int, r.split())) for r in rows[:-1]), _sparse_expression(variables, rows[-1])

    def reference():
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

    return _checked(native, reference, variables)


def inverse_map(variables, mapping, base):
    """Return coordinate fractions in the base, each with a monic denominator."""
    variables, base = tuple(variables), tuple(base)

    def native():
        try:
            rows = _ask("inverse", variables, mapping, base, ())
        except _AffineUnavailable:
            if not hasattr(_library(), "atlas_wide_abi"):
                raise
            if len(variables) != len(base):
                raise ArithmeticError("Expected a square chart map.")
            known = {f: x for f, x in zip(mapping, base) if f in variables}
            api = chart_expressions(variables)
            unknown = {v: api.Dummy() for v in variables if v not in known}
            renamed = {**known, **unknown}
            graph = tuple(api.sympify(f).xreplace(renamed) - x for f, x in zip(mapping, base) if f not in known)
            out = []
            for v in variables:
                if v in known:
                    out.append((known[v], api.S.One))
                    continue
                u = unknown[v]
                others = tuple(w for w in unknown.values() if w != u)
                coords = (*others, u, *base)
                for relation in gb.groebner(graph, coords, order="lex").exprs:
                    if set(others).isdisjoint(relation.free_symbols):
                        parts = frame_linear(coords, relation, u)
                        if parts is not None:
                            den, constant = parts
                            lead = sparse_terms(base, den)[0][1]
                            out.append((frame_function(base, -constant/lead), frame_function(base, den/lead)))
                            break
                else:
                    raise ArithmeticError("The chart map has no rational inverse.")
            return tuple(out)
        parts = [_sparse_expression(base, row) for row in rows]
        return tuple(zip(parts[::2], parts[1::2]))

    def reference():
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

    return _checked(native, reference, variables)


def transition_pole(variables, mapping, base, inverse):
    """Return the squarefree monic equation of the complement of the overlap."""
    variables = tuple(variables)

    def native():
        rows = _ask("pole", variables, mapping, base, (f for pair in inverse for f in pair))
        return _sparse_expression(variables, rows[0])

    def reference():
        if len(inverse) != len(base):
            raise ArithmeticError("Expected one fraction per coordinate.")
        denominators = []
        for num, den in inverse:
            a, b = (_ref_image(f, variables, mapping, base, None) for f in (num, den))
            if not b:
                raise ArithmeticError("A transition denominator vanished identically.")
            denominators.append(sp.cancel(a/b).as_numer_denom()[1])
        return sp.Poly(sp.prod(denominators), *variables, domain=sp.QQ).sqf_part().monic().as_expr()

    return _checked(native, reference, variables)


@lru_cache(maxsize=128)
def frame_locus(variables, equations, units, *, closure=False):
    """Emptiness or the grevlex basis of the contracted ideal on an open frame."""
    variables, equations, units = tuple(variables), tuple(equations), tuple(units)
    if not variables:
        empty = any(f != 0 for f in equations) or any(u == 0 for u in units)
        return ((1,) if empty else ()) if closure else empty

    def native():
        ring = _ring(variables)
        lines = ["frame_basis" if closure else "frame_empty", ring.header,
                 str(len(equations)), *map(ring.text, equations), *map(ring.text, units)]
        try:
            result = _request(lines)
        except _AffineUnavailable:
            if not hasattr(_library(), "atlas_wide_block"):
                raise
            # Adjoin the inverse of the product of units and contract back.
            # A first-variable elimination order induces grevlex on the frame.
            if not units:
                rows = tuple(gb.groebner(equations, variables).exprs)
            else:
                api = chart_expressions(variables)
                t = api.Dummy("inverse")
                lifted = (*equations, t*api.Mul(*units)-1)
                elimination = gb.groebner(lifted, (t, *variables), order="first" if closure else "grevlex")
                rows = tuple(f for f in elimination.exprs if t not in f.free_symbols)
            return rows if closure else rows == (1,)
        return tuple(_sparse_expression(variables, row) for row in result) if closure else result == ["1"]

    def reference():
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

    return _checked(native, reference, variables)


def _frame_polys(op, variables, columns, functions):
    ring = _ring(variables)
    lines = [op, ring.header]
    if columns is not None:
        lines.append(" ".join(str(variables.index(v)) for v in columns))
    rows = _request([*lines, *map(ring.text, functions)])
    return tuple(_sparse_expression(variables, row) for row in rows)


def frame_function(variables, value):
    """Expand a polynomial in the kernel; its term list can be reduced by a held basis."""
    variables = tuple(variables)
    if not variables:
        return value if isinstance(value, (np.Expr, int, Fraction)) else sp.sympify(value)
    return _checked(lambda: _frame_polys("function", variables, None, (value,))[0],
                    lambda: sp.expand(value), variables)


def frame_minor(variables, rows, columns):
    """A Jacobian minor, with its sign and rational scale."""
    variables = tuple(variables)
    return _checked(lambda: _frame_polys("minor", variables, columns, rows)[0],
                    lambda: sp.expand(sp.Matrix(rows).jacobian(columns).det(method="berkowitz")), variables)


def frame_field(variables, rows, columns):
    """Signed complementary minors for the implicit derivatives in these columns."""
    variables = tuple(variables)

    def reference():
        jac = sp.Matrix(rows).jacobian(columns)
        return tuple(sp.expand((-1)**(len(rows) + j)*jac[:, [i for i in range(len(columns)) if i != j]].det())
                     for j in range(len(columns)))

    values = _checked(lambda: _frame_polys("field", variables, columns, rows), reference, variables)
    return {v: c for v, c in zip(columns, values) if c != 0}


def frame_derivative(variables, f, field):
    """Apply a polynomial vector field before reducing by the frame's rows."""
    variables = tuple(variables)
    return _checked(lambda: _frame_polys("derive", variables, field, (f, *field.values()))[0],
                    lambda: sp.expand(sum(c*sp.diff(f, v) for v, c in field.items())), variables)


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


def _factor_order(left, right, n):
    """Lexicographic order of nested descending coefficient lists, skipping zero runs."""
    if not n:
        a, b = left.get((), 0), right.get((), 0)
        return (a > b) - (a < b)

    def offsets(terms):
        degree = max((m[0] for m in terms), default=-1 if n == 1 else 0)
        groups = {}
        for (exponent, *rest), value in terms.items():
            groups.setdefault(degree-exponent, {})[tuple(rest)] = value
        return degree, groups

    da, a = offsets(left)
    db, b = offsets(right)
    for k in sorted(a.keys() | b.keys()):
        if k > min(da, db):
            break
        order = _factor_order(a.get(k, {}), b.get(k, {}), n-1)
        if order:
            return order
    return (da > db) - (da < db)


@lru_cache(maxsize=512)
def rational_factor_rows(variables, expression):
    """Scalar and (sparse row, multiplicity, primitive-vertex certificate) factors over Q.

    The certificate applies to rationally irreducible factors. The geometric
    factors are Galois conjugate, so every vertex of the Newton polytope of
    their product is divisible by their number. A gcd of one proves that
    there is just one. The order of the rows is the archived QQ factor order.
    """
    def reference():
        poly = sp.Poly(expression, *variables, domain=sp.QQ)
        scalar, factors = poly.factor_list()
        rows = []
        for p, m in factors:
            terms = p.terms()
            row = ";".join(f"{c} " + " ".join(map(str, e)) for e, c in terms)
            rows.append((row, m, _primitive_support(tuple(e for e, _ in terms))))
        return Fraction(scalar), tuple(rows)

    def native():
        ring = _ring(variables)
        supports = hasattr(gb._kernel(), "atlas_geometric_factors_abi")
        operation = "geometric_factors" if supports else "factor"
        lines = _request([operation, ring.header, ring.text(expression)])
        scalar = dict(_sparse_terms(lines[0])).get((0,)*len(variables), Fraction(0))
        parts = []
        for info, row in zip(lines[1::2], lines[2::2]):
            fields = tuple(map(int, info.split()))
            primitive = bool(fields[1]) if supports else _primitive_support(tuple(e for e, _ in _sparse_terms(row)))
            parts.append((row, fields[0], primitive))
        if len(parts) > 1:
            order = cmp_to_key(lambda a,b: _factor_order(a, b, len(variables)))
            def key(part):
                terms = dict(_sparse_terms(part[0]))
                return max(e[0] for e in terms), part[1], order(terms)
            parts.sort(key=key)
        return scalar, tuple(parts)
    return _checked(native, reference, variables)


def rational_factors(poly):
    """Primitive factors over Q, in the branch order of existing records."""
    if poly.domain.is_AlgebraicField:
        from .field_factors import factors
        return poly.LC(), factors(poly)
    if poly.domain != sp.QQ:
        return poly.factor_list()

    scalar, rows = rational_factor_rows(poly.gens, poly.as_expr())
    return sp.Rational(scalar), [(gb._poly(poly.gens, row)[0], m) for row, m, _ in rows]


def frame_linear(variables, f, k):
    """The coefficient and constant of f in k, or None unless its degree is one."""
    variables = tuple(variables)

    def native():
        ring = _ring(variables)
        rows = _request(["linear", ring.header, str(variables.index(k)), ring.text(f)])
        return None if rows == ["none"] else tuple(_sparse_expression(variables, r) for r in rows)

    def reference():
        p = sp.Poly(f, k)
        if p.degree() == 1:
            return sp.expand(p.nth(1)), sp.expand(p.nth(0))
        return None

    return _checked(native, reference, variables)


def frame_restrict(variables, f, k, a, b):
    """Restrict f to k=-b/a, clearing its denominator up to an invertible power of a."""
    variables = tuple(variables)

    def reference():
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

    return _checked(lambda: _frame_polys("restrict", variables, (k,), (f, a, b))[0], reference, variables)


def frame_quotient(variables, f, k):
    """The polynomial f/k, or None when there is a remainder."""
    variables = tuple(variables)

    def native():
        ring = _ring(variables)
        rows = _request(["divide_coordinate", ring.header, str(variables.index(k)), ring.text(f)])
        return None if rows == ["none"] else _sparse_expression(variables, rows[0])

    def reference():
        quotient, rem = sp.div(f, k, *variables, domain=sp.QQ)
        return quotient if rem == 0 else None

    return _checked(native, reference, variables)
