"""Arithmetic over K=Q(theta) using a rational polynomial presentation.

Order x-monomials first, then powers of theta. If G is a reduced monic
K-basis, its canonical lifts and the minimal polynomial m form a reduced
Q-basis for (I,m): the heads are LM_x(g) and theta^deg(m), and the
standard monomials are precisely x^a theta^j with x^a standard for G
and 0 <= j < deg(m). These give a Q-basis of K[x]/I. Uniqueness of the
reduced basis and normal form proves the native basis and remainder,
including their coefficients. For the unit ideal the lifted basis is [1].
"""
import ctypes
from functools import cached_property, lru_cache

from .reference import QQ, sp
from .candidate_field import FieldPolynomial, NativeField
from . import native_polynomial as np
from .polynomial import sparse_row

from . import groebner as gb


def _oracle(name, *args):
    from tests import field_polynomial_oracle
    return getattr(field_polynomial_oracle, name)(*args)


def _display_sum(terms):
    """Reconstruct an already computed symbolic sum without symbolic arithmetic."""
    from .field_display import _held
    return _held(sp.Add, [leaf for term in terms if term != 0 for leaf in sp.Add.make_args(term)])


def _display_term(coefficient, variables, powers):
    from .field_display import _held, _pow
    if coefficient == 0:
        return sp.S.Zero
    factors = list(sp.Mul.make_args(coefficient)) if coefficient != 1 else []
    factors.extend(_pow(v, k) for v, k in zip(variables, powers) if k)
    return _held(sp.Mul, factors)


def _in_theta(coefficients, theta):
    return _display_sum(_display_term(sp.Rational(c), (theta,), (j,))
                        for j, c in enumerate(reversed(coefficients)) if c)


@lru_cache(maxsize=128)
def modulus(domain, theta):
    """The minimal polynomial defining the coefficient field, written in theta."""
    if isinstance(domain, NativeField):
        return np.from_terms((theta,), {(i,): c for i,c in enumerate(reversed(domain.coefficients))})
    return _in_theta(domain.mod.to_list(), theta)


@lru_cache(maxsize=8192)
def _coefficient(value, domain, theta):
    from .field_coefficient import generator_expression
    generator = generator_expression(domain.ext)
    if value == generator or value == domain.ext:
        return theta
    from .field_coefficient import coefficients
    result = coefficients(value, domain)
    converted = _in_theta(tuple(reversed(result)), theta)
    if gb.CHECK and converted != _in_theta(_oracle("coefficient", value, domain), theta):
        raise AssertionError("The marked algebraic coefficient differs from its reference.")
    return converted


@lru_cache(maxsize=16384)
def _lift(expr, gens, domain, theta):
    if expr.is_Rational or expr in gens:
        return expr
    if not expr.free_symbols:
        return _coefficient(expr, domain, theta)
    if expr.is_Add:
        return _display_sum(_lift(a, gens, domain, theta) for a in expr.args)
    if expr.is_Mul:
        from .field_display import _held
        return _held(sp.Mul, [factor for a in expr.args
                              for factor in sp.Mul.make_args(_lift(a, gens, domain, theta))])
    if expr.is_Pow and expr.exp.is_Integer and int(expr.exp) >= 0:
        return sp.Pow(_lift(expr.base, gens, domain, theta), expr.exp, evaluate=False)
    raise sp.PolynomialError("Expected a polynomial over the supplied number field.")


def lift(expr, variables, domain, theta):
    """Represent a K-polynomial in Q[x,theta], keeping products and powers held."""
    if isinstance(expr, str):
        raise TypeError("Supply a parsed polynomial expression.")
    if isinstance(domain, NativeField):
        return np.coerce(expr).xreplace({domain.parameter: theta})
    if isinstance(expr, sp.Poly):
        if expr.domain == domain:
            terms = ((powers, _in_theta(c.to_list(), theta)) for powers, c in expr.rep.to_dict().items())
        elif expr.domain.is_AlgebraicField:
            terms = ((powers, _lift(_scalar_expression(c, expr.domain), tuple(variables), domain, theta))
                     for powers, c in expr.rep.to_dict().items())
        else:
            # Each entry is a stored coefficient; keep the outer polynomial held.
            terms = ((powers, _lift(c, tuple(variables), domain, theta)) for powers, c in expr.as_dict().items())
        return _display_sum(_display_term(c, expr.gens, powers) for powers, c in terms)
    expr = sp.sympify(expr)
    return _lift(expr, tuple(variables), domain, theta)


@lru_cache(maxsize=128)
def _parameter(domain):
    return domain.parameter if isinstance(domain, NativeField) else sp.Dummy("coefficient_field_generator")


@lru_cache(maxsize=8192)
def _field_terms(row, domain):
    # QQ terms have exponent vectors (a,j). Group the theta coefficients
    # into field elements exactly, without symbolic substitutions of roots.
    groups = {}
    for powers, c in sparse_row(row).items():
        groups.setdefault(powers[:-1], {})[powers[-1]] = sp.QQ(c.numerator, c.denominator)
    return tuple((m,domain([terms.get(j, 0) for j in range(max(terms), -1, -1)]))
                 for m,terms in groups.items())


def _line_polynomial(row, variables, domain):
    if isinstance(domain, NativeField):
        return FieldPolynomial(_line_expression(row, variables, domain), tuple(variables), domain)
    return sp.Poly.from_dict(dict(_field_terms(row, domain)), *variables, domain=domain)


def polynomial(expression, variables, domain):
    """Build the compatibility Poly from native power-basis coefficients.

    Coercing a radical-valued expression with Poly(expr, domain=K) can
    solve a new field isomorphism for each coefficient. Reduce the lift
    once modulo the marked relation and supply the field elements directly.
    """
    if not hasattr(gb._kernel(),"atlas_coefficient_block_abi"):
        raise ArithmeticError("Rebuild the native number-field kernel with make groebner.")
    from .affine_kernel import pullbacks
    variables = tuple(variables)
    theta = _parameter(domain)
    extended = (*variables, theta)
    value, = pullbacks(extended, extended, extended, (lift(expression, variables, domain, theta),),
                       modulus=modulus(domain, theta))
    return _line_polynomial(gb._line(extended, value), variables, domain)


@lru_cache(maxsize=8192)
def _scalar_expression(value, domain):
    """Render only the powers present, preserving the field's marked generator."""
    coefficients = value.to_list()
    if len(coefficients) <= 1:
        return sp.Rational(coefficients[0]) if coefficients else sp.S.Zero
    from .field_coefficient import generator_expression
    from .field_display import scalar
    result = scalar(tuple(coefficients), generator_expression(domain.ext))
    if gb.CHECK and result != _oracle("scalar", value, domain):
        from .field_coefficient import coefficients as coordinates
        # A complete integer factorization may shorten a historical
        # bounded-search radical. Verify its marked power-basis value.
        if coordinates(result,domain) != tuple(reversed(coefficients)):
            raise AssertionError("The displayed scalar differs from its marked power-basis coordinates.")
    return result


def _line_expression(row, variables, domain):
    if isinstance(domain, NativeField):
        return np.from_row((*variables, domain.parameter), row)
    return _display_sum(_display_term(_scalar_expression(c, domain), variables, m)
                        for m, c in _field_terms(row, domain))


def lower(expr, variables, domain, theta):
    """Evaluate a rational lift at the marked algebraic generator."""
    from .affine_kernel import pullbacks
    variables = tuple(variables)
    extended = (*variables, theta)
    reduced, = pullbacks(extended, extended, extended, (expr,), modulus=modulus(domain, theta))
    return _line_expression(gb._line(extended, reduced), variables, domain)


class FieldBasis:
    def __init__(self, library, handle, variables, domain, theta, order):
        self.library, self.handle = library, handle
        self.gens, self.domain, self.theta, self.order = variables, domain, theta, order
        self.extended = (*variables, theta)
        rows = library.atlas_groebner_text(handle).decode("ascii").splitlines()
        # In every proper ideal the last row is the minimal polynomial,
        # which has a pure-theta leading monomial and vanishes in K.
        if rows and int(rows[-1].split(";")[0].split()[-1]) > 0:
            rows.pop()
        self.rows = rows
        self.exprs = [_line_expression(row, variables, domain) for row in rows]
        self.unit = self.exprs == [1]

    def __len__(self):
        return len(self.exprs)

    def __iter__(self):
        return iter(self.exprs)

    def __getitem__(self, index):
        return self.exprs[index]

    @cached_property
    def polys(self):
        return [_line_polynomial(row, self.gens, self.domain) for row in self.rows]

    def __del__(self):
        handle, self.handle = self.handle, None
        if handle:
            self.library.atlas_groebner_free(handle)

    def _rows(self, expr, quotients):
        lifted = lift(expr, self.gens, self.domain, self.theta)
        if quotients and self.unit:
            # A rational lifted basis [1] has lost the modulus row. Reduce
            # the single quotient's coefficients on the original field sheet.
            value = lower(lifted, self.gens, self.domain, self.theta)
            zero = np.S.Zero if isinstance(self.domain, NativeField) else sp.S.Zero
            return [value, zero] if value else [zero]
        payload = (gb._line(self.extended, lifted) + "\n").encode("ascii")
        operation = self.library.atlas_field_division if quotients else self.library.atlas_groebner_reduce
        answer = operation(self.handle, payload, len(payload))
        if not answer:
            reason = self.library.atlas_groebner_error()
            if reason not in (b"Exponent out of range.", b"Degree out of range.", b"Unsupported number of variables."):
                raise ArithmeticError(reason.decode())
            if hasattr(self.library, "atlas_wide_block"):
                from .field_wide import wide_block_basis
                wide = wide_block_basis(self.exprs, self.gens, self.order, self.domain, self.theta)
                if quotients:
                    qs, r = wide.reduce(expr)
                    return [*qs, r]
                return [wide.remainder(expr)]
            raise ArithmeticError("Rebuild the wide native coefficient-field kernel with make groebner.")
        try:
            rows = ctypes.string_at(answer).decode("ascii").splitlines()
        finally:
            self.library.atlas_groebner_release(answer)
        if quotients and not any(rows):
            return [np.S.Zero if isinstance(self.domain, NativeField) else sp.S.Zero]
        return [_line_expression(row, self.gens, self.domain) for row in rows]

    def contains(self, expr):
        return self._rows(expr, False)[0] == 0

    def reduce(self, expr):
        *quotients, remainder = self._rows(expr, True)
        return quotients, remainder


def native_basis(ideal, variables, order, domain):
    library = gb._kernel()
    if (library is None or not hasattr(library, "atlas_field_division")
            or order not in gb._ORDERS or not variables or domain.dom != QQ):
        return None
    library.atlas_field_division.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    library.atlas_field_division.restype = ctypes.c_void_p
    theta = _parameter(domain)
    if order == "grlex" or len(variables) >= gb._MAX_VARIABLES:
        from .field_wide import wide_block_basis
        return wide_block_basis(ideal, variables, order, domain, theta)
    extended = (*variables, theta)
    generators = (modulus(domain, theta), *(lift(f, variables, domain, theta) for f in ideal))
    # A coefficient-field basis prefix need not be monic on input. Pairs
    # with m must therefore be computed, even for a declared prefix.
    header = f"{len(extended)} {3 + gb._ORDERS[order]} 0\n"
    payload = (header + "".join(gb._line(extended, f) + "\n" for f in generators)).encode("ascii")
    handle = library.atlas_groebner(payload, len(payload))
    if not handle and library.atlas_groebner_error() in (b"Degree out of range.", b"Exponent out of range."):
        from .field_wide import wide_block_basis
        return wide_block_basis(ideal, variables, order, domain, theta)
    return FieldBasis(library, handle, variables, domain, theta, order) if handle else None


def native_colon(ideal, by, variables, domain):
    """Lift (I:J) through the surjection Q[x,theta] -> K[x]."""
    theta = _parameter(domain)
    relation = modulus(domain, theta)
    left = [relation, *(lift(f, variables, domain, theta) for f in ideal)]
    right = [relation, *(lift(f, variables, domain, theta) for f in by)]
    result = gb.colon(left, right, (*variables, theta))
    if result is None:
        return None
    lowered = tuple(lower(f, variables, domain, theta) for f in result)
    return tuple(gb.groebner(lowered, variables, domain=domain).exprs)
