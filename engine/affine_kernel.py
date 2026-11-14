"""Rational affine maps, their divisorial data and transition poles (KERNEL.md §9.5).

FLINT receives the polynomial expressions, including products and powers. The
unpacked native kernel handles rings or monomials outside the packed ABI.
"""
import ctypes
import re
from fractions import Fraction
from functools import lru_cache
from operator import index

from . import native_polynomial as np
from .polynomial import integer_from_decimal, integer_text, rational_text
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
            return rational_text(value)
        if not isinstance(value, sp.Basic):
            raise TypeError("Supply a parsed rational polynomial.")
        if value.is_Symbol:
            if value not in self.names:
                raise ArithmeticError("A polynomial uses an undeclared coordinate.")
            return self.names[value]
        if value.is_Rational:
            text = rational_text(value)
            return text if value.q == 1 else f"({text})"
        if value.is_Add or value.is_Mul:
            sep = "+" if value.is_Add else "*"
            return "(" + sep.join(self.text(a) for a in value.args) + ")"
        if value.is_Pow and value.exp.is_Integer:
            exponent = int(value.exp)
            power = f"({self.text(value.base)})^{integer_text(abs(exponent))}"
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


def _oracle(name, *args):
    from tests import affine_oracle
    return getattr(affine_oracle, name)(*args)


def _checked(native, reference, variables=()):
    if np.native_ring(variables):
        return native()
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

    return _checked(native, lambda: _oracle("lattice_dimension", rows))


@lru_cache(maxsize=16384)
def _sparse_terms(row):
    """Decode an expanded native polynomial without a dense coefficient array."""
    from .polynomial import sparse_row
    return tuple(sparse_row(row).items())


def _polynomial(terms, variables):
    if np.native_ring(variables):
        return np.from_terms(variables, terms)
    from .field_polynomial import _display_sum, _display_term
    return _display_sum(_display_term(sp.Rational(c.numerator, c.denominator), variables, powers)
                        for powers, c in terms.items())


@lru_cache(maxsize=64)
def _coordinate_rows(size):
    return {"1 " + " ".join("1" if j == i else "0" for j in range(size)): i for i in range(size)}


@lru_cache(maxsize=256)
def _can_reuse_coordinates(variables):
    return len(set(variables)) == len(variables) and all(v.is_Symbol for v in variables)


@lru_cache(maxsize=16384)
def _sparse_expression(variables, row):
    # Wide rings can have arbitrarily many variables. Only build the table for packed rings.
    coordinate = _coordinate_rows(len(variables)).get(row) if len(variables) <= gb._MAX_VARIABLES else None
    if coordinate is not None and _can_reuse_coordinates(variables):
        value = variables[coordinate]
    else:
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

    return _checked(native, lambda: _oracle("sparse_terms", variables, value), variables)


def decode(variables, text):
    """Read one record expression with the polynomial grammar over Q."""
    from .polynomial import InputError
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

    return _checked(native, lambda: _oracle("decode", variables, names, text), variables)


def pullbacks(variables, mapping, base, functions, modulus=None):
    """Compose each function on the base with the chart map, optionally over Q[theta]/m."""
    variables, functions = tuple(variables), tuple(functions)

    def native():
        rows = _ask("pullbacks", variables, mapping, base, functions, modulus=modulus or 0)
        return tuple(_sparse_expression(variables, row) for row in rows)

    return _checked(native, lambda: _oracle("pullbacks", functions, variables, mapping, base, modulus), variables)


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
        return tuple(tuple(map(integer_from_decimal, r.split())) for r in rows[:-1]), _sparse_expression(variables, rows[-1])

    return _checked(native, lambda: _oracle("chart_data", variables, mapping, divisors, equation, base, modulus, indices), variables)


def inverse_map(variables, mapping, base):
    """Return coordinate fractions in the base, each with a monic denominator."""
    variables, base = tuple(variables), tuple(base)

    def native():
        rows = _ask("inverse", variables, mapping, base, ())
        parts = [_sparse_expression(base, row) for row in rows]
        return tuple(zip(parts[::2], parts[1::2]))

    return _checked(native, lambda: _oracle("inverse_map", variables, mapping, base), variables)


def transition_pole(variables, mapping, base, inverse):
    """Return the squarefree monic equation of the complement of the overlap."""
    variables = tuple(variables)

    def native():
        rows = _ask("pole", variables, mapping, base, (f for pair in inverse for f in pair))
        return _sparse_expression(variables, rows[0])

    return _checked(native, lambda: _oracle("transition_pole", variables, mapping, base, inverse), variables)


@lru_cache(maxsize=128)
def frame_locus(variables, equations, units, *, closure=False):
    """Emptiness or the grevlex basis of the contracted ideal on an open frame."""
    variables, equations, units = tuple(variables), tuple(equations), tuple(units)
    if not variables:
        empty = any(f != 0 for f in equations) or any(u == 0 for u in units)
        return ((1,) if empty else ()) if closure else empty

    def native():
        op = "frame_basis" if closure else "frame_empty"
        ring = _ring(variables)
        if (gb._MODE == "native" and np.native_ring(variables) and len(variables) <= gb._MAX_VARIABLES
                and all(isinstance(f, (np.Expr, int, Fraction)) for f in (*equations, *units))):
            from .native_library import _kernel, ideal_request
            if hasattr(_kernel(), "atlas_frame_locus_abi"):
                try:
                    rows = [np.row(variables, f) for f in (*equations, *units)]
                    result = ideal_request(f"{op} {len(variables)} {len(equations)} {len(units)}", rows)
                except (_AffineUnavailable, KeyError):
                    pass  # The general affine reader supports wide exponents and checks coordinates.
                else:
                    if closure:
                        return tuple(_sparse_expression(variables, row) for row in result)
                    if result in (["1"], ["0"]):
                        return result == ["1"]
        lines = [op, ring.header, str(len(equations)), *map(ring.text, equations), *map(ring.text, units)]
        result = _request(lines)
        return tuple(_sparse_expression(variables, row) for row in result) if closure else result == ["1"]

    return _checked(native, lambda: _oracle("frame_locus", variables, equations, units, closure), variables)


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
    if gb._MODE == "native" and np.native_ring(variables) and isinstance(value, (np.Expr, int, Fraction)):
        value = np.coerce(value)
        if value.free_symbols.issubset(_ring(variables).names):
            return value  # Native chart polynomials already carry their expanded coefficients.
    return _checked(lambda: _frame_polys("function", variables, None, (value,))[0],
                    lambda: _oracle("frame_function", value), variables)


def frame_minor(variables, rows, columns):
    """A Jacobian minor, with its sign and rational scale."""
    variables = tuple(variables)
    return _checked(lambda: _frame_polys("minor", variables, columns, rows)[0],
                    lambda: _oracle("frame_minor", rows, columns), variables)


def frame_field(variables, rows, columns):
    """Signed complementary minors for the implicit derivatives in these columns."""
    variables = tuple(variables)

    values = _checked(lambda: _frame_polys("field", variables, columns, rows), lambda: _oracle("frame_field", rows, columns), variables)
    return {v: c for v, c in zip(columns, values) if c != 0}


def frame_derivative(variables, f, field):
    """Apply a polynomial vector field before reducing by the frame's rows."""
    variables = tuple(variables)
    return _checked(lambda: _frame_polys("derive", variables, field, (f, *field.values()))[0],
                    lambda: _oracle("frame_derivative", f, field), variables)


def archived_factor_order(parts, dimension):
    """Indices of the sparse rational factors in the original Atlas branch order."""
    from .kernel_json import kernel_json
    from .point_geometry import encode_terms
    from .polynomial import integer_text
    payload = [[encode_terms(terms), integer_text(m)] for terms, m in parts]
    return kernel_json("atlas_identity_divisor", "branch_order", width=dimension, parts=payload)


@lru_cache(maxsize=512)
def rational_factor_rows(variables, expression):
    """Scalar and (sparse row, multiplicity, primitive-vertex certificate) factors over Q.

    The certificate applies to rationally irreducible factors. The geometric
    factors are Galois conjugate, so every vertex of the Newton polytope of
    their product is divisible by their number. A gcd of one proves that
    there is just one. The order of the rows is the archived QQ factor order.
    """
    def native():
        ring = _ring(variables)
        lines = _request(["geometric_factors", ring.header, ring.text(expression)])
        scalar = dict(_sparse_terms(lines[0])).get((0,)*len(variables), Fraction(0))
        parts = []
        for info, row in zip(lines[1::2], lines[2::2]):
            fields = tuple(map(integer_from_decimal, info.split()))
            primitive = bool(fields[1])
            parts.append((row, fields[0], primitive))
        if len(parts) > 1:
            order = archived_factor_order(((dict(_sparse_terms(row)), m) for row, m, _ in parts), len(variables))
            parts = [parts[i] for i in order]
        return scalar, tuple(parts)
    return _checked(native, lambda: _oracle("rational_factor_rows", variables, expression), variables)


def rational_factors(poly):
    """Primitive factors over Q, in the branch order of existing records."""
    if poly.domain.is_AlgebraicField:
        from .field_factors import factors
        return poly.LC(), factors(poly)
    if poly.domain not in (sp.QQ, sp.ZZ):
        if gb._MODE in ("sympy", "check"):
            from tests.field_polynomial_oracle import factors
            return factors(poly)
        raise ValueError("Native factorization accepts QQ, ZZ, or a marked number field.")

    scalar, rows = rational_factor_rows(poly.gens, poly.as_expr())
    return sp.Rational(scalar), [(sp.Poly.from_dict(dict(_sparse_terms(row)), *poly.gens, domain=poly.domain), m)
                                 for row, m, _ in rows]


def frame_linear(variables, f, k):
    """The coefficient and constant of f in k, or None unless its degree is one."""
    variables = tuple(variables)

    def native():
        ring = _ring(variables)
        rows = _request(["linear", ring.header, str(variables.index(k)), ring.text(f)])
        return None if rows == ["none"] else tuple(_sparse_expression(variables, r) for r in rows)

    return _checked(native, lambda: _oracle("frame_linear", f, k), variables)


@lru_cache(maxsize=512)
def _native_restriction(variables, f, k, a, b):
    return _frame_polys("restrict", variables, (k,), (f, a, b))[0]


def frame_restrict(variables, f, k, a, b):
    """Restrict f to k=-b/a, clearing its denominator up to an invertible power of a."""
    variables = tuple(variables)

    if (gb._MODE == "native" and np.native_ring(variables)
            and all(isinstance(value, (np.Expr, int, Fraction)) for value in (f, k, a, b))):
        return _native_restriction(variables, f, k, a, b)
    return _checked(lambda: _frame_polys("restrict", variables, (k,), (f, a, b))[0], lambda: _oracle("frame_restrict", variables, f, k, a, b), variables)


def frame_quotient(variables, f, k):
    """The polynomial f/k, or None when there is a remainder."""
    variables = tuple(variables)

    def native():
        ring = _ring(variables)
        rows = _request(["divide_coordinate", ring.header, str(variables.index(k)), ring.text(f)])
        return None if rows == ["none"] else _sparse_expression(variables, rows[0])

    return _checked(native, lambda: _oracle("frame_quotient", variables, f, k), variables)
