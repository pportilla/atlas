"""Exact qqbar interchange for the public SymPy algebraic-expression syntax."""
from functools import lru_cache

import sympy as sp
from sympy.polys.polyerrors import CoercionFailed

from .native_library import Unavailable, field_request
from .polynomial import integer_from_decimal
from .rational_point import coordinate_fraction


class UnsupportedExpression(CoercionFailed):
    """A symbolic spelling outside the native algebraic literal syntax."""

    def __init__(self, value):
        super().__init__(
            f"Unsupported native algebraic literal {type(value).__name__}. Supply an indexed "
            "CRootOf of its rational minimal polynomial, or a polynomial in the marked field "
            "generator with rational coefficients.")


def _call(name, lines):
    row = field_request(name, lines)
    if row is None:
        raise Unavailable(f"The native field kernel lacks {name}. Rebuild it with make groebner.")
    return row


@lru_cache(maxsize=512)
def generator_expression(number):
    """Read an AlgebraicNumber's stored coordinates without calling as_expr()."""
    coeffs = tuple(number.native_coeffs())
    if coeffs == (1,0):
        return number.root
    if not coeffs:
        return sp.S.Zero
    from .field_display import scalar
    return scalar(coeffs,number.root)


@lru_cache(maxsize=8192)
def program(value):
    """Postfix QQbar operations with indexed exact algebraic root literals."""
    if value.is_Rational:
        return ("q", str(value))
    if value == sp.I:
        return ("i",)
    if value.is_Add or value.is_Mul:
        op = "+" if value.is_Add else "*"
        first,*rest = value.args
        return (*program(first), *(t for a in rest for t in (*program(a), op)))
    if value.is_Pow and value.exp.is_Rational:
        return (*program(value.base), "pow", str(value.exp))
    if isinstance(value, sp.AlgebraicNumber):
        # Keep the marked root exact, including a non-primitive element's
        # polynomial in its own primitive generator.
        coeffs = value.native_coeffs()
        answer = ("q", str(coeffs[0])) if coeffs else ("q","0")
        generator = program(value.root)
        for c in coeffs[1:]:
            answer = (*answer,*generator,"*","q",str(c),"+")
        return answer
    if isinstance(value, sp.CRootOf):
        coeffs = tuple(reversed(value.poly.all_coeffs()))
        display = marked_root_display(coeffs)
        flint_index = display[2].index(value.index)
        return ("flintroot", str(value.poly.degree()), *map(str, coeffs), str(flint_index))
    if value.func in (sp.exp, sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc):
        from .marked_field import program as marked_program
        from .polynomial import InputError
        try:
            return tuple(marked_program(str(value)).split())
        except InputError as exc:
            raise UnsupportedExpression(value) from exc
    raise UnsupportedExpression(value)


@lru_cache(maxsize=512)
def rational_sqrt(value, scale=1, offset=0):
    """offset + scale times the principal radical in the historical display syntax."""
    q = sp.Rational(value)
    scale, offset = sp.Rational(scale), sp.Rational(offset)

    row = _call("atlas_field_scaled_radical", [f"{q} {scale} {offset}"])
    num, den, raw_d, same_spelling, rational_sum = row.split()
    num, den, d = map(integer_from_decimal, (num, den, raw_d))
    scalar = sp.Rational(num, den)
    # The radicand has already been normalized by FLINT. Keep the power
    # unevaluated so constructing the display expression does not factor it.
    if not scalar or d == 1:
        result = sp.Rational(coordinate_fraction(rational_sum))
    else:
        factors = [scalar] if scalar != 1 else []
        if d < 0:
            factors.append(sp.I)
        if abs(d) != 1:
            factors.append(sp.Pow(sp.Integer(abs(d)), sp.S.Half, evaluate=False))
        radical = sp.Mul(*factors, evaluate=False)
        result = sp.Add(offset, radical, evaluate=False) if offset else radical
    return result


def coefficients(value, domain):
    """Ascending power-basis coordinates in the marked field."""
    alpha, expr = program(domain.ext), program(value)
    row = _call("atlas_field_coefficient", (" ".join(alpha), " ".join(expr)))
    if row == "outside":
        raise CoercionFailed(f"{value} does not lie in {domain}.")
    return tuple(map(coordinate_fraction, row.split()))


@lru_cache(maxsize=128)
def marked_root_display(coeffs):
    """CRootOf's scale, primitive polynomial and indices in FLINT root order.

    The input is an irreducible rational minimal polynomial, in ascending
    powers. Positive normalization preserves the ordering of FLINT's roots,
    but the subdivision defining the historical indices uses the normalized
    polynomial's own coefficient bound.
    """
    n = len(coeffs) - 1
    row = _call("atlas_field_normalize", [" ".join(map(str, (n, *coeffs)))])
    scale, *normalized = map(int, row.split())
    positions = _call("atlas_field_display_order", [" ".join(map(str, (n, *reversed(normalized))))])
    return scale, tuple(normalized), tuple(map(int, positions.split()))


@lru_cache(maxsize=128)
def indexed_field(coeffs, variable, index):
    """Make the public domain of a certified irreducible polynomial and indexed root."""
    poly = sp.PurePoly.from_list(coeffs, variable, domain=sp.QQ)
    if not 0 <= index < poly.degree():
        raise IndexError("Root index exceeds the defining polynomial's degree.")

    # Retain CRootOf's historical spelling, including positive rational
    # rescaling. Irreducibility makes refactorization and index adjustment
    # unnecessary. A positive scale preserves all real and conjugate indices
    # in the normalized polynomial, just as in CRootOf's public constructor.
    if poly.degree() == 1:
        row = _call("atlas_field_low_degree_root", [f"1 0 {poly.LC()} {poly.TC()}"])
        root = sp.Rational(coordinate_fraction(row))
    else:
        row = _call("atlas_field_normalize", [" ".join(map(str, (poly.degree(), *reversed(coeffs))))])
        scale, *normalized_coeffs = map(int, row.split())
        normalized = sp.PurePoly.from_list(normalized_coeffs, variable, domain=sp.ZZ)
        root = scale * sp.CRootOf._new(normalized, index)
    result = sp.QQ.algebraic_field((poly, root)), root
    return result


@lru_cache(maxsize=128)
def root_position(poly, primitive):
    """Index of the marked embedding in the historical CRootOf numbering."""
    # Both defining polynomials are known minimal polynomials of the same
    # literal. Scalar multiples have identical ordered roots.
    if isinstance(primitive, sp.CRootOf):
        return primitive.index

    generator = " ".join(program(primitive))
    coeffs = tuple(reversed(poly.all_coeffs()))
    lines = [generator, " ".join(map(str, (poly.degree(), *coeffs)))]
    display = marked_root_display(coeffs)
    return display[2][int(_call("atlas_field_flint_index", lines))]
