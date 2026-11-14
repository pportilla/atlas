"""Exact qqbar interchange for the public SymPy algebraic-expression syntax."""
from fractions import Fraction
from functools import lru_cache

import sympy as sp
from sympy.polys.polyerrors import CoercionFailed

from . import groebner as gb
from .native_library import Unavailable, field_request


class UnsupportedExpression(CoercionFailed):
    """A symbolic spelling outside the native algebraic literal syntax."""

    def __init__(self, value):
        super().__init__(
            f"Unsupported native algebraic literal {type(value).__name__}. Supply an indexed "
            "CRootOf of its rational minimal polynomial, or a polynomial in the marked field "
            "generator with rational coefficients. Set ATLAS_GROEBNER=sympy for symbolic coercion.")


def _call(name, lines):
    row = field_request(name, lines)
    if row is None and gb._MODE == "native":
        raise Unavailable(f"The native field kernel lacks {name}. Rebuild it with make groebner.")
    return row


def _unity_power(q):
    # exp(i*pi*q) is the principal rational power (-1)**q.
    return ("q", "-1", "pow", str(q))


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
        if gb._MODE != "sympy":
            display = marked_root_display(coeffs)
            if display is not None:
                flint_index = display[2].index(value.index)
                return ("flintroot", str(value.poly.degree()), *map(str, coeffs), str(flint_index))
        # Portable reader for a kernel predating native indexed-root selection.
        bounds = value._get_interval()
        box = (bounds.a, bounds.b, 0, 0) if value.is_real else (bounds.ax, bounds.bx, bounds.ay, bounds.by)
        return ("root", str(value.poly.degree()), *map(str,reversed(value.poly.all_coeffs())), *map(str,box))
    if value.func == sp.exp:
        q = value.args[0]/(sp.I*sp.pi)
        if q.is_Rational:
            return _unity_power(q)
    if value.func in (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc):
        q = value.args[0]/sp.pi
        if q.is_Rational:
            # Euler's identities at a rational multiple of pi use only
            # roots of unity and field operations. QQbar evaluates the
            # reciprocal and detects a pole when its denominator vanishes.
            z, inv = _unity_power(q), _unity_power(-q)
            cosine = (*z, *inv, "+", "q", "1/2", "*")
            sine = (*z, *inv, "q", "-1", "*", "+", "i", "*", "q", "-1/2", "*")
            formulas = {sp.cos: cosine, sp.sin: sine, sp.tan: (*sine, *cosine, "pow", "-1", "*"),
                        sp.cot: (*cosine, *sine, "pow", "-1", "*"), sp.sec: (*cosine, "pow", "-1"),
                        sp.csc: (*sine, "pow", "-1")}
            return formulas[value.func]
    raise UnsupportedExpression(value)


@lru_cache(maxsize=512)
def rational_sqrt(value, scale=1, offset=0):
    """offset + scale times the principal radical in the historical display syntax."""
    q = sp.Rational(value)
    scale, offset = sp.Rational(scale), sp.Rational(offset)
    def reference():
        return offset + scale * sp.sqrt(q)

    if gb._MODE == "sympy":
        return reference()
    row = _call("atlas_field_radical", [str(q)])
    if row is None:
        return reference()
    num, den, d, same_spelling = map(int, row.split())
    scalar = scale * sp.Rational(num, den)
    # The radicand has already been normalized by FLINT. Keep the power
    # unevaluated so constructing the display expression does not factor it.
    if not scalar or d == 1:
        result = offset + scalar
    else:
        factors = [scalar] if scalar != 1 else []
        if d < 0:
            factors.append(sp.I)
        if abs(d) != 1:
            factors.append(sp.Pow(sp.Integer(abs(d)), sp.S.Half, evaluate=False))
        radical = sp.Mul(*factors, evaluate=False)
        result = sp.Add(offset, radical, evaluate=False) if offset else radical
    if gb.CHECK and result != reference():
        # A bounded historical search can leave a large square inside a
        # radical. Since both branches use the principal root, comparing
        # its square suffices even for negative rational radicands.
        if same_spelling or not scale:
            raise AssertionError("The marked quadratic radical differs from its historical spelling.")
        if (sp.simplify(((result - offset) / scale)**2 - q) != 0
                or sp.simplify(((reference() - offset) / scale)**2 - q) != 0):
            raise AssertionError("Quadratic radicals differ after normalization.")
    return result


def coefficients(value, domain):
    """Ascending power-basis coordinates. Diagnostics can defer unknown literals to SymPy."""
    try:
        alpha, expr = program(domain.ext), program(value)
    except UnsupportedExpression:
        if gb._MODE == "native":
            raise
        return None
    row = _call("atlas_field_coefficient", (" ".join(alpha), " ".join(expr)))
    if row is None:
        return None
    if row == "outside":
        raise CoercionFailed(f"{value} does not lie in {domain}.")
    return tuple(Fraction(word) for word in row.split())


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
    if row is None:
        return None
    scale, *normalized = map(int, row.split())
    positions = _call("atlas_field_display_order", [" ".join(map(str, (n, *reversed(normalized))))])
    if positions is None:
        return None
    return scale, tuple(normalized), tuple(map(int, positions.split()))


@lru_cache(maxsize=128)
def indexed_field(coeffs, variable, index):
    """Make the public domain of a certified irreducible polynomial and indexed root."""
    poly = sp.PurePoly.from_list(coeffs, variable, domain=sp.QQ)
    if not 0 <= index < poly.degree():
        raise IndexError("Root index exceeds the defining polynomial's degree.")

    def reference():
        root = sp.CRootOf(poly.as_expr(), index)
        return sp.QQ.algebraic_field(root), root

    if gb._MODE == "sympy":
        return reference()
    # Retain CRootOf's historical spelling, including positive rational
    # rescaling. Irreducibility makes refactorization and index adjustment
    # unnecessary. A positive scale preserves all real and conjugate indices
    # in the normalized polynomial, just as in CRootOf's public constructor.
    if poly.degree() == 1:
        root = -poly.TC() / poly.LC()
    else:
        row = _call("atlas_field_normalize", [" ".join(map(str, (poly.degree(), *reversed(coeffs))))])
        if row is None:
            return reference()
        scale, *normalized_coeffs = map(int, row.split())
        normalized = sp.PurePoly.from_list(normalized_coeffs, variable, domain=sp.ZZ)
        root = scale * sp.CRootOf._new(normalized, index)
    result = sp.QQ.algebraic_field((poly, root)), root
    if gb.CHECK and result != reference():
        raise AssertionError("The reconstructed primitive field differs from its reference.")
    return result


@lru_cache(maxsize=128)
def root_position(poly, primitive):
    """Index of the marked embedding in the historical CRootOf numbering."""
    # Both defining polynomials are known minimal polynomials of the same
    # literal. Scalar multiples have identical ordered roots.
    if isinstance(primitive, sp.CRootOf):
        return primitive.index

    def reference():
        if poly.degree() == 1 and primitive == -poly.TC()/poly.LC():
            return 0
        return next(i for i in range(poly.degree())
                    if poly.same_root(primitive, sp.CRootOf(poly.as_expr(), i)))

    if gb._MODE == "sympy":
        return reference()
    try:
        generator = " ".join(program(primitive))
    except UnsupportedExpression:
        if gb._MODE == "native":
            raise
        return reference()
    coeffs = tuple(reversed(poly.all_coeffs()))
    lines = [generator, " ".join(map(str, (poly.degree(), *coeffs)))]
    display = (marked_root_display(coeffs) if gb._MODE == "native"
               or hasattr(gb._kernel(), "atlas_field_flint_index") else None)
    if display is not None:
        row = str(display[2][int(_call("atlas_field_flint_index", lines))])
    else:
        row = _call("atlas_field_position", lines)
    if row is None:
        return reference()
    if row.startswith("boxes "):
        # With several conjugate pairs, SymPy sorts by the lower corners
        # of its isolating rectangles. Compare those marked roots exactly.
        for i in range(int(row.split()[1]), poly.degree()):
            root = sp.CRootOf(poly.as_expr(), i)
            lines.append(f"{i} " + " ".join(program(root)))
        row = _call("atlas_field_position", lines)
        if row.startswith("boxes "):
            raise ArithmeticError("The marked root has no matching complex isolator.")
    index = int(row)
    if gb.CHECK and index != reference():
        raise AssertionError("The marked field root index differs from its reference.")
    return index
