"""Reduced Groebner bases over Q from the native kernel.

The native kernel computes canonical bases over Q and simple number fields,
including sparse block-order calculations outside the packed monomial range.
Independent basis and remainder comparisons live in the tests. When the exact strategies
swell, the kernel switches to modular images and certifies the lifted basis;
with ATLAS_GROEBNER_GPU=1 the replays for further primes run on an OpenCL
device (make gpu).
"""
import ctypes
from functools import cached_property, lru_cache

from . import native_polynomial as np
from .polynomial import integer_text, rational_text, sparse_row
from .reference import sp
from .reference import QQ
from .algebra_execution import current_execution, packed_basis, record_fallback, record_stage

from .native_library import Unavailable, _kernel

# "first": the exponent of the first variable is compared before the grevlex
# order of the others, so the first variable is eliminated.
_ORDERS = {"grevlex": 0, "lex": 1, "first": 2, "grlex": 3}
_MAX_VARIABLES = 32
_MAX_DEGREE = 32767  # Native packed exponent/total-degree limit.


# Generators recur from one basis to the next, and converting a SymPy
# expression costs far more than the kernel spends on it.
_lines = {}
_polys = {}
_CACHE_LIMIT = 50_000


@lru_cache(maxsize=256)
def _monomial_positions(gens):
    if not all(v.is_Symbol and v.is_commutative for v in gens):
        return None
    positions = {v: i for i, v in enumerate(gens)}
    return positions if len(positions) == len(gens) else None


def _monomial_row(gens, expr):
    """Encode a symbolic rational monomial in the declared coordinate order."""
    positions = _monomial_positions(gens)
    if not positions:
        return None
    if expr.is_Rational and expr.is_zero:
        return ""
    powers = ["0"] * len(gens)
    coefficient = None
    for factor in expr.args if expr.is_Mul else (expr,):
        if factor.is_Rational:
            if coefficient is not None:
                return None  # Unevaluated repeated factors go through the native parser.
            coefficient = str(factor)
            continue
        base, exponent = (factor.base, factor.exp) if factor.is_Pow else (factor, 1)
        if factor.is_Pow and not (exponent.is_Integer and exponent.is_positive):
            return None
        i = positions.get(base)
        if i is None or powers[i] != "0":
            return None
        powers[i] = str(exponent)
    return "" if coefficient == "0" else (coefficient or "1") + " " + " ".join(powers)


def _expanded_qq_line(gens, expr):
    """Encode parsed rational expressions as sparse FLINT terms."""
    if not isinstance(expr, sp.Expr) or not gens:
        return None
    # The FLINT expression reader also accepts exact polynomial division.
    # Restrict it to the QQ expression tree so symbols named I, pi and
    # similar cannot be confused with undeclared constants or functions.
    if not all(g.is_Symbol and g.is_commutative for g in gens):
        return None
    if any(t.is_Pow and t.exp.is_negative for t in sp.preorder_traversal(expr)):
        return None
    from .affine_kernel import _request, _ring
    ring = _ring(gens)
    try:
        row, = _request(["function", ring.header, ring.text(expr)])
    except Unavailable:
        raise
    except ArithmeticError as exc:
        raise sp.PolynomialError("Expected a polynomial in the specified rational coordinates.") from exc
    return row


def _line(gens, expr):
    """One polynomial in the kernel's text form: factors joined by '*', each a
    list of terms 'coefficient e1 ... en' joined by ';'."""
    if np.native_ring(gens):
        return np.row(gens, expr)
    if not isinstance(expr, sp.Basic):
        expr = sp.sympify(expr)
    if isinstance(expr, sp.Poly) and expr.domain in (QQ,sp.ZZ) and set(expr.gens).issubset(gens):
        positions = tuple(expr.gens.index(g) if g in expr.gens else None for g in gens)
        return ";".join(rational_text(c) + " " + " ".join(integer_text(m[i]) if i is not None else "0" for i in positions)
                        for m,c in expr.rep.to_dict().items())
    key = (gens, expr)
    line = _lines.get(key)
    if line is not None:
        return line
    line = _monomial_row(gens, expr)
    if line is None:
        if expr.is_Mul and any(factor.is_Add for factor in expr.args):
            # The kernel multiplies; expanding a product here is the slow part.
            line = "*".join(_line(gens, factor) for factor in expr.args)
        elif (line := _expanded_qq_line(gens, expr)) is None:
            raise sp.PolynomialError("Expected a polynomial in the specified rational coordinates.")
    if len(_lines) >= _CACHE_LIMIT:
        _lines.clear()
    _lines[key] = line
    return line


def _canonical_qq(gens, expr):
    """Reuse a decoded expression or expand a compound generator for its caller."""
    if np.native_ring(gens):
        value = np.coerce(expr)
        if not value.free_symbols.issubset(gens):
            raise ValueError("The polynomial uses an undeclared coordinate.")
        return value
    row = _lines.get((gens, expr))
    decoded = _polys.get((gens, row)) if row is not None else None
    if decoded is not None:
        return decoded[1]
    if row is None or "*" in row:
        row = _expanded_qq_line(gens, expr)
    if row is None:
        return None
    from .affine_kernel import _sparse_expression
    return _sparse_expression(gens, row)


def _poly(gens, line):
    """The polynomial of a kernel line, with its expression."""
    key = (gens, line)
    found = _polys.get(key)
    if found is None:
        poly = sp.Poly.from_dict(sparse_row(line), *gens, domain=QQ)
        found = poly, poly.as_expr()
        if len(_polys) >= _CACHE_LIMIT:
            _polys.clear()
        _polys[key] = found
        _lines.setdefault((gens, found[1]), line)
    return found


class NativeBasis:
    """A reduced Groebner basis held by the kernel.

    It offers what the verifier uses of SymPy's GroebnerBasis: polys, exprs,
    contains and the remainder of reduce.
    """
    def __init__(self, library, handle, gens, order):
        self._library, self._handle, self.gens = library, handle, gens
        self._order = order
        self._wide = None
        from .affine_kernel import _sparse_expression
        self.rows = library.atlas_groebner_text(handle).decode("ascii").splitlines()
        self.exprs = [_sparse_expression(gens, row) for row in self.rows]

    @cached_property
    def polys(self):
        if np.native_ring(self.gens):
            return self.exprs
        return [_poly(self.gens, row)[0] for row in self.rows]

    def __del__(self):
        handle, self._handle = self._handle, None
        if handle:
            try:
                self._library.atlas_groebner_free(handle)
            except Exception:  # the interpreter is shutting down
                pass

    def _device_remainder(self, row):
        run = current_execution()
        if run is not None and run.replay.compute == "gpu":
            from .gpu_normal_form import normal_form
            return normal_form(self.gens, self.rows, row)
        return None

    def _wide_divisors_on_overflow(self):
        if self._library.atlas_groebner_error() not in (b"Degree out of range.", b"Exponent out of range."):
            return None
        if self._wide is None:
            from .wide_basis import wide_basis_from_rows
            self._wide = wide_basis_from_rows(self.rows, self.gens, self._order)
        return self._wide

    def _cpu_remainder(self, expr, row):
        payload = (row + "\n").encode("ascii")
        answer = self._library.atlas_groebner_reduce(self._handle, payload, len(payload))
        if not answer:
            wide = self._wide_divisors_on_overflow()
            if wide is not None:
                return _line(self.gens, wide.remainder(expr))
            raise ArithmeticError("The native normal form could not be computed.")
        try:
            result = ctypes.string_at(answer).decode("ascii")[:-1]
        finally:
            self._library.atlas_groebner_release(answer)
        record_stage("normal_form", "cpu", divisors=len(self.rows))
        return result

    def contains(self, expr):
        row = _line(self.gens, expr)
        device = self._device_remainder(row)
        if device is not None:
            return not device
        contains = getattr(self._library, "atlas_groebner_contains", None)
        if contains is None:
            return not self._cpu_remainder(expr, row)
        payload = (row + "\n").encode("ascii")
        result = contains(self._handle, payload, len(payload))
        if result < 0:
            wide = self._wide_divisors_on_overflow()
            if wide is not None:
                return wide.contains(expr)
            raise ArithmeticError("The native membership test could not be completed.")
        record_stage("membership", "cpu", divisors=len(self.rows))
        return bool(result)

    def remainder(self, expr):
        from .affine_kernel import _sparse_expression
        row = _line(self.gens, expr)
        result = self._device_remainder(row)
        if result is None:
            result = self._cpu_remainder(expr, row)
        return _sparse_expression(self.gens, result)

    def reduce(self, expr):
        """Return (None, remainder): the quotients are not computed."""
        return None, self.remainder(expr)


def _wide_qq_basis(ideal, gens, order, reason):
    from .wide_basis import wide_basis
    basis = wide_basis(ideal, gens, order)
    if basis is not None:
        record_fallback(reason)
        record_stage("groebner_basis", "cpu", algorithm="wide_block" if order == "first" else "wide",
                     generators=len(ideal), order=order)
    return basis


def _native(ideal, gens, order, known):
    library = _kernel()
    if library is None or order not in _ORDERS or not gens:
        return None
    if order == "grlex":
        from .field_wide import wide_block_basis
        return wide_block_basis(ideal, gens, order)
    if len(gens) > _MAX_VARIABLES:
        return _wide_qq_basis(ideal, gens, order, "The basis exceeds the packed GPU variable range.")
    if np.native_ring(gens):
        lines = [np.row(gens, f) for f in ideal]
    else:
        lines = [_line(gens, f) for f in ideal]
    payload = (f"{len(gens)} {_ORDERS[order]} {known}\n" + "".join(line + "\n" for line in lines)).encode("ascii")
    handle = packed_basis(library, payload, lines, can_homogenize=len(gens) < _MAX_VARIABLES,
                          known=known, order=order)
    if not handle and library.atlas_groebner_error() in (b"Degree out of range.", b"Exponent out of range."):
        return _wide_qq_basis(ideal, gens, order, "The basis exceeds the packed GPU monomial range.")
    return NativeBasis(library, handle, gens, order) if handle else None


def groebner(ideal, gens, *, order="grevlex", domain=QQ, known=0):
    """The reduced Groebner basis of the ideal, largest leading monomial first.

    `known` promises that the first that many generators already form a
    Groebner basis for this order; the kernel then skips their S-polynomials.
    """
    gens = tuple(gens)
    ideal = tuple(ideal)
    if not 0 <= known <= len(ideal):
        raise ValueError("known must count leading generators of the ideal.")
    native = None
    if domain == QQ:
        native = _native(ideal, gens, order, known)
    elif getattr(domain, "is_AlgebraicField", False):
        from .field_polynomial import native_basis
        native = native_basis(ideal, gens, order, domain)
    if native is not None:
        return native
    raise Unavailable("Native Groebner computation unavailable for the specified ring or order. "
                      "Build the current kernel with make groebner.")


def colon(ideal, by, gens, *, domain=QQ):
    """Generators of the ideal quotient (ideal) : (by), or None when only SymPy can do it."""
    gens = tuple(gens)
    library = _kernel()
    if (getattr(domain, "is_AlgebraicField", False) and library is not None
            and hasattr(library, "atlas_coefficient_block_abi")):
        from .field_polynomial import native_colon
        return native_colon(ideal, by, gens, domain)
    if domain != QQ or library is None or not gens:
        return None
    if len(gens) >= _MAX_VARIABLES:
        from .wide_basis import wide_colon
        return wide_colon(ideal, by, gens)
    if np.native_ring(gens):
        lines = [np.row(gens, f) for f in (*ideal, *by)]
    else:
        try:
            lines = [_line(gens, f) for f in (*ideal, *by)]
        except sp.polys.polyerrors.BasePolynomialError:
            return None
    payload = (f"{len(gens)} {len(ideal)}\n" + "".join(line + "\n" for line in lines)).encode("ascii")
    answer = library.atlas_groebner_colon(payload, len(payload))
    if not answer:
        from .wide_basis import wide_colon
        return wide_colon(ideal, by, gens)
    try:
        text = ctypes.string_at(answer).decode("ascii")
    finally:
        library.atlas_groebner_release(answer)
    from .affine_kernel import _sparse_expression
    return tuple(_sparse_expression(gens, line) for line in text.splitlines())
