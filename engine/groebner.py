"""Reduced Groebner bases over Q from the native kernel.

The native kernel computes canonical bases over Q and simple number fields,
including sparse block-order calculations outside the packed monomial range.
SymPy supplies the explicit comparison and reference modes.
Set ATLAS_GROEBNER=sympy to use SymPy only, or =check to compute with both
and compare every basis and every remainder. When the exact strategies
swell, the kernel switches to modular images and certifies the lifted basis;
with ATLAS_GROEBNER_GPU=1 the replays for further primes run on an OpenCL
device (make gpu).
"""
import ctypes
from functools import cached_property, lru_cache

from . import native_polynomial as np
from .reference import sp
from .reference import QQ

from .native_library import CHECK, Unavailable, _LIBRARY, _MODE, _kernel, _use_gpu

# "first": the exponent of the first variable is compared before the grevlex
# order of the others, so the first variable is eliminated.
_ORDERS = {"grevlex": 0, "lex": 1, "first": 2, "grlex": 3}


def _reference_order(order):
    if order != "first":
        return order
    from sympy.polys.orderings import ProductOrder, grevlex, lex
    return ProductOrder((lex, lambda monomial: monomial[:1]), (grevlex, lambda monomial: monomial[1:]))
_MAX_VARIABLES = 32


# Generators recur from one basis to the next, and converting a SymPy
# expression costs far more than the kernel spends on it.
_lines = {}
_polys = {}
_CACHE_LIMIT = 50_000


def _expanded_qq_line(gens, expr):
    """Encode parsed rational expressions as sparse FLINT terms."""
    if _MODE == "sympy" or not isinstance(expr, sp.Expr) or not gens:
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
        if _MODE == "native":
            raise
        return None
    except ArithmeticError as exc:
        if _MODE == "native":
            raise sp.PolynomialError("Expected a polynomial in the specified rational coordinates.") from exc
        return None
    if CHECK:
        from .affine_kernel import _sparse_terms
        from sympy.polys.rings import ring
        sparse, *_ = ring(gens, QQ)
        if dict(_sparse_terms(row)) != dict(sparse.from_expr(expr)):
            raise AssertionError("Native polynomial expansion differs from the QQ term list.")
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
        return ";".join(f"{c} " + " ".join(str(m[i]) if i is not None else "0" for i in positions)
                        for m,c in expr.rep.to_dict().items())
    key = (gens, expr)
    line = _lines.get(key)
    if line is None:
        if expr.is_Mul and any(factor.is_Add for factor in expr.args):
            # The kernel multiplies; expanding a product here is the slow part.
            line = "*".join(_line(gens, factor) for factor in expr.args)
        elif (line := _expanded_qq_line(gens, expr)) is None:
            if _MODE == "native":
                raise sp.PolynomialError("Expected a polynomial in the specified rational coordinates.")
            terms = sp.Poly(expr, *gens, domain=QQ).as_dict(native=True)
            line = ";".join(f"{c} {' '.join(map(str, m))}" for m, c in terms.items())
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
    if _MODE == "sympy":
        return None
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
        terms = {}
        for term in line.split(";") if line else ():
            coefficient, *powers = term.split(" ")
            numerator, _, denominator = coefficient.partition("/")
            terms[tuple(map(int, powers))] = QQ(int(numerator), int(denominator or 1))
        poly = sp.Poly.from_dict(terms, *gens, domain=QQ)
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

    def _remainder(self, expr):
        payload = (_line(self.gens, expr) + "\n").encode("ascii")
        answer = self._library.atlas_groebner_reduce(self._handle, payload, len(payload))
        if not answer:
            if self._library.atlas_groebner_error() in (b"Degree out of range.", b"Exponent out of range."):
                from .wide_basis import wide_basis
                if self._wide is None:
                    self._wide = wide_basis(self.exprs, self.gens, self._order)
                if self._wide is not None:
                    return _line(self.gens, self._wide.reduce(expr)[1])
            raise ArithmeticError("The native normal form could not be computed.")
        try:
            return ctypes.string_at(answer).decode("ascii")[:-1]
        finally:
            self._library.atlas_groebner_release(answer)

    def contains(self, expr):
        return not self._remainder(expr)

    def reduce(self, expr):
        """Return (None, remainder): the quotients are not computed."""
        from .affine_kernel import _sparse_expression
        return None, _sparse_expression(self.gens, self._remainder(expr))


class _Checked:
    """Both engines side by side; any disagreement is an error."""
    def __init__(self, native, reference):
        self._native, self._reference = native, reference
        if native.exprs != reference.exprs:
            raise AssertionError("The native Groebner basis differs from SymPy's.")
        self.exprs = native.exprs

    @property
    def polys(self):
        return self._native.polys

    def contains(self, expr):
        answer = self._native.contains(expr)
        if answer != self._reference.contains(expr):
            raise AssertionError("The native ideal membership differs from SymPy's.")
        return answer

    def reduce(self, expr):
        answer = self._native.reduce(expr)
        expected = self._reference.reduce(expr)
        if answer[1] != expected[1] or (answer[0] is not None and answer[0] != expected[0]):
            raise AssertionError("The native normal form differs from SymPy's.")
        return answer


def _native(ideal, gens, order, known):
    library = _kernel()
    if library is None or order not in _ORDERS or not gens:
        return None
    if order == "grlex":
        from .field_wide import wide_block_basis
        return wide_block_basis(ideal, gens, order)
    if len(gens) > _MAX_VARIABLES:
        from .wide_basis import wide_basis
        return wide_basis(ideal, gens, order)
    if np.native_ring(gens):
        lines = [np.row(gens, f) for f in ideal]
    else:
        try:
            lines = [_line(gens, f) for f in ideal]
        except sp.polys.polyerrors.BasePolynomialError:
            if _MODE == "native":
                raise
            return None
    payload = (f"{len(gens)} {_ORDERS[order]} {known}\n" + "".join(line + "\n" for line in lines)).encode("ascii")
    handle = library.atlas_groebner(payload, len(payload))
    if not handle and library.atlas_groebner_error() in (b"Degree out of range.", b"Exponent out of range."):
        from .wide_basis import wide_basis
        return wide_basis(ideal, gens, order)
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
    if _MODE != "sympy":
        if domain == QQ:
            native = _native(ideal, gens, order, known)
        elif getattr(domain, "is_AlgebraicField", False):
            from .field_polynomial import native_basis
            native = native_basis(ideal, gens, order, domain)
    if native is not None and _MODE != "check":
        return native
    if _MODE == "native" or np.native_ring(gens):
        raise Unavailable("Native Groebner computation unavailable for the specified ring or order. "
                          "Build the current kernel with make groebner, or select ATLAS_GROEBNER=sympy.")
    reference = sp.groebner(ideal, *gens, order=_reference_order(order), domain=domain)
    return reference if native is None else _Checked(native, reference)


def colon(ideal, by, gens, *, domain=QQ):
    """Generators of the ideal quotient (ideal) : (by), or None when only SymPy can do it."""
    gens = tuple(gens)
    library = _kernel()
    if (getattr(domain, "is_AlgebraicField", False) and _MODE != "sympy" and library is not None
            and hasattr(library, "atlas_coefficient_block_abi")):
        from .field_polynomial import native_colon
        return native_colon(ideal, by, gens, domain)
    if domain != QQ or _MODE == "sympy" or library is None or not gens:
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
