"""Reduced Groebner bases over Q from the native kernel.

The kernel (native/groebner.cpp) is exact, so its reduced basis is the one
SymPy returns, element for element. SymPy remains the engine for number
fields, for rings beyond the kernel's packing, and when the library is absent.
Set ATLAS_GROEBNER=sympy to use SymPy only, or =check to compute with both
and compare every basis and every remainder. When the exact strategies
swell, the kernel switches to modular images and certifies the lifted basis;
with ATLAS_GROEBNER_GPU=1 the replays for further primes run on an OpenCL
device (make gpu).
"""
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import sys

import sympy as sp
from sympy import QQ
from sympy.polys.polyerrors import BasePolynomialError

_LIBRARY = Path(__file__).resolve().parents[1] / "native" / "build" / (
    "libgroebner.dylib" if sys.platform == "darwin" else "libgroebner.so")
# "first": the exponent of the first variable is compared before the grevlex
# order of the others, so the first variable is eliminated.
_ORDERS = {"grevlex": 0, "lex": 1, "first": 2}


def _reference_order(order):
    if order != "first":
        return order
    from sympy.polys.orderings import ProductOrder, grevlex, lex
    return ProductOrder((lex, lambda monomial: monomial[:1]), (grevlex, lambda monomial: monomial[1:]))
_MAX_VARIABLES = 32
_MODE = os.getenv("ATLAS_GROEBNER", "native")
if _MODE not in ("native", "sympy", "check"):
    raise ValueError("ATLAS_GROEBNER must be native, sympy, or check.")
CHECK = _MODE == "check"


@lru_cache(maxsize=1)
def _kernel():
    try:
        library = ctypes.CDLL(str(_LIBRARY))
    except OSError:
        return None
    library.atlas_groebner.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    library.atlas_groebner.restype = ctypes.c_void_p
    library.atlas_groebner_text.argtypes = [ctypes.c_void_p]
    library.atlas_groebner_text.restype = ctypes.c_char_p
    library.atlas_groebner_seconds.argtypes = [ctypes.c_void_p]
    library.atlas_groebner_seconds.restype = ctypes.c_double
    library.atlas_groebner_reduce.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    library.atlas_groebner_reduce.restype = ctypes.c_void_p
    library.atlas_groebner_colon.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    library.atlas_groebner_colon.restype = ctypes.c_void_p
    try:
        library.atlas_ideal.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        library.atlas_ideal.restype = ctypes.c_void_p
    except AttributeError:  # built before native/ideal.cpp existed: run make
        return None
    library.atlas_groebner_release.argtypes = [ctypes.c_void_p]
    library.atlas_groebner_release.restype = None
    library.atlas_groebner_free.argtypes = [ctypes.c_void_p]
    library.atlas_groebner_free.restype = None
    wanted = os.getenv("ATLAS_GROEBNER_GPU")
    if wanted == "1":
        _use_gpu(library)
    elif wanted in ("triton", "reference"):
        # The replay in small parts (engine/gpu_replay.py): Triton kernels on
        # a CUDA device, or the same parts with NumPy.
        from . import gpu_replay
        gpu_replay.use(library, wanted)
    return library


def _use_gpu(library):
    """Hand the replays of the modular route to the OpenCL plugin (make gpu), if there is one."""
    plugin = _LIBRARY.with_name(_LIBRARY.name.replace("libgroebner", "libgroebner_opencl"))
    try:
        device = ctypes.CDLL(str(plugin))
        if not device.atlas_replay_opencl_available():
            return False
        library.atlas_groebner_use_batch.argtypes = [ctypes.c_void_p]
        library.atlas_groebner_use_batch.restype = None
        library.atlas_groebner_use_batch(ctypes.cast(device.atlas_replay_opencl, ctypes.c_void_p))
    except (OSError, AttributeError):
        return False
    library._replay_device = device  # keep the plugin loaded
    return True


# Generators recur from one basis to the next, and converting a SymPy
# expression costs far more than the kernel spends on it.
_lines = {}
_polys = {}
_CACHE_LIMIT = 50_000


def _line(gens, expr):
    """One polynomial in the kernel's text form: factors joined by '*', each a
    list of terms 'coefficient e1 ... en' joined by ';'."""
    key = (gens, expr)
    line = _lines.get(key)
    if line is None:
        if expr.is_Mul and any(factor.is_Add for factor in expr.args):
            # The kernel multiplies; expanding a product here is the slow part.
            line = "*".join(_line(gens, factor) for factor in expr.args)
        else:
            terms = sp.Poly(expr, *gens, domain=QQ).as_dict(native=True)
            line = ";".join(f"{c} {' '.join(map(str, m))}" for m, c in terms.items())
        if len(_lines) >= _CACHE_LIMIT:
            _lines.clear()
        _lines[key] = line
    return line


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
    def __init__(self, library, handle, gens):
        self._library, self._handle, self.gens = library, handle, gens
        lines = library.atlas_groebner_text(handle).decode("ascii").split("\n")[:-1]
        pairs = [_poly(gens, line) for line in lines]
        self.polys = [poly for poly, _ in pairs]
        self.exprs = [expr for _, expr in pairs]

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
            raise ArithmeticError("The native normal form could not be computed.")
        try:
            return ctypes.string_at(answer).decode("ascii")[:-1]
        finally:
            self._library.atlas_groebner_release(answer)

    def contains(self, expr):
        return not self._remainder(expr)

    def reduce(self, expr):
        """Return (None, remainder): the quotients are not computed."""
        return None, _poly(self.gens, self._remainder(expr))[1]


class _Checked:
    """Both engines side by side; any disagreement is an error."""
    def __init__(self, native, reference):
        self._native, self._reference = native, reference
        if native.exprs != reference.exprs:
            raise AssertionError("The native Groebner basis differs from SymPy's.")
        self.polys, self.exprs = native.polys, native.exprs

    def contains(self, expr):
        answer = self._native.contains(expr)
        if answer != self._reference.contains(expr):
            raise AssertionError("The native ideal membership differs from SymPy's.")
        return answer

    def reduce(self, expr):
        answer = self._native.reduce(expr)
        if answer[1] != self._reference.reduce(expr)[1]:
            raise AssertionError("The native normal form differs from SymPy's.")
        return answer


def _native(ideal, gens, order, known):
    library = _kernel()
    if library is None or order not in _ORDERS or not 1 <= len(gens) <= _MAX_VARIABLES:
        return None
    try:
        lines = [_line(gens, f) for f in ideal]
    except BasePolynomialError:
        return None
    payload = (f"{len(gens)} {_ORDERS[order]} {known}\n" + "".join(line + "\n" for line in lines)).encode("ascii")
    handle = library.atlas_groebner(payload, len(payload))
    # A null handle means an exponent beyond the packed range.
    return NativeBasis(library, handle, gens) if handle else None


def groebner(ideal, gens, *, order="grevlex", domain=QQ, known=0):
    """The reduced Groebner basis of the ideal, largest leading monomial first.

    `known` promises that the first that many generators already form a
    Groebner basis for this order; the kernel then skips their S-polynomials.
    """
    gens = tuple(gens)
    ideal = tuple(ideal)
    if not 0 <= known <= len(ideal):
        raise ValueError("known must count leading generators of the ideal.")
    native = _native(ideal, gens, order, known) if domain == QQ and _MODE != "sympy" else None
    if native is not None and _MODE != "check":
        return native
    reference = sp.groebner(ideal, *gens, order=_reference_order(order), domain=domain)
    return reference if native is None else _Checked(native, reference)


def colon(ideal, by, gens, *, domain=QQ):
    """Generators of the ideal quotient (ideal) : (by), or None when only SymPy can do it."""
    gens = tuple(gens)
    library = _kernel()
    if domain != QQ or _MODE == "sympy" or library is None or not 1 <= len(gens) < _MAX_VARIABLES:
        return None
    try:
        lines = [_line(gens, f) for f in (*ideal, *by)]
    except BasePolynomialError:
        return None
    payload = (f"{len(gens)} {len(ideal)}\n" + "".join(line + "\n" for line in lines)).encode("ascii")
    answer = library.atlas_groebner_colon(payload, len(payload))
    if not answer:
        return None
    try:
        text = ctypes.string_at(answer).decode("ascii")
    finally:
        library.atlas_groebner_release(answer)
    return tuple(_poly(gens, line)[1] for line in text.split("\n")[:-1])
