"""Owned, immutable native ideals for arithmetic on live session values."""
import ctypes
from dataclasses import dataclass
from functools import cache, cached_property, partial
from importlib.machinery import EXTENSION_SUFFIXES, ExtensionFileLoader, FileFinder
from importlib.util import module_from_spec
from threading import Lock

from . import native_polynomial as polynomial
from .native_library import _LIBRARY, _kernel


_compile_assignments = None


def _load_binding():
    global _compile_assignments
    finder = FileFinder(str(_LIBRARY.parent), (ExtensionFileLoader, EXTENSION_SUFFIXES))
    spec = finder.find_spec("_atlas_ideal")
    if spec is not None:
        try:
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            _compile_assignments = getattr(module, "compile_assignments", None)
            return module.Ideal
        except (ImportError, OSError):
            pass
    return None


_unloaded = object()
_binding = _unloaded
_binding_lock = Lock()


def _compiled():
    """Acquire one handle type even if two session threads first declare ideals together."""
    global _binding
    if _binding is _unloaded:
        with _binding_lock:
            if _binding is _unloaded:
                _binding = _load_binding()
    return _binding


def assignment_runner():
    return _compile_assignments if _compiled() is not None else None


@cache
def _api():
    library = _kernel()
    if library is None or not hasattr(library, "atlas_ideal_expressions"):
        return None
    ptr = ctypes.c_void_p
    size = ctypes.c_size_t
    signatures = (
        ("atlas_ideal_prepare", ptr, (size, ctypes.c_char_p, size)),
        ("atlas_ideal_binary", ptr, (ptr, ptr, ctypes.c_int)),
        ("atlas_ideal_power_small", ptr, (ptr, ctypes.c_ulong, ctypes.c_int)),
        ("atlas_ideal_size", size, (ptr,)),
        ("atlas_ideal_rows", ptr, (ptr,)),
        ("atlas_ideal_expressions", ptr, (ptr, ctypes.c_char_p)),
        ("atlas_ideal_error", ctypes.c_char_p, ()),
        ("atlas_ideal_release", None, (ptr,)),
    )
    if hasattr(library, "atlas_ideal_from_polynomials"):
        signatures += (
            ("atlas_ideal_ring", ptr, (size,)),
            ("atlas_ideal_ring_release", None, (ptr,)),
            ("atlas_ideal_polynomial", ptr, (ptr, ctypes.c_char_p)),
            ("atlas_ideal_polynomial_release", None, (ptr,)),
            ("atlas_ideal_from_polynomials", ptr, (ptr, ctypes.POINTER(ptr), size)),
        )
    for name, result, args in signatures:
        function = getattr(library, name)
        function.restype, function.argtypes = result, args
    return library


class _CtypesOwner:
    __slots__ = ("ptr", "_release")

    def __init__(self, ptr, release):
        self.ptr = ptr
        self._release = release

    def __del__(self):
        self._release(self.ptr)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class _CtypesRing(_CtypesOwner):
    __slots__ = ("library",)

    def __init__(self, library, width):
        ptr = library.atlas_ideal_ring(width)
        if not ptr:
            raise ArithmeticError(library.atlas_ideal_error().decode())
        super().__init__(ptr, library.atlas_ideal_ring_release)
        self.library = library

    def polynomial(self, row):
        ptr = self.library.atlas_ideal_polynomial(self.ptr, row)
        if not ptr:
            raise ArithmeticError(self.library.atlas_ideal_error().decode())
        return _CtypesOwner(ptr, self.library.atlas_ideal_polynomial_release)

    def ideal(self, generators):
        count = len(generators)
        if count > 128:
            return None
        pointers = (ctypes.c_void_p * count)(*(g.ptr for g in generators))
        ptr = self.library.atlas_ideal_from_polynomials(self.ptr, pointers, count)
        return NativeIdeal(self.library, ptr) if ptr else NativeIdeal._declined(self.library)


def ideal_constructor(fallback, polynomial_type, ideal_type):
    binding = _compiled()
    if binding is None or not hasattr(binding, "constructor"):
        return fallback

    @dataclass(frozen=True, eq=False, repr=False, init=False)
    class DeclaredIdeal(ideal_type, binding):
        """A declared session ideal owns its immutable native generators directly."""
        __slots__ = ()
        _native_ideal = binding._native_ideal

        @cached_property
        def _values(self):
            return tuple(f._value for f in self.generators)

    return binding.constructor(DeclaredIdeal, polynomial_type, fallback)


class _CompiledRing:
    __slots__ = ("polynomial", "ideal")

    def __init__(self, binding, width):
        ring = binding.ring(width)
        self.polynomial = partial(binding.polynomial, ring)
        self.ideal = partial(binding.from_polynomials, ring)


def native_ring(width):
    binding = _compiled()
    if binding is not None and hasattr(binding, "constructor"):
        return _CompiledRing(binding, width)
    library = _api()
    if library is not None and hasattr(library, "atlas_ideal_from_polynomials"):
        return _CtypesRing(library, width)
    return None


class NativeIdeal:
    __slots__ = ("library", "ptr", "_values")

    def __init__(self, library, ptr):
        self.library = library
        self.ptr = ptr

    def __del__(self):
        self.library.atlas_ideal_release(self.ptr)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @staticmethod
    def _declined(library):
        message = library.atlas_ideal_error()
        if message:
            raise ArithmeticError(message.decode())
        return None  # The caller can retry in the isolated worker.

    def sum(self, right):
        ptr = self.library.atlas_ideal_binary(self.ptr, right.ptr, False)
        return NativeIdeal(self.library, ptr) if ptr else self._declined(self.library)

    def product(self, right):
        ptr = self.library.atlas_ideal_binary(self.ptr, right.ptr, True)
        return NativeIdeal(self.library, ptr) if ptr else self._declined(self.library)

    def power(self, exponent, lex):
        ptr = self.library.atlas_ideal_power_small(self.ptr, exponent, lex)
        return NativeIdeal(self.library, ptr) if ptr else self._declined(self.library)

    def size(self):
        return self.library.atlas_ideal_size(self.ptr)

    def rows(self):
        return self._read_lines(self.library.atlas_ideal_rows(self.ptr))

    def expressions(self, names):
        return self._read_lines(self.library.atlas_ideal_expressions(self.ptr, " ".join(names).encode("ascii")))

    def _read_lines(self, ptr):
        if not ptr:
            raise ArithmeticError(self.library.atlas_ideal_error().decode())
        try:
            return ctypes.string_at(ptr).decode("ascii").splitlines()
        finally:
            self.library.atlas_groebner_release(ptr)


def values(handle, variables):
    """A handle and its retained values belong to one live coordinate space."""
    try:
        return handle._values
    except AttributeError:
        handle._values = tuple(polynomial.from_row(variables, row) for row in handle.rows())
        return handle._values
