"""Owned, immutable native ideals for arithmetic on live session values."""
import ctypes
from functools import cache
from importlib.machinery import EXTENSION_SUFFIXES, ExtensionFileLoader, FileFinder
from importlib.util import module_from_spec
from threading import Lock

from . import native_polynomial as polynomial
from .native_library import _LIBRARY, _kernel


def _load_binding():
    finder = FileFinder(str(_LIBRARY.parent), (ExtensionFileLoader, EXTENSION_SUFFIXES))
    spec = finder.find_spec("_atlas_ideal")
    if spec is not None:
        try:
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
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
    for name, result, args in signatures:
        function = getattr(library, name)
        function.restype, function.argtypes = result, args
    return library


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

    @classmethod
    def prepare(cls, variables, generators):
        # Larger input is parsed and computed in the cancellable worker.
        if len(variables) > 32 or len(generators) > 128:
            return None
        compiled = _compiled()
        library = None if compiled is not None else _api()
        if compiled is None and library is None:
            return None
        values = tuple(f._value for f in generators)
        if sum(len(value._terms) for value in values) > 512:
            return None
        # Before formatting an integer with thousands of decimal digits,
        # use its stored bit length to tell whether the transport fits.
        for value in values:
            for powers, coefficient in value._terms:
                if (coefficient.numerator.bit_length() > 16384 or coefficient.denominator.bit_length() > 16384
                        or any(power.bit_length() > 16384 for power in powers)):
                    return None
        rows = [polynomial.row(variables, value) for value in values]
        if sum(map(len, rows)) > 4096:
            return None
        payload = "".join(row + "\n" for row in rows).encode("ascii")
        if compiled is not None:
            handle = compiled.prepare(len(variables), payload, len(rows))
            if handle is None:
                return None
        else:
            ptr = library.atlas_ideal_prepare(len(variables), payload, len(rows))
            if not ptr:
                return cls._declined(library)
            handle = cls(library, ptr)
        handle._values = values
        return handle

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
