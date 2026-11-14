"""Owned native sparse inputs for repeated local-dimension and smoothness solves."""

import ctypes
from collections import OrderedDict
from functools import lru_cache
from threading import Lock
from weakref import finalize

from . import native_polynomial as np
from .native_library import Unavailable, _library
from .polynomial import InputError


@lru_cache(maxsize=1)
def _functions():
    lib = _library()
    try:
        prepare = lib.atlas_local_input_prepare
        solve = lib.atlas_local_cpu_invariants_scoped
        free = lib.atlas_local_input_free
        needs_scope = lib.atlas_local_input_needs_scope
    except AttributeError as exc:
        raise Unavailable("Rebuild the native kernel for prepared local dimension inputs.") from exc
    prepare.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    prepare.restype = ctypes.c_void_p
    solve.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                      ctypes.POINTER(ctypes.c_uint64)]
    solve.restype = ctypes.c_int
    free.argtypes = [ctypes.c_void_p]
    free.restype = None
    needs_scope.argtypes = [ctypes.c_void_p]
    needs_scope.restype = ctypes.c_int
    return prepare, solve, free, lib.atlas_groebner_error, needs_scope


class _PreparedInput:
    __slots__ = ("pointer", "needs_scope", "__weakref__")

    def __init__(self, data):
        prepare, _, free, error, needs_scope = _functions()
        self.pointer = prepare(data, len(data))
        if not self.pointer:
            raise ArithmeticError(error().decode("utf-8"))
        try:
            self.needs_scope = bool(needs_scope(self.pointer))
            finalize(self, free, self.pointer)
        except BaseException:
            free(self.pointer)
            raise


# These constrain only retained input. Large requests are solved with a
# temporary handle. The byte threshold bounds both the point and sparse rows.
_MAX_ENTRIES = 64
_MAX_ENTRY_BYTES = 32 * 1024
_cache = OrderedDict()
_recent = None
_lock = Lock()
_Outputs = ctypes.c_int * 4
cpu_activity_buffer = ctypes.c_uint64 * 8


def _input(variables, generators, point):
    global _recent
    key = (tuple(variables), tuple(generators), tuple(point))
    # Most interactive solves reuse the immediately preceding ideal. The
    # immutable pair is also in the bounded cache; a local reference keeps
    # its handle alive across concurrent lookups, evictions and native calls.
    recent = _recent
    if recent is not None and recent[0] == key:
        return recent[1]
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            _recent = (key, cached)
            return cached

    variables, generators, point = key
    lines = [" ".join(v.name for v in variables), " ".join(point)]
    # This cache owns its lifetime and budget. Avoid retaining huge operands
    # separately in the general polynomial row encoder's LRU.
    lines.extend(np._encoded_row.__wrapped__(variables, f) for f in generators)
    data = ("\n".join(lines) + "\n").encode("ascii")
    prepared = _PreparedInput(data)
    if len(data) <= _MAX_ENTRY_BYTES:
        with _lock:
            _cache[key] = prepared
            _cache.move_to_end(key)
            _recent = (key, prepared)
            if len(_cache) > _MAX_ENTRIES:
                _cache.popitem(last=False)
    return prepared


@lru_cache(maxsize=1)
def _ambient():
    fn = _library().atlas_local_ambient_invariants
    fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    fn.restype = None
    return fn


def cpu_invariants(variables, generators, point, lex, smoothness, activity=None):
    """Compute fresh dimension, embedding dimension, Jacobian rank and smoothness.

    Point coordinates are canonical rational strings in the given variable
    order. A dimension-only run can leave the tangent metadata at -1.
    """
    if not generators and activity is None:
        result = _Outputs()
        _ambient()(len(variables), result)
        return result[0], result[1], result[2], bool(result[3])
    prepared = _input(variables, generators, point)
    return _solve(prepared, lex, smoothness, activity)


def measured_cpu_invariants(variables, generators, point, lex, smoothness):
    if not generators:
        return cpu_invariants(variables, generators, point, lex, smoothness), None
    prepared = _input(variables, generators, point)
    counts = cpu_activity_buffer() if prepared.needs_scope else None
    return _solve(prepared, lex, smoothness, counts), counts


def _solve(prepared, lex, smoothness, activity):
    _, solve, _, error, _ = _functions()
    result = _Outputs()
    if activity is not None and not isinstance(activity, cpu_activity_buffer):
        raise TypeError("Pass an eight-counter buffer from cpu_activity_buffer().")
    status = solve(prepared.pointer, lex, smoothness, result, activity)
    if status == 1:
        raise InputError("The point must lie on the scheme.")
    if status != 0:
        raise ArithmeticError(error().decode("utf-8"))
    return result[0], result[1], result[2], bool(result[3])
