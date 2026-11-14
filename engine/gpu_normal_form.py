"""Device normal forms for a reduced monomial basis. QQ coefficients stay on the host."""
import ctypes
from array import array
from functools import lru_cache
from itertools import chain
from time import perf_counter_ns

from .native_library import _LIBRARY, _library, current_replay_scope


@lru_cache(maxsize=1)
def _opencl_kernel():
    path = _LIBRARY.with_name(_LIBRARY.name.replace("libgroebner", "libgroebner_opencl"))
    function = ctypes.CDLL(str(path)).atlas_monomial_divisors_opencl_index
    ptr = ctypes.POINTER(ctypes.c_int32)
    function.argtypes = [ptr, ptr, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ptr, ctypes.c_size_t]
    function.restype = ctypes.c_int
    return function


def _buffer(rows):
    values = array("i", chain.from_iterable(rows))
    return (ctypes.c_int32 * len(values)).from_buffer(values)


def _opencl_witnesses(terms, basis, index):
    out = (ctypes.c_int32 * len(terms))()
    if _opencl_kernel()(_buffer(terms), _buffer(basis), len(terms), len(basis), len(basis[0]), out, index):
        raise RuntimeError("OpenCL monomial divisor kernel or device unavailable")
    return out


@lru_cache(maxsize=1)
def _verification_kernel():
    function = _library().atlas_monomial_witnesses_valid
    ptr = ctypes.POINTER(ctypes.c_int32)
    function.argtypes = [ptr, ctypes.c_size_t, ptr, ctypes.c_size_t, ctypes.c_size_t, ptr]
    function.restype = ctypes.c_int
    return function


def _certify(terms, basis, witnesses):
    """Check every proposed deletion and every retained term in the exact host kernel."""
    if len(terms) != len(witnesses):
        raise ArithmeticError("Incomplete monomial divisibility witnesses")
    width = len(basis[0]) if basis else 0
    if not width or any(len(row) != width for row in chain(terms, basis)):
        raise ArithmeticError("Invalid monomial divisibility witness dimensions")
    # Avoid narrowing a malformed witness into an apparently valid divisor index.
    if any(j < -1 or j >= len(basis) for j in witnesses):
        raise ArithmeticError("Host rejected a monomial divisibility witness")
    indices = (ctypes.c_int32 * len(witnesses))(*witnesses)
    if not _verification_kernel()(_buffer(terms), len(terms), _buffer(basis), len(basis), width, indices):
        raise ArithmeticError("Host rejected a monomial divisibility witness")


def _eligible(variables, terms, divisors):
    # Tiny reductions are cheaper on the CPU. Bound the worst-case search and
    # the size of the term/witness buffers, even for very sparse basis inputs.
    return (1 <= variables <= 32 and 64 <= terms <= 65536 and divisors > 0
            and variables * terms * divisors <= 1 << 24)


@lru_cache(maxsize=1)
def _device_bounds_kernel():
    function = _library().atlas_monomials_fit_device
    function.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_size_t]
    function.restype = ctypes.c_int
    return function


def monomial_witnesses(variables, terms, basis):
    """Return verified divisor indices (-1 for retained terms), or None for CPU fallback."""
    scope = current_replay_scope()
    if scope is None or scope.compute != "gpu" or not _eligible(variables, len(terms), len(basis)):
        return None
    started = perf_counter_ns()
    if any(len(row) != variables for row in chain(terms, basis)):
        return None
    try:
        if not all(_device_bounds_kernel()(_buffer(rows), len(rows), variables) for rows in (terms, basis)):
            return None
    except OverflowError:
        return None  # An exponent does not fit the device's int32 buffer.

    from .algebra_execution import record_fallback, record_stage
    choices = ("triton", "opencl") if scope.backend == "auto" else (scope.backend,)
    for backend in choices:
        try:
            if backend == "opencl":
                witnesses = _opencl_witnesses(terms, basis, scope.index)
                device = None
            elif backend == "triton":
                from .gpu_devices import triton_device
                device = triton_device(scope.index)
                if device is None:
                    raise RuntimeError("Triton device unavailable")
                from .triton_monomial import witnesses as execute
                witnesses = execute(terms, basis, device["torch_device"])
            else:
                return None
            checking = perf_counter_ns()
            _certify(terms, basis, witnesses)
            verify_ns = perf_counter_ns() - checking
        except Exception as error:
            record_fallback(f"{backend} monomial normal form: {error}")
            continue
        record_stage("monomial_normal_form", backend, device=scope.record_device(backend, device),
                     terms=len(terms), divisors=len(basis), removed=sum(w >= 0 for w in witnesses),
                     host_verified=True, verify_ns=verify_ns, elapsed_ns=perf_counter_ns() - started)
        return witnesses
    return None


def normal_form(gens, basis_rows, row):
    """Return a verified sparse remainder, or None to use ordinary exact division."""
    scope = current_replay_scope()
    if scope is None or scope.compute != "gpu" or not row or not basis_rows:
        return None
    if "*" in row or any(not head or ";" in head or "*" in head for head in basis_rows):
        return None
    variables = len(gens)
    if not _eligible(variables, 1 + row.count(";"), len(basis_rows)):
        return None
    chunks = row.split(";")
    try:
        terms = [tuple(map(int, chunk.split()[1:])) for chunk in chunks]
        basis = [tuple(map(int, head.split()[1:])) for head in basis_rows]
    except ValueError:
        return None
    witnesses = monomial_witnesses(variables, terms, basis)
    if witnesses is None:
        return None
    return ";".join(chunk for chunk, witness in zip(chunks, witnesses) if witness == -1)
