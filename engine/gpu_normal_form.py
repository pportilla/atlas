"""Device normal forms for a reduced monomial basis. QQ coefficients stay on the host."""
import ctypes
from array import array
from functools import lru_cache
from itertools import chain
from time import perf_counter_ns

from .native_library import _LIBRARY, current_replay_scope


@lru_cache(maxsize=1)
def _opencl_kernel():
    path = _LIBRARY.with_name(_LIBRARY.name.replace("libgroebner", "libgroebner_opencl"))
    function = ctypes.CDLL(str(path)).atlas_monomial_divisors_opencl_index
    ptr = ctypes.POINTER(ctypes.c_int32)
    function.argtypes = [ptr, ptr, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ptr, ctypes.c_size_t]
    function.restype = ctypes.c_int
    return function


def _opencl_witnesses(terms, basis, index):
    def buffer(rows):
        values = array("i", chain.from_iterable(rows))
        return (ctypes.c_int32 * len(values)).from_buffer(values)

    out = (ctypes.c_int32 * len(terms))()
    if _opencl_kernel()(buffer(terms), buffer(basis), len(terms), len(basis), len(basis[0]), out, index):
        raise RuntimeError("OpenCL monomial divisor kernel or device unavailable")
    return out


def _exponents(term, variables):
    coefficient, *powers = term.split()
    if not coefficient or len(powers) != variables:
        raise ValueError("Expected a sparse monomial")
    vector = tuple(map(int, powers))
    if any(p < 0 for p in vector) or sum(vector) > 32767:
        raise ValueError("Outside the packed rational monomial range")
    return vector


def _certify(terms, basis, witnesses):
    """Check every proposed deletion, and every retained term's irreducibility."""
    if len(terms) != len(witnesses):
        raise ArithmeticError("Incomplete monomial divisibility witnesses")
    # Each 16-bit component is below 2^15. With its guard bit set a
    # subtraction cannot borrow from the next component. A guard survives
    # exactly when the term's corresponding exponent is >= the divisor's.
    def packed(powers):
        value = 0
        for exponent in powers:
            value = (value << 16) | exponent
        return value

    guard = packed([1 << 15] * len(basis[0]))
    heads = [packed(b) for b in basis]
    for powers, j in zip(terms, witnesses):
        word = packed(powers) | guard
        if j == -1:
            valid = all((word - head) & guard != guard for head in heads)
        else:
            valid = 0 <= j < len(heads) and (word - heads[j]) & guard == guard
        if not valid:
            raise ArithmeticError("Host rejected a monomial divisibility witness")


def normal_form(gens, basis_rows, row):
    """Return a verified sparse remainder, or None to use ordinary exact division."""
    scope = current_replay_scope()
    if scope is None or scope.compute != "gpu" or not row or not basis_rows:
        return None
    if "*" in row or any(not head or ";" in head or "*" in head for head in basis_rows):
        return None
    variables = len(gens)
    size = 1 + row.count(";")
    # Tiny reductions are cheaper on the CPU. Bound the worst-case search and
    # the size of the term/witness buffers, even for very sparse basis inputs.
    if (not 1 <= variables <= 32 or not 64 <= size <= 65536 or
            variables * size * len(basis_rows) > 1 << 24):
        return None
    started = perf_counter_ns()
    chunks = row.split(";")
    try:
        terms = [_exponents(chunk, variables) for chunk in chunks]
        basis = [_exponents(head, variables) for head in basis_rows]
    except ValueError:
        return None

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
        kept = [chunk for chunk, witness in zip(chunks, witnesses) if witness == -1]
        record_stage("monomial_normal_form", backend, device=scope.record_device(backend, device),
                     terms=size, divisors=len(basis), removed=size - len(kept),
                     host_verified=True, verify_ns=verify_ns, elapsed_ns=perf_counter_ns() - started)
        return ";".join(kept)
    return None
