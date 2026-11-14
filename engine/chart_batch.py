"""Exact exponent substitutions for all cones of a toric atlas in dimensions 2–8."""
from array import array
import ctypes
from functools import lru_cache
from itertools import chain, islice
import os
from struct import iter_unpack
from time import perf_counter

from .gpu_devices import gpu_preference, triton_device
from .native_backend import _LIBRARY, _load, _gpu_config, _gpu_library, gpu_devices

_I32_MAX = 2**31 - 1
_WORD = ctypes.POINTER(ctypes.c_int32)


@lru_cache(maxsize=1)
def _cpu_functions():
    try:
        lib = _load(_LIBRARY)
        run = lib.resolution_chart_cpu_nd
        valid = lib.resolution_chart_valid
    except (OSError, AttributeError) as exc:
        raise RuntimeError("The toric chart batch engine is not built. Run make.") from exc
    run.argtypes = [_WORD]*4 + [ctypes.c_int]*3 + [_WORD]
    run.restype = None
    valid.argtypes = [_WORD, ctypes.c_int]
    valid.restype = ctypes.c_int
    return run, valid


@lru_cache(maxsize=1)
def _gpu_function():
    run = getattr(_gpu_library(), "resolution_gpu_charts_nd", None)
    if run:
        run.argtypes = [_WORD]*4 + [ctypes.c_int]*5 + [_WORD]
        run.restype = ctypes.c_int
    return run


def _pack(values):
    packed = array("i", values)
    if packed.itemsize != 4:
        raise RuntimeError("This platform's int representation is not 32-bit.")
    return (ctypes.c_int32 * len(packed)).from_buffer(packed)


def _validate_vectors(values, bound, name, dimension):
    for v in values:
        if len(v) != dimension or any(type(x) != int or not 0 <= x <= bound for x in v):
            raise ValueError(f"{name} must contain nonnegative integer {dimension}-tuples with entries <= {bound}.")


def _accelerate(inputs, out, nt, nr, nc, dimension, index, stats):
    preference = gpu_preference()
    if preference != "opencl":
        device = triton_device(index)
        if device is not None:
            try:
                from .gpu_charts import transform
                transform(inputs, out, nt, dimension, device["torch_device"])
            except Exception as error:
                # Optional runtimes can fail during compilation, launch or
                # transfer. The next backend overwrites the entire buffer.
                stats["triton_error"] = f"{type(error).__name__}: {error}"
            else:
                stats.update(backend="triton", device=device)
                return True
        if preference == "triton":
            stats["backend"] = "cpu_after_triton"
            stats.setdefault("triton_error", "Triton device unavailable at the selected index.")
            return False
    gpu = _gpu_function()
    if gpu and gpu(*inputs, nt, nr, nc, dimension, index, out) == 0:
        device = next((d for d in gpu_devices() if d["index"] == index), {"index": index})
        stats.update(backend="opencl", device=device)
        return True
    stats["backend"] = "cpu_after_opencl"
    stats["opencl_error"] = "OpenCL chart kernel or device unavailable at the selected index."
    return False


def transform_batch(exps, rays, cones, orders, compute=None):
    """Return strict exponent vectors by chart and term, plus execution timings.

    The caller supplies the vanishing order on each ray. Negative transformed
    powers are rejected. Coefficients stay in the caller's rational polynomial.
    """
    started = perf_counter()
    nt, nr, nc = len(exps), len(rays), len(cones)
    pairs = nt*nc
    if min(nt, nr, nc) < 1 or pairs > 2_000_000 or nr > 100_000:
        raise ValueError("Invalid chart batch size (up to 2,000,000 chart-term pairs).")
    dimension = len(exps[0])
    if not 2 <= dimension <= 8:
        raise ValueError("Chart dimension must be 2–8.")
    entries = dimension * pairs
    if entries > 6_000_000:
        raise ValueError("Chart batch exceeds the limit of 6,000,000 output exponents.")
    _validate_vectors(exps, _I32_MAX, "Exponents", dimension)
    _validate_vectors(rays, _I32_MAX, "Rays", dimension)
    _validate_vectors(cones, nr-1, "Cones", dimension)
    if len(orders) != nr or any(type(x) != int or not 0 <= x <= _I32_MAX for x in orders):
        raise ValueError("Expected one nonnegative 32-bit order per ray.")
    # All input coordinates are nonnegative. Each partial product and sum
    # is at most max |e|_1 * max r_i. Subtracting order stays in signed range.
    if max(map(sum, exps))*max(map(max, rays)) > _I32_MAX:
        raise OverflowError("The chart exponent bound exceeds the signed 32-bit kernel range.")

    inputs = [_pack(chain.from_iterable(group)) for group in (exps, rays, cones)] + [_pack(orders)]
    out = (ctypes.c_int32 * entries)()
    cpu, valid = _cpu_functions()
    _, index, _ = _gpu_config(compute)
    policy = ("auto" if os.getenv("RESOLUTION_GPU", "off") == "auto" else "cpu") if compute is None else compute
    threshold = int(os.getenv("RESOLUTION_GPU_MIN_PAIRS", 50000)) if policy == "auto" else 1
    if threshold < 1:
        raise ValueError("RESOLUTION_GPU_MIN_PAIRS must be positive.")
    attempt = policy != "cpu" and pairs >= threshold
    stats = {"chart_term_pairs": pairs, "dimension": dimension, "backend": "cpu"}
    native_start = perf_counter()
    if not attempt or not _accelerate(inputs, out, nt, nr, nc, dimension, index, stats):
        cpu(*inputs, nt, nc, dimension, out)
    if not valid(out, entries):
        raise ValueError("A ray order exceeds the valuation of an input monomial.")
    stats["native_elapsed_ms"] = (perf_counter()-native_start)*1000
    vectors = iter_unpack(f"={dimension}i", out)
    result = [list(islice(vectors, nt)) for _ in range(nc)]
    stats["elapsed_ms"] = (perf_counter()-started)*1000
    return result, stats
