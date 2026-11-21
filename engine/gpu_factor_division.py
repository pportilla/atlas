"""Transport native modular-root and binomial-division batches to Triton."""
from contextlib import contextmanager
import ctypes
from functools import lru_cache
from time import perf_counter_ns

from .native_library import _library, current_replay_scope

_Ints = ctypes.POINTER(ctypes.c_int32)
_RootScreen = ctypes.CFUNCTYPE(ctypes.c_int, _Ints, ctypes.c_int, _Ints, ctypes.c_int)
_BinomialSteps = ctypes.CFUNCTYPE(ctypes.c_int, _Ints, ctypes.c_int, ctypes.c_int, _Ints)


def _tensor(ptr, size, device):
    import torch
    # The synchronous host copy below keeps the native callback buffers alive
    # until all asynchronous transfers on this stream have completed.
    host = (ctypes.c_int32 * size).from_address(ctypes.addressof(ptr.contents))
    from .triton_buffers import to_device
    return to_device(torch.frombuffer(host, dtype=torch.int32), device)


def _run(name, parameters, buffers, count, output, extra):
    from .algebra_execution import record_stage, record_fallback
    from .gpu_devices import triton_device, torch_device_context
    from .triton_ir import launch_kernel
    scope = current_replay_scope()
    if scope is None or scope.compute != "gpu" or scope.backend not in ("auto", "triton"):
        return 0
    try:
        device = triton_device(scope.index)
        if device is None:
            record_fallback(f"No Triton device at index {scope.index} for {name}.")
            return 0
        import torch
        started = perf_counter_ns()
        with torch_device_context(device["torch_device"]):
            arrays = [_tensor(ptr, length, device["torch_device"]) for ptr, length in buffers]
            result = torch.empty(count, dtype=torch.int32, device=device["torch_device"])
            block = 128
            launch_kernel(name, (*parameters, block), ((count + block - 1)//block,), *arrays, result, *extra)
            from .triton_buffers import to_host
            host = to_host(result).contiguous()
            ctypes.memmove(output, host.data_ptr(), ctypes.sizeof(ctypes.c_int32) * count)
        record_stage(name, "triton", device=scope.record_device("triton", device), lanes=count,
                     elapsed_ns=perf_counter_ns() - started)
        return 1
    except (KeyboardInterrupt, SystemExit) as error:
        scope.interrupted = error
        return 0
    except Exception as error:
        record_fallback(f"Triton {name}: {error}")
        return 0


@_RootScreen
def _roots(coeffs, length, output, count):
    return _run("poly_root_screen", (), ((coeffs, length),), count, output, (length,))


@_BinomialSteps
def _steps(packed, count, width, output):
    return _run("poly_binomial_steps", (width,), ((packed, (2 + count) * width),),
                count, output, (count,))


@lru_cache(maxsize=1)
def _swap():
    function = _library().atlas_swap_factor_division_hooks
    function.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    function.restype = None
    return function


@contextmanager
def factor_division_hooks():
    scope = current_replay_scope()
    if scope is None or scope.compute != "gpu" or scope.backend not in ("auto", "triton"):
        yield
        return
    fn = _swap()
    roots = ctypes.cast(_roots, ctypes.c_void_p)
    steps = ctypes.cast(_steps, ctypes.c_void_p)
    fn(ctypes.byref(roots), ctypes.byref(steps))
    try:
        yield
    finally:
        fn(ctypes.byref(roots), ctypes.byref(steps))
