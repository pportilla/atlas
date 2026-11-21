"""Finite-field S-pair rows against the shared, triangular reducer table for one degree."""
from .triton_buffers import to_device, to_host
import os

from .triton_ir import launch_kernel


def _plan(width, offsets, targets, coefficients, tails):
    import ctypes
    import numpy as np
    from .native_library import _kernel

    function = getattr(_kernel(), "atlas_groebner_trace_levels", None)
    if function is None:
        raise RuntimeError("Rebuild the native kernel for GPU trace reduction.")
    ptr = ctypes.POINTER(ctypes.c_uint32)
    function.argtypes = [ctypes.c_uint32] + [ptr] * 8
    function.restype = ctypes.c_uint32
    arrays = [np.empty(n, dtype=np.int32) for n in
              (width + 1, max(1, tails), max(1, tails), width, width + 1)]
    depth = function(width, offsets, targets, coefficients,
                     *(a.ctypes.data_as(ptr) for a in arrays))
    if not depth:
        raise ValueError("Invalid triangular trace graph")
    arrays[-1] = arrays[-1][:depth + 1]
    return arrays, depth


def reduce_rows(args, device):
    import numpy as np
    import torch
    from .gpu_devices import torch_device_context

    p, pairs, width, initial, offsets, pivots, targets, coefficients, output = args
    words = pairs * width
    tails = offsets[width]
    def upload(pointer, count):
        if not count:
            return torch.empty(1, dtype=torch.int32, device=device)
        host = np.ctypeslib.as_array(pointer, shape=(count,)).view(np.int32)
        return to_device(torch.from_numpy(host), device)

    with torch_device_context(device):
        arrays, depth = _plan(width, offsets, targets, coefficients, tails)
        result = torch.empty(words, dtype=torch.int32, device=device)
        table = [to_device(torch.from_numpy(a), device) for a in arrays]
        work = torch.empty_like(result)
        buffers = (upload(initial, words), upload(pivots, width), *table, work, result, width, depth)
        if torch.device(device).type == "cpu" and os.getenv("TRITON_INTERPRET") == "1":
            from tests.triton_trace_oracle import _level_rows
            _level_rows[(pairs,)](*buffers, p, 16, 4, num_warps=2)
        else:
            launch_kernel("trace_levels", (p,), (pairs,), *buffers, warps=2)
        np.ctypeslib.as_array(output, shape=(words,))[:] = to_host(result).numpy()
