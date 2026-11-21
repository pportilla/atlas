"""Transport exact sparse polynomial requests and native coefficient buffers to Triton."""

import ctypes
from functools import lru_cache
from time import perf_counter_ns

from .algebra_execution import cpu_stage, record_fallback, record_stage
from .gpu_devices import torch_device_context, triton_device
from .native_library import _library, current_replay_scope


_Word = ctypes.c_int64
_Ptr = ctypes.POINTER(_Word)
_Callback = ctypes.CFUNCTYPE(ctypes.c_int, *([ctypes.c_int] * 6), _Ptr, _Ptr, _Ptr)


@lru_cache(maxsize=1)
def _entry():
    entry = _library().atlas_polynomial_arithmetic
    entry.argtypes = [ctypes.c_char_p, ctypes.c_size_t, _Callback]
    entry.restype = ctypes.c_void_p
    return entry


def _call(payload, callback):
    handle = _entry()(payload, len(payload), callback)
    if not handle:
        raise ArithmeticError(_library().atlas_groebner_error().decode())
    try:
        return ctypes.string_at(handle).decode("ascii")
    finally:
        _library().atlas_groebner_release(handle)


def _host_view(pointer, count):
    import torch

    return torch.frombuffer((_Word * count).from_address(ctypes.addressof(pointer.contents)), dtype=torch.int64)


def _launch(device, kind, n, m, aw, bw, ow, a, b, out):
    import torch
    from .triton_buffers import to_device, to_host
    from .triton_ir import launch_kernel

    if kind == 0:
        kernel, count = "poly_add_words", n * ow
    elif kind == 1:
        kernel, count = ("poly_outer_product" if ow == 1 else "poly_outer_words"), n * m * ow
    elif kind == 2:
        kernel, count = ("poly_point_product" if ow == 1 else "poly_point_words"), n * ow
    else:
        kernel, count = "poly_dense_convolution", n + m - 1
    block = 128 if count >= 128 else 32
    with torch_device_context(device):
        left = to_device(_host_view(a, n * aw), device)
        right = to_device(_host_view(b, m * bw), device)
        result = torch.empty(count, dtype=torch.int64, device=device)
        launch_kernel(kernel, (block,), ((count + block - 1) // block,),
                      left, right, result, n, m, aw, bw, ow, count)
        _host_view(out, count).copy_(to_host(result))
    return kernel


def compute_row(operation, source_width, target_width, rows, parameter=None, *, modulus=None):
    """Execute the native arithmetic operation and return its canonical sparse row."""
    opcode = operation
    if modulus is not None:
        opcode = "marked-" + operation
        rows = [modulus, *rows]
    head = f"{opcode} {source_width} {target_width}"
    if parameter is not None:
        head += " " + str(parameter)
    payload = ("\n".join([head, *rows]) + "\n").encode("ascii")
    scope = current_replay_scope()
    if scope is not None and scope.compute == "gpu" and scope.backend in ("auto", "triton"):
        device = triton_device(scope.index)
        if device is None:
            record_fallback("Polynomial arithmetic requires a usable Triton device.")
        else:
            stages = {}
            errors = []

            @_Callback
            def batch(*args):
                try:
                    kernel = _launch(device["torch_device"], *args)
                    stages[kernel] = stages.get(kernel, 0) + 1
                    return 1
                except Exception as exc:
                    errors.append(str(exc))
                    return 0

            started = perf_counter_ns()
            try:
                result = _call(payload, batch)
            except ArithmeticError as exc:
                if str(exc) != "Triton coefficient batch declined.":
                    raise
                record_fallback(errors[0] if errors else "Polynomial batch exceeds Triton working storage.")
            else:
                if stages:
                    field_work = {"coefficient_field_reduction": "cpu"} if modulus is not None else {}
                    record_stage("polynomial_" + operation, "triton", device=scope.record_device("triton", device),
                                 kernels=stages, elapsed_ns=perf_counter_ns() - started, **field_work)
                else:
                    record_stage("polynomial_" + operation, "cpu", algorithm="sparse identity",
                                 elapsed_ns=perf_counter_ns() - started)
                return result
    with cpu_stage("polynomial_" + operation, algorithm="flint"):
        return _call(payload, _Callback())
