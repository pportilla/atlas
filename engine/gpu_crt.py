"""Transport paired prime images to the native Triton CRT kernel."""
import ctypes
from time import perf_counter_ns

from .gpu_devices import torch_device_context, triton_device
from .triton_ir import launch_kernel


_U32 = ctypes.POINTER(ctypes.c_uint32)
_CRT = ctypes.CFUNCTYPE(ctypes.c_int, _U32, _U32, ctypes.c_uint32,
                        ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64))


def merge_pairs(images, moduli, pairs, width, output, device):
    import torch
    import triton
    from .triton_buffers import to_device, to_host

    def upload(pointer, count):
        buffer = (ctypes.c_uint32 * count).from_address(ctypes.addressof(pointer.contents))
        return to_device(torch.frombuffer(buffer, dtype=torch.int32), device)

    with torch_device_context(device):
        residues = upload(images, 2 * pairs * width)
        parameters = upload(moduli, 3 * pairs)
        result = torch.empty((pairs * width,), dtype=torch.int64, device=device)
        grid = (triton.cdiv(width, 256), pairs)
        launch_kernel("crt_pair", (256,), grid, residues, parameters, result, width)
        host = to_host(result)
        ctypes.memmove(output, host.data_ptr(), host.numel() * host.element_size())


@_CRT
def dispatch_crt(images, moduli, pairs, width, output):
    from .algebra_execution import record_fallback, record_stage
    from .native_library import current_replay_scope

    scope = current_replay_scope()
    if scope is None or scope.compute != "gpu" or scope.interrupted is not None:
        return 1
    try:
        device = triton_device(scope.index)
        if device is None:
            return 1
        started = perf_counter_ns()
        merge_pairs(images, moduli, pairs, width, output, device["torch_device"])
        record_stage("crt_pair_lift", "triton", device=scope.record_device("triton", device),
                     pairs=pairs, coefficients=width, elapsed_ns=perf_counter_ns()-started)
        return 0
    except Exception as exc:
        record_fallback(f"Triton CRT lift: {exc}")
        return 1
    except BaseException as exc:
        scope.interrupted = exc
        return 1
