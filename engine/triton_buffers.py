"""Transfer contiguous integer buffers for Triton, including HIP pageable-host memory."""
import ctypes
import os
from ctypes.util import find_library
from functools import lru_cache


@lru_cache(maxsize=1)
def initialize_hip():
    # Initialize the driver before torch loads its allocator. Triton can also
    # locate a wheel's bundled HIP runtime without importing torch.
    path = os.environ.get("TRITON_LIBHIP_PATH") or find_library("amdhip64")
    if not path:
        try:
            from triton.backends.amd.driver import _get_path_to_hip_runtime_dylib
            path = _get_path_to_hip_runtime_dylib()
        except (ImportError, OSError, RuntimeError):
            return None
    if path:
        try:
            runtime = ctypes.CDLL(path)
            if runtime.hipInit(0) == 0:
                return runtime
        except (OSError, AttributeError):
            pass
    return None


@lru_cache(maxsize=1)
def _hip_runtime():
    from triton.backends.amd.driver import _get_path_to_hip_runtime_dylib
    runtime = ctypes.CDLL(_get_path_to_hip_runtime_dylib())
    runtime.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                       ctypes.c_int, ctypes.c_void_p]
    runtime.hipMemcpyAsync.restype = ctypes.c_int
    runtime.hipGetErrorString.argtypes = [ctypes.c_int]
    runtime.hipGetErrorString.restype = ctypes.c_char_p
    return runtime


def _synchronous_copy(destination, source, kind, device):
    import torch
    if source.numel():
        runtime = _hip_runtime()
        # Order transfers with the caller's kernels, including nondefault streams.
        # Synchronizing keeps pageable host storage alive and ready for immediate use.
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            error = runtime.hipMemcpyAsync(destination.data_ptr(), source.data_ptr(),
                                           source.nbytes, kind, stream.cuda_stream)
            if error:
                raise RuntimeError(runtime.hipGetErrorString(error).decode())
            stream.synchronize()
    return destination


def to_device(tensor, device):
    """Upload CPU storage; keep the input independent when testing kernels on CPU."""
    import torch
    device = torch.device(device)
    if device.type == "cuda" and torch.version.hip:
        if tensor.device.type != "cpu":
            raise ValueError("Upload a CPU tensor to the Triton device.")
        tensor = tensor.contiguous()
        return _synchronous_copy(torch.empty_like(tensor, device=device), tensor, 1, device)
    return tensor.to(device, copy=True)


def to_host(tensor):
    """Download a kernel result after its current stream has finished."""
    import torch
    if tensor.device.type == "cuda" and torch.version.hip:
        if not tensor.is_contiguous():
            raise ValueError("Download a contiguous Triton output buffer.")
        return _synchronous_copy(torch.empty_like(tensor, device="cpu"), tensor, 2, tensor.device)
    return tensor.cpu()
