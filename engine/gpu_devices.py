"""Discovery of accelerator backends. All GPU dependencies are optional."""
from functools import lru_cache
import os


def gpu_preference():
    backend = os.getenv("RESOLUTION_GPU_BACKEND", "auto")
    if backend not in ("auto", "triton", "opencl"):
        raise ValueError("RESOLUTION_GPU_BACKEND must be auto, triton, or opencl.")
    return backend


@lru_cache(maxsize=1)
def _triton_candidates():
    """CUDA/HIP ordinals retain their indices. XPU ordinals follow them."""
    try:
        import triton
        import torch
    except Exception:
        return ()
    devices = []
    for kind in ("cuda", "xpu"):
        try:
            count = getattr(torch, kind).device_count()
        except Exception:
            # A broken or absent optional runtime needn't hide the other one.
            continue
        devices.extend(f"{kind}:{ordinal}" for ordinal in range(count))
    return tuple(devices)


def torch_device_context(device):
    """Scope PyTorch allocations and Triton launches to the probed device."""
    import torch
    from contextlib import nullcontext
    kind, _, ordinal = str(device).partition(":")
    if kind == "cpu":  # Kernel tests in the Triton interpreter.
        return nullcontext()
    runtime = getattr(torch, kind)
    return runtime.device(int(ordinal) if ordinal else runtime.current_device())


def triton_device(index):
    """Return the backend-local device after a successful integer kernel launch."""
    if os.getenv("TRITON_INTERPRET") == "1":
        return None
    candidates = _triton_candidates()
    if not 0 <= index < len(candidates):
        return None
    return _probe_triton(index, candidates[index])


@lru_cache(maxsize=32)
def _probe_triton(index, device):
    try:
        import torch
        from triton.runtime import driver
        from .triton_probe import square_modulo

        kind, ordinal = device.split(":")
        with torch_device_context(device):
            target = driver.active.get_current_target()
            runtimes = ("xpu",) if kind == "xpu" else ("cuda", "hip")
            if target.backend not in runtimes:
                return None
            name = getattr(torch, kind).get_device_name(int(ordinal))
            probe = torch.full((1,), 2_000_000_001, dtype=torch.int64, device=device)
            square_modulo[(1,)](probe)
            if probe.item() != pow(2_000_000_001, 2, 1_000_000_007):
                return None
        arch = target.arch
        if isinstance(arch, dict):
            arch = arch.get("arch") or arch.get("architecture") or name
        return {"index": index, "name": name, "backend": "triton", "runtime": target.backend,
                "architecture": str(arch), "torch_device": device}
    except Exception:
        # Imports alone cannot detect a missing driver or unsupported compiler target.
        return None


def available_devices():
    """List usable backend/device pairs. Indices are local to each backend."""
    from .native_backend import gpu_devices
    devices = [dict(device, backend="opencl") for device in gpu_devices()]
    for index in range(len(_triton_candidates())):
        device = triton_device(index)
        if device is not None:
            devices.append(device)
    return devices
