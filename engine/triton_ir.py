"""Compile the native kernel's portable Triton IR for the active GPU."""
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

from .native_library import Unavailable, field_request


@lru_cache(maxsize=512)
def _compile(name, parameters, device, target_key, warps):
    from triton.compiler import compile
    from triton.runtime import driver

    target = driver.active.get_current_target()

    source = native_ir(name, parameters)
    # Triton accepts IR as a file and caches compiled binaries by the IR's
    # contents and target. The module parses the file during this call.
    with TemporaryDirectory(prefix="atlas-triton-") as directory:
        path = Path(directory, name + ".ttir")
        path.write_text(source, encoding="ascii")
        return compile(str(path), target=target, options={"num_warps": warps})


def native_ir(name, parameters):
    request = " ".join([name, *(str(i) for i in parameters)])
    source = field_request("atlas_triton_ir", [request])
    if source is None:
        raise Unavailable("Rebuild the Atlas kernel for the selected Triton device.")
    return source


def launch_kernel(name, parameters, grid, *args, warps=4):
    from triton.runtime import driver

    active = driver.active
    target = active.get_current_target()
    target_key = (target.backend, repr(target.arch), target.warp_size)
    compiled = _compile(name, parameters, active.get_current_device(), target_key, warps)
    compiled[tuple(grid) + (1,) * (3-len(grid))](*args)
