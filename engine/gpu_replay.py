"""Replays of the modular Groebner route for many primes at once, in small
data-parallel parts.

The modular route (native/modular.cpp) records one Buchberger computation
modulo a prime as a program, and replays that program modulo many other
primes. Every prime executes the same instructions on its own coefficients:
the replay is the part of the kernel that a GPU can take. A program works on
two arrays per prime, `pool` (32-bit residues: the polynomials known so far)
and `work` (64-bit entries below p^2: the polynomial being reduced), with
five instructions:

  1 width                        work[0:width] = 0
  2 src n i_1..i_n               work[i_t] += pool[src + t]
  3 src n i_1..i_n               work[i_t] -= pool[src + t]
  4 col src n i_1..i_n           f = work[col] mod p; work[col] = 0;
                                 work[i_t] -= f * pool[src + t]
  5 n i_1..i_n                   every work[i_t] must be 0 mod p
  6 dst n i_1..i_n               pool[dst + t] = work[i_t] / work[i_1] mod p

(additions are made modulo p^2 without division: an entry stays below p^2 < 2^62).
A prime for which a check of 5 fails, or the leading coefficient of 6 is zero,
does not agree with the recorded computation and is dropped (status 0).

Each instruction is one small kernel over (primes) x (terms):

  pivot      per prime: the factor p - f of instruction 4, and work[col] = 0
  axpy       per prime and term: work[i_t] += factor * pool[src + t]  (2, 3, 4)
  check      per prime: whether all the entries listed vanish mod p   (5)
  inverse    per prime: the inverse of work[i_1] mod p, by Fermat      (6)
  scale      per prime and term: pool[dst + t] = work[i_t] * inverse   (6)

The production kernels are emitted as Triton IR by native/triton_kernels.cpp.
The NumPy and Triton interpreter implementations used for comparison live in tests.

`use(library, backend)` hands a backend to the kernel as its batch replay.
"""
from .triton_buffers import to_device, to_host
import ctypes
from functools import lru_cache
import os

import numpy as np

_HOOK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32,
                         ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                         ctypes.POINTER(ctypes.c_uint32))


def instructions(code):
    """The instructions of a program: (op, scalar arguments, start and end of its index list)."""
    at, size = 0, len(code)
    while at < size:
        op = int(code[at])
        if op == 1:
            yield op, (int(code[at + 1]),), at + 2, at + 2
            at += 2
        elif op in (2, 3):
            count = int(code[at + 2])
            yield op, (int(code[at + 1]),), at + 3, at + 3 + count
            at += 3 + count
        elif op == 4:
            count = int(code[at + 3])
            yield op, (int(code[at + 1]), int(code[at + 2])), at + 4, at + 4 + count
            at += 4 + count
        elif op == 5:
            count = int(code[at + 1])
            yield op, (), at + 2, at + 2 + count
            at += 2 + count
        elif op == 6:
            count = int(code[at + 2])
            yield op, (int(code[at + 1]),), at + 3, at + 3 + count
            at += 3 + count
        else:
            raise ValueError("Unknown instruction in a replay program.")


def triton_kernels():
    """The parts as Triton kernels, compiled on first use: a dict of name -> kernel.

    The launch grid has one program per prime (pivot, check, inverse work on
    blocks of primes instead) and, for the parts over terms, one per block
    of BLOCK terms. The arrays are int32 (pool, primes, factors, status,
    index lists) and int64 (work). All values are nonnegative; a product has
    factors below 2^31 and a sum is below 2 p^2 < 2^63, so the signed 64-bit
    arithmetic never overflows and % is the mathematical remainder.
    """
    return _triton_kernels(os.environ.get("TRITON_INTERPRET") == "1")


@lru_cache(maxsize=2)
def _triton_kernels(interpreted):
    import triton
    names = ("pivot", "axpy", "check", "inverse", "scale")
    if interpreted:
        from tests import triton_replay_oracle
        kernels = {name: getattr(triton_replay_oracle, name) for name in names}
    else:
        kernels = {name: _ReplayKernel(name) for name in names}
    return {**kernels, "cdiv": triton.cdiv}


class _ReplayKernel:
    def __init__(self, name):
        self.name = "replay_" + name

    def __getitem__(self, grid):
        from .triton_ir import launch_kernel

        def run(*args, BLOCK, opcode=None):
            parameters = (BLOCK,) if opcode is None else (BLOCK, opcode)
            return launch_kernel(self.name, parameters, grid, *args)
        return run


def replay_triton(code, pool_size, work_size, primes, pools, status, block=256, device="cuda"):
    """The replay on the GPU that PyTorch uses, with the Triton kernels:
    the host walks the program and launches one part per instruction."""
    import torch
    from .gpu_devices import torch_device_context
    selected = torch.device(device)
    with torch_device_context(selected):
        return _launch_replay(code, pool_size, work_size, primes, pools, status, block, selected)


def _launch_replay(code, pool_size, work_size, primes, pools, status, block, device):
    import torch
    k = triton_kernels()
    items = len(primes)
    # Primes are below 2^31: every 32-bit value fits in int32.
    prime = to_device(torch.as_tensor(np.asarray(primes, dtype=np.int32)), device)
    pool = to_device(torch.as_tensor(np.asarray(pools, dtype=np.int32).reshape(-1)), device)
    listing = to_device(torch.as_tensor(np.asarray(code, dtype=np.int32)), device)
    work = to_device(torch.zeros(items*work_size, dtype=torch.int64), device)
    factor = to_device(torch.ones(items, dtype=torch.int32), device)
    state = to_device(torch.ones(items, dtype=torch.int32), device)
    by_primes = (k["cdiv"](items, block),)

    def by_terms(count):
        return items, k["cdiv"](count, block)

    for op, scalars, start, end in instructions(code):
        count = end - start
        index = listing[start:end]
        if op == 1:
            if os.environ.get("TRITON_INTERPRET") == "1":
                work.view(items, work_size)[:, :scalars[0]] = 0
            elif scalars[0]:
                from .triton_ir import launch_kernel
                launch_kernel("replay_clear", (block,), by_terms(scalars[0]), work, work_size, scalars[0])
        elif op in (2, 3):
            k["axpy"][by_terms(count)](work, pool, index, factor, prime, work_size, pool_size, scalars[0], count,
                                       BLOCK=block, opcode=op)
        elif op == 4:
            k["pivot"][by_primes](work, factor, prime, work_size, scalars[0], items, BLOCK=block)
            k["axpy"][by_terms(count)](work, pool, index, factor, prime, work_size, pool_size, scalars[1], count,
                                       BLOCK=block)
        elif op == 5:
            k["check"][by_terms(count)](work, index, prime, state, work_size, count, BLOCK=block)
        else:
            k["inverse"][by_primes](work, factor, prime, state, work_size, int(code[start]), items, BLOCK=block)
            k["scale"][by_terms(count)](work, pool, index, factor, prime, work_size, pool_size, scalars[0], count,
                                        BLOCK=block)
    # Both blocking transfers complete before any native inputs are modified.
    output = to_host(pool.view(items, pool_size)).numpy()
    accepted = to_host(state).numpy()
    pools[:] = output
    status[:] = accepted
    return 0


def triton_available():
    from .gpu_devices import triton_device
    return triton_device() is not None


def _arrays(args):
    code, length, pool_size, work_size, primes, count, pools, status = args
    return (np.ctypeslib.as_array(code, shape=(length,)), pool_size, work_size,
            np.ctypeslib.as_array(primes, shape=(count,)),
            np.ctypeslib.as_array(pools, shape=(count, pool_size)),
            np.ctypeslib.as_array(status, shape=(count,)))


def run_batch(args, device):
    """Dispatch an actual native batch, preserving its input on failed transfers."""
    return replay_triton(*_arrays(args), device=device["torch_device"])


def hook(backend):
    """A C callback retained by the caller for the duration of its use."""
    @_HOOK
    def call(*args):
        try:
            return backend(*_arrays(args))
        except Exception:  # the kernel then replays on the host
            return 1
    return call


def use(library, backend="triton"):
    """Hand the replays of the modular route to a backend: "triton" (when it
    can run here) or "reference". Returns whether it was installed."""
    if backend == "triton":
        from functools import partial
        from .gpu_devices import triton_device
        device = triton_device()
        if device is None:
            return False
        chosen = partial(replay_triton, device=device["torch_device"])
    elif backend == "reference":
        host = getattr(library, "atlas_groebner_replay_host", None)
        if host is None:
            return False
        library.atlas_groebner_use_batch(ctypes.cast(host, ctypes.c_void_p))
        return True
    else:
        raise ValueError("Unknown replay backend.")
    library.atlas_groebner_use_batch.argtypes = [ctypes.c_void_p]
    library.atlas_groebner_use_batch.restype = None
    callback = hook(chosen)
    library.atlas_groebner_use_batch(ctypes.cast(callback, ctypes.c_void_p))
    library._replay_callback = callback
    return True
