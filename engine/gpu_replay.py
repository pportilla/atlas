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

`replay_reference` runs them with NumPy, and is what the tests compare with
the kernel's own replay (atlas_groebner_replay_compare). `triton_kernels`
holds the same parts written in Triton (https://triton-lang.org); they need
PyTorch with CUDA and the `triton` package, neither of which the tests can
assume: on a machine without them the Triton parts are not exercised.

`use(library, backend)` hands a backend to the kernel as its batch replay.
"""
import ctypes

import numpy as np

_HOOK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32,
                         ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                         ctypes.POINTER(ctypes.c_uint32))
_kept = []  # the callbacks handed to the kernel must stay alive


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


def replay_reference(code, pool_size, work_size, primes, pools, status):
    """Run the program for all the primes. `pools` (primes x pool_size,
    uint32) is updated in place and `status` (uint32) receives 1 for the
    primes that agree with the program."""
    primes = np.asarray(primes, dtype=np.uint64)
    p = primes[:, None]
    square = p*p
    work = np.zeros((len(primes), work_size), dtype=np.uint64)
    status[:] = 1

    def axpy(factor, source, index):
        # factor < p and pool < p: the product is below p^2, the sum below 2 p^2 < 2^63.
        w = work[:, index] + factor*pools[:, source:source + len(index)].astype(np.uint64)
        work[:, index] = np.where(w >= square, w - square, w)

    for op, scalars, start, end in instructions(code):
        index = np.asarray(code[start:end], dtype=np.intp)
        if op == 1:
            work[:, :scalars[0]] = 0
        elif op == 2:
            axpy(np.ones_like(p), scalars[0], index)
        elif op == 3:
            axpy(p - np.uint64(1), scalars[0], index)
        elif op == 4:
            column, source = scalars
            f = work[:, column] % primes
            work[:, column] = 0
            axpy(np.where(f == 0, np.uint64(0), primes - f)[:, None], source, index)
        elif op == 5:
            status[((work[:, index] % p) != 0).any(axis=1)] = 0
        else:
            target = scalars[0]
            base = work[:, index[0]] % primes
            status[base == 0] = 0
            inverse = np.array([pow(int(b), int(q) - 2, int(q)) if b else 0 for b, q in zip(base, primes)],
                               dtype=np.uint64)[:, None]
            pools[:, target] = 1
            pools[:, target + 1:target + len(index)] = ((work[:, index[1:]] % p)*inverse % p).astype(np.uint32)
    return 0


def triton_kernels():
    """The parts as Triton kernels, compiled on first use: a dict of name -> kernel.

    The launch grid has one program per prime (pivot, check, inverse work on
    blocks of primes instead) and, for the parts over terms, one per block
    of BLOCK terms. The arrays are int32 (pool, primes, factors, status,
    index lists) and int64 (work). All values are nonnegative; a product has
    factors below 2^31 and a sum is below 2 p^2 < 2^63, so the signed 64-bit
    arithmetic never overflows and % is the mathematical remainder.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def pivot(work_ptr, factor_ptr, prime_ptr, work_size, column, items, BLOCK: tl.constexpr):
        # factor = p - (work[column] mod p), or 0; work[column] = 0.
        offsets = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        mask = offsets < items
        p = tl.load(prime_ptr + offsets, mask=mask, other=1).to(tl.int64)
        place = work_ptr + offsets.to(tl.int64)*work_size + column
        f = tl.load(place, mask=mask, other=0) % p
        tl.store(place, f*0, mask=mask)
        tl.store(factor_ptr + offsets, tl.where(f == 0, f, p - f).to(tl.int32), mask=mask)

    @triton.jit
    def axpy(work_ptr, pool_ptr, index_ptr, factor_ptr, prime_ptr, work_size, pool_size, source, count,
             BLOCK: tl.constexpr):
        # work[index[t]] += factor * pool[source + t], modulo p^2 without division.
        item = tl.program_id(0).to(tl.int64)
        offsets = tl.program_id(1)*BLOCK + tl.arange(0, BLOCK)
        mask = offsets < count
        p = tl.load(prime_ptr + item).to(tl.int64)
        factor = tl.load(factor_ptr + item).to(tl.int64)
        index = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
        src = tl.load(pool_ptr + item*pool_size + source + offsets, mask=mask, other=0).to(tl.int64)
        place = work_ptr + item*work_size + index
        w = tl.load(place, mask=mask, other=0) + factor*src
        square = p*p
        tl.store(place, tl.where(w >= square, w - square, w), mask=mask)

    @triton.jit
    def check(work_ptr, index_ptr, prime_ptr, status_ptr, work_size, count, BLOCK: tl.constexpr):
        # status = 0 for a prime with a listed entry that is not zero mod p.
        item = tl.program_id(0).to(tl.int64)
        offsets = tl.program_id(1)*BLOCK + tl.arange(0, BLOCK)
        mask = offsets < count
        p = tl.load(prime_ptr + item).to(tl.int64)
        index = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
        w = tl.load(work_ptr + item*work_size + index, mask=mask, other=0) % p
        wrong = tl.sum(((w != 0) & mask).to(tl.int32), axis=0)
        # Programs of the same prime may all write the same value 0.
        tl.store(status_ptr + item, wrong*0, mask=wrong > 0)

    @triton.jit
    def inverse(work_ptr, factor_ptr, prime_ptr, status_ptr, work_size, column, items, BLOCK: tl.constexpr):
        # factor = (work[column] mod p)^(p-2) mod p; status = 0 where that entry is 0.
        offsets = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        mask = offsets < items
        p = tl.load(prime_ptr + offsets, mask=mask, other=3).to(tl.int64)
        base = tl.load(work_ptr + offsets.to(tl.int64)*work_size + column, mask=mask, other=1) % p
        tl.store(status_ptr + offsets, (base*0).to(tl.int32), mask=mask & (base == 0))
        exponent = p - 2
        result = base*0 + 1
        for _ in range(31):  # p < 2^31
            result = tl.where((exponent & 1) == 1, result*base % p, result)
            base = base*base % p
            exponent = exponent >> 1
        tl.store(factor_ptr + offsets, result.to(tl.int32), mask=mask)

    @triton.jit
    def scale(work_ptr, pool_ptr, index_ptr, factor_ptr, prime_ptr, work_size, pool_size, target, count,
              BLOCK: tl.constexpr):
        # pool[target + t] = work[index[t]] * factor mod p, and 1 for t = 0.
        item = tl.program_id(0).to(tl.int64)
        offsets = tl.program_id(1)*BLOCK + tl.arange(0, BLOCK)
        mask = offsets < count
        p = tl.load(prime_ptr + item).to(tl.int64)
        factor = tl.load(factor_ptr + item).to(tl.int64)
        index = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
        w = tl.load(work_ptr + item*work_size + index, mask=mask, other=0) % p
        value = tl.where(offsets == 0, w*0 + 1, w*factor % p)
        tl.store(pool_ptr + item*pool_size + target + offsets, value.to(tl.int32), mask=mask)

    return {"pivot": pivot, "axpy": axpy, "check": check, "inverse": inverse, "scale": scale, "cdiv": triton.cdiv}


_compiled = None


def replay_triton(code, pool_size, work_size, primes, pools, status, block=256):
    """The replay on the GPU that PyTorch uses, with the Triton kernels:
    the host walks the program and launches one part per instruction."""
    global _compiled
    import torch
    if _compiled is None:
        _compiled = triton_kernels()
    k = _compiled
    device = torch.device("cuda")
    items = len(primes)
    # Primes are below 2^31: every 32-bit value fits in int32.
    prime = torch.as_tensor(np.asarray(primes, dtype=np.int32), device=device)
    pool = torch.as_tensor(np.asarray(pools, dtype=np.int32).reshape(-1), device=device)
    listing = torch.as_tensor(np.asarray(code, dtype=np.int32), device=device)
    work = torch.zeros(items*work_size, dtype=torch.int64, device=device)
    factor = torch.ones(items, dtype=torch.int32, device=device)
    state = torch.ones(items, dtype=torch.int32, device=device)
    by_primes = (k["cdiv"](items, block),)

    def by_terms(count):
        return items, k["cdiv"](count, block)

    for op, scalars, start, end in instructions(code):
        count = end - start
        index = listing[start:end]
        if op == 1:
            work.view(items, work_size)[:, :scalars[0]] = 0
        elif op in (2, 3):
            factor.copy_(torch.ones_like(prime) if op == 2 else prime - 1)
            k["axpy"][by_terms(count)](work, pool, index, factor, prime, work_size, pool_size, scalars[0], count,
                                       BLOCK=block)
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
    torch.cuda.synchronize()
    pools[:] = pool.view(items, pool_size).cpu().numpy()
    status[:] = state.cpu().numpy()
    return 0


def triton_available():
    """Whether the Triton kernels can run here: PyTorch with CUDA and the triton package."""
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def hook(backend):
    """A C callback with the kernel's batch signature that runs `backend`."""
    def call(code, length, pool_size, work_size, primes, count, pools, status):
        try:
            program = np.ctypeslib.as_array(code, shape=(length,))
            moduli = np.ctypeslib.as_array(primes, shape=(count,))
            table = np.ctypeslib.as_array(pools, shape=(count, pool_size))
            answer = np.ctypeslib.as_array(status, shape=(count,))
            return backend(program, pool_size, work_size, moduli, table, answer)
        except Exception:  # the kernel then replays on the host
            return 1
    callback = _HOOK(call)
    _kept.append(callback)
    return callback


def use(library, backend="triton"):
    """Hand the replays of the modular route to a backend: "triton" (when it
    can run here) or "reference". Returns whether it was installed."""
    if backend == "triton":
        if not triton_available():
            return False
        chosen = replay_triton
    elif backend == "reference":
        chosen = replay_reference
    else:
        raise ValueError("Unknown replay backend.")
    library.atlas_groebner_use_batch.argtypes = [ctypes.c_void_p]
    library.atlas_groebner_use_batch.restype = None
    library.atlas_groebner_use_batch(ctypes.cast(hook(chosen), ctypes.c_void_p))
    return True
