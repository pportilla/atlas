"""Finite-field S-pair rows against the shared, triangular reducer table for one degree."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["width"])
def _trace_rows(initial, offsets, pivots, targets, coefficients, work, output, width,
                PRIME: tl.constexpr, BLOCK: tl.constexpr):
    # Each program owns a row. A reducer tail has distinct destination columns,
    # all strictly after its pivot. The next pivot sees every previous update.
    row = tl.program_id(0) * width
    lanes = tl.arange(0, BLOCK)
    for start in range(0, width, BLOCK):
        j = start + lanes
        x = tl.load(initial + row + j, j < width, 0)
        tl.store(work + row + j, x.to(tl.int64), j < width)
    tl.debug_barrier()
    for col in range(width):
        if tl.load(pivots + col) != 0:
            value = tl.load(work + row + col) % PRIME
            if value != 0:
                begin = tl.load(offsets + col)
                end = tl.load(offsets + col + 1)
                for start in range(begin, end, BLOCK):
                    i = start + lanes
                    active = i < end
                    target = tl.load(targets + i, active, 0)
                    coeff = tl.load(coefficients + i, active, 0).to(tl.int64)
                    dest = work + row + target
                    # x < 2*p² < 2^63, so signed 64-bit arithmetic is exact.
                    x = tl.load(dest, active, 0) + (PRIME - value) * coeff
                    tl.store(dest, tl.where(x >= PRIME * PRIME, x - PRIME * PRIME, x), active)
                tl.debug_barrier()
    for start in range(0, width, BLOCK):
        j = start + lanes
        active = j < width
        x = tl.load(work + row + j, active, 0)
        pivot = tl.load(pivots + j, active, 0)
        tl.store(output + row + j, tl.where(pivot != 0, 0, x % PRIME).to(tl.int32), active)


@triton.jit(do_not_specialize=["width", "depth"])
def _level_rows(initial, pivots, starts, parents, factors, order, levels, work, output, width, depth,
                PRIME: tl.constexpr, COLUMNS: tl.constexpr, TERMS: tl.constexpr):
    # Each program owns one row. Four lanes sum incoming terms per column and
    # the group advances several independent columns of one level together.
    row = tl.program_id(0) * width
    cols = tl.arange(0, COLUMNS)
    terms = tl.arange(0, TERMS)
    for d in range(depth):
        first = tl.load(levels + d)
        limit = tl.load(levels + d + 1)
        for base in range(first, limit, COLUMNS):
            pos = base + cols
            live = pos < limit
            col = tl.load(order + pos, live, 0)
            begin = tl.load(starts + col, live, 0)
            end = tl.load(starts + col + 1, live, 0)
            longest = tl.max(end - begin, 0)
            sums = tl.full((COLUMNS, TERMS), 0, tl.int64)
            for k in range(0, longest, TERMS):
                edge = begin[:, None] + k + terms[None, :]
                mask = edge < end[:, None]
                src = tl.load(parents + edge, mask, 0)
                coeff = tl.load(factors + edge, mask, 0).to(tl.int64)
                value = tl.load(work + row + src, mask, 0).to(tl.int64)
                sums += coeff * value
                sums = tl.where(sums >= PRIME * PRIME, sums - PRIME * PRIME, sums)
            # Reduce residues before the cross-lane sum so it fits in int64.
            result = (tl.sum(sums % PRIME, 1) + tl.load(initial + row + col, live, 0)) % PRIME
            pivot = tl.load(pivots + col, live, 0)
            tl.store(work + row + col, (PRIME - result).to(tl.int32), live)
            tl.store(output + row + col, tl.where(pivot != 0, 0, result).to(tl.int32), live)
        tl.debug_barrier()


def _plan(width, offsets, targets, coefficients, tails):
    import ctypes
    import numpy as np
    from .native_library import _kernel

    function = getattr(_kernel(), "atlas_groebner_trace_levels", None)
    if function is None:
        return None
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
        return torch.from_numpy(host).to(device)

    with torch_device_context(device):
        plan = _plan(width, offsets, targets, coefficients, tails)
        result = torch.empty(words, dtype=torch.int32, device=device)
        if plan is None:
            # An earlier native library still supplies the outgoing-edge layout.
            inputs = [upload(pointer, n) for pointer, n in
                      ((initial, words), (offsets, width+1), (pivots, width),
                       (targets, tails), (coefficients, tails))]
            work = torch.empty(words, dtype=torch.int64, device=device)
            _trace_rows[(pairs,)](*inputs, work, result, width, p, 64)
        else:
            arrays, depth = plan
            table = [torch.from_numpy(a).to(device) for a in arrays]
            work = torch.empty_like(result)
            _level_rows[(pairs,)](upload(initial, words), upload(pivots, width), *table,
                                  work, result, width, depth, p, 16, 4, num_warps=2)
        np.ctypeslib.as_array(output, shape=(words,))[:] = result.cpu().numpy()
