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
        inputs = [upload(pointer, n) for pointer, n in
                  ((initial, words), (offsets, width+1), (pivots, width),
                   (targets, tails), (coefficients, tails))]
        work = torch.empty(words, dtype=torch.int64, device=device)
        result = torch.empty(words, dtype=torch.int32, device=device)
        _trace_rows[(pairs,)](*inputs, work, result, width, p, 64)
        np.ctypeslib.as_array(output, shape=(words,))[:] = result.cpu().numpy()
