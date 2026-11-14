"""Exact Triton exponent transforms. The dispatcher imports this on eligible devices only."""
import triton
import triton.language as tl


@triton.jit
def _strict_powers(exponents, rays, cones, orders, output, terms, size,
                   DIM: tl.constexpr, RAY_TABLE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < size
    if RAY_TABLE:
        ray = offsets // terms
        term = offsets % terms
    else:
        pair = offsets // DIM
        term = pair % terms
        position = (pair // terms) * DIM + offsets % DIM
        ray = tl.load(cones + position, mask=active, other=0)

    # transform_batch proves every partial sum fits signed int32.
    power = tl.full((BLOCK,), 0, tl.int32)
    for k in tl.static_range(DIM):
        exponent = tl.load(exponents + term * DIM + k, mask=active, other=0)
        direction = tl.load(rays + ray * DIM + k, mask=active, other=0)
        power += exponent * direction
    order = tl.load(orders + ray, mask=active, other=0)
    tl.store(output + offsets, power - order, mask=active)


@triton.jit
def _gather_cones(table, cones, output, terms, size, DIM: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < size
    pair = offsets // DIM
    position = (pair // terms) * DIM + offsets % DIM
    ray = tl.load(cones + position, mask=active, other=0)
    value = tl.load(table + ray * terms + pair % terms, mask=active, other=0)
    tl.store(output + offsets, value, mask=active)


def transform(inputs, output, terms, dimension, device):
    """Fill the supplied chart-major host buffer, synchronously on the selected GPU."""
    import torch

    from .gpu_devices import torch_device_context
    with torch_device_context(device):
        exponents, rays, cones, orders = [
            torch.frombuffer(buffer, dtype=torch.int32).to(device) for buffer in inputs
        ]
        size = len(output)
        result = torch.empty(size, dtype=torch.int32, device=device)
        ray_terms = terms * len(orders)
        block = 256
        if 2 * ray_terms <= size:
            # Shared cone rays reuse each pairing. The temporary is at most half
            # the output size, even when the caller supplies many unused rays.
            table = torch.empty(ray_terms, dtype=torch.int32, device=device)
            _strict_powers[(triton.cdiv(ray_terms, block),)](
                exponents, rays, cones, orders, table, terms, ray_terms, dimension, True, block)
            _gather_cones[(triton.cdiv(size, block),)](
                table, cones, result, terms, size, dimension, block)
        else:
            _strict_powers[(triton.cdiv(size, block),)](
                exponents, rays, cones, orders, result, terms, size, dimension, False, block)
        torch.frombuffer(output, dtype=torch.int32).copy_(result)
