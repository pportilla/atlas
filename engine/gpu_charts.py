"""Exact Triton exponent transforms. The dispatcher imports this on eligible devices only."""
from .triton_buffers import to_device, to_host
import os
import triton

from .triton_ir import launch_kernel


def transform(inputs, output, terms, dimension, device):
    """Fill the supplied chart-major host buffer, synchronously on the selected GPU."""
    import torch

    from .gpu_devices import torch_device_context
    with torch_device_context(device):
        exponents, rays, cones, orders = [
            to_device(torch.frombuffer(buffer, dtype=torch.int32), device) for buffer in inputs
        ]
        size = len(output)
        result = torch.empty(size, dtype=torch.int32, device=device)
        ray_terms = terms * len(orders)
        block = 256
        interpreted = os.environ.get("TRITON_INTERPRET") == "1"
        if interpreted:
            from tests.triton_chart_oracle import _gather_cones, _strict_powers

        if 2 * ray_terms <= size:
            # Shared cone rays reuse each pairing. The temporary is at most half
            # the output size, even when the caller supplies many unused rays.
            table = torch.empty(ray_terms, dtype=torch.int32, device=device)
            ray_grid = (triton.cdiv(ray_terms, block),)
            chart_grid = (triton.cdiv(size, block),)
            if interpreted:
                _strict_powers[ray_grid](exponents, rays, cones, orders, table, terms, ray_terms, dimension, True, block)
                _gather_cones[chart_grid](table, cones, result, terms, size, dimension, block)
            else:
                launch_kernel("ray_pairs", (dimension, block), ray_grid, exponents, rays, cones, orders, table, terms, ray_terms)
                launch_kernel("gather_charts", (dimension, block), chart_grid, table, cones, result, terms, size)
        else:
            grid = (triton.cdiv(size, block),)
            if interpreted:
                _strict_powers[grid](exponents, rays, cones, orders, result, terms, size, dimension, False, block)
            else:
                launch_kernel("strict_charts", (dimension, block), grid, exponents, rays, cones, orders, result, terms, size)
        torch.frombuffer(output, dtype=torch.int32).copy_(to_host(result))
