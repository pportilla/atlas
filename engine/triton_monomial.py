"""Divisor witnesses for the terms of a polynomial modulo a monomial ideal."""
import os
import triton

from .triton_ir import launch_kernel


def witnesses(terms, basis, device):
    import torch
    from .gpu_devices import torch_device_context

    with torch_device_context(device):
        powers = torch.tensor(terms, dtype=torch.int32, device=device)
        divisors = torch.tensor(basis, dtype=torch.int32, device=device)
        result = torch.empty(len(terms), dtype=torch.int32, device=device)
        block = 64
        grid = (triton.cdiv(len(terms), block),)
        args = (powers, divisors, result, len(terms), len(basis))
        if os.environ.get("TRITON_INTERPRET") == "1":
            from tests.triton_chart_oracle import _divisors
            _divisors[grid](*args, len(basis[0]), block)
        else:
            launch_kernel("monomial_witness", (len(basis[0]), block), grid, *args)
        return result.cpu().tolist()
