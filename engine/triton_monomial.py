"""Divisor witnesses for the terms of a polynomial modulo a monomial ideal."""
import triton
import triton.language as tl


@triton.jit
def _divisors(powers, basis, output, terms, divisors, VARIABLES: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = rows < terms
    witness = tl.full((BLOCK,), -1, tl.int32)
    for j in range(divisors):
        divides = live & (witness < 0)
        for k in tl.static_range(VARIABLES):
            divisor = tl.load(basis + j * VARIABLES + k)
            power = tl.load(powers + rows * VARIABLES + k, live, 0)
            divides = divides & (divisor <= power)
        witness = tl.where(divides, j, witness)
    tl.store(output + rows, witness, live)


def witnesses(terms, basis, device):
    import torch
    from .gpu_devices import torch_device_context

    with torch_device_context(device):
        powers = torch.tensor(terms, dtype=torch.int32, device=device)
        divisors = torch.tensor(basis, dtype=torch.int32, device=device)
        result = torch.empty(len(terms), dtype=torch.int32, device=device)
        block = 64
        _divisors[(triton.cdiv(len(terms), block),)](powers, divisors, result, len(terms),
                                                      len(basis), len(basis[0]), block)
        return result.cpu().tolist()
