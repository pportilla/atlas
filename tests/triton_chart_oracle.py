"""Independent Triton DSL kernels for CPU interpreter checks."""
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


