"""Exact row operations shared by all primes in a modular Gröbner replay."""
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
    # Multiple term blocks can reject a prime concurrently.
    tl.atomic_and(status_ptr + item, 0, mask=wrong > 0, sem="relaxed")

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
