"""Triton interpreter comparison for the compiled trace reducer."""
import triton
import triton.language as tl


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
