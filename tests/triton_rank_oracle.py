"""Independent Triton DSL rank kernels for its CPU interpreter."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["col"])
def _pivot(matrix, permutation, rows, columns, col, PRIME: tl.constexpr,
           ROW_BLOCK: tl.constexpr, COL_BLOCK: tl.constexpr, ROW_BASIS: tl.constexpr):
    # permutation[rows] is a failure flag shared by all later pivots.
    if ROW_BASIS:
        target = tl.load(permutation + rows)  # number of pivot rows found
    else:
        target = col
    if ROW_BASIS or tl.load(permutation + rows) >= 0:
        indices = tl.arange(0, ROW_BLOCK)
        candidates = tl.load(matrix + indices * columns + col,
                             mask=(indices >= target) & (indices < rows), other=0)
        selected = tl.min(tl.where(candidates != 0, indices, rows), 0)
        if selected == rows:
            if ROW_BASIS:
                tl.store(permutation + rows + 1, -1)
            else:
                tl.store(permutation + rows, -1)
        else:
            base = tl.load(matrix + selected * columns + col).to(tl.uint64)
            inverse = tl.full((), 1, tl.uint64)
            for bit in tl.static_range(30):
                if ((PRIME - 2) >> bit) & 1:
                    inverse = (inverse * base) % PRIME
                base = (base * base) % PRIME

            # Load both tails before swapping. When selected == col only
            # the normalization is needed.
            j = tl.arange(0, COL_BLOCK)
            active = (j >= col) & (j < columns)
            value = tl.load(matrix + selected * columns + j, active, 0)
            if selected != target:
                old = tl.load(matrix + target * columns + j, active, 0)
                tl.store(matrix + selected * columns + j, old, active)
                old_id = tl.load(permutation + target)
                new_id = tl.load(permutation + selected)
                tl.store(permutation + selected, old_id)
                tl.store(permutation + target, new_id)
            normalized = (value.to(tl.uint64) * inverse) % PRIME
            tl.store(matrix + target * columns + j, normalized.to(tl.int32), active)
            if ROW_BASIS:
                tl.store(permutation + rows + 1, target)
                tl.store(permutation + rows, target + 1)


@triton.jit(do_not_specialize=["col"])
def _eliminate(matrix, permutation, rows, columns, col, PRIME: tl.constexpr,
               TILE_ROWS: tl.constexpr, TILE_COLS: tl.constexpr, ROW_BASIS: tl.constexpr):
    if ROW_BASIS:
        target = tl.load(permutation + rows + 1)
    else:
        target = col
    if target >= 0 and (ROW_BASIS or tl.load(permutation + rows) >= 0):
        i = target + 1 + tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
        j = col + 1 + tl.program_id(1) * TILE_COLS + tl.arange(0, TILE_COLS)
        factor = tl.load(matrix + i * columns + col, i < rows, 0)
        pivot = tl.load(matrix + target * columns + j, j < columns, 0)
        product = factor[:, None].to(tl.uint64) * pivot[None, :].to(tl.uint64)
        offset = i[:, None] * columns + j[None, :]
        active = (i[:, None] < rows) & (j[None, :] < columns)
        entry = tl.load(matrix + offset, active, 0)
        difference = entry - (product % PRIME).to(tl.int32)
        tl.store(matrix + offset, tl.where(difference < 0, difference + PRIME, difference), active)
