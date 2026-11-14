"""Full-column-rank witnesses over the two prime fields used by the Jacobian test."""
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


def _factorize(matrix, permutation, rows, columns, prime, *, row_basis=False):
    """Every pivot depends on the previous update. The trailing tiles are independent."""
    row_block = triton.next_power_of_2(rows)
    col_block = triton.next_power_of_2(columns)
    tile_rows, tile_cols = 8, 32
    for col in range(columns):
        _pivot[(1,)](matrix, permutation, rows, columns, col, prime, row_block, col_block, row_basis)
        trailing_rows = rows - 1 if row_basis else rows - col - 1
        if col + 1 < columns and trailing_rows > 0:
            grid = (triton.cdiv(trailing_rows, tile_rows), triton.cdiv(columns - col - 1, tile_cols))
            _eliminate[grid](matrix, permutation, rows, columns, col, prime, tile_rows, tile_cols, row_basis)


def full_rank_rows(matrix_ptr, rows, cols, prime, device_dict):
    """Return original rows of a full-rank minor, or None when a pivot is missing."""
    import numpy as np
    import torch

    if not 1 <= cols <= min(rows, 512) or rows > 16384 or prime not in (1000000007, 1000000009):
        raise ValueError("Unsupported Jacobian matrix shape or prime.")
    device = device_dict["torch_device"]
    from .gpu_devices import torch_device_context
    with torch_device_context(device):
        host = np.ctypeslib.as_array(matrix_ptr, shape=(rows * cols,)).view(np.int32)
        matrix = torch.from_numpy(host).to(device)
        permutation = torch.arange(rows + 1, dtype=torch.int32, device=device)
        _factorize(matrix, permutation, rows, cols, prime)
        witness = permutation.cpu().tolist()
    return None if witness[rows] < 0 else witness[:cols]


def independent_rows(matrix_ptr, rows, cols, prime, device_dict):
    """Original row indices spanning a rectangular matrix modulo a supported prime."""
    import numpy as np
    import torch
    from .gpu_devices import torch_device_context

    if not (1 <= rows <= 16384 and 1 <= cols <= 512 and prime in (1000000007, 1000000009)):
        raise ValueError("Unsupported Jacobian matrix shape or prime.")
    device = device_dict["torch_device"]
    with torch_device_context(device):
        host = np.ctypeslib.as_array(matrix_ptr, shape=(rows*cols,)).view(np.int32)
        matrix = torch.from_numpy(host).to(device, copy=True)
        permutation = torch.arange(rows + 2, dtype=torch.int32, device=device)
        permutation[rows:] = 0
        _factorize(matrix, permutation, rows, cols, prime, row_basis=True)
        ids = permutation.cpu().tolist()
    return ids[:ids[rows]]
