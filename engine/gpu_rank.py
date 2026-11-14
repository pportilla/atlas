"""Launch the native modular pivot and elimination kernels on a Triton device."""
import os

import triton

from .triton_ir import launch_kernel


def _factorize(matrix, permutation, rows, columns, prime, *, row_basis=False):
    """Every pivot depends on the previous update. The trailing tiles are independent."""
    row_block = triton.next_power_of_2(rows)
    col_block = triton.next_power_of_2(columns)
    tile_rows, tile_cols = 8, 32
    interpreted = os.environ.get("TRITON_INTERPRET") == "1"
    if interpreted:
        from tests.triton_rank_oracle import _eliminate, _pivot
    for col in range(columns):
        arguments = matrix, permutation, rows, columns, col
        if interpreted:
            _pivot[(1,)](*arguments, prime, row_block, col_block, row_basis)
        else:
            launch_kernel("rank_pivot", (prime, row_block, col_block, int(row_basis)), (1,), *arguments)
        trailing_rows = rows - 1 if row_basis else rows - col - 1
        if col + 1 < columns and trailing_rows > 0:
            grid = (triton.cdiv(trailing_rows, tile_rows), triton.cdiv(columns - col - 1, tile_cols))
            if interpreted:
                _eliminate[grid](*arguments, prime, tile_rows, tile_cols, row_basis)
            else:
                launch_kernel("rank_eliminate", (prime, tile_rows, tile_cols, int(row_basis)), grid, *arguments)


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
