import ctypes
import os
import unittest

from engine.gpu_devices import triton_device
from native.test_opencl import reference_rank


class TritonRank(unittest.TestCase):
    def test_rectangular_row_basis_skips_free_columns(self):
        interpreter = os.getenv("TRITON_INTERPRET") == "1"
        device = {"torch_device": "cpu"} if interpreter else triton_device(int(os.getenv("RESOLUTION_GPU_INDEX", 0)))
        if device is None:
            self.skipTest("No Triton GPU")
        import numpy as np
        from engine.gpu_rank import independent_rows

        prime = 1000000007
        for matrix in ([[0, 1, 2, 0, 4], [0, 2, 4, 0, 8], [0, 1, 2, 0, 5]],
                       [[0, 0, 0, 1]], [[0, 0], [0, 0]], [[0, 1], [1, 0], [2, 1]]):
            with self.subTest(matrix=matrix):
                data = np.array(matrix, dtype=np.uint32)
                ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
                ids = independent_rows(ptr, *data.shape, prime, device)
                rank = reference_rank(matrix, prime)
                self.assertEqual(len(set(ids)), rank)
                self.assertEqual(reference_rank([matrix[i] for i in ids], prime) if ids else 0, rank)
                self.assertEqual(data.tolist(), matrix)

    def test_witnesses(self):
        interpreter = os.getenv("TRITON_INTERPRET") == "1"
        device = triton_device(int(os.getenv("RESOLUTION_GPU_INDEX", 0))) if not interpreter else None
        if not device and not interpreter:
            self.skipTest("No Triton GPU")
        import numpy as np
        import torch
        from engine.gpu_rank import _factorize, full_rank_rows

        rng = np.random.default_rng(3719)
        for prime in (1000000007, 1000000009):
            for rows, cols in [(1, 1), (6, 6), (35, 22), (40, 33)]:
                for deficient in (False, True):
                    with self.subTest(prime=prime, shape=(rows, cols), deficient=deficient):
                        matrix = rng.integers(prime, size=(rows, cols), dtype=np.int32)
                        if rows > 1:
                            matrix[0, 0] = 0  # forces a swap for full-rank examples
                        if deficient:
                            matrix[:, -1] = matrix[:, 0] if cols > 1 else 0
                        original = matrix.tolist()
                        if interpreter:
                            permutation = torch.arange(rows + 1, dtype=torch.int32)
                            _factorize(torch.from_numpy(matrix), permutation, rows, cols, prime)
                            ids = permutation.tolist()
                            witness = ids[:cols] if ids[-1] >= 0 else None
                        else:
                            ptr = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
                            witness = full_rank_rows(ptr, rows, cols, prime, device)
                        self.assertEqual(witness is not None, reference_rank(original, prime) == cols)
                        if witness is not None:
                            self.assertEqual(len(set(witness)), cols)
                            self.assertEqual(reference_rank([original[i] for i in witness], prime), cols)
