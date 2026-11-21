import ctypes
import os
import unittest

from engine.gpu_devices import triton_device
from tests.test_native_opencl import reference_rank


class TritonRank(unittest.TestCase):
    def test_transfers_and_kernel_on_nondefault_stream(self):
        selected = triton_device()
        if selected is None:
            self.skipTest("No physical Triton GPU")
        import torch
        from engine.triton_buffers import to_device, to_host
        from engine.triton_ir import launch_kernel

        device = torch.device(selected["torch_device"])
        runtime = getattr(torch, device.type)
        original_device = runtime.current_device()
        host = torch.tensor([0, 2_000_000_001, 0, 7], dtype=torch.int64)[1::2]
        with runtime.stream(runtime.Stream(device=device)):
            uploaded = to_device(host, device)
            launch_kernel("trace_probe", (), (1,), uploaded)
            self.assertEqual(to_host(uploaded).tolist(), [169, 7])
            self.assertEqual(to_host(to_device(host[:0], device)).numel(), 0)
        self.assertEqual(host.tolist(), [2_000_000_001, 7])
        self.assertEqual(runtime.current_device(), original_device)

    def test_rectangular_row_basis_skips_free_columns(self):
        interpreter = os.getenv("TRITON_INTERPRET") == "1"
        device = {"torch_device": "cpu"} if interpreter else triton_device()
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
        device = triton_device() if not interpreter else None
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
