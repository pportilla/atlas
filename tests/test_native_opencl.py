"""Check exact OpenCL full-column-rank decisions and independently verify minors."""
import ctypes
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.native_backend import _gpu_library, gpu_devices, resolve_native


def reference_rank(matrix, prime):
    matrix = [row.copy() for row in matrix]
    rows, columns = len(matrix), len(matrix[0])
    rank = 0
    for col in range(columns):
        pivot = next((i for i in range(rank, rows) if matrix[i][col]), None)
        if pivot is None:
            continue
        matrix[pivot], matrix[rank] = matrix[rank], matrix[pivot]
        inverse = pow(matrix[rank][col], -1, prime)
        for i in range(rank + 1, rows):
            scale = matrix[i][col] * inverse % prime
            for j in range(col, columns):
                matrix[i][j] = (matrix[i][j] - scale*matrix[rank][j]) % prime
        rank += 1
    return rank


@unittest.skipUnless(gpu_devices(), "OpenCL GPU unavailable (build with make gpu and install a device ICD)")
class GPUParity(unittest.TestCase):
    def test_independent_modular_matrices(self):
        random_source = random.Random(1278)
        for device in gpu_devices():
            for prime in (1000000007, 1000000009):
                for rows, cols in ((1,1), (6,6), (35,22)):
                    matrix = [[random_source.randrange(prime) for _ in range(cols)] for _ in range(rows)]
                    if rows > 1:
                        matrix[0][0] = 0  # force the first pivot swap
                    for deficient in (False, True):
                        if deficient:
                            for row in matrix:
                                row[-1] = row[0] if cols > 1 else 0
                        with self.subTest(device=device["name"], prime=prime, shape=(rows,cols), deficient=deficient):
                            flat = (ctypes.c_uint32 * (rows*cols))(*(n for row in matrix for n in row))
                            minor = (ctypes.c_int * cols)()
                            result = _gpu_library().resolution_gpu_rank(flat, rows, cols, prime, device["index"], minor)
                            self.assertEqual(result, int(reference_rank(matrix, prime) == cols))
                            if result == 1:
                                self.assertEqual(reference_rank([matrix[i] for i in minor], prime), cols)

    def test_geometry_matches_cpu_and_device_is_reported(self):
        terms = [((2,1,0), "1"), ((0,2,1), "1"), ((1,0,2), "1")]
        variables = ["x","y","z"]
        with patch.dict(os.environ, RESOLUTION_GPU="off"):
            cpu = resolve_native(terms, variables, "intrinsic")
        for device in gpu_devices():
            with patch.dict(os.environ, RESOLUTION_GPU="auto", RESOLUTION_GPU_BACKEND="opencl", RESOLUTION_GPU_MIN_COLS="1",
                            RESOLUTION_GPU_INDEX=str(device["index"])):
                gpu = resolve_native(terms, variables, "intrinsic")
            self.assertEqual(gpu["certificate"]["rank_backend"], "opencl")
            self.assertEqual(gpu["certificate"]["device"]["name"], device["name"])
            self.assertEqual(gpu["charts"], cpu["charts"])
            self.assertEqual(gpu["dual_complex"], cpu["dual_complex"])


class CPUFallback(unittest.TestCase):
    def test_missing_device_is_recoverable(self):
        terms = [((1,1,0), "1"), ((0,0,2), "1")]
        with patch.dict(os.environ, RESOLUTION_GPU="auto", RESOLUTION_GPU_BACKEND="opencl",
                        RESOLUTION_GPU_MIN_COLS="1", RESOLUTION_GPU_INDEX="999"):
            result = resolve_native(terms, ["x","y","z"], "intrinsic")
        self.assertIn(result["certificate"]["rank_backend"], ("cpu", "cpu_after_opencl"))


if __name__ == "__main__":
    unittest.main()
