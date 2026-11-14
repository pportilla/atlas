"""Check the finite-field row remainders independently of the trace builder."""
import ctypes
import os
import random
import unittest

from engine.gpu_devices import available_devices
from engine.native_library import _opencl_trace


def check_rows(test, execute):
    rng = random.Random(7901)
    for prime in (101, 2147483629):
        # Several non-pivot columns, empty and long tails, masked final block.
        width, pairs = 75, 13
        pivots, offsets, targets, coeffs = [], [0], [], []
        for j in range(width):
            is_pivot = j % 4 != 2
            pivots.append(int(is_pivot))
            dest = list(range(j+1, width)) if is_pivot and j % 7 == 0 else []
            if is_pivot and j % 7 > 1:
                dest = rng.sample(range(j+1, width), min(4, width-j-1))
            targets.extend(dest)
            coeffs.extend(rng.randrange(1, prime) for _ in dest)
            offsets.append(len(targets))
        rows = [[rng.randrange(prime) if rng.randrange(4) else 0 for _ in pivots] for _ in range(pairs)]
        expected = []
        for initial in rows:
            row = initial.copy()
            for j in range(width):
                if pivots[j]:
                    factor, row[j] = row[j], 0
                    for k in range(offsets[j], offsets[j+1]):
                        target = targets[k]
                        row[target] = (row[target] - factor*coeffs[k]) % prime
            expected.extend(row)
        array = lambda values: (ctypes.c_uint32 * len(values))(*values)
        result = (ctypes.c_uint32 * (pairs*width))()
        args = (prime, pairs, width, array([c for row in rows for c in row]), array(offsets),
                array(pivots), array(targets), array(coeffs), result)
        execute(args)
        test.assertEqual(list(result), expected)


def check_wide_rows(test, execute):
    # Exceeds typical workgroup local memory. The two pivots each update the
    # last column, and the second pivot depends on the first.
    width, pairs, prime = 16384, 3, 2147483629
    array = lambda values: (ctypes.c_uint32 * len(values))(*values)
    initial = (ctypes.c_uint32 * (width * pairs))()
    expected = [0] * len(initial)
    for i in range(pairs):
        start = i * width
        a, b, c = prime - 1 - i, i + 2, i + 5
        initial[start], initial[start + 1], initial[start + width - 1] = a, b, c
        expected[start + width - 1] = (c - 7*a - (b - 3*a)*11) % prime
    result = (ctypes.c_uint32 * len(initial))()
    args = (prime, pairs, width, initial, array([0, 2] + [3]*(width - 1)),
            array([1, 1] + [0]*(width - 2)), array([1, width - 1, width - 1]), array([3, 7, 11]), result)
    execute(args)
    test.assertEqual(list(result), expected)


def check_dense_levels(test, execute):
    # Four independent lanes each accumulate many nearly p²-sized products.
    # Every column of the second level depends on all pivots of the first,
    # and the free columns depend on both levels. Use the trace prime and a
    # different large prime to exercise generic reduction as well.
    width, pairs = 85, 3
    array = lambda entries: (ctypes.c_uint32 * len(entries))(*entries)
    for prime in (2147483629, 2147483647):
        starts, targets, coeffs = [0], [], []
        for j in range(width):
            begin = 32 if j < 32 else 69 if j < 69 else width
            for i in range(begin, width):
                targets.append(i)
                coeffs.append(prime - 1 - ((i + j) % 7))
            starts.append(len(targets))
        initial = [1 + j % 11 for j in range(pairs * width)]
        expected = []
        for row in range(pairs):
            values = initial[row * width:(row + 1) * width]
            for j in range(69):
                residue, values[j] = values[j], 0
                for t in range(starts[j], starts[j + 1]):
                    i = targets[t]
                    values[i] = (values[i] - residue * coeffs[t]) % prime
            expected.extend(values)
        out = array([0] * len(initial))
        execute((prime, pairs, width, array(initial), array(starts), array([1]*69 + [0]*16),
                 array(targets), array(coeffs), out))
        test.assertEqual(list(out), expected)


class TraceRows(unittest.TestCase):
    def test_physical_devices(self):
        devices = available_devices()
        if not devices:
            self.skipTest("No usable GPU")
        for device in devices:
            with self.subTest(device=device):
                if device["backend"] == "opencl":
                    run = _opencl_trace()
                    self.assertIsNotNone(run, "Build the current OpenCL Gröbner plugin")
                    def execute(args):
                        self.assertEqual(run(*args, device["index"]), 0)
                    check_rows(self, execute)
                    check_wide_rows(self, execute)
                    check_dense_levels(self, execute)
                else:
                    from engine.gpu_trace import reduce_rows
                    check_rows(self, lambda args: reduce_rows(args, device["torch_device"]))
                    check_dense_levels(self, lambda args: reduce_rows(args, device["torch_device"]))

    @unittest.skipUnless(os.getenv("TRITON_INTERPRET") == "1", "Requires Triton interpreter")
    def test_interpreter(self):
        from engine.gpu_trace import reduce_rows
        check_rows(self, lambda args: reduce_rows(args, "cpu"))
        check_dense_levels(self, lambda args: reduce_rows(args, "cpu"))
