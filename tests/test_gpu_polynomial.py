"""Exact agreement of the coefficient transports, native composition, and FLINT."""
import random
import unittest
from fractions import Fraction
from unittest.mock import patch

from engine.algebra_execution import algebra_execution
from engine.gpu_devices import triton_device
from engine.gpu_polynomial import _Callback, _call, compute_row
from engine.native_library import _kernel


def reference_batch(kind, n, m, aw, bw, ow, a, b, out):
    if kind == 0:
        for i in range(n * ow):
            out[i] = a[i] + b[i]
    elif kind == 3:
        for i in range(n + m - 1):
            out[i] = sum(a[j] * b[i - j] for j in range(max(0, i - m + 1), min(n, i + 1)))
    else:
        for i in range(n):
            for j in range(m) if kind == 1 else (i,):
                row = i * m + j if kind == 1 else i
                for k in range(ow):
                    out[row * ow + k] = sum(a[i * aw + h] * b[j * bw + k - h]
                                            for h in range(max(0, k - bw + 1), min(aw, k + 1)))
    return 1


def examples():
    a, b = (1 << 303) + 157, -((1 << 201) + 139)
    e = (1 << 80) + 3
    f = f"{a}/41 {e} 5;2/5 0 0;{b}/7 2 5"
    g = f"{b}/37 {e} 5;13/17 0 1"
    for op in ("add", "subtract", "multiply"):
        yield op, 2, 2, (f, g), None
    yield "divide", 2, 2, (f, "-23/67 0 0"), None
    yield "differentiate", 2, 2, (f,), 0
    yield "differentiate", 2, 2, ("2 1 1;-3 2 2",), 1
    yield "power", 2, 2, ("-2/7 0 1;1 2 0",), 13
    yield "power", 1, 1, ("-1 1",), e
    yield "multiply", 1, 1, ("5 1;-7/3 2;1 3", "-9 9;3 10"), None
    yield "multiply", 0, 0, (str(a), str(b)), None
    yield "substitute", 2, 2, ("-3 2 1;1 3 2", "1 0 1;-1 1 0", "2 1 1;3 0 0"), None
    yield "substitute", 2, 0, ("-3 2 1;1 3 2", "-14/5", "3/19"), None
    yield "substitute", 1, 1, (f"{a}/{b} 1;3 2", ""), None
    yield "substitute", 0, 0, ("-13/11",), None
    yield "multiply", 1, 1, ("", "2 0"), None
    yield "power", 0, 0, ("",), 0
    yield "scalar-power", 0, 0, ("-13/7",), -7
    yield "compare", 0, 0, (str(a), str(b)), None


@unittest.skipUnless(_kernel() is not None, "Build the native Atlas kernel")
class PolynomialGPU(unittest.TestCase):
    def test_wide_sparse_coefficients_and_exponents(self):
        cb = _Callback(reference_batch)
        for op, n, m, rows, parameter in examples():
            with self.subTest(op=op, rows=rows):
                head = f"{op} {n} {m}" + ("" if parameter is None else f" {parameter}")
                payload = ("\n".join([head, *rows]) + "\n").encode("ascii")
                self.assertEqual(_call(payload, cb), compute_row(op, n, m, rows, parameter))

    def test_random_convolution_and_carry(self):
        rng = random.Random(8124)
        cb = _Callback(reference_batch)
        for trial in range(24):
            width = (1, 16, 30, 61, 62, 63, 128, 511)[trial % 8]
            def row(count):
                return ";".join(f"{Fraction(rng.randint(-2**width,2**width), rng.randint(1,17))} {i}"
                                for i in range(count))
            a, b = row(5), row(3)
            data = f"multiply 1 1\n{a}\n{b}\n".encode("ascii")
            self.assertEqual(_call(data, cb), compute_row("multiply", 1, 1, [a,b]))

    def test_decline_restarts_cleanly_on_cpu(self):
        mock = {"index": 0, "backend": "triton", "torch_device": "cuda:0", "name": "test"}
        with (algebra_execution("gpu", backend="triton", index=0) as run,
              patch("engine.gpu_polynomial.triton_device", return_value=mock),
              patch("engine.gpu_polynomial._launch", side_effect=RuntimeError("launch failed"))):
            self.assertEqual(compute_row("multiply", 1, 1, ["2 1", "-3/7 2"]), "-6/7 3")
            report = run.report()
            self.assertEqual(report["backend"], "cpu")
            self.assertIn("launch failed", report["fallback_reasons"])

    def test_actual_triton_matches_flint(self):
        if triton_device() is None:
            self.skipTest("Requires a usable physical Triton device")
        kernels = set()
        for op, n, m, rows, parameter in examples():
            with self.subTest(op=op, rows=rows):
                with algebra_execution("cpu"):
                    expected = compute_row(op, n, m, rows, parameter)
                with algebra_execution("gpu", backend="triton", index=0) as run:
                    self.assertEqual(compute_row(op, n, m, rows, parameter), expected)
                    report = run.report()
                    self.assertFalse(report["fallback_reasons"])
                    for stage in report["stages"]:
                        kernels.update(stage.get("kernels", {}))
        self.assertEqual(kernels, {"poly_add_words", "poly_outer_words", "poly_point_words",
                                   "poly_dense_convolution", "poly_point_product", "poly_outer_product"})
