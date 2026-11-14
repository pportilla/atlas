"""Compare chart-major native exponent tables with Python integer dot products."""
import json
import os
from contextlib import nullcontext
from ctypes import c_int32
from itertools import chain
from pathlib import Path
from random import Random
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.chart_batch import _cpu_functions, _pack, transform_atlas, transform_batch
from engine.gpu_devices import available_devices
from engine.native_backend import gpu_devices


def example():
    fan = json.loads(subprocess.check_output([ROOT/"native/build/toricfan"], input=b"15 10 6\n"))
    return ([(2,0,0), (0,3,0), (0,0,5), (2,2,2)], fan["rays"], fan["cones"])


def random_batch(dim=3):
    rng = Random(91843)
    exps = [tuple(rng.randrange(45) for _ in range(dim)) for _ in range(221)]
    rays = [tuple(rng.randrange(30) for _ in range(dim)) for _ in range(51)]
    cones = [tuple(rng.sample(range(len(rays)), dim)) for _ in range(91)]
    return exps, rays, cones


def projections(exps, rays, cones):
    values = [[sum(a*b for a,b in zip(e,r)) for e in exps] for r in rays]
    orders = list(map(min, values))
    return orders, [[tuple(values[i][t]-orders[i] for i in cone) for t in range(len(exps))] for cone in cones]


class ChartBatchTests(unittest.TestCase):
    def test_rendered_charts_preserve_coefficients_and_graded_order(self):
        from fractions import Fraction
        from engine.polynomial import format_poly

        # Both cones permute the two tied leading monomials. The larger
        # power also exercises the arbitrary-precision chart table.
        rays, cones = [(1, 0), (0, 1)], [(0, 1), (1, 0)]
        names = ("zz", "alpha")
        coefficients = [Fraction(-1), Fraction(2, 3), Fraction(-10**4500, 7), Fraction(1)]
        for power in (7, 2**70, 10**4500):
            exponents = [(power, 0), (0, power), (1, 1), (0, 0)]
            values = list(zip(*exponents))
            expected = [format_poly({tuple(e[i] for i in cone): c for e, c in zip(exponents, coefficients)},
                                    names) for cone in cones]
            policies = ("cpu", "auto", "gpu") if power.bit_length() < 1000 else ("cpu",)
            for policy in policies:
                with self.subTest(bits=power.bit_length(), policy=policy):
                    equations, _ = transform_atlas(exponents, rays, cones, [0, 0], values, policy,
                                                   equations=(names, coefficients))
                    self.assertEqual(equations, expected)

    def test_wide_cpu_auto_and_absent_device_reuse_valuations(self):
        exps, rays, cones = [(2**90, 5), (3, 7)], [(11, 2**75), (0, 1)], [(1, 0)]
        values = [[sum(a*b for a,b in zip(e,r)) for e in exps] for r in rays]
        orders, expected = projections(exps, rays, cones)
        with patch("engine.chart_batch._accelerate", side_effect=AssertionError("unexpected device launch")), \
             patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton"), \
             patch("engine.chart_batch.triton_device", return_value=None):
            for policy in ("cpu", "auto", "gpu"):
                result, stats = transform_atlas(exps, rays, cones, orders, values, policy)
                self.assertEqual(list(result), expected)
                self.assertEqual(stats["backend"], "cpu")

    def test_wide_device_failure_discards_partial_results(self):
        exps, rays, cones = [(2**90, 5), (3, 7)], [(11, 2**75), (0, 1)], [(1, 0)]
        values = [[sum(a*b for a,b in zip(e,r)) for e in exps] for r in rays]
        orders, expected = projections(exps, rays, cones)
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton"), \
             patch("engine.chart_batch.triton_device", return_value={"index": 0}), \
             patch("engine.chart_batch._accelerate", side_effect=[True, False]) as device:
            result, stats = transform_atlas(exps, rays, cones, orders, values, "gpu")
        self.assertEqual(list(result), expected)
        self.assertEqual(device.call_count, 2)
        self.assertEqual(stats["limb_batches"], 1)
        self.assertIn("did not complete", stats["fallback_reason"])

    def test_device_products_must_pass_the_native_exact_check(self):
        exps, rays = [(2**90, 5), (3, 7)], [(11, 2**75), (0, 1)]
        cones = [(1, 0)]
        values = [[sum(a*b for a, b in zip(e, r)) for e in exps] for r in rays]
        orders, expected = projections(exps, rays, cones)
        # A device reporting success without filling its output leaves zeros.
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton"), \
             patch("engine.chart_batch.triton_device", return_value={"index": 0}), \
             patch("engine.chart_batch._accelerate", return_value=True):
            charts, stats = transform_atlas(exps, rays, cones, orders, values, "gpu")
        self.assertEqual(list(charts), expected)
        self.assertEqual(stats["backend"], "cpu")
        self.assertIn("exact check", stats["fallback_reason"])

    def test_wide_recombination_with_a_packed_device_callback(self):
        magnitude = 16383 + (16383 << 70)
        exps = [(magnitude,)*8, tuple(magnitude-(i+1) for i in range(8))]
        rays = [(magnitude,)*8, tuple(magnitude if i%2 else 0 for i in range(8))]
        cones = [tuple(i%2 for i in range(8))]
        values = [[sum(a*b for a,b in zip(e,r)) for e in exps] for r in rays]
        orders, expected = projections(exps, rays, cones)

        def launch(inputs, out, nt, nr, nc, dimension, index, stats):
            _cpu_functions()[0](*inputs, nt, nc, dimension, out)
            stats["backend"] = "packed test device"
            return True

        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton"), \
             patch("engine.chart_batch.triton_device", return_value={"index": 0}), \
             patch("engine.chart_batch._accelerate", side_effect=launch):
            charts, stats = transform_atlas(exps, rays, cones, orders, values, "gpu")
        self.assertEqual(list(charts), expected)
        self.assertEqual((stats["backend"], stats["limb_batches"]), ("packed test device", 4))

    def test_oversized_chart_count_still_uses_exact_known_valuations(self):
        exps, rays, cones = [(2,0), (0,3)], [(1,0), (0,1)], [(1,0), (0,1)]
        orders, expected = projections(exps, rays, cones)
        with patch("engine.chart_batch._MAX_PAIRS", 1), patch("engine.chart_batch._accelerate") as device:
            result, stats = transform_atlas(exps, rays, cones, orders, [(2,0), (0,3)], "gpu")
        self.assertEqual(list(result), expected)
        self.assertEqual(stats["backend"], "cpu")
        self.assertIn("batch size", stats["fallback_reason"])
        device.assert_not_called()

    @unittest.skipUnless(gpu_devices(), "OpenCL GPU unavailable")
    def test_wide_eight_axis_digit_sums_on_device(self):
        # Eight maximum digit products in one sum, with nonzero low and high
        # limbs on both sides and a zero limb between them.
        magnitude = 16383 + (16383 << 70)
        exps = [(magnitude,)*8, tuple(magnitude-(i+1) for i in range(8))]
        rays = [(magnitude,)*8, tuple(magnitude if i%2 else 0 for i in range(8))]
        cones = [tuple(i%2 for i in range(8))]
        values = [[sum(a*b for a,b in zip(e,r)) for e in exps] for r in rays]
        orders, expected = projections(exps, rays, cones)
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="opencl", RESOLUTION_GPU_INDEX="0"):
            result, stats = transform_atlas(exps, rays, cones, orders, values, "gpu")
        self.assertEqual(list(result), expected)
        self.assertEqual((stats["backend"], stats["limb_batches"]), ("opencl", 4))

    def test_cpu_exact(self):
        for exps, rays, cones in [example(), random_batch(3), random_batch(4), random_batch(8)]:
            orders, expected = projections(exps, rays, cones)
            actual, stats = transform_batch(exps, rays, cones, orders, "cpu")
            self.assertEqual(actual, expected)
            self.assertEqual(stats["backend"], "cpu")

    @unittest.skipUnless(gpu_devices(), "OpenCL GPU unavailable")
    def test_gpu_exact(self):
        for d in gpu_devices():
            for exps, rays, cones in [example(), random_batch(3), random_batch(4), random_batch(8)]:
                orders, expected = projections(exps, rays, cones)
                with patch.dict(os.environ, RESOLUTION_GPU_INDEX=str(d["index"]), RESOLUTION_GPU_BACKEND="opencl"):
                    actual, stats = transform_batch(exps, rays, cones, orders, "gpu")
                self.assertEqual(actual, expected)
                self.assertEqual(stats["backend"], "opencl")
                self.assertEqual(stats["device"]["name"], d["name"])

    def test_triton_exact(self):
        devices = [d for d in available_devices() if d["backend"] == "triton"]
        if not devices:
            self.skipTest("No Triton-capable GPU")
        for device in devices:
            with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton", RESOLUTION_GPU_INDEX=str(device["index"])):
                for dimension in range(2, 9):
                    exps, rays, cones = random_batch(dimension)
                    orders, expected = projections(exps, rays, cones)
                    # Many cones use the temporary ray table. A single cone
                    # uses the direct kernel with the same input buffers.
                    for count in [1, len(cones)]:
                        with self.subTest(device=device["name"], dimension=dimension, cones=count):
                            actual, stats = transform_batch(exps, rays, cones[:count], orders, "gpu")
                            self.assertEqual(stats["backend"], "triton", stats)
                            self.assertEqual(stats["device"], device)
                            self.assertEqual(actual, expected[:count])
                limit = 2**31 - 1
                actual, _ = transform_batch([(limit, 0)], [(1, 0)], [(0, 0)], [0], "gpu")
                self.assertEqual(actual, [[(limit, limit)]])
                with self.assertRaisesRegex(ValueError, "ray order"):
                    transform_batch([(0, 1)], [(1, 0)], [(0, 0)], [1], "gpu")

    @unittest.skipUnless(os.getenv("TRITON_INTERPRET") == "1", "Triton interpreter disabled")
    def test_triton_interpreter(self):
        try:
            import torch
            from engine.gpu_charts import transform
        except ImportError:
            self.skipTest("Triton interpreter and PyTorch required")
        # Exercise the real transfer/launch code with host tensors. Only the
        # device-selection context needs replacing in Triton's interpreter.
        with patch.object(torch.cuda, "device", return_value=nullcontext()):
            for dimension in range(2, 9):
                exps, rays, cones = random_batch(dimension)
                exps = exps[:37]
                for group in [cones[:1], cones[:23]]:
                    orders, expected = projections(exps, rays, group)
                    inputs = [_pack(chain.from_iterable(g)) for g in (exps, rays, group)] + [_pack(orders)]
                    out = (c_int32 * (dimension*len(exps)*len(group)))()
                    transform(inputs, out, len(exps), dimension, "cpu")
                    self.assertEqual(list(out), [x for chart in expected for vector in chart for x in vector])

    def test_unavailable_triton_preserves_cpu_result(self):
        exps, rays, cones = example()
        orders, expected = projections(exps, rays, cones)
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton"), \
             patch("engine.chart_batch.triton_device", return_value=None), \
             patch("engine.chart_batch._gpu_function") as opencl:
            actual, stats = transform_batch(exps, rays, cones, orders, "gpu")
        opencl.assert_not_called()
        self.assertEqual(actual, expected)
        self.assertEqual(stats["backend"], "cpu_after_triton")

    def test_auto_threshold_skips_triton(self):
        exps, rays, cones = example()
        orders, expected = projections(exps, rays, cones)
        with patch.dict(os.environ, RESOLUTION_GPU_MIN_PAIRS=str(len(exps)*len(cones) + 1)), \
             patch("engine.chart_batch.triton_device") as lookup:
            actual, stats = transform_batch(exps, rays, cones, orders, "auto")
        lookup.assert_not_called()
        self.assertEqual(stats["backend"], "cpu")
        self.assertEqual(actual, expected)

    def test_fallback_and_bounds(self):
        exps = [(0, 0, 1), (1, 0, 0)]
        rays = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        cone = [(0, 1, 2)]
        with patch.dict(os.environ, RESOLUTION_GPU_INDEX="999"):
            actual, stats = transform_batch(exps, rays, cone, [0,0,0], "gpu")
        self.assertEqual(stats["backend"], "cpu_after_opencl")
        self.assertEqual(actual, [exps])
        with self.assertRaises(OverflowError):
            transform_batch([(2**31-1,0,0)], [(2,0,0)], [(0,0,0)], [0], "cpu")
        with self.assertRaises(ValueError):
            transform_batch(exps, rays, cone, [1,0,0], "cpu")
        with self.assertRaises(ValueError):
            transform_batch(exps, rays, [(0,1,3)], [0,0,0], "cpu")


if __name__ == "__main__":
    unittest.main()
