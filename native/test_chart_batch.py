"""Compare chart-major native exponent tables with Python integer dot products."""
import json
import os
from pathlib import Path
from random import Random
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.chart_batch import transform_batch
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
                with patch.dict(os.environ, RESOLUTION_GPU_INDEX=str(d["index"])):
                    actual, stats = transform_batch(exps, rays, cones, orders, "gpu")
                self.assertEqual(actual, expected)
                self.assertEqual(stats["backend"], "opencl")
                self.assertEqual(stats["device"]["name"], d["name"])

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
