import os
import unittest
from fractions import Fraction
from unittest.mock import patch

from engine.gpu_devices import available_devices
from engine.resolve import resolve


class GPUResolution(unittest.TestCase):
    def test_wide_newton_and_binomial_chart_certificates(self):
        from engine.binomial_backend import resolve_binomial
        from engine.newton_backend import resolve_newton
        from engine.newton_faces import face_search
        degree = 2**80 + 5
        newton = {(degree,0,0): Fraction(1), (0,degree,0): Fraction(1), (0,0,degree): Fraction(1)}
        binomial = {(degree,0,0): Fraction(1), (0,1,0): Fraction(1)}
        devices = available_devices()
        with face_search(True):
            for solve, polynomial in ((resolve_newton, newton), (resolve_binomial, binomial)):
                reference = solve(polynomial, list("xyz"), "embedded", compute="cpu")
                self.assertEqual(reference["status"], "resolved")
                self.assertTrue(any(str(degree) in chart["strict_transform"] for chart in reference["charts"]))
                for device in devices:
                    with self.subTest(solver=solve.__name__, backend=device["backend"]), patch.dict(
                        os.environ, RESOLUTION_GPU_BACKEND=device["backend"], RESOLUTION_GPU_INDEX=str(device["index"])
                    ):
                        actual = solve(polynomial, list("xyz"), "embedded", compute="gpu")
                        stats = actual["certificate"]["chart_batch"]
                        self.assertEqual(stats["backend"], device["backend"], stats)
                        self.assertEqual(stats["limb_batches"], 2)
                        for field in ("charts", "fan", "dual_complex"):
                            self.assertEqual(actual[field], reference[field])

    def test_device_work_preserves_owned_atlas_and_zetas(self):
        devices = available_devices()
        if not devices:
            self.skipTest("No GPU device available")
        # The four pure powers give a convenient Newton polyhedron. Both
        # resolution modes use chart transforms on its regular fan.
        request = dict(polynomial="x^2+y^3+z^5+t^7", variables=list("xyzt"),
                       search="quick", monodromy=True)
        fields = ("atlas", "charts", "dual_complex", "fan", "root_families",
                  "monodromy_zeta", "topological_zeta")
        for mode in ("embedded", "intrinsic"):
            cpu = resolve(dict(request, mode=mode, compute="cpu"))
            self.assertEqual(cpu["status"], "resolved")
            for device in devices:
                backend = device["backend"]
                with self.subTest(mode=mode, device=device), patch.dict(
                    os.environ, RESOLUTION_GPU_BACKEND=backend, RESOLUTION_GPU_INDEX=str(device["index"])
                ):
                    gpu = resolve(dict(request, mode=mode, compute="gpu"))
                    self.assertEqual(gpu["status"], "resolved")
                    self.assertEqual(gpu["execution"]["chart_backend"], backend, gpu["execution"])
                    stage = next(s for s in gpu["execution"]["stages"] if s["operation"] == "chart_transform")
                    self.assertEqual((stage["backend"], stage["device"]["index"]), (backend, device["index"]))
                    for field in fields:
                        self.assertEqual(gpu[field], cpu[field], field)

    def test_cpu_only_resolution_under_gpu_request(self):
        result = resolve(dict(polynomial="x+y", variables=["x", "y"], compute="gpu"))
        self.assertEqual(result["execution"]["device"], "CPU")
        self.assertEqual(result["execution"]["stages"], [])
