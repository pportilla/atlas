import os
import unittest
from unittest.mock import patch

from engine.gpu_devices import available_devices
from engine.resolve import resolve


class GPUResolution(unittest.TestCase):
    def test_device_work_preserves_owned_atlas_and_zetas(self):
        devices = available_devices()
        if not devices:
            self.skipTest("No GPU device available")
        request = dict(polynomial="x^5+z*y^3+z^15+t^4*x+t^8", variables=list("xyzt"),
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
