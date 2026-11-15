"""Request-local device selection and accounting through the public dispatcher."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import os
from threading import Barrier
import unittest
from unittest.mock import patch

from engine import algebra_groebner
from engine.algebra import dispatch
from engine.gpu_devices import available_devices
from engine.native_library import _kernel


@unittest.skipUnless(_kernel() is not None, "requires the native rational kernel")
class AlgebraExecutionTests(unittest.TestCase):
    request = {"operation": "groebner", "ring": {"variables": ["x", "y"], "order": "lex"},
               "operands": {"ideal": ["x^2+2147483629*y-1", "x*y-2"]}}

    def overlap(self, backend):
        ready = Barrier(3)
        handle = algebra_groebner.handle

        def synchronized(*args, **kwargs):
            # Each request has installed its policy on its own native thread.
            ready.wait(timeout=15)
            return handle(*args, **kwargs)

        policies = ("gpu", "cpu", "auto")
        with patch.object(algebra_groebner, "handle", synchronized), ThreadPoolExecutor(3) as workers:
            runs = list(workers.map(lambda mode: dispatch(dict(self.request, compute=mode)), policies))
        for record, policy in zip(runs, policies):
            self.assertEqual(record["result"], runs[0]["result"])
            execution = record["execution"]
            self.assertEqual(execution["requested"], policy)
            self.assertFalse(execution["fallback_reasons"], execution)
            if policy != "gpu":
                self.assertEqual(execution["backend"], "cpu")
                self.assertTrue(all(s["backend"] == "cpu" for s in execution["stages"]))
            else:
                self.assertEqual(execution["backend"], backend)
                stage, = [s for s in execution["stages"] if s["operation"] == "groebner_replay"]
                self.assertEqual(stage["backend"], backend)
                self.assertGreater(stage["primes"], 0)
                if backend != "reference":
                    self.assertEqual(stage["device"]["backend"], backend)
                if hasattr(_kernel(), "atlas_groebner_certification_counts"):
                    proof, = [s for s in execution["stages"] if s["operation"] == "groebner_certification"]
                    self.assertEqual(proof["bases"], 1)

    def test_overlapping_reference_cpu_auto(self):
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="auto", RESOLUTION_GPU_INDEX="0",
                        ATLAS_GROEBNER_GPU="reference"):
            self.overlap("reference")

    def test_overlapping_physical_gpu_cpu_auto(self):
        devices = available_devices()
        if not devices:
            self.skipTest("requires a physical GPU")
        for device in devices:
            with self.subTest(device=device), patch.dict(os.environ, RESOLUTION_GPU_BACKEND=device["backend"],
                                                         RESOLUTION_GPU_INDEX=str(device["index"])):
                self.overlap(device["backend"])

    def test_separate_unavailable_index(self):
        cpu = dispatch(dict(self.request, compute="cpu"))
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="opencl", RESOLUTION_GPU_INDEX=str(2**31-1)):
            gpu = dispatch(dict(self.request, compute="gpu"))
        self.assertEqual(cpu["result"], gpu["result"])
        execution = gpu["execution"]
        self.assertEqual(execution["backend"], "cpu")
        self.assertTrue(execution["fallback_reasons"])
        self.assertTrue(any(s["operation"] == "groebner_replay" and s["backend"] == "cpu" and s["primes"] > 0
                            for s in execution["stages"]))

    def test_failed_modular_route_retries_certified_work_on_cpu(self):
        # Coefficient swell drives even the ordinary solver to modular images.
        # Simulate decline of the first route and observe the native CPU retry.
        data = {**self.request, "ring": {"variables": ["x", "y"], "order": "grevlex"},
                "operands": {"ideal": [f"x^2+{2**260+1}*y", "x*y+1"]}}
        expected = dispatch(dict(data, compute="cpu"))
        with patch.object(_kernel(), "atlas_groebner_modular", return_value=ctypes.c_void_p()), \
                patch.dict(os.environ, RESOLUTION_GPU_BACKEND="auto", ATLAS_GROEBNER_GPU="reference"):
            record = dispatch(dict(data, compute="gpu"))
        self.assertEqual(record["result"], expected["result"])
        report = record["execution"]
        self.assertEqual(report["backend"], "cpu")
        self.assertTrue(any("Modular basis declined" in reason for reason in report["fallback_reasons"]))
        self.assertTrue(any(s["operation"] == "groebner_replay" and s["primes"] > 0
                            for s in report["stages"]))

    def test_repeated_labeled_bases_have_disjoint_counts(self):
        from engine.algebra_execution import algebra_execution, stage_label
        from engine.native_polynomial import Symbol
        x, y = Symbol("x"), Symbol("y")
        with algebra_execution("gpu", backend="reference") as run:
            for label in ("first_basis", "second_basis"):
                with stage_label(label):
                    algebra_groebner.reduced_basis((x**2+2147483629*y-1, x*y-2), (x,y), order="lex")
            replays = [s for s in run.report()["stages"] if s["operation"] == "groebner_replay"]
            self.assertEqual([s["label"] for s in replays], ["first_basis", "second_basis"])
            self.assertEqual(replays[0]["primes"], replays[1]["primes"])
            self.assertGreater(replays[0]["primes"], 0)
            self.assertEqual(run.report()["stages"], run.report()["stages"])
