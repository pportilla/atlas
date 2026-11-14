"""Public compute policies for local algebras and quotient bases.

Set ATLAS_TEST_ALGEBRA_GPU=1 on a physical GPU to require verified device work.
"""
import os
import unittest
from unittest.mock import patch

from engine.algebra import dispatch


def requests():
    # In the independent coordinates u=x+2y and v=3x+y the equation is
    # u^4+v^4(v-1). Its Jacobian ideal is (u^3, v^3(5v-4)), so its local
    # length is 3*3 and its global length is 3*4. The equation is in the
    # localized Jacobian ideal since 5v-4 is invertible at the origin.
    u, v = "(x+2*y)", "(3*x+y)"
    ideal = {"ideal": [f"{u}^3", f"{v}^3*(5*{v}-4)"]}
    polynomial = {"polynomial": f"{u}^4+{v}^4*({v}-1)"}
    for operation, operands, expected in (
        ("quotient-basis", ideal, 12),
        ("local-quotient-basis", ideal, 9),
        ("local-jacobian-algebra", polynomial, 9),
        ("milnor-number", polynomial, 9),
        ("tjurina-number", polynomial, 9),
    ):
        yield {"operation": operation, "ring": {"variables": ["x", "y"]},
               "operands": operands}, expected


class LocalAlgebraDeviceTests(unittest.TestCase):
    def test_cpu_auto_and_unavailable_gpu_agree(self):
        # Use an unassigned OpenCL index to exercise the same fallback on hosts
        # with a device and on build workers with no OpenCL implementation.
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="opencl",
                        RESOLUTION_GPU_INDEX=str(2**31 - 1)):
            for request, expected in requests():
                with self.subTest(operation=request["operation"]):
                    cpu = dispatch(dict(request, compute="cpu"))
                    result = cpu["result"]
                    self.assertTrue(result["finite"])
                    if request["operation"] in ("milnor-number", "tjurina-number"):
                        self.assertEqual(result["value"], expected)
                    else:
                        self.assertEqual(result["dimension"], expected)
                        self.assertEqual(len(result["basis"]), expected)
                    self.assertEqual(cpu["execution"]["backend"], "cpu")
                    for policy in ("auto", "gpu"):
                        record = dispatch(dict(request, compute=policy))
                        self.assertEqual(record["result"], result)
                        report = record["execution"]
                        self.assertEqual(report["requested"], policy)
                        self.assertEqual(report["backend"], "cpu")
                        self.assertEqual(report["device"], "CPU")
                        self.assertFalse(any(stage["backend"] in ("opencl", "triton")
                                             for stage in report["stages"]))
                        if policy == "gpu":
                            self.assertTrue(report["note"])

    @unittest.skipUnless(os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1",
                         "Set ATLAS_TEST_ALGEBRA_GPU=1 to require a physical GPU")
    def test_physical_device_exact_results_and_modular_certification(self):
        for request, _ in requests():
            operation = request["operation"]
            with self.subTest(operation=operation):
                cpu = dispatch(dict(request, compute="cpu"))["result"]
                for policy in ("auto", "gpu"):
                    record = dispatch(dict(request, compute=policy))
                    self.assertEqual(record["result"], cpu)
                    self.assertEqual(record["execution"]["requested"], policy)
                    if policy == "gpu":
                        stages = record["execution"]["stages"]
                        device = [s for s in stages if s["backend"] in ("opencl", "triton")
                                  and s.get("device")]
                        self.assertTrue(any(s.get("primes", 0) > 0 or s.get("pairs", 0) > 0
                                            for s in device), stages)
                        self.assertTrue(any(s["operation"] == "groebner_certification" and
                                            s["backend"] == "cpu" and s["bases"] > 0
                                            for s in stages), stages)
