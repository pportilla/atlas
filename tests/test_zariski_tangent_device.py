"""Public rational Jacobian rank, device selection and exact witness verification."""
import os
import unittest
from unittest.mock import patch

from engine.algebra import dispatch


OPERATIONS = ("zariski-tangent-space", "embedding-dimension")


def calculate(operation, compute, *, bad_prime=False, zero=False):
    if bad_prime:
        # The primitive integer rows are equal modulo 1,000,000,007 but
        # their determinant over QQ is that prime. The true rank is two.
        names, point, ideal = ["x", "y"], [0, 0], ["x+y", "x+1000000008*y"]
    elif zero:
        names, point, ideal = ["x", "y"], [0, 0], ["x^2", "y^3"]
    else:
        names = ["a", "x", "c", "y", "z"]
        point = [0, "1/2", 0, "-2/3", 1]
        x, y, z = "(x-1/2)", "(y+2/3)", "(z-1)"
        f, g = f"({x}+2*{y})", f"({y}-3*{z})"
        # Rectangular rank two matrix with its first and third columns zero.
        ideal = [f + "+a^2", g + "+c^2", f + "+7*" + g]
    return dispatch({"operation": operation, "compute": compute, "ring": {"variables": names},
                     "operands": {"ideal": ideal, "point": point}})


class ZariskiTangentDeviceTests(unittest.TestCase):
    def test_missing_device_and_zero_jacobian(self):
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="opencl", RESOLUTION_GPU_INDEX=str(2**31-1)):
            for operation in OPERATIONS:
                with self.subTest(operation=operation):
                    cpu = calculate(operation, "cpu")["result"]
                    if operation == "zariski-tangent-space":
                        self.assertEqual(cpu["equations"], ["x+6*z", "y-3*z"])
                        self.assertEqual(cpu["basis"], [["1","0","0","0","0"], ["0","0","1","0","0"],
                                                       ["0","-6","0","3","1"]])
                    else:
                        self.assertEqual(cpu["value"], 3)
                    for policy in ("auto", "gpu"):
                        report = calculate(operation, policy)
                        self.assertEqual(report["result"], cpu)
                        self.assertEqual(report["execution"]["backend"], "cpu")
                        if policy == "gpu":
                            self.assertTrue(report["execution"]["note"])
                    empty = calculate(operation, "gpu", zero=True)
                    self.assertEqual(empty["result"]["jacobian_rank"], 0)
                    if operation == "zariski-tangent-space":
                        self.assertEqual(empty["result"]["basis"], [["1","0"], ["0","1"]])
                    self.assertFalse(any(s["operation"] == "jacobian_row_selection"
                                         for s in empty["execution"]["stages"]))

    @unittest.skipUnless(os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1",
                         "Set ATLAS_TEST_ALGEBRA_GPU=1 to require a physical GPU")
    def test_rectangular_device_selection_and_bad_prime_recovery(self):
        for operation in OPERATIONS:
            for bad_prime in (False, True):
                with self.subTest(operation=operation, bad_prime=bad_prime):
                    cpu = calculate(operation, "cpu", bad_prime=bad_prime)["result"]
                    self.assertEqual(calculate(operation, "auto", bad_prime=bad_prime)["result"], cpu)
                    gpu = calculate(operation, "gpu", bad_prime=bad_prime)
                    self.assertEqual(gpu["result"], cpu)
                    self.assertEqual(cpu["jacobian_rank"], 2)
                    stages = gpu["execution"]["stages"]
                    selection, = [s for s in stages if s["operation"] == "jacobian_row_selection"]
                    check, = [s for s in stages if s["operation"] == "certify_jacobian_row_space"]
                    self.assertIn(selection["backend"], ("opencl", "triton"))
                    self.assertTrue(selection["device"])
                    self.assertEqual(selection["prime"], 1_000_000_007)
                    self.assertEqual((selection["rows"], selection["columns"]), (2, 2) if bad_prime else (3, 5))
                    self.assertEqual(selection["candidate_rank"], 1 if bad_prime else 2)
                    self.assertEqual((check["backend"], check["candidate_rank"]), ("cpu", selection["candidate_rank"]))
                    self.assertLess(stages.index(selection), stages.index(check))
                    if bad_prime:
                        self.assertIn("Recomputed over QQ", " ".join(gpu["execution"]["fallback_reasons"]))
                        if operation == "zariski-tangent-space":
                            self.assertEqual(cpu["basis"], [])
                        else:
                            self.assertEqual(cpu["value"], 0)
