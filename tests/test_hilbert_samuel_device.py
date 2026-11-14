"""Public compute policies for Hilbert–Samuel functions and multiplicities."""
import os
import unittest
from unittest.mock import patch

from engine.algebra import dispatch


OPERATIONS = ("hilbert-samuel-function", "hilbert-samuel-multiplicity")


def calculate(operation, compute):
    # The independent linear forms u,v vanish at the point. Since 5v-4
    # is a local unit, the associated graded algebra has basis u^i*v^j
    # with 0 <= i,j < 3. Its Hilbert series is (1+t+t^2)^2.
    x, y = "(x-1/2)", "(y+2/3)"
    u, v = f"({x}+2*{y})", f"(3*{x}+{y})"
    operands = {"ideal": [f"{u}^3", f"{v}^3*(5*{v}-4)"], "point": ["1/2", "-2/3"]}
    if operation == "hilbert-samuel-function":
        operands["degrees"] = list(range(6))
    return dispatch({"operation": operation, "operands": operands,
                     "ring": {"variables": ["x", "y"]}, "compute": compute})


class HilbertSamuelDeviceTests(unittest.TestCase):
    def test_expected_lengths_and_missing_gpu(self):
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="opencl", RESOLUTION_GPU_INDEX=str(2**31-1)):
            for operation in OPERATIONS:
                with self.subTest(operation=operation):
                    cpu = calculate(operation, "cpu")
                    result = cpu["result"]
                    self.assertEqual(cpu["execution"]["backend"], "cpu")
                    self.assertEqual((result["dimension"], result["point"]), (0, ["1/2", "-2/3"]))
                    if operation == "hilbert-samuel-function":
                        self.assertEqual(result["values"], [1, 3, 6, 8, 9, 9])
                        self.assertEqual(result["graded_values"], [1, 2, 3, 2, 1, 0])
                        self.assertEqual(result["multiplicity"], 9)
                    else:
                        self.assertEqual(result["value"], 9)
                    for policy in ("auto", "gpu"):
                        record = calculate(operation, policy)
                        self.assertEqual(record["result"], result)
                        report = record["execution"]
                        self.assertEqual((report["requested"], report["backend"], report["device"]),
                                         (policy, "cpu", "CPU"))
                        if policy == "gpu":
                            self.assertTrue(report["note"])

    @unittest.skipUnless(os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1",
                         "Set ATLAS_TEST_ALGEBRA_GPU=1 to require a physical GPU")
    def test_physical_modular_work_and_certification_for_both_operations(self):
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                cpu = calculate(operation, "cpu")["result"]
                self.assertEqual(calculate(operation, "auto")["result"], cpu)
                record = calculate(operation, "gpu")
                self.assertEqual(record["result"], cpu)
                stages = [s for s in record["execution"]["stages"] if s.get("label") == "local_standard_basis"]
                device = [s for s in stages if s["backend"] in ("opencl", "triton") and s.get("device")]
                self.assertTrue(any(s.get("primes", 0) or s.get("pairs", 0) for s in device), stages)
                self.assertTrue(any(s["operation"] == "groebner_certification" and s["backend"] == "cpu"
                                    and s["bases"] > 0 and s["attempts"] > 0 for s in stages), stages)
