"""Certified modular device work for both tangent cone operations at a rational point."""
import os
import unittest
from unittest.mock import patch

from engine.algebra import dispatch


OPERATIONS = ("tangent-cone", "tangent-cone-hilbert-series")
RING = {"variables": ["x", "y"], "order": "grevlex"}


def calculate(operation, compute):
    x, y = "(x-1/2)", "(y+2/3)"
    u, v = f"({x}+2*{y})", f"(3*{x}+{y})"
    return dispatch({"operation": operation, "ring": RING, "compute": compute,
                     "operands": {"ideal": [f"{u}^3", f"{v}^3*(5*{v}-4)"],
                                  "point": ["1/2", "-2/3"]}})


def reduced_generators(generators):
    return dispatch({"operation": "groebner", "ring": RING, "compute": "cpu",
                     "operands": {"ideal": generators}})["result"]["generators"]


class TangentConeDeviceTests(unittest.TestCase):
    def test_exact_ideal_series_and_unavailable_device(self):
        # The two centered linear forms are independent and 5v-4 is a unit.
        # Thus the tangent ideal in direction coordinates is (u^3,v^3),
        # with numerator (1-t^3)^2 over the ambient denominator (1-t)^2.
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="opencl", RESOLUTION_GPU_INDEX=str(2**31-1)):
            for operation in OPERATIONS:
                with self.subTest(operation=operation):
                    cpu = calculate(operation, "cpu")
                    result = cpu["result"]
                    self.assertEqual((result["scope"], result["point"]), ("point", ["1/2", "-2/3"]))
                    self.assertEqual((result["dimension"], result["multiplicity"]), (0, 9))
                    self.assertEqual(cpu["execution"]["backend"], "cpu")
                    if operation == "tangent-cone":
                        self.assertEqual(result["coordinate_system"], "tangent")
                        self.assertEqual(reduced_generators(result["generators"]),
                                         reduced_generators(["(x+2*y)^3", "(3*x+y)^3"]))
                    else:
                        self.assertEqual(result["denominator_power"], 2)
                        self.assertEqual(result["numerator"], [{"degree": 0, "coefficient": 1},
                                                               {"degree": 3, "coefficient": -2},
                                                               {"degree": 6, "coefficient": 1}])
                    for policy in ("auto", "gpu"):
                        candidate = calculate(operation, policy)
                        self.assertEqual(candidate["result"], result)
                        report = candidate["execution"]
                        self.assertEqual((report["requested"], report["backend"], report["device"]),
                                         (policy, "cpu", "CPU"))
                        if policy == "gpu":
                            self.assertTrue(report["note"])

    @unittest.skipUnless(os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1",
                         "Set ATLAS_TEST_ALGEBRA_GPU=1 to require a physical GPU")
    def test_physical_local_basis_work_for_both_operations(self):
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                cpu = calculate(operation, "cpu")["result"]
                self.assertEqual(calculate(operation, "auto")["result"], cpu)
                record = calculate(operation, "gpu")
                self.assertEqual(record["result"], cpu)
                stages = [s for s in record["execution"]["stages"] if s.get("label") == "local_standard_basis"]
                devices = [s for s in stages if s["backend"] in ("opencl", "triton") and s.get("device")]
                self.assertTrue(any(s.get("primes", 0) > 0 or s.get("pairs", 0) > 0 for s in devices), stages)
                self.assertTrue(any(s["operation"] == "groebner_certification" and s["backend"] == "cpu"
                                    and s["bases"] > 0 and s["attempts"] > 0 for s in stages), stages)
