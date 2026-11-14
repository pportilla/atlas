"""Exact public local standard bases and weak normal forms across compute policies."""
from fractions import Fraction
import os
import unittest
from unittest.mock import patch

from engine.algebra import dispatch
from engine.polynomial import parse_expression


OPERATIONS = ("local-standard-basis", "local-normal-form")
POINT = (Fraction(1, 2), Fraction(-2, 3))


def calculate(operation, compute):
    # At POINT the independent linear forms u and v both vanish. Here 5v-4
    # is a unit, so u^3+v^3 belongs to the local ideal. It is nonzero at
    # u=0, v=4/5, where both generators vanish, ruling out global membership.
    u = "((x-1/2)+2*(y+2/3))"
    v = "(3*(x-1/2)+(y+2/3))"
    operands = {"ideal": [f"{u}^3", f"{v}^3*(5*{v}-4)"],
                "point": [str(q) for q in POINT]}
    if operation == "local-normal-form":
        operands["polynomial"] = f"{u}^3+{v}^3"
    return dispatch({"operation": operation, "ring": {"variables": ["x", "y"]},
                     "operands": operands, "compute": compute})


def value_at(polynomial, point):
    x, y = point
    return sum(c * x**i * y**j for (i, j), c in parse_expression(polynomial, ["x", "y"]).items())


def associate_division(compute):
    # y^8(1+x) is associated to y^8 at (1/2,-2/3) in centered
    # coordinates. Only the first eight of the 64 centered terms survive.
    return dispatch({
        "operation": "local-normal-form", "compute": compute,
        "ring": {"variables": ["x", "y"]},
        "operands": {"ideal": ["(y+2/3)^8*(x+1/2)"], "point": ["1/2", "-2/3"],
                     "polynomial": "(1+(y+2/3))^63/7"},
    })


class LocalStandardDeviceTests(unittest.TestCase):
    def test_cpu_auto_and_unavailable_gpu(self):
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="opencl",
                        RESOLUTION_GPU_INDEX=str(2**31 - 1)):
            for operation in OPERATIONS:
                with self.subTest(operation=operation):
                    cpu = calculate(operation, "cpu")
                    result = cpu["result"]
                    self.assertEqual(result["point"], [str(q) for q in POINT])
                    self.assertEqual(cpu["execution"]["backend"], "cpu")
                    if operation == "local-standard-basis":
                        self.assertTrue(result["generators"])
                        for g in result["generators"]:
                            self.assertEqual(value_at(g, POINT), 0)
                    else:
                        self.assertEqual(result["remainder"], "0")
                        self.assertNotEqual(value_at(result["unit"], POINT), 0)
                    for policy in ("auto", "gpu"):
                        record = calculate(operation, policy)
                        self.assertEqual(record["result"], result)
                        report = record["execution"]
                        self.assertEqual(report["requested"], policy)
                        self.assertEqual(report["backend"], "cpu")
                        self.assertEqual(report["device"], "CPU")
                        if policy == "gpu":
                            self.assertTrue(report["note"])

    def test_rejected_local_division_witness_retries_on_cpu(self):
        cpu = associate_division("cpu")["result"]
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="opencl"), \
             patch("engine.gpu_normal_form._opencl_witnesses", return_value=[-1] * 64) as launch:
            self.assertEqual(associate_division("auto")["result"], cpu)
            launch.assert_not_called()
            record = associate_division("gpu")
            launch.assert_called_once()
        self.assertEqual(record["result"], cpu)
        self.assertEqual(record["execution"]["backend"], "cpu")
        self.assertIn("Host rejected", " ".join(record["execution"]["fallback_reasons"]))

    @unittest.skipUnless(os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1",
                         "Set ATLAS_TEST_ALGEBRA_GPU=1 to require a physical GPU")
    def test_local_associate_division_on_device(self):
        from math import comb
        cpu = associate_division("cpu")["result"]
        gpu = associate_division("gpu")
        self.assertEqual(gpu["result"], cpu)
        expected = "+".join(f"{comb(63, i)}*(y+2/3)^{i}/7" for i in range(8))
        self.assertEqual(parse_expression(cpu["remainder"], ["x", "y"]),
                         parse_expression(expected, ["x", "y"]))
        self.assertEqual((cpu["unit"], cpu["reduction"]), ("1", "standard"))
        witnesses = [s for s in gpu["execution"]["stages"] if s["backend"] in ("opencl", "triton")
                     and s.get("device") and s["operation"] == "monomial_normal_form"
                     and s.get("label") == "local_normal_form" and s["host_verified"]]
        self.assertEqual([(s["terms"], s["divisors"], s["removed"]) for s in witnesses], [(64, 1, 56)])

    @unittest.skipUnless(os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1",
                         "Set ATLAS_TEST_ALGEBRA_GPU=1 to require a physical GPU")
    def test_both_operations_execute_certified_modular_work_on_device(self):
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                cpu = calculate(operation, "cpu")["result"]
                self.assertEqual(calculate(operation, "auto")["result"], cpu)
                gpu = calculate(operation, "gpu")
                self.assertEqual(gpu["result"], cpu)
                stages = gpu["execution"]["stages"]
                physical = [s for s in stages if s["backend"] in ("opencl", "triton")
                            and s.get("device")]
                self.assertTrue(any(s.get("primes", 0) > 0 or s.get("pairs", 0) > 0
                                    for s in physical), stages)
                self.assertTrue(any(s["operation"] == "groebner_certification" and
                                    s["backend"] == "cpu" and s["bases"] > 0
                                    for s in stages), stages)
