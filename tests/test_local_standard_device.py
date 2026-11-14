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
    def test_large_positive_dimensional_requests_bypass_python_translation(self):
        n = 2**64+7
        for compute in ("cpu", "auto", "gpu"):
            with self.subTest(compute=compute), patch("engine.local_standard._translate",
                    side_effect=AssertionError("expanded a wide point shift before native dispatch")):
                request = {"ring": {"variables": ["x", "y", "z"]}, "compute": compute,
                           "operands": {"ideal": [f"x^{n}*z-y^2"], "point": ["1","0","0"]}}
                basis = dispatch({**request, "operation": "local-standard-basis"})["result"]
                self.assertTrue(basis["generators"])
                nf = dispatch({**request, "operation": "local-normal-form",
                               "operands": {**request["operands"], "polynomial": "z"}})["result"]
                self.assertEqual((nf["reduction"], nf["unit"], nf["remainder"]), ("standard", f"x^{n}", "y^2"))
                # The large shift can belong entirely to the query. Its
                # centered terms involve only x, which is free modulo (y+z)².
                free = dispatch({**request, "operation": "local-normal-form",
                                 "operands": {"ideal": ["(y+z)^2"], "point": ["1","0","0"],
                                              "polynomial": f"x^{n}"}})["result"]
                self.assertEqual((free["unit"], free["remainder"]), ("1", f"x^{n}"))

    @unittest.skipUnless(os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1",
                         "Set ATLAS_TEST_ALGEBRA_GPU=1 to require a physical GPU")
    def test_wide_shifted_basis_and_member_on_device(self):
        exponent = 2**64+7
        a = f"(x^{exponent}-1+2*(y-1/2))"
        b = f"(3*(x^{exponent}-1)+y-1/2)"
        ideal = [f"{a}^3", f"{b}^3*(5*{b}-4)"]
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                operands = {"ideal": ideal, "point": ["1", "1/2"]}
                if operation == "local-normal-form":
                    operands["polynomial"] = f"{a}^3+{b}^3"
                request = {"operation": operation, "ring": {"variables": ["x", "y"]}, "operands": operands}
                cpu = dispatch({**request, "compute": "cpu"})
                gpu = dispatch({**request, "compute": "gpu"})
                self.assertEqual(gpu["result"], cpu["result"])
                result = gpu["result"]
                self.assertEqual(result["point"], operands["point"])
                if operation == "local-normal-form":
                    self.assertEqual(result["remainder"], "0")
                    self.assertEqual(value_at(result["unit"], (1, Fraction(1,2))), 1)
                else:
                    self.assertTrue(result["generators"])
                    self.assertTrue(all(value_at(g, (1, Fraction(1,2))) == 0 for g in result["generators"]))
                stages = gpu["execution"]["stages"]
                self.assertTrue(any(s["operation"] == "groebner_replay" and s["backend"] in ("opencl", "triton")
                                    and s["primes"] > 0 and s["device"] for s in stages), stages)
                self.assertTrue(any(s["operation"] == "groebner_certification" and
                                    s["backend"] == "cpu" and s["bases"] > 0 for s in stages), stages)

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

    @unittest.skipUnless(os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1",
                         "Set ATLAS_TEST_ALGEBRA_GPU=1 to require a physical GPU")
    def test_oblique_parameter_basis_on_device(self):
        for exponent in (1001, 2**64 + 7):
            with self.subTest(exponent=exponent):
                u = f"(x^{exponent}*y^{exponent}+x+y-3+z)"
                v, w = "(x-y+z)", "(z+x-1)"
                request = {"operation": "local-standard-basis", "ring": {"variables": ["x", "y", "z"]},
                           "operands": {"ideal": [f"{u}*{v}", f"{u}*{w}", f"{v}*{w}"],
                                        "point": ["1", "1", "0"]}}
                cpu = dispatch({**request, "compute": "cpu"})["result"]
                self.assertEqual(len(cpu["generators"]), 3)
                self.assertEqual(dispatch({**request, "compute": "auto"})["result"], cpu)
                gpu = dispatch({**request, "compute": "gpu"})
                self.assertEqual(gpu["result"], cpu)
                stages = gpu["execution"]["stages"]
                self.assertTrue(any(s["operation"] == "groebner_replay" and s.get("device")
                                    and s["backend"] in ("opencl", "triton") and s["primes"] > 0
                                    for s in stages), stages)
                self.assertTrue(any(s["operation"] == "groebner_certification" and s["backend"] == "cpu"
                                    and s["bases"] == 1 for s in stages), stages)

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

    def test_gpu_witness_for_monomial_basis_derived_from_linear_forms(self):
        request = {"operation": "local-normal-form", "ring": {"variables": ["x", "y"]},
                   "operands": {"ideal": ["x+y", "x-y"], "polynomial": "(1+x+y)^12"}}
        cpu = dispatch(dict(request, compute="cpu"))["result"]
        self.assertEqual((cpu["unit"], cpu["remainder"], cpu["reduction"]), ("1", "1", "standard"))
        # The derived basis is (x,y). Only the constant among the 91 terms survives.
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="opencl",
                        RESOLUTION_GPU_INDEX=str(2**31 - 1)), \
             patch("engine.gpu_normal_form._opencl_witnesses", return_value=[-1] * 91) as launch:
            record = dispatch(dict(request, compute="gpu"))
        launch.assert_called_once()
        terms, divisors, _ = launch.call_args.args
        self.assertEqual(len(terms), 91)
        self.assertCountEqual(divisors, [(1, 0), (0, 1)])
        self.assertIn("Host rejected", " ".join(record["execution"]["fallback_reasons"]))
        self.assertEqual(record["result"], cpu)

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
