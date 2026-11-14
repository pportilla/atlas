"""Public compute policies for local algebras and quotient bases.

Set ATLAS_TEST_ALGEBRA_GPU=1 on a physical GPU to require verified device work.
"""
import os
import unittest
from unittest.mock import patch

from engine.algebra import dispatch
from engine.polynomial import parse_expression


def translated_requests():
    # Singular's vdim(std(jacob(f))) is 12, and adding f drops it to 11.
    x, y = "(x-1/2)", "(y+2/3)"
    f = f"{x}^4+{y}^5+{x}^2*{y}^3"
    derivatives = [f"4*{x}^3+2*{x}*{y}^3", f"5*{y}^4+3*{x}^2*{y}^2"]
    for op in ("local-quotient-basis", "local-jacobian-algebra", "milnor-number", "tjurina-number"):
        operands = {"ideal": derivatives} if op == "local-quotient-basis" else {"polynomial": f}
        yield {"operation": op, "ring": {"variables": ["x", "y"]},
               "operands": dict(operands, point=["2/4", "-6/9"])}, derivatives


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

    def test_translated_local_algebras_and_unavailable_gpu(self):
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="opencl",
                        RESOLUTION_GPU_INDEX=str(2**31 - 1)):
            for request, derivatives in translated_requests():
                op = request["operation"]
                with self.subTest(operation=op):
                    cpu = dispatch(dict(request, compute="cpu"))
                    result = cpu["result"]
                    self.assertEqual(result["scope"], "point")
                    self.assertEqual(result["point"], ["1/2", "-2/3"])
                    self.assertTrue(result["finite"])
                    if op in ("milnor-number", "tjurina-number"):
                        self.assertEqual(result["value"], 11 if op == "tjurina-number" else 12)
                        self.assertNotIn("basis", result)
                    else:
                        self.assertEqual(result["dimension"], 12)
                        rows = [parse_expression(s, ["x", "y"]) for s in result["basis"]]
                        self.assertEqual(len(rows), 12)
                        self.assertIn(parse_expression("x-1/2", ["x", "y"]), rows)
                        self.assertIn(parse_expression("y+2/3", ["x", "y"]), rows)
                    if op == "local-jacobian-algebra":
                        self.assertEqual([parse_expression(g, ["x", "y"]) for g in result["generators"]],
                                         [parse_expression(g, ["x", "y"]) for g in derivatives])
                    for policy in ("auto", "gpu"):
                        record = dispatch(dict(request, compute=policy))
                        self.assertEqual(record["result"], result)
                        report = record["execution"]
                        self.assertEqual((report["requested"], report["backend"], report["device"]),
                                         (policy, "cpu", "CPU"))
                        if policy == "gpu":
                            self.assertTrue(report["note"])

    def test_large_exponent_at_fractional_point(self):
        # Shift only x, so the sparse 65-bit y exponent survives translation.
        # y^N and the square of the independent coordinate (x-1/2)+y
        # generate the Jacobian ideal; the quotient has length 2N.
        n = 2**64
        for op in ("milnor-number", "tjurina-number"):
            request = {"operation": op, "ring": {"variables": ["x", "y"]},
                       "operands": {"polynomial": f"(x-1/2+y)^3+y^{n+1}", "point": ["1/2", "0"]}}
            for policy in ("cpu", "auto", "gpu"):
                with self.subTest(operation=op, compute=policy):
                    record = dispatch(dict(request, compute=policy))
                    self.assertEqual(record["result"]["value"], 2*n)
                    self.assertEqual(record["result"]["point"], ["1/2", "0"])
                    self.assertNotIn("basis", record["result"])

    @unittest.skipUnless(os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1",
                         "Set ATLAS_TEST_ALGEBRA_GPU=1 to require a physical GPU")
    def test_physical_device_exact_results_and_modular_certification(self):
        for request, _ in (*requests(), *translated_requests()):
            operation = request["operation"]
            with self.subTest(operation=operation, point=request["operands"].get("point")):
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
