"""Verify lifted factors in ZZ[x] modulo the squared prime using CPU and Triton."""

import os
import unittest
from unittest.mock import patch
from engine.algebra import dispatch
from engine.gpu_devices import triton_device
from engine.session import new_session
from engine.session_help import algebra_value_help
from tests.cli_support import run_script


def request(f, g, h, p, variables=("x",), mode="cpu"):
    return {
        "operation": "hensel-factors",
        "ring": {"variables": list(variables)},
        "compute": mode,
        "operands": {"polynomial": f, "factors": [g, h], "modulus": str(p)},
    }


class HenselFactorsTests(unittest.TestCase):
    def test_quadratic_and_ownership(self):
        script = (
            'x, y=coordinates("x","y")\nP=x.parent\n'
            "H=hensel_factors(y^2-2,y-3,y+3,7)\nprint(H.left,H.right,H.modulus,H.left.parent==P)\n"
        )
        proc = run_script(script)
        self.assertEqual((proc.returncode, proc.stdout.strip()), (0, "y+39 y+10 49 True"), proc.stderr)
        session = new_session(None)
        list(
            session.run(
                script.split("print(")[0] + 'use_ring(P)\nC=hensel_factors(1,1,1,"7")\n'
                'X,Y=coordinates("x","y",fresh=True)'
            )
        )
        H = session.variables["H"]
        self.assertIs(session.variables["C"].left.parent, session.variables["P"])
        with self.assertRaisesRegex(Exception, "different"):
            list(session.run("H=hensel_factors(y^2-2,y-3,Y+3,7)"))
        self.assertIs(session.variables["H"], H)
        card = algebra_value_help("H", H, session.variables)
        for attr in ("left", "right", "modulus"):
            self.assertIn(f"H.{attr}", card)
            list(session.run(f"entry=H.{attr}"))
        self.assertIn("examples/polynomial-ideals.md#lift-two-coprime-polynomial-factors", card)

    def test_wide_sparse_degree_prime_and_units(self):
        n = 2**64 + 3
        p = 2**127 - 1
        for mode in ("cpu", "gpu"):
            result = dispatch(request(f"x^{n + 1}+{p + 1}*x", f"x^{n}+1", "x", p, mode=mode))["result"]
            self.assertEqual(
                (result["left"], result["right"], result["modulus"]),
                (f"x^{n}+{p + 1}", "x", str(p * p)),
            )
            loose = request("50*x^2-2", "7*x^3-6*x-3", "x+3", 7, mode=mode)
            self.assertEqual(dispatch(loose)["result"]["left"], "x+39")
            unit = dispatch(request("x^2-2", "1", "x^2-2", 7, mode=mode))["result"]
            self.assertEqual((unit["left"], unit["right"]), ("1", "x^2+47"))
            constant = dispatch(request("1", "1", "1", 2, variables=(), mode=mode))["result"]
            self.assertEqual(
                (constant["left"], constant["right"], constant["modulus"]), ("1", "1", "4")
            )

    def test_invalid_moduli_factors_and_congruences(self):
        cases = [
            (request("x^2-2", "x-3", "x+3", 1), "positive prime"),
            (request("x^2-2", "x-3", "x+3", 9), "prime"),
            (request("x^2-2", "x-3", "-x-3", 7), "monic"),
            (request("0", "0", "x", 7), "monic"),
            (request("8*x^2-2", "x-3", "x+3", 7), "target modulo"),
            (request("x^3-2", "x-3", "x+3", 7), "degree"),
            (request("x^2-2", "x^2-2", "y", 7, ("x", "y")), "one common variable"),
            (request("x^2+1/3", "x-3", "x+3", 7), "integer polynomial coefficients"),
            (request("x^2-3", "x-3", "x+3", 7), "do not multiply"),
            (request("x^2+7", "x", "x", 7), "coprime"),
        ]
        for case, message in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, message):
                    dispatch(case)

    def test_faulty_convolution_is_detected_and_recovered(self):
        # A multiple of p added to the device product passes the first modular
        # test but changes the lifted coefficients, so the final check must reject it.
        from engine.gpu_polynomial import compute_row

        def incorrect(op, n, m, rows):
            self.assertEqual(compute_row(op, n, m, rows), "12 0;7 1;1 2")
            return "19 0;7 1;1 2"

        with patch("engine.gpu_polynomial.compute_row", side_effect=incorrect):
            result = dispatch(request("x^2-2", "x-3", "x+3", 7, mode="gpu"))
        self.assertEqual((result["result"]["left"], result["result"]["right"]), ("x+39", "x+10"))
        self.assertTrue(any("rejected" in s for s in result["execution"]["fallback_reasons"]))

    def test_physical_dense_convolution_matches_cpu(self):
        if not triton_device(0):
            self.skipTest("Requires a physical Triton device.")
        g = "(1+" + "+".join(f"x^{i}" for i in range(1, 65)) + ")"
        h = f"({g}+1)"
        a, b = f"({g}+7*x^6)", f"({h}+21*x^13)"
        case = request(f"{a}*{b}", g, h, 7)
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton", RESOLUTION_GPU_INDEX="0"):
            result = dispatch(dict(case, compute="gpu"))
        self.assertEqual(result["result"], dispatch(case)["result"])
        self.assertTrue(
            any(
                s["backend"] == "triton" and "poly_dense_convolution" in s.get("kernels", {})
                for s in result["execution"]["stages"]
            ),
            result["execution"],
        )
