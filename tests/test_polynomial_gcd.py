"""Monic multivariate GCDs through the rational algebra dispatcher."""
from fractions import Fraction
import os
import shutil
import subprocess
import unittest
from unittest.mock import patch

from engine.algebra import calculate_algebra, dispatch
from engine.gpu_devices import available_devices
from engine.polynomial import parse_expression, rational_text


def gcd_request(left, right, variables=("x", "y", "z"), order="grevlex", compute="cpu"):
    return {"operation": "gcd", "ring": {"field": "QQ", "variables": list(variables), "order": order},
            "operands": {"polynomial": left, "other_polynomial": right}, "compute": compute}


def terms(expression, names):
    return parse_expression(expression, list(names), _reference=True)


def replay_workload():
    # Consecutive cofactors are coprime. Their shared factor and the device's
    # reconstructed LCM have multiple terms, and the internal coefficients
    # require many modular images even after scalar content is discarded.
    h = "3*x^2+4*y*z-5*z+1"
    a = f"({2**400 + 31}*(x^3+4*x^2*y+3*y^2*z-3*x*y+2*z+5)+x+y)"
    return f"({h})*{a}", f"({h})*({a}+1)", f"({h})/3"


class PolynomialGCD(unittest.TestCase):
    def test_elimination_recovers_common_factor_with_wide_degree(self):
        h = "x^18446744073709551617+x*y^2+z^2"
        request = gcd_request(f"({h})*(x+y)", f"({h})*(x+z)", order="lex")
        expected = parse_expression(h, ["x", "y", "z"])
        result = dispatch(request)["result"]
        self.assertEqual(parse_expression(result["polynomial"], ["x", "y", "z"]), expected)

    def test_zero_units_and_repeated_factors(self):
        cases = [
            ("0", "0", "0"),
            ("0", "-12*x^2*y/7", "x^2*y"),
            ("-12*x^2*y/7", "0", "x^2*y"),
            ("-2/3", "5*x+z^3", "1"),
            ("(x+y)^2*z", "(x+y)^3*(z+1)", "(x+y)^2"),
            ("x+y", "x-y", "1"),
        ]
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                request = gcd_request(left, right)
                result = dispatch(request)["result"]
                self.assertEqual(result["type"], "polynomial")
                self.assertEqual(result["ring"], request["ring"])
                self.assertEqual(terms(result["polynomial"], result["ring"]["variables"]),
                                 terms(expected, result["ring"]["variables"]))
        for left, right, expected in [("0", "0", "0"), ("-2/7", "0", "1"), ("-2", "3/7", "1")]:
            with self.subTest(scalars=(left, right)):
                result = dispatch(gcd_request(left, right, variables=()))["result"]
                self.assertEqual(result["polynomial"], expected)
                self.assertEqual(result["ring"]["variables"], [])

    def test_monic_in_the_declared_order(self):
        # With equal total degrees, grevlex uses y^2 ahead of x*z.
        common = "(2*x*z+3*y^2)"
        for order, expected in [("lex", "x*z+3*y^2/2"), ("grevlex", "2*x*z/3+y^2")]:
            for compute in ("cpu", "gpu"):
                with self.subTest(order=order, compute=compute):
                    request = gcd_request(common + "*(-2*z/7)", common + "*((z+1)/3)",
                                          order=order, compute=compute)
                    record = dispatch(request)
                    self.assertEqual(record["result"]["ring"], request["ring"])
                    self.assertEqual(terms(record["result"]["polynomial"], ("x", "y", "z")),
                                     terms(expected, ("x", "y", "z")))
                    self.assertEqual(record["execution"]["requested"], compute)

    def test_large_rational_coefficient(self):
        # The reduced result has a coefficient larger than Python's decimal
        # literal conversion limit, both above and below the fraction bar.
        h = "(x+((2^20000+1)/(2^19000+1))*y)"
        request = gcd_request(h + "*z", h + "*(z+1)", order="lex")
        result = dispatch(request)["result"]
        coefficient = Fraction(2**20000 + 1, 2**19000 + 1)
        self.assertEqual(result["polynomial"], "x+" + rational_text(coefficient) + "*y")

    def test_certified_shortcuts_and_their_failed_candidates(self):
        from engine.session_algebra import gcd, polynomial
        from engine.session_polynomial import coordinates

        prime = 65537
        names = ("x", "y")
        product = "(x+2)*(y+2)"
        cases = [
            # The common divisor evaluates to one on x=y=t. Both operands lose degree.
            ("diagonal degree drop", "1+x^10-y^10", "2*x+3*y+1", "x-y+2", "1+x^10-y^10"),
            # Reduction modulo the probe prime can hide a genuine rational factor.
            ("prime degree drop", f"{prime}*x^7+1", "x+y+3", "x-y+2", f"x^7+1/{prime}"),
            ("dense prime degree drop", f"1+{prime}*({product})", "(x+1)*(y+1)",
             "(x+3)*(y+3)+11", f"({product})+1/{prime}"),
            # Kronecker substitution gives both cofactors a root at t=1. Lifting t-1
            # yields a factor of the first cofactor that does not divide the second.
            ("spurious Kronecker factor", product + "+1", "(x-1)*(y-1)",
             "(x+3)*(y+3)-16", product + "+1"),
            # A centered modular lift changes this coefficient. Exact division rejects it.
            ("coefficient wraps", product + "+40000", "(x+1)*(y+1)",
             "(x+3)*(y+3)+11", product + "+40000"),
            # Both operands are exact squares, with a nonmonic rational common factor.
            ("two squares", "(3*x+2*y-1)^2", "(x+2*y+3)^2/9",
             "(2*x-y+1)^2/25", "(3*x+2*y-1)^2/9"),
            # Just one square: taking both multiplicities to be even would lose one copy.
            ("one square", "(x+y+1)^3", "(x+y+1)*(x-2)^2", "y-3", "(x+y+1)^3"),
            # Cancelling the leading terms produces h*y. It cannot divide the input h*A.
            ("sparse difference nondivisor", "x+y+1", f"(-{2**200+3}/7)*(x^3+3*x*y^2+1)",
             "(5/9)*(x^3+3*x*y^2+1+y)", "x+y+1"),
        ]
        ring = coordinates(*names)[0].parent
        for name, h, a, b, expected in cases:
            with self.subTest(name=name):
                left, right = f"({h})*({a})", f"({h})*({b})"
                f, g = (polynomial(source, ring=ring) for source in (left, right))
                want = polynomial(expected, ring=ring)
                result = gcd(f, g, compute="cpu")
                self.assertEqual(result, want)
                record = dispatch(gcd_request(left, right, variables=names))
                self.assertEqual(record["result"], result.result)

    @unittest.skipUnless(os.getenv("ATLAS_GROEBNER", "native") == "native",
                         "The native sparse representation handles multi-word exponents")
    def test_sparse_exponent_beyond_one_machine_word(self):
        h = "x^18446744073709551633+y^2"
        for compute in ("cpu", "gpu"):
            with self.subTest(compute=compute):
                request = gcd_request(f"({h})*z", f"({h})*(z+1)", compute=compute)
                status, record = calculate_algebra(request, timeout=30)
                self.assertEqual(status, 200, record)
                self.assertEqual(record["result"]["polynomial"], h)
                self.assertEqual(record["execution"]["backend"], "cpu")
                if compute == "gpu":
                    self.assertTrue(record["execution"]["fallback_reasons"] or record["execution"].get("note"))

    def test_unavailable_device_uses_exact_cpu(self):
        f, g, expected = replay_workload()
        with patch.dict(os.environ, {"RESOLUTION_GPU_BACKEND": "opencl", "RESOLUTION_GPU_INDEX": "2147483646"}):
            record = dispatch(gcd_request(f, g, compute="gpu"))
        self.assertEqual(terms(record["result"]["polynomial"], ("x", "y", "z")),
                         terms(expected, ("x", "y", "z")))
        execution = record["execution"]
        self.assertEqual(execution["backend"], "cpu")
        from engine.native_library import algebra_device_unavailable
        self.assertTrue(all(stage["backend"] == "cpu" for stage in execution["stages"]))
        self.assertTrue(execution["fallback_reasons"])
        if algebra_device_unavailable("opencl", 2147483646):
            self.assertEqual([(s["operation"], s.get("algorithm")) for s in execution["stages"]],
                             [("polynomial_gcd", "flint")])

    def test_gcd_uses_algebra_plugin_device_status(self):
        from engine import native_library
        f, g, expected = replay_workload()
        with (patch.dict(os.environ, {"RESOLUTION_GPU_BACKEND": "opencl", "RESOLUTION_GPU_INDEX": "1"}),
              patch.object(native_library, "_opencl_replay", return_value=None),
              patch.object(native_library, "_opencl_trace", return_value=lambda *args: 1),
              patch("engine.native_backend.gpu_devices", return_value=[])):
            for status in (0, 1, -1, None):
                # An older plugin has no indexed probe. A probe can also
                # report an enumeration error. Both retain modular dispatch.
                probe = None if status is None else lambda index: status
                with self.subTest(status=status), patch.object(native_library, "_opencl_device_probe", return_value=probe):
                    record = dispatch(gcd_request(f, g, compute="gpu"))
                    operations = [s["operation"] for s in record["execution"]["stages"]]
                    self.assertEqual(terms(record["result"]["polynomial"], ("x", "y", "z")),
                                     terms(expected, ("x", "y", "z")))
                    if status == 0:
                        self.assertEqual(operations, ["polynomial_gcd"])
                    else:
                        self.assertIn("gcd_from_lcm", operations)

    @unittest.skipUnless(shutil.which("Singular"), "Singular is unavailable")
    def test_against_singular(self):
        left, right, _ = replay_workload()
        for order, singular_order in [("grevlex", "dp"), ("lex", "lp")]:
            with self.subTest(order=order):
                pairs = [(left, right),
                         ("(-3/7)*(2*x+3*y^2)*(x+y)^2", "(2/9)*(2*x+3*y^2)*(x+y)*(z+1)"),
                         ("0", "0"), ("x", "y")]
                commands = [f"ring R=0,(x,y,z),{singular_order};", "short=0;", "poly d;"]
                for f, g in pairs:
                    commands += [f"d=gcd(poly({f}),poly({g}));", "if(d!=0){d=d/leadcoef(d);}", "print(d);"]
                singular = subprocess.run(["Singular", "-q"], input="\n".join([*commands, "quit;"]),
                                          text=True, capture_output=True, check=True, timeout=30)
                answers = singular.stdout.strip().splitlines()
                self.assertEqual(len(answers), len(pairs), singular.stdout + singular.stderr)
                for (f, g), answer in zip(pairs, answers):
                    result = dispatch(gcd_request(f, g, order=order))["result"]
                    self.assertEqual(terms(result["polynomial"], ("x", "y", "z")),
                                     terms(answer, ("x", "y", "z")))

    @unittest.skipUnless(os.getenv("ATLAS_GROEBNER", "native") == "native",
                         "Physical-device checks use the production native backend")
    def test_certified_gcd_on_physical_device(self):
        devices = available_devices()
        if not devices:
            self.skipTest("No physical Triton or OpenCL device")
        f, g, expected = replay_workload()
        cpu = dispatch(gcd_request(f, g))
        self.assertEqual(cpu["execution"]["backend"], "cpu")
        self.assertEqual(terms(cpu["result"]["polynomial"], ("x", "y", "z")),
                         terms(expected, ("x", "y", "z")))
        for device in devices:
            backend = device["backend"]
            with self.subTest(device=device), patch.dict(os.environ, {
                "RESOLUTION_GPU_BACKEND": backend, "RESOLUTION_GPU_INDEX": str(device["index"])
            }):
                record = dispatch(gcd_request(f, g, compute="gpu"))
                self.assertEqual(record["result"], cpu["result"])
                stages = record["execution"]["stages"]
                on_device = [stage for stage in stages if stage["backend"] == backend]
                self.assertTrue(any(s.get("primes", 0) or s.get("pairs", 0) for s in on_device), stages)
                self.assertTrue(all(s["device"]["index"] == device["index"] for s in on_device), stages)
                self.assertTrue(any(s["operation"] == "groebner_certification" and
                                    s["backend"] == "cpu" and s["bases"] > 0 for s in stages), stages)
