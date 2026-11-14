"""Exact resultants in the coefficient ring left after eliminating one variable."""
from fractions import Fraction
import shutil
import subprocess
import unittest

from engine.algebra import dispatch
from engine.polynomial import parse_expression, rational_text


def resultant(left, right, variable="x", names=("x", "y", "z"), order="grevlex"):
    return dispatch({"operation": "resultant", "ring": {"variables": list(names), "order": order},
                     "operands": {"polynomial": left, "other_polynomial": right, "variables": [variable]}})


class PolynomialResultantTests(unittest.TestCase):
    def check(self, left, right, expected, variable="x", names=("x", "y", "z"), order="grevlex"):
        record = resultant(left, right, variable, names, order)
        result = record["result"]
        survivors = [v for v in names if v != variable]
        self.assertEqual(result["ring"], {"field": "QQ", "variables": survivors, "order": order})
        self.assertEqual(result["type"], "polynomial")
        self.assertEqual(parse_expression(result["polynomial"], survivors, _reference=True),
                         parse_expression(expected, survivors, _reference=True))
        stage, = record["execution"]["stages"]
        self.assertEqual((stage["operation"], stage["variable"], stage["backend"]),
                         ("polynomial_resultant", variable, "cpu"))

    def test_sign_rational_content_and_nonconstant_leading_coefficient(self):
        self.check("x-y", "x-z", "y-z")
        self.check("x-z", "x-y", "z-y")
        self.check("x*y+y", "z*(x^2+1)/3", "2*y^2*z/3")
        self.check("x^2-y", "x^3+z", "z^2-y^3")
        self.check("(x-y)*(x^3+1)", "(x-y)*(x^2+1)", "0")

    def test_choice_of_middle_variable_and_scalar_ring(self):
        self.check("x*y+z", "x-y^2", "x^3-z^2", "y", ("z", "y", "x"), "lex")
        self.check("x*y+z", "x-y^2", "-y^3-z", "x", ("z", "y", "x"), "lex")
        self.check("2*x^2-1", "x-1/3", "-7/9", names=("x",))

    def test_constants_independent_polynomials_and_zero(self):
        for left, right, expected in (("0", "0", "0"), ("0", "1", "0"), ("y", "0", "0"),
                                      ("y+z", "y-z", "1"), ("y", "x^3+z", "y^3"),
                                      ("x^2+z", "-z/3", "z^2/9")):
            with self.subTest(left=left, right=right):
                self.check(left, right, expected)

    def test_wide_exponent_and_large_rational_content(self):
        n = 2**64 + 17
        self.check(f"x^{n}-y", "x-z", f"y-z^{n}")
        self.check("x-z", f"x^{n}-y", f"z^{n}-y")
        q = Fraction(2**19000+1, 2**18000+1)
        result = resultant("x+(2^19000+1)/(2^18000+1)*y", "x")["result"]
        self.assertEqual(result["polynomial"], rational_text(-q) + "*y")

    def test_single_occupied_power_in_the_elimination_variable(self):
        n = 2**64 + 17
        self.check(f"x^{n}", "x^2+1", "1")
        self.check(f"(y+1)*x^{n}", "x^3+z", f"(y+1)^3*z^{n}")
        self.check("x^3+z", f"(y+1)*x^{n}", f"-(y+1)^3*z^{n}")
        self.check(f"x^{n}*(x+1)", "x*(x^2-z)", "0")

    @unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
    def test_against_singular(self):
        left = "(y+x)^2+z"
        right = "y^3-x*z"
        for variable in ("x", "y", "z"):
            script = (f"ring R=0,(z,y,x),dp; short=0;\n"
                      f"print(resultant({left},{right},{variable}));\nquit;\n")
            oracle = subprocess.run(["Singular", "-q"], input=script, text=True,
                                    capture_output=True, check=True, timeout=20)
            with self.subTest(variable=variable):
                self.check(left, right, oracle.stdout.strip(), variable, ("z", "y", "x"))
