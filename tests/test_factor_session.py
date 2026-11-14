"""Polynomial factorizations are reusable in the shared prompt and script session."""
from fractions import Fraction
from textwrap import dedent
import unittest

from engine.session_algebra import factor, gcd
from engine.session_polynomial import coordinates
from tests.test_cli_session import run_prompt, run_script


class FactorSessionTests(unittest.TestCase):
    def test_prompt_and_script_reconstruct_the_polynomial(self):
        source = dedent("""
            x, y = coordinates("x", "y", coeffs=QQ)
            f = (-3/2)*(x-y)^3*(x^2+1)
            D = factor(f, order="lex")
            print(D.type, D.unit, D.coordinates == (x,y), len(D.factors))
            product = D.unit
            for entry in D.factors:
                product = product * entry.polynomial^entry.multiplicity

            print(product == f)
        """).lstrip()
        expected = ["factorization -3/2 True 2", "True"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").replace("... ", "").strip().splitlines(), expected)

    def test_each_factor_keeps_its_space_and_normalization_order(self):
        x, y = coordinates("x", "y")
        p = x + 2*y**2
        D = factor(-p**3/7, order="lex")
        self.assertEqual(D.unit, Fraction(-1, 7))
        (q, exponent), = D.factors
        self.assertEqual((q, exponent), (p, 3))
        self.assertEqual(gcd(q, 0), q)
        graded = factor(q, order="grevlex")
        self.assertEqual(graded.unit, 2)
        self.assertEqual(graded.factors[0].polynomial, p/2)
        other_x, _ = coordinates("x", "y")
        with self.assertRaisesRegex(ValueError, "different coordinate spaces"):
            _ = other_x + q

    def test_constant_and_undefined_factorizations(self):
        x = coordinates("x")
        huge = Fraction(-10**5000, 3)
        D = factor(huge, variables=x)
        self.assertEqual((D.unit, D.factors, D.coordinates), (huge, (), (x,)))
        self.assertEqual(factor(1).variables, ())
        with self.assertRaisesRegex(ValueError, "zero"):
            factor(x-x)
