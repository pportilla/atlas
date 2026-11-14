"""Resultant values through the shared prompt and script evaluator."""
from fractions import Fraction
from textwrap import dedent
import unittest

from engine.session_algebra import eliminate, ideal, resultant
from engine.session_polynomial import coordinates
from tests.test_cli_session import run_prompt, run_script


class ResultantSessionTests(unittest.TestCase):
    def test_choose_either_coordinate_at_prompt_and_in_scripts(self):
        source = dedent("""
            x, y = coordinates("x", "y", coeffs=QQ)
            f = x^2-y
            g = x-y
            R = resultant(f, g, x, order="lex")
            Y, = R.coordinates
            S = resultant(f, g, "y")
            X, = S.coordinates
            print(R.variables, R.order, R.polynomial == Y^2-Y)
            print(S.variables, S.polynomial == X^2-X)
            T = resultant(R, Y-2, Y)
            print(T.polynomial == 2, T.order, T.variables)
        """).lstrip()
        expected = ["('y',) lex True", "('x',) True", "True lex ()"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_result_space_matches_elimination_and_preserves_unused_coordinates(self):
        z, y, x = coordinates("z", "y", "x")
        r = resultant(y-x, y-z, y)
        Z, X = r.coordinates
        self.assertEqual(r.variables, ("z", "x"))
        self.assertEqual(r, X-Z)
        self.assertEqual(r.coordinates, eliminate(ideal(y-x, y-z), y).coordinates)
        self.assertEqual(resultant(y-x, y-1, "y").coordinates, r.coordinates)
        with self.assertRaisesRegex(ValueError, "different coordinate spaces"):
            _ = r + x

    def test_constants_in_owned_and_scalar_rings(self):
        x, y = coordinates("x", "y")
        for r in (resultant(2, Fraction(-1, 3), x),
                  resultant(2, Fraction(-1, 3), "x", variables=(x, y))):
            self.assertEqual(r, 1)
            self.assertEqual(r.variables, ("y",))
        t = coordinates("t")
        r = resultant(t**2+1, Fraction(1, 3), t, order="lex")
        self.assertEqual(r.ring, {"variables": [], "field": "QQ", "order": "lex"})
        self.assertEqual(r.polynomial, Fraction(1, 9))

    def test_invalid_selector_and_unrelated_spaces(self):
        x, y = coordinates("x", "y")
        a, b = coordinates("x", "y")
        for value in (None, 1, "", "z", (x, y), x+y, a):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resultant(x, y, value)
        with self.assertRaisesRegex(ValueError, "different coordinate spaces"):
            resultant(x, b, x)
        with self.assertRaisesRegex(ValueError, "different coordinate spaces"):
            resultant(2, 3, a, variables=(x, y))
