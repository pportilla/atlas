"""Quotient arithmetic, coordinate ownership, and saved requests at the prompt and in scripts."""
import json
from fractions import Fraction
from textwrap import dedent
import unittest
from unittest.mock import patch

from engine.algebra import calculate_algebra
from engine.session_algebra import algebra_request, ideal
from engine.session_polynomial import coordinates
from engine.session_quotient import quotient_ring
from tests.cli_support import run_prompt, run_script


class QuotientRingSessionTests(unittest.TestCase):
    def test_prompt_and_script(self):
        source = dedent("""
            x, y = coordinates("x", "y", coeffs=QQ)
            A = quotient_ring(ideal(x*y-1, order="lex"))
            u, v = A.gens
            print(u.inverse() == v, (1/u).representative == y, u^-3 == v^3)
            print(A(x^2*y) == u, (x-u).representative, (x+v)*u == x^2+1)
            print(u.parent == A, A.order, A.defining_ideal.generators == (x*y-1,))
            D = quotient_ring(ideal(x^2))
            t = D(x)
            print(t^2 == 0, t.is_unit(), (1+t).inverse() == 1-t, -t == 0-t)
            print(D.apply("unit", 1+t).value, (1/t.parent(2)).representative)
        """).lstrip()
        expected = ["True True True", "True 0 True", "True lex True", "True False True True", "True 1/2"]
        run = run_script(source)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.splitlines(), expected)
        status, output, err = run_prompt(source + "quit\n")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_ownership_and_division_errors(self):
        x = coordinates("x")
        a = quotient_ring(ideal(x**2))
        b = quotient_ring(ideal(x**2))
        u, v = a(x), b(x)
        for mixed in (lambda: a(v), lambda: u+v, lambda: u == v):
            with self.assertRaisesRegex(ValueError, "different quotient rings"):
                mixed()
        with self.assertRaisesRegex(ValueError, "different coordinate spaces"):
            a(coordinates("x"))
        for operation in (lambda: 1/u, lambda: u**-1, lambda: u.inverse(), lambda: 0/u):
            with self.assertRaisesRegex(ValueError, "unit|invert"):
                operation()

    def test_scalar_ring_zero_ring_and_exact_wide_inputs(self):
        q = quotient_ring(ideal(variables=()))
        self.assertEqual(q.gens, ())
        self.assertEqual((q(Fraction(1, 3)) / 2).representative, Fraction(1, 6))
        zero = quotient_ring(ideal(1, variables=()))
        self.assertTrue(zero(0).is_unit())
        self.assertEqual(zero(0).inverse().representative, 0)
        x = coordinates("x")
        a = quotient_ring(ideal(x**2-1))
        u = a(x)
        self.assertEqual((u**-(2**70+1)).representative, x)
        big = Fraction(2**91+3, 17)
        self.assertEqual((big * u / u).representative, big)

    def test_replay_has_a_fresh_parent_and_reuses_verified_basis(self):
        data = {"operation": "quotient-ring", "ring": {"variables": ["x", "y"]}, "compute": "cpu",
                "operands": {"ideal": ["x-y", "x+y"], "action": "add", "polynomial": "y", "other_polynomial": "1"}}
        value = algebra_request(json.dumps(data))
        self.assertEqual(value.representative, 1)
        with patch("engine.algebra.calculate_algebra", wraps=calculate_algebra) as call:
            again = value + 1
            self.assertTrue(call.call_args.kwargs["known_basis"])
        self.assertIs(again.parent, value.parent)
        self.assertEqual(again.representative, 2)
        external = value.parent.result
        external["relations"].clear()
        self.assertEqual(value.parent(value.parent.coordinates[0]).representative, 0)
