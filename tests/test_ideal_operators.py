"""Ideal arithmetic at the Atlas prompt and in scripts."""
import unittest
from textwrap import dedent

from tests.cli_support import run_prompt, run_script


class IdealOperatorTests(unittest.TestCase):
    def test_zero_and_unit_ideals_in_scalar_and_polynomial_rings(self):
        source = dedent("""
            x = coordinates("x")
            Z = ideal(ring=x.parent)
            U = ideal(7, ring=x.parent)
            I = ideal(x^2)
            print((Z+I).generators == I.generators, (I+U).generators == (x^0,))
            print((Z*I).generators == (), (U*I).generators == I.generators)
            print((Z^0).generators == (x^0,), (Z^2).generators == ())
            Z = ideal(ring=QQ)
            U = ideal(polynomial("3/7", ring=QQ))
            one = polynomial("1", ring=QQ)
            print((Z+Z).generators == (), (Z+U).generators == (one,))
            print((U*Z).generators == (), (U*U).generators == (one,))
            print((Z^0).generators == (one,), (Z^2).generators == ())
            print((Z+U).parent == QQ, (Z*U).parent == QQ)
        """).lstrip()
        expected = ["True True"] * 7
        script = run_script(source)
        self.assertEqual((script.returncode, script.stderr), (0, ""))
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, err = run_prompt(source + "quit\n")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_arithmetic_composes_and_preserves_the_parent(self):
        source = dedent("""
            x, y = coordinates("x", "y")
            I = ideal(x^2, y, order="lex")
            J = ideal(x, y^2)
            S = I + J
            P = I * J
            print(S.parent == I.parent, P.parent == J.parent, S.order, P.order)
            print((J+I).order, (J*I).order)
            print(groebner(S).generators == groebner(ideal(x,y,order="lex")).generators)
            print(groebner(P).generators == groebner(ideal(x^3,x*y,y^3,order="lex")).generators)
            print(S.generators == ideal_sum(I, J).generators, P.generators == ideal_product(I, J).generators)
            print(((I + J)^2).generators == ideal_power(S, 2).generators)
            print((I^0).generators == (x^0,), (I^(4/2)).generators == ideal_power(I, 4/2).generators)
            print(groebner(I + J*I).generators == groebner(I).generators)
            n = 2^72+3
            k = 2^140+1
            W = ideal(k*x^n+y) * ideal(x^2)
            print(W.generators == (k*x^(n+2)+x^2*y,))
        """).lstrip()
        expected = ["True True lex lex", "grevlex grevlex", "True", "True", "True True",
                    "True", "True True", "True", "True"]
        script = run_script(source)
        self.assertEqual((script.returncode, script.stderr), (0, ""))
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, err = run_prompt(source + "quit\n")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_ideal_operator_errors_leave_existing_bindings_intact(self):
        setup = dedent("""
            x = coordinates("x")
            u = coordinates("x", fresh=True)
            I = ideal(x)
            J = ideal(u)
            A = quotient_ring(I)
            a = A(1)
            f = x+1
            op = groebner
            keep = I
        """).lstrip()
        cases = [(op, "different parent rings") for op in
                 ("I + J", "J * I", "ideal_sum(I, J)", "ideal_product(J, I)")]
        for value in ("1/2", "f", "op", "a"):
            cases.extend(((f"I + {value}", "requires two ideals"), (f"{value} + I", "requires two ideals")))
        cases.extend((("I * f", "requires two ideals"), ("a * I", "requires two ideals"),
                      ("I^True", "nonnegative integer"), ("I^(3/2)", "nonnegative integer"),
                      ("I^(-1)", "nonnegative integer"), ("I^f", "nonnegative integer")))
        for expression, message in cases:
            with self.subTest(expression=expression):
                source = setup + "keep = " + expression + "\n"
                script = run_script(source + 'print("unexpected")\n')
                self.assertEqual((script.returncode, script.stdout), (2, ""))
                self.assertIn(message, script.stderr)
                status, out, err = run_prompt(source + "print(keep == I, I.parent == x.parent)\nquit\n")
                self.assertEqual(status, 0)
                self.assertIn("True True", out)
                self.assertIn(message, err)
