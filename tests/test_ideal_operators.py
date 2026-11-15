"""Ideal arithmetic at the Atlas prompt and in scripts."""
import json
import unittest
from textwrap import dedent
from unittest.mock import patch

from tests.cli_support import run_prompt, run_script


class IdealOperatorTests(unittest.TestCase):
    def test_session_arithmetic_with_and_without_python_extension(self):
        from engine import session_ideal_native
        from engine.session_algebra import ideal
        from engine.session_polynomial import coordinates

        x, y = coordinates("x", "y")

        def calculate():
            left, right = ideal(x, y**2), ideal(x**2 + y, ring=x.parent)
            value = ((left * right) + left) ** 2
            self.assertIs(value.parent, x.parent)
            return value.result, value.record["replay"], value.generators

        expected = calculate()
        with patch.object(session_ideal_native, "_compiled", return_value=None):
            self.assertEqual(calculate(), expected)

    def test_growing_a_power_result_reaches_the_worker(self):
        from engine import session_algebra as algebra
        from engine.session_polynomial import coordinates

        x = coordinates("x")
        original = algebra.ideal_power(algebra.ideal(x), 2)
        value = original
        with patch.object(algebra, "_evaluate", wraps=algebra._evaluate) as worker:
            for _ in range(8):
                value = algebra.ideal_sum(value, value)
            self.assertTrue(worker.called)
        self.assertEqual(value.generators, (x**2,) * 256)
        self.assertEqual(original.generators, (x**2,))

    def test_replay_of_derived_ideals_after_rebinding_coordinates(self):
        # Keep proper order and ownership across an intermediate result when
        # its old names and source ideals are no longer in the workspace.
        source = dedent("""
            z, a = coordinates("z", "a")
            I = ideal(z^2 + a, 0, order="lex")
            J = ideal(a, ring=z.parent)
            P = ideal_product(I, J)
            S = ideal_sum(P, J)
            K = ideal_power(S, 2)
            I, J, P = 0, 0, 0
            z, a = coordinates("z", "a", fresh=True)
            print(S.json)
            print(K.json)
        """).lstrip()
        run = run_script(source)
        self.assertEqual((run.returncode, run.stderr), (0, ""))
        for line, operation in zip(run.stdout.splitlines(), ("ideal-sum", "ideal-power"), strict=True):
            record = json.loads(line)
            self.assertEqual(record["operation"], operation)
            self.assertEqual(record["result"]["ring"], {"variables": ["z", "a"], "order": "lex", "field": "QQ"})
            replay = run_script(record["replay"]["script"])
            self.assertEqual((replay.returncode, replay.stderr), (0, ""))
            repeated = json.loads(replay.stdout)
            self.assertEqual((repeated["result"], repeated["replay"]), (record["result"], record["replay"]))

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
