"""Compose rational ideal operations in the same identity session at both entry points."""
import unittest
from textwrap import dedent

from tests.cli_support import run_prompt, run_script


class IdealSessionTests(unittest.TestCase):
    def test_live_parents_and_rational_scalar_embeddings_at_both_entry_points(self):
        source = dedent("""
            x, y = coordinates("x", "y")
            q = polynomial("1/2", ring=QQ)
            print(q.parent == QQ, q == polynomial("1/2", variables=[]))
            print((q+x).parent == x.parent, q+x == x+q, q-x == -(x-q), x/q == 2*x)
            I = ideal(q, x)
            print(I.parent == x.parent, I.generators[0].parent == I.parent)
            J = ideal(x^2, ring=I.parent)
            print(polynomial("x^2", ring=J.parent) == J.generators[0], groebner(J).parent == J.parent)
            print(ideal(ring=QQ).parent == QQ, ideal(1, ring=x.parent).parent == x.parent)
            A = quotient_ring(J)
            print(A.ambient == J.parent, A(x).parent == A, A(q) == q, q == A(q))
        """).lstrip()
        expected = ["True True", "True True True True", "True True", "True True", "True True", "True True True True"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_ideal_construction_respects_an_explicit_ambient_ring(self):
        setup = 'x = coordinates("x")\na = coordinates("x", fresh=True)\n'
        for command, message in (
            ("ideal(x-x, ring=a.parent)", "different coordinate spaces"),
            ("ideal(ring=x.parent, variables=(x,))", "exactly one"),
            ("ideal(ring=ideal(x).ring)", "parent ring"),
        ):
            with self.subTest(command=command):
                result = run_script(setup + command)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)

    def test_ground_field_results_supply_coefficients_and_point_coordinates(self):
        source = dedent("""
            x, y = coordinates("x", "y")
            a = coordinates("a")
            q = resultant(a, a+1, a, order="lex")
            f = gcd(x^2, x, order="grevlex")
            print(q.parent == QQ, ideal(q).order, ideal(q, f).order)
            print(ideal(q, variables=(x,y)).order, ideal(q, ring=x.parent).order)
            half = polynomial("1/2", ring=QQ)
            I = ideal(x-1, 2*y-1)
            V = zariski_tangent_space(I, point=(q,half))
            print(V.dimension, V.point)
        """).lstrip()
        expected = ["True lex grevlex", "grevlex grevlex", "0 ['1', '1/2']"]
        script = run_script(source)
        self.assertEqual((script.returncode, script.stderr), (0, ""))
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

        for value in ("x-x+1", "a^0"):
            with self.subTest(value=value):
                result = run_script(source + f"V = zariski_tangent_space(I, point=({value},half))\n")
                self.assertEqual(result.returncode, 2)
                self.assertIn("must have parent QQ", result.stderr)

    def test_prompt_and_script_preserve_result_rings(self):
        source = dedent("""
            x, y = coordinates("x", "y", coeffs=QQ)
            D = ideal(x^2*y)
            Q = quotient(D, ideal(x))
            M = intersection(ideal(x), ideal(y))
            S = saturation(D, x)
            print(Q.generators == M.generators, Q.generators[0] == x*y, S.generators[0] == y)
            E = eliminate(ideal(x^2-1, y-x, order="lex"), x)
            v = E.coordinates[0]
            print(E.variables, E.order, E.generators[0] == v^2-1)
            print(dimension(E).value, dimension(ideal(1, variables=(x,y))).value)
            C = eliminate(E, v)
            print(C.variables, C.generators, dimension(C).value)
        """).lstrip()
        expected = ["True True True", "('y',) lex True", "0 -1", "() () 0"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)
