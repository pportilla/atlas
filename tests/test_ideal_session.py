"""Compose rational ideal operations in the same identity session at both entry points."""
import unittest
from textwrap import dedent

from tests.cli_support import run_prompt, run_script


class IdealSessionTests(unittest.TestCase):
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
