"""Radicals are reusable global ideals in the input coordinate space."""
import json
from textwrap import dedent
import unittest

from engine.session import SessionError, new_session
from tests.test_cli_session import run_prompt, run_script


class RadicalSessionTests(unittest.TestCase):
    def test_prompt_and_script_reuse_the_radical_of_a_mixed_dimensional_ideal(self):
        # The common zero set is the line x=0 and the isolated point (1,1).
        source = dedent("""
            x, y = coordinates("x", "y", coeffs=QQ)
            I = ideal(x*(x-1), x*(y-1)^2, order="lex")
            J = radical(I, compute="cpu")
            print(J.type, J.operation, J.order, J.coordinates == I.coordinates)
            print(membership(x*(y-1), I).value, membership(x*(y-1), J).value)
            print(membership(x, J).value, membership(y-1, J).value)
            print(groebner(J).generators == J.generators)
        """).lstrip()
        expected = ["ideal radical lex True", "False True", "False False", "True"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        code, output, error = run_prompt(source + "quit\n")
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_radical_of_saved_basis_after_elimination(self):
        session = new_session(None)
        list(session.run(dedent("""
            t, x, y = coordinates("t", "x", "y")
            E = eliminate(ideal(t-x, x+y^2, x-y^2, order="lex"), t)
            B = groebner(E)
            J = radical(B)
            u, v = J.coordinates
            K = intersection(J, ideal(v, variables=E.coordinates, order="lex"))
        """)))
        j, k, e = (session.variables[name] for name in ("J", "K", "E"))
        self.assertEqual(j.coordinates, e.coordinates)
        self.assertEqual(j.generators, j.coordinates)
        self.assertEqual(k.generators, (session.variables["v"],))
        with self.assertRaisesRegex(SessionError, "coordinate"):
            list(session.run("membership(y, J)"))
        replay = run_script(j.record["replay"]["script"])
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout)["result"], j.result)
