"""Global finite-colength multiplicity at the Atlas prompt and in scripts."""

import json
from decimal import Decimal
from textwrap import dedent
import unittest

from engine.session import SessionError, new_session
from tests.test_cli_session import run_prompt, run_script


class MultiplicitySessionTests(unittest.TestCase):
    def test_global_length_and_infinite_colength_at_both_entry_points(self):
        source = dedent("""
            x, y = coordinates("x", "y", coeffs=QQ)
            I = ideal((x^2+1)^2, y-x, order="lex")
            m = multiplicity(I, compute="cpu")
            print(m, m.value, m.finite, m.scope)
            print(m.type, m.order, m.coordinates == (x,y), m.execution.requested)
            line = multiplicity(ideal(y-x))
            print(line, line.finite, line.value)
        """).lstrip()
        expected = ["4 4 True global", "integer lex True cpu", "infinite colength False None"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_trusted_basis_uses_projected_ring_and_replay_is_portable(self):
        session = new_session(None)
        list(session.run(dedent("""
            z, x, y = coordinates("z", "x", "y")
            J = eliminate(ideal(z-x, y-x, (x-1)^3, order="lex"), z)
            B = groebner(J)
            m = multiplicity(B, compute="cpu")
        """)))
        result = session.variables["m"]
        self.assertEqual((result.coordinates, result.order, result.value), (session.variables["J"].coordinates, "lex", 3))
        self.assertNotEqual(result.coordinates, (session.variables["x"], session.variables["y"]))
        self.assertEqual([s["operation"] for s in result.execution["stages"]], ["restore_global_basis", "quotient_length"])
        replay = run_script(result.record["replay"]["script"])
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout)["result"], result.result)
        for call, error in (("m.point", "point"), ("multiplicity(local_standard_basis(J))", "ideal")):
            with self.subTest(call=call), self.assertRaisesRegex(SessionError, error):
                list(session.run(call))

    def test_exact_large_length_as_a_value_and_in_json(self):
        digits = "1" + "0" * 4500
        session = new_session(None)
        list(session.run('x = coordinates("x")\nm = multiplicity(ideal(x^(10^4500)))'))
        m = session.variables["m"]
        self.assertEqual(repr(m), digits)
        self.assertEqual(m.value, 10**4500)
        self.assertEqual(json.loads(m.json, parse_int=Decimal)["result"]["value"], m.value)
