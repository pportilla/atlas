"""Standard-graded Hilbert series keep their structured fields and coordinate space."""
import json
from textwrap import dedent
import unittest

from engine.session import SessionError, new_session
from tests.cli_support import run_prompt, run_script


class HilbertSeriesSessionTests(unittest.TestCase):
    def test_prompt_and_script_expose_signed_sparse_terms(self):
        source = dedent("""
            x, y = coordinates("x", "y", coeffs=QQ)
            S = hilbert_series(ideal(x^2, y^3, order="lex"), compute="cpu")
            print(S)
            print(S.type, S.scope, S.dimension, S.multiplicity, S.denominator_power)
            print(S.coordinates == (x,y), S.order, S.execution.requested)
            for term in S.numerator:
                print(term.degree, term.coefficient)

        """).lstrip()
        expected = ["(1 - t^2 - t^3 + t^5)/(1 - t)^2", "hilbert-series global 0 6 2",
                    "True lex cpu", "0 1", "2 -1", "3 -1", "5 1"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        code, output, error = run_prompt(source + "quit\n")
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").replace("... ", "").strip().splitlines(), expected)

    def test_projected_ring_basis_reuse_and_record_ownership(self):
        session = new_session(None)
        list(session.run(dedent("""
            x, y, z = coordinates("x", "y", "z")
            J = eliminate(ideal(x-y, z^2, order="lex"), x)
            G = groebner(J)
            H = hilbert_series(G)
            u, v = H.coordinates
            again = hilbert_series(ideal(v^2, variables=J.coordinates, order=J.order))
        """)))
        h, other = (session.variables[name] for name in ("H", "again"))
        self.assertEqual(h.result, other.result)
        self.assertEqual(h.coordinates, session.variables["J"].coordinates)
        self.assertNotEqual(h.coordinates, (session.variables["y"], session.variables["z"]))
        self.assertEqual(h.numerator, [{"degree": 0, "coefficient": 1}, {"degree": 2, "coefficient": -1}])
        h.numerator[1]["degree"] = 99
        self.assertEqual(h.numerator[1]["degree"], 2)
        replay = run_script(h.record["replay"]["script"])
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout)["result"], h.result)

    def test_global_quotient_requires_a_global_ideal(self):
        session = new_session(None)
        list(session.run('x = coordinates("x")\nB = local_standard_basis(ideal(x*(1-x)))'))
        with self.assertRaisesRegex(SessionError, "ideal"):
            list(session.run("hilbert_series(B)"))
