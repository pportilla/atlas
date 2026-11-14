"""Tangent coordinates, associated graded ideals and their Hilbert series in Atlas."""
import json
import unittest
from textwrap import dedent

from engine.session import new_session
from engine.session_expressions import SessionError
from tests.test_cli_session import run_prompt, run_script


class TangentConeSessionTests(unittest.TestCase):
    def test_translated_cusp_at_prompt_and_in_script(self):
        source = dedent("""
            x, y = coordinates("x", "y")
            I = ideal((y+2/3)^2-(x-1/2)^3, order="lex")
            C = tangent_cone(I, point=["2/4",-2/3], compute="cpu")
            u, v = C.coordinates
            print(C.type, C.scope, C.point, C.coordinate_system, C.dimension, C.multiplicity)
            print(u == x, v == y, C.generators[0] == v^2, C.generators[0] == y^2)
            S = tangent_cone_hilbert_series(C)
            T = tangent_cone_hilbert_series(C, point=[0,0])
            print(S.type, S.scope, S.point, T.scope, T.point, S.dimension, S.multiplicity)
            print(S.denominator_power, S.numerator[0].degree, S.numerator[0].coefficient)
            print(S.numerator[1].degree, S.numerator[1].coefficient)
        """).lstrip()
        expected = ["tangent-cone point ['1/2', '-2/3'] tangent 1 2", "False False True False",
                    "tangent-cone-hilbert-series origin ['0', '0'] point ['0', '0'] 1 2",
                    "2 0 1", "2 -1"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        code, output, error = run_prompt(source + "quit\n")
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_saved_local_basis_point_replay_and_cone_coordinate_ownership(self):
        session = new_session(None)
        list(session.run(dedent("""
            x, y = coordinates("x", "y")
            I = ideal(x+1/2+(y-1)^2, x+1/2+(y-1)^3)
            B = local_standard_basis(I, point=(-1/2,1))
            C = tangent_cone(B)
            S = tangent_cone_hilbert_series(B, point=["-2/4",1], compute="cpu")
            u, v = C.coordinates
            M = ideal(C.generators)
            member = membership(v^2,M)
        """)))
        c, s = session.variables["C"], session.variables["S"]
        u, v = c.coordinates
        self.assertCountEqual(c.generators, (u, v**2))
        self.assertTrue(session.variables["member"].value)
        self.assertEqual((c.scope, c.point, c.dimension, c.multiplicity), ("point", ["-1/2", "1"], 0, 2))
        self.assertEqual((s.scope, s.point, s.dimension, s.multiplicity), ("point", c.point, 0, 2))
        numerator = s.numerator
        numerator[0]["coefficient"] = 0
        self.assertEqual(s.numerator[0], {"degree": 0, "coefficient": 1})
        for expr in ("tangent_cone(B, point=[0,0])", "tangent_cone_hilbert_series(B, point=[0,0])",
                     "tangent_cone_hilbert_series(C, point=B.point)", "membership(y^2,M)"):
            with self.subTest(expr=expr), self.assertRaises(SessionError):
                list(session.run(expr))
        for value in (s, c):
            with self.subTest(operation=value.operation):
                replay = run_script(value.record["replay"]["script"])
                self.assertEqual(replay.returncode, 0, replay.stderr)
                self.assertEqual(json.loads(replay.stdout)["result"], value.result)

    def test_origin_and_explicit_zero_for_empty_and_full_cones(self):
        session = new_session(None)
        list(session.run(dedent("""
            x, y = coordinates("x", "y")
            full = tangent_cone(ideal(variables=(x,y)))
            explicit = tangent_cone(ideal(variables=(x,y)), point=[0,0])
            plane = tangent_cone_hilbert_series(full)
            empty = tangent_cone(ideal(1, variables=(x,y)))
            zero = tangent_cone_hilbert_series(empty)
        """)))
        values = session.variables
        full, explicit = values["full"], values["explicit"]
        self.assertEqual((full.scope, full.point, full.dimension, full.multiplicity), ("origin", ["0", "0"], 2, 1))
        self.assertNotIn("point", full.result)
        self.assertEqual((explicit.scope, explicit.result["point"]), ("point", ["0", "0"]))
        self.assertFalse(full.generators)
        self.assertEqual(values["plane"].result["numerator"], [{"degree": 0, "coefficient": 1}])
        self.assertEqual(tuple(values["empty"].generators), (1,))
        zero = values["zero"]
        self.assertEqual((zero.dimension, zero.multiplicity, zero.result["numerator"]), (-1, 0, []))

    def test_series_at_cone_vertex_exports_exact_degrees_and_replays(self):
        session = new_session(None)
        list(session.run(dedent("""
            x = coordinates("x")
            C = tangent_cone(ideal(x^9007199254740993))
            S = tangent_cone_hilbert_series(C, point=["-0/5"])
        """)))
        c, s = session.variables["C"], session.variables["S"]
        self.assertEqual(s.coordinates, c.coordinates)
        self.assertEqual((s.scope, s.point, s.denominator_power), ("point", ["0"], 1))
        self.assertEqual(s.numerator[-1], {"degree": 9007199254740993, "coefficient": -1})
        self.assertEqual(s.multiplicity, 9007199254740993)
        replay = run_script(s.record["replay"]["script"])
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout)["result"], s.result)
