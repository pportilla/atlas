"""Hilbert–Samuel calculations in the shared Atlas mathematical session."""
import json
import unittest
from textwrap import dedent

from engine.session import new_session
from engine.session_expressions import SessionError
from tests.test_cli_session import run_prompt, run_script


class HilbertSamuelSessionTests(unittest.TestCase):
    def test_cusp_at_prompt_and_in_script(self):
        source = dedent("""
            x, y = coordinates("x", "y")
            I = ideal(y^2-x^3)
            H = hilbert_samuel_function(I, degrees=[2,0,2,4], compute="cpu")
            e = hilbert_samuel_multiplicity(I, compute="cpu")
            print(H.type, H.scope, H.point, H.dimension, H.multiplicity, H.coordinates == (x,y))
            for i in [0,1,2,3]:
                print(H.degrees[i], H.values[i], H.graded_values[i])

            print(e.type, e.value, e.dimension, e.scope, e.point, e.execution.requested)
        """).lstrip()
        expected = ["hilbert-samuel-function origin ['0', '0'] 1 2 True", "2 5 2", "0 1 1", "2 5 2", "4 9 2",
                    "integer 2 1 origin ['0', '0'] cpu"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        code, out, err = run_prompt(source + "quit\n")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(out.replace("atlas> ", "").replace("... ", "").strip().splitlines(), expected)

    def test_saved_local_basis_inherits_point_and_replays(self):
        session = new_session(None)
        list(session.run(dedent("""
            x = coordinates("x")
            I = ideal((x-1/2)^3*(x+1))
            B = local_standard_basis(I, point=[1/2])
            H = hilbert_samuel_function(B, degrees=[0,1,2,3], compute="cpu")
            E = hilbert_samuel_multiplicity(B, point=["2/4"], compute="cpu")
        """)))
        h, e = session.variables["H"], session.variables["E"]
        self.assertEqual((h.scope, h.point, h.dimension, h.multiplicity), ("point", ["1/2"], 0, 3))
        self.assertEqual(tuple(h.values), (1, 2, 3, 3))
        self.assertEqual(tuple(h.graded_values), (1, 1, 1, 0))
        self.assertEqual((e.point, e.dimension, e.value), (h.point, 0, 3))
        for command in ("hilbert_samuel_function(B, degrees=[1], point=[-1])",
                        "hilbert_samuel_multiplicity(B, point=[-1])"):
            with self.subTest(command=command), self.assertRaisesRegex(SessionError, "point"):
                list(session.run(command))
        for value in (h, e):
            with self.subTest(operation=value.operation):
                replay = run_script(value.record["replay"]["script"])
                self.assertEqual(replay.returncode, 0, replay.stderr)
                record = json.loads(replay.stdout)
                self.assertEqual((record["operation"], record["result"]), (value.operation, value.result))

    def test_zero_ring_and_empty_degree_selection(self):
        session = new_session(None)
        list(session.run(dedent("""
            x = coordinates("x")
            I = ideal(1, variables=x)
            H = hilbert_samuel_function(I, degrees=[0,3])
            e = hilbert_samuel_multiplicity(I)
            empty = hilbert_samuel_function(ideal(x^2), degrees=[])
        """)))
        h, e, empty = (session.variables[k] for k in ("H", "e", "empty"))
        self.assertEqual((h.dimension, h.multiplicity, e.dimension, e.value), (-1, 0, -1, 0))
        self.assertEqual(tuple(h.values), (0, 0))
        self.assertEqual(tuple(h.graded_values), (0, 0))
        self.assertEqual((tuple(empty.degrees), tuple(empty.values), tuple(empty.graded_values)), ((), (), ()))
        self.assertEqual((empty.dimension, empty.multiplicity), (0, 2))

    def test_default_degrees_explicit_origin_and_exact_decimal_input(self):
        session = new_session(None)
        list(session.run(dedent("""
            x = coordinates("x")
            I = ideal(variables=x)
            H = hilbert_samuel_function(I)
            E = hilbert_samuel_multiplicity(I, point=[0])
            N = hilbert_samuel_function(I, degrees=["9007199254740993"])
        """)))
        h, e, n = (session.variables[k] for k in ("H", "E", "N"))
        self.assertEqual(h.degrees, list(range(6)))
        self.assertNotIn("point", h.result)
        self.assertEqual((e.scope, e.result["point"], e.value), ("point", ["0"], 1))
        self.assertEqual(n.values, [9007199254740994])
        h.values.clear()
        self.assertEqual(h.values, list(range(1, 7)))
        for degrees in ("None", "[True]", "[-1]", "[1/2]"):
            with self.subTest(degrees=degrees), self.assertRaises(SessionError):
                list(session.run(f"hilbert_samuel_function(I, degrees={degrees})"))
