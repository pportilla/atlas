"""Coordinate ownership and exact rational vectors in Zariski tangent sessions."""
import json
import unittest
from fractions import Fraction
from textwrap import dedent

from engine.session import new_session
from engine.session_expressions import SessionError
from tests.cli_support import run_prompt, run_script


class ZariskiTangentSessionTests(unittest.TestCase):
    def test_fractional_tangent_direction_at_prompt_and_in_script(self):
        source = dedent("""
            x, y = coordinates("x", "y")
            u, v = x-1/2, y+2/3
            I = ideal(u^2+3*u-2*v, order="lex")
            V = zariski_tangent_space(I, point=[1/2,"-4/6"], compute="cpu")
            a, b = V.coordinates
            print(V.type, V.coordinate_system, V.scope, V.point, V.ambient_dimension, V.jacobian_rank, V.dimension)
            print(a == x, b == y, V.equations[0] == a-2*b/3, V.equations[0] == x-2*y/3)
            direction = V.basis[0]
            print(direction[0], direction[1], 3*direction[0]-2*direction[1] == 0)
            e = embedding_dimension(I, point=V.point, compute="cpu")
            print(e.type, e.value, e.ambient_dimension, e.jacobian_rank, e.point, e.coordinates == (x,y))
        """).lstrip()
        expected = ["zariski-tangent-space tangent point ['1/2', '-2/3'] 2 1 1",
                    "False False True False", "2/3 1 True", "integer 1 2 1 ['1/2', '-2/3'] True"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        code, output, error = run_prompt(source + "quit\n")
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_saved_basis_and_cone_localize_at_different_points(self):
        session = new_session(None)
        list(session.run(dedent("""
            x, y = coordinates("x", "y")
            I = ideal(3*(x-1/2)-2*(y+2/3)+(x-1/2)^2)
            B = local_standard_basis(I, point=[1/2,-2/3])
            V = zariski_tangent_space(B)
            e = embedding_dimension(B, point=["2/4","-4/6"])
            C = tangent_cone(B)
            at_vertex = zariski_tangent_space(C)
            explicit = zariski_tangent_space(C, point=[0,0])
            cone_dim = embedding_dimension(C, point=[0,0])
        """)))
        values = session.variables
        v, e, c, vertex = (values[name] for name in ("V", "e", "C", "at_vertex"))
        self.assertEqual((v.scope, v.point, e.scope, e.point), ("point", ["1/2", "-2/3"], "point", v.point))
        self.assertEqual(v.basis, ((Fraction(2, 3), Fraction(1)),))
        self.assertTrue(all(isinstance(q, Fraction) for q in v.basis[0]))
        self.assertEqual((vertex.scope, vertex.point, vertex.basis), ("origin", ["0", "0"], v.basis))
        self.assertEqual(vertex.dimension, values["cone_dim"].value)
        for result in (values["explicit"], values["cone_dim"]):
            self.assertEqual((result.scope, result.result["point"]), ("point", ["0", "0"]))
        self.assertNotIn("point", vertex.result)
        for a, b in zip(vertex.coordinates, c.coordinates):
            self.assertNotEqual(a, b)
        for operation in ("zariski_tangent_space", "embedding_dimension"):
            for expression in (f"{operation}(B, point=[0,0])", f"{operation}(C, point=B.point)"):
                with self.subTest(expression=expression), self.assertRaisesRegex(SessionError, "point|origin|vertex"):
                    list(session.run(expression))
        for result in (v, values["cone_dim"]):
            with self.subTest(operation=result.operation):
                replay = run_script(result.record["replay"]["script"])
                self.assertEqual(replay.returncode, 0, replay.stderr)
                self.assertEqual(json.loads(replay.stdout)["result"], result.result)

    def test_zero_dimensional_tangent_and_off_support_error(self):
        session = new_session(None)
        list(session.run(dedent("""
            x, y = coordinates("x", "y")
            I = ideal(x,y)
            V = zariski_tangent_space(I)
            e = embedding_dimension(I)
        """)))
        v = session.variables["V"]
        self.assertEqual((v.dimension, v.jacobian_rank, v.ambient_dimension, v.basis), (0, 2, 2, ()))
        self.assertEqual(tuple(v.equations), v.coordinates)
        self.assertEqual(session.variables["e"].value, 0)
        for op in ("zariski_tangent_space", "embedding_dimension"):
            with self.subTest(op=op), self.assertRaisesRegex(SessionError, "point must lie on the scheme"):
                list(session.run(f"{op}(I, point=[1,0])"))

    def test_direction_coefficient_beyond_uint64_stays_exact(self):
        session = new_session(None)
        list(session.run(dedent("""
            x, y = coordinates("x", "y")
            n = 2^64+7
            I = ideal(n*x-y+x^n)
            V = zariski_tangent_space(I)
            a, b = V.coordinates
            ok = V.equations[0] == a-b/n
        """)))
        v = session.variables["V"]
        self.assertTrue(session.variables["ok"])
        self.assertEqual(v.basis, ((Fraction(1, 2**64 + 7), Fraction(1)),))
