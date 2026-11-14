"""Point-aware local quotient and hypersurface invariants in Atlas sessions."""
import json
import unittest
from textwrap import dedent

from engine.session import new_session
from engine.session_expressions import SessionError
from tests.cli_support import run_prompt, run_script


class SessionPointInvariantTests(unittest.TestCase):
    def test_translated_cusp_at_prompt_and_in_script(self):
        source = dedent("""
            y, x = coordinates("y", "x", coeffs=QQ)
            p = (-2/3, 1/2)
            u, v = x-1/2, y+2/3
            f = v^2-u^3
            Q = local_quotient_basis(ideal(v,u^2), point=p, compute="cpu")
            A = local_jacobian_algebra(f, point=p, order="lex", compute="cpu")
            mu = milnor_number(f, point=("-4/6", "2/4"), compute="cpu")
            tau = tjurina_number(f, point=A.point, compute="cpu")
            print(Q.point, Q.scope, Q.dimension, Q.basis[1] == u)
            print(A.point == Q.point, A.scope, A.dimension, A.basis[1] == u)
            print(A.coordinates == (y,x), A.generators[0] == 2*v, A.generators[1] == -3*u^2)
            print(mu.point == Q.point, tau.point == Q.point, mu.scope, tau.scope, mu.value, tau.value)
        """).lstrip()
        expected = [
            "['-2/3', '1/2'] point 2 True", "True point 2 True",
            "True True True", "True True point point 2 2",
        ]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        code, out, err = run_prompt(source + "quit\n")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(out.replace("atlas> ", "").strip().splitlines(), expected)

    def test_implicit_origin_and_explicit_zero_have_distinct_records(self):
        session = new_session(None)
        list(session.run('x = coordinates("x")\nI = ideal(x^2)'))
        for operation, operand in (("local_quotient_basis", "I"), ("local_jacobian_algebra", "x^3"),
                                   ("milnor_number", "x^3"), ("tjurina_number", "x^3")):
            with self.subTest(operation=operation):
                list(session.run(f'implicit = {operation}({operand})\n'
                                 f'explicit = {operation}({operand}, point=["-0/7"])'))
                implicit, explicit = (session.variables[k] for k in ("implicit", "explicit"))
                self.assertEqual((implicit.scope, implicit.point), ("origin", ["0"]))
                self.assertNotIn("point", implicit.result)
                self.assertEqual((explicit.scope, explicit.point), ("point", ["0"]))
                self.assertEqual(explicit.result["point"], ["0"])
                explicit.point[0] = "9"
                self.assertEqual(explicit.point, ["0"])
        list(session.run("G = quotient_basis(I)"))
        for expression in ("G.point", "quotient_basis(I, point=[0])"):
            with self.subTest(expression=expression), self.assertRaises(SessionError):
                list(session.run(expression))
        list(session.run("B = local_standard_basis(I)\nat_zero = local_quotient_basis(B)"))
        at_zero = session.variables["at_zero"]
        self.assertEqual((at_zero.scope, at_zero.result["point"]), ("point", ["0"]))

    def test_saved_basis_inherits_its_rational_point_and_replays(self):
        session = new_session(None)
        list(session.run(dedent("""
            x = coordinates("x")
            I = ideal((x+1/2)^2*(x-1))
            B = local_standard_basis(I, point=["-2/4"])
            Q = local_quotient_basis(B, compute="cpu")
            repeated = local_quotient_basis(B, point=(-3/6,))
        """)))
        q = session.variables["Q"]
        self.assertEqual((q.point, q.dimension), (["-1/2"], 2))
        self.assertEqual(q.basis, session.variables["repeated"].basis)
        with self.assertRaisesRegex(SessionError, "point"):
            list(session.run("local_quotient_basis(B, point=[1])"))
        replay = run_script(q.record["replay"]["script"])
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout)["result"], q.result)

    def test_nonisolated_germ_at_fractional_point(self):
        session = new_session(None)
        list(session.run(dedent("""
            x, y = coordinates("x", "y")
            p = (2/3, -1/7)
            f = (x-2/3)^2
            Q = local_quotient_basis(ideal(x-2/3), point=p)
            A = local_jacobian_algebra(f, point=p)
            mu = milnor_number(f, point=p)
            tau = tjurina_number(f, point=p)
        """)))
        for name in ("Q", "A", "mu", "tau"):
            value = session.variables[name]
            self.assertEqual(value.point, ["2/3", "-1/7"])
            self.assertFalse(value.finite)
            if name in ("Q", "A"):
                self.assertIsNone(value.basis)
                self.assertIsNone(value.dimension)
            else:
                self.assertIsNone(value.value)
