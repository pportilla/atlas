"""Ideals and their operations in a live quotient ring."""
import json
import shlex
import unittest
from textwrap import dedent
from unittest.mock import patch

from engine.session import new_session
from engine.session_algebra import _evaluate
from engine.session_help import algebra_help_values, algebra_value_help
from engine.session_panel import workspace
from tests.cli_support import run_prompt, run_script, shell
from tests.http_support import endpoint, post


class QuotientIdealTests(unittest.TestCase):
    def check_session(self, source, expected):
        source = dedent(source).lstrip()
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stderr, "")
        self.assertEqual(script.stdout.splitlines(), expected)
        status, out, err = run_prompt(source + "quit\n")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(out.replace("atlas> ", "").strip().splitlines(), expected)

    def test_cusp_ideal_and_quotient_arithmetic(self):
        self.check_session("""
            x,y = coordinates("x","y")
            K = ideal(y^2+x^5).quotient_ring
            u,v = K.coordinates
            I = ideal((y,x^5), ring=K)
            J = ideal(u)
            print(I.parent == K, I.generators == (v,-v^2), J.parent == K)
            print(ideal((u,y),ring=K).generators == (u,v), ideal(1/7,v).parent == K)
            print(I.generators[0].parent == K, I.coordinates.parent == K)
            print(membership(y,I).value, membership(x,I).value)
            print(membership(u,I+J).value, membership(1,ideal_sum(I,J)).value)
            print(membership(u*v,I*J).value, membership(y,ideal_product(I,J)).value)
            print(membership(x^5,I^2).value, membership(y,ideal_power(I,2)).value)
            B = groebner(I)
            r = normal_form(y^2+x^5+x,B)
            print(B.parent == K, r.parent == K, r == u)
        """, ["True True True", "True True", "True True", "True False", "True False", "True False",
              "True False", "True True True"])

    def test_nilpotents_ground_quotient_and_wide_values(self):
        self.check_session("""
            x,y = coordinates("x","y")
            D = ideal(y^2).quotient_ring
            u,t = D.coordinates
            J = ideal(t)
            Z = ideal(ring=D)
            print(membership(0,Z).value, membership(1,Z).value)
            print(membership(t^2,J^2).value, membership(t,J^2).value)
            print(membership(t,ideal_power(J,18446744073709551617)).value)
            print(membership(1,J^0).value, membership(1,ideal(1/7, ring=D)).value)
            f = (18446744073709551629/7)*x^18446744073709551617
            H = ideal(f, ring=D)
            print(H.generators[0].parent == D, H.generators[0].representative == f)
            Q = ideal(1,ring=QQ).quotient_ring
            print(membership(1,ideal(ring=Q)).value)
        """, ["True False", "True False", "False", "True True", "True True", "True"])

    def test_failed_operations_preserve_the_existing_binding(self):
        setup = dedent("""
            x,y = coordinates("x","y")
            X,Y = coordinates("x","y",fresh=True)
            I = ideal(y^2)
            K = I.quotient_ring
            L = quotient_ring(I,fresh=True)
            a,b = K.coordinates
            z = L(y)
            keep = ideal(b)
        """).lstrip()
        for expression in (
            "ideal(X,ring=K)", "ideal(b,z)", "ideal(b,ring=L)", "ideal(b,x)", "ideal(x,b)",
            "ideal(b,ring=K,order='lex')", "keep + ideal(z)", "ideal(y) * keep",
            "membership(z,keep)", "normal_form(Y,keep)", "local_standard_basis(keep)",
        ):
            with self.subTest(expression=expression):
                source = setup + "keep = " + expression + "\n"
                script = run_script(source + 'print("continued")\n')
                self.assertNotEqual(script.returncode, 0)
                self.assertNotIn("continued", script.stdout)
                status, output, error = run_prompt(source + "print(keep.parent == K, keep.generators == (b,))\nquit\n")
                self.assertEqual(status, 0)
                self.assertRegex(error, r"ring|quotient|coordinate|order")
                self.assertIn("True True", output)

    def test_further_quotient_uses_the_defining_ideal_and_the_natural_projection(self):
        self.check_session("""
            x,y = coordinates("x","y")
            K = ideal(x^2-y).quotient_ring
            u,v = K.coordinates
            J = ideal(v)
            L = J.quotient_ring
            print(L.defining_ideal == J, J.parent == K, L == quotient_ring(J))
            print(L.ambient == K.ambient, L(v) == 0, L(u)^2 == 0, L(u) == L(x))
            print(L.projection.domain == K, L.projection.codomain == L, L.projection(u) == L(x))
            H = ideal((u,v),ring=L)
            print(H.parent == L, H.generators == (L(x),L(0)))
        """, ["True True True", "True True True True", "True True True", "True True"])

    def test_generators_and_remainders_respect_the_parent_monomial_order(self):
        self.check_session("""
            x,y = coordinates("x","y")
            A = ideal(x-y^2,order="lex").quotient_ring
            B = ideal(x-y^2,order="grevlex").quotient_ring
            I = ideal(x,ring=A)
            J = ideal(y^2,ring=B)
            print(I.generators[0].representative == y^2, J.generators[0].representative == x)
            print(normal_form(y+x,I) == A(y), normal_form(y+y^2,J) == B(y))
        """, ["True True", "True True"])

    def test_workspace_tracks_quotient_ideals_and_their_generators_without_reduction(self):
        session = new_session(None)
        list(session.run('x,y=coordinates("x","y")\nK=ideal(y^2+x^5).quotient_ring\n'
                         'J=ideal((y,x^5),ring=K)'))
        with patch("engine.session_quotient._evaluate", side_effect=AssertionError("inspecting a generator recomputed it")):
            list(session.run('g=J.generators[1]\nB=groebner(J)'))
            generator = session.variables["g"]
            self.assertEqual((str(generator), generator.result["representative"]), ("-y^2", "-y^2"))
            self.assertIn("residue", algebra_value_help("g", algebra_help_values(session)["g"]).lower())
            inventory = workspace(session.variables)
            self.assertEqual(inventory.header()[0], ["R1 = QQ[x, y] → x, y", "K = R1/(x^5+y^2) → J, g, B"])
            self.assertEqual([binding.role for binding in inventory.rings[session.variables["K"]].members],
                             ["Ideals", "Residue classes", "Ideals"])
        record = generator.record
        self.assertEqual(record["result"], generator.result)
        replay = run_script(record["replay"]["script"])
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout)["result"], generator.result)

    def test_live_and_worker_routes_return_the_same_ideal_in_the_same_parent(self):
        session = new_session(None)
        list(session.run('x,y=coordinates("x","y")\nK=ideal(y^2+x^5).quotient_ring'))
        with patch("engine.session_algebra._evaluate", side_effect=AssertionError("unexpected worker")):
            list(session.run("J=ideal((y,x^5),ring=K)"))
        with (patch("engine.algebra.dispatch", return_value=None),
              patch("engine.session_algebra._evaluate", wraps=_evaluate) as worker):
            list(session.run("L=ideal((y,x^5),ring=K)"))
        worker.assert_called_once()
        first, second = session.variables["J"], session.variables["L"]
        self.assertIs(first.parent, second.parent)
        self.assertEqual(first.result, second.result)

    def test_http_cli_and_replay_keep_the_quotient_presentation(self):
        ring = {"field": "QQ", "variables": ["x", "y"], "order": "grevlex", "relations": ["y^2+x^5"]}
        cases = (
            ("ideal", {"ideal": ["y", "x^5"]}, "quotient-ideal"),
            ("ideal-sum", {"ideal": ["y"], "other_ideal": ["x"]}, "quotient-ideal"),
            ("ideal-product", {"ideal": ["y"], "other_ideal": ["y"]}, "quotient-ideal"),
            ("ideal-power", {"ideal": ["y"], "exponent": "2"}, "quotient-ideal"),
            ("groebner", {"ideal": ["x^5", "y"]}, "quotient-ideal"),
            ("membership", {"ideal": ["y"], "polynomial": "x^5"}, "boolean"),
            ("normal-form", {"ideal": ["y"], "polynomial": "x^5+x"}, "quotient-element"),
        )
        with endpoint() as url:
            for operation, operands, expected_type in cases:
                with self.subTest(operation=operation):
                    status, record = post(url, operation, operands, ring=ring)
                    self.assertEqual(status, 200, record)
                    result = record["result"]
                    self.assertEqual(result["type"], expected_type)
                    self.assertEqual(result["ring"]["relations"], ["x^5+y^2"])
                    if operation == "ideal":
                        self.assertEqual(result["generators"], ["y", "-y^2"])
                    elif operation == "normal-form":
                        self.assertEqual(result["representative"], "x")
                    elif operation == "membership":
                        self.assertTrue(result["value"])
                    args = shlex.split(record["replay"]["command"])
                    self.assertIn("--relation", args)
                    for run in (shell(args), shell(["python3", *args]), run_script(record["replay"]["script"])):
                        self.assertEqual(run.returncode, 0, run.stderr)
                        self.assertEqual(json.loads(run.stdout)["result"], result)
            status, record = post(url, "ideal", {"ideal": ["x"]}, ring=dict(ring, relations=[]))
            self.assertEqual(status, 200, record)
            self.assertEqual(record["result"]["ring"]["relations"], [])
            self.assertIn("--quotient-ring", record["replay"]["command"])
            status, record = post(url, "dimension", {"ideal": ["x"]}, ring=ring)
            self.assertEqual(status, 400, record)
            self.assertIn("quotient", record["error"]["message"].lower())
