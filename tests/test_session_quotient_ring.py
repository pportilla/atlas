"""Quotient arithmetic, coordinate ownership, and saved requests at the prompt and in scripts."""
import json
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
from fractions import Fraction
from textwrap import dedent
from threading import Barrier
import unittest
from unittest.mock import patch

from engine.algebra import calculate_algebra
from engine.session_algebra import algebra_request, ideal
from engine.session_polynomial import coordinates
from engine.session_quotient import quotient_ring
from tests.cli_support import run_prompt, run_script


class QuotientRingSessionTests(unittest.TestCase):
    def test_prompt_and_script(self):
        source = dedent("""
            x, y = coordinates("x", "y", coeffs=QQ)
            A = quotient_ring(ideal(x*y-1, order="lex"))
            u, v = A.gens
            print(u.inverse() == v, (1/u).representative == y, u^-3 == v^3)
            print(A(x^2*y) == u, (x-u).representative, (x+v)*u == A(x^2+1))
            print(u.parent == A, A.order, A.defining_ideal.generators == (x*y-1,))
            D = quotient_ring(ideal(x^2))
            t = D(x)
            print(t^2 == 0, t.is_unit(), (1+t).inverse() == 1-t, -t == 0-t)
            print(D.apply("unit", 1+t).value, (1/t.parent(2)).representative)
            I = ideal(y^2+x^5)
            K = I.quotient_ring
            L = quotient_ring(I, fresh=True)
            print(K == quotient_ring(I), K.defining_ideal == I, L != K, L.ambient == K.ambient)
            B = quotient_ring(I, compute="cpu")
            print(B == K, B(x) + K(x) == K(2*x), K.declaration_executions.cpu.requested)
        """).lstrip()
        expected = ["True True True", "True 0 True", "True lex True", "True False True True", "True 1/2",
                    "True True True True", "True True cpu"]
        run = run_script(source)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.splitlines(), expected)
        status, output, err = run_prompt(source + "quit\n")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_ownership_and_division_errors(self):
        x = coordinates("x")
        a = quotient_ring(ideal(x**2))
        b = quotient_ring(ideal(x**2))
        u, v = a(x), b(x)
        for mixed in (lambda: a(v), lambda: u+v, lambda: u == v):
            with self.assertRaisesRegex(ValueError, "different quotient rings"):
                mixed()
        for mixed in (lambda: x == u, lambda: u == x, lambda: x != u):
            with self.assertRaisesRegex(ValueError, "A\\(f\\)"):
                mixed()
        with self.assertRaisesRegex(ValueError, "different coordinate spaces"):
            a(coordinates("x"))
        for operation in (lambda: 1/u, lambda: u**-1, lambda: u.inverse(), lambda: 0/u):
            with self.assertRaisesRegex(ValueError, "unit|invert"):
                operation()

    def test_default_quotient_belongs_to_the_declaring_ideal(self):
        x = coordinates("x")
        relations = ideal(2*x, x**2)
        with patch("engine.algebra.calculate_algebra", wraps=calculate_algebra) as compute:
            ring = quotient_ring(relations)
            self.assertIs(relations.quotient_ring, ring)
            self.assertIs(quotient_ring(relations), ring)
            self.assertEqual(compute.call_count, 1)
        self.assertIs(ring.defining_ideal, relations)
        self.assertIs(copy(ring), ring)
        self.assertIs(deepcopy(ring(x)).parent, ring)
        derived = relations + relations
        derived_parent = derived.quotient_ring
        for clone in (copy, deepcopy):
            copied = clone(derived)
            self.assertIs(copied.quotient_ring, derived_parent)
            self.assertIs(copied.quotient_ring.defining_ideal, copied)
        self.assertEqual(ring.defining_ideal.generators, (2*x, x**2))
        self.assertEqual(ring.relations, (x,))
        independent = quotient_ring(relations, fresh=True)
        for other in (independent, quotient_ring(ideal(2*x, x**2))):
            self.assertIsNot(other, ring)
            with self.assertRaisesRegex(ValueError, "different quotient rings"):
                ring(other(x))
        first = ring.record
        self.assertIs(quotient_ring(relations, compute="cpu"), ring)
        self.assertEqual(ring.record, first)
        self.assertEqual(ring.declaration_executions["cpu"]["requested"], "cpu")
        self.assertIs(independent.apply("declare", compute="cpu"), independent)
        self.assertIs(ring.apply("declare"), ring)
        saved = ring.declaration_executions
        self.assertEqual(set(saved), {"cpu", "auto"})
        saved["auto"].clear()
        self.assertEqual(ring.declaration_executions["auto"]["requested"], "auto")
        with self.assertRaisesRegex(ValueError, "Choose cpu, auto or gpu"):
            quotient_ring(relations, compute="misspelled")
        with patch("engine.algebra.calculate_algebra", return_value=(503, {"error": {"message": "GPU unavailable"}})):
            with self.assertRaisesRegex(ValueError, "GPU unavailable"):
                quotient_ring(relations, compute="gpu")
        self.assertEqual(set(ring.declaration_executions), {"cpu", "auto"})
        with self.assertRaisesRegex(ValueError, "fresh must"):
            quotient_ring(relations, fresh=1)
        self.assertIs(relations.quotient_ring, ring)

    def test_concurrent_modes_share_the_first_published_parent(self):
        x = coordinates("x")
        relations = ideal(x**2)
        entered = Barrier(2)

        def evaluate(*args, **kwargs):
            entered.wait(timeout=10)
            return calculate_algebra(*args, **kwargs)

        with patch("engine.algebra.calculate_algebra", side_effect=evaluate), ThreadPoolExecutor(2) as pool:
            parents = list(pool.map(lambda mode: quotient_ring(relations, compute=mode), ("cpu", "auto")))
        self.assertIs(parents[0], parents[1])
        self.assertIs(relations.quotient_ring, parents[0])
        self.assertEqual(set(parents[0].declaration_executions), {"cpu", "auto"})

    def test_scalar_ring_zero_ring_and_exact_wide_inputs(self):
        q = quotient_ring(ideal(variables=()))
        self.assertEqual(q.gens, ())
        self.assertEqual((q(Fraction(1, 3)) / 2).representative, Fraction(1, 6))
        zero = quotient_ring(ideal(1, variables=()))
        self.assertTrue(zero(0).is_unit())
        self.assertEqual(zero(0).inverse().representative, 0)
        x = coordinates("x")
        a = quotient_ring(ideal(x**2-1))
        u = a(x)
        self.assertEqual((u**-(2**70+1)).representative, x)
        big = Fraction(2**91+3, 17)
        self.assertEqual((big * u / u).representative, big)

    def test_replay_has_a_fresh_parent_and_reuses_verified_basis(self):
        data = {"operation": "quotient-ring", "ring": {"variables": ["x", "y"]}, "compute": "cpu",
                "operands": {"ideal": ["x-y", "x+y"], "action": "add", "polynomial": "y", "other_polynomial": "1"}}
        value = algebra_request(json.dumps(data))
        self.assertEqual(value.representative, 1)
        self.assertIs(value.parent.defining_ideal.quotient_ring, value.parent)
        x, y = value.parent.ambient.coordinates
        self.assertEqual(value.parent.defining_ideal.generators, (x-y, x+y))
        self.assertIn("--gen x-y --gen x+y", value.parent.record["replay"]["command"])
        declaration = dict(data, operands={"ideal": ["2*x", "x^2"], "action": "declare"})
        replayed = algebra_request(json.dumps(declaration))
        self.assertIs(replayed.defining_ideal.quotient_ring, replayed)
        x, y = replayed.ambient.coordinates
        self.assertEqual(replayed.defining_ideal.generators, (2*x, x**2))
        self.assertEqual(replayed.relations, (x,))
        self.assertIsNot(replayed.ambient, value.parent.ambient)
        with patch("engine.algebra.calculate_algebra", wraps=calculate_algebra) as call:
            again = value + 1
            self.assertTrue(call.call_args.kwargs["known_basis"])
        self.assertIs(again.parent, value.parent)
        self.assertEqual(again.representative, 2)
        external = value.parent.result
        external["relations"].clear()
        self.assertEqual(value.parent(value.parent.coordinates[0]).representative, 0)

    def test_quotient_coordinates_satisfy_the_relation_at_prompt_and_in_scripts(self):
        source = dedent("""
            x, y = coordinates("x", "y")
            I = ideal(y^2+x^5)
            K = quotient_ring(I)
            u, v = K.coordinates
            print(K.coordinates, K.ambient.coordinates)
            print(u.parent == K, v.parent == K, K.ambient.coordinates == (x, y))
            print(v^2+u^5 == 0, y^2+x^5 == 0)
            print(K.projection(x) == u, u.coordinates == K.gens, v.representative == y)
        """).lstrip()
        expected = ["([x], [y]) (x, y)", "True True True", "True False", "True True True"]
        run = run_script(source)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.splitlines(), expected)
        status, output, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_parsing_uses_the_selected_parent_even_for_repeated_images_or_no_generators(self):
        from engine.session import new_session
        from engine.session_expressions import SessionError

        session = new_session(None)
        source = dedent("""
            x, y = coordinates("x", "y")
            K = ideal(x-y).quotient_ring
            G = K.coordinates
            u, v = G
            a = polynomial("x-y", ring=K)
            b = polynomial("x-y", variables=(u,v))
            print(G.parent == K, a.parent == b.parent, a == 0, b == 0)
            print(polynomial("x-y", variables=K.ambient.coordinates) == x-y)
            print(a.ring == K.ring, K.ring.relations == K.result.relations)
            Q = ideal(2, ring=QQ).quotient_ring
            z = polynomial("18446744073709551629/7", variables=Q.coordinates)
            print(z.parent == Q, z == 0)
        """).lstrip()
        expected = ["True True True True", "True", "True True", "True True"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        status, out, err = run_prompt(source + "quit\n")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(out.replace("atlas> ", "").strip().splitlines(), expected)

        list(session.run('x,y=coordinates("x","y")\nI=ideal(x-y)\nK=I.quotient_ring\nu,v=K.coordinates\nkeep=17'))
        for coords in ("(v,u)", "(u,u)", "(x,v)", "(K(x),v)", "(quotient_ring(I,fresh=True).gens[0],v)"):
            with self.subTest(coords=coords), self.assertRaisesRegex(SessionError, "declared order"):
                list(session.run(f'keep=polynomial("x", variables={coords})'))
            self.assertEqual(session.variables["keep"], 17)
        with self.assertRaisesRegex(SessionError, "ambient polynomial"):
            list(session.run("keep=ideal(u,v)"))
        self.assertEqual(session.variables["keep"], 17)
