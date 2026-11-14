"""Exercise local algebra results in the shared Atlas prompt and script namespace."""
import json
import os
import subprocess
import sys
import unittest
from textwrap import dedent

from engine.session import new_session
from engine.session_algebra import algebra_request
from tests.cli_support import ROOT, run_prompt, run_script


class LocalAlgebraSessionTests(unittest.TestCase):
    def test_embedded_milnor_number_preserves_4501_digits_with_a_decimal_guard(self):
        source = dedent("""
            import json, sys
            from decimal import Decimal
            from engine.session_algebra import milnor_number
            from engine.session_polynomial import coordinates

            n = 10**4500 + 7
            x = coordinates("x")
            mu = milnor_number(x**(n+1), compute="cpu")
            assert type(mu.value) is int and mu.value == n
            assert repr(mu) == "1" + "0"*4499 + "7"
            record = json.loads(mu.json, parse_int=Decimal)
            assert isinstance(record["result"]["value"], Decimal) and record == mu.record
            if hasattr(sys, "get_int_max_str_digits"):
                assert sys.get_int_max_str_digits() == 4300
        """)
        env = dict(os.environ, ATLAS_GROEBNER="native", PYTHONINTMAXSTRDIGITS="4300")
        result = subprocess.run([sys.executable, "-c", source], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_quotient_bases_and_cusp_invariants_at_both_entry_points(self):
        source = dedent("""
            x, y = coordinates("x", "y")
            I = ideal(x^2*(x-1), y, order="lex")
            G = quotient_basis(I, compute="cpu")
            L = local_quotient_basis(I, compute="cpu")
            print(G.type, G.scope, G.finite, G.dimension, G.order)
            print(L.type, L.scope, L.finite, L.dimension, L.order)
            print(G.basis[0]+G.basis[1]+G.basis[2] == 1+x+x^2, L.basis[0]+L.basis[1] == 1+x)
            f = y^2-x^3
            A = local_jacobian_algebra(f, order="lex", compute="cpu")
            print(A.type, A.scope, A.dimension, A.generators[0] == -3*x^2, A.generators[1] == 2*y)
            print(A.basis[0]+A.basis[1] == 1+x, A.coordinates[0] == x)
            mu = milnor_number(f, compute="cpu")
            tau = tjurina_number(f, compute="cpu")
            print(mu.type, mu.finite, mu.value, tau.type, tau.finite, tau.value)
        """).lstrip()
        expected = [
            "quotient-basis global True 3 lex", "quotient-basis origin True 2 lex", "True True",
            "local-jacobian-algebra origin 2 True True", "True True", "integer True 2 integer True 2",
        ]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_empty_and_infinite_results_and_constant_polynomial_coordinates(self):
        session = new_session(None)
        list(session.run(dedent("""
            x = coordinates("x")
            empty = local_quotient_basis(ideal(x-1))
            infinite = quotient_basis(ideal(variables=x))
            loc = local_quotient_basis(ideal(variables=x))
            A = local_jacobian_algebra(0, variables=(x,), order="lex", compute="cpu")
            mu = milnor_number(0, variables=(x,), order="lex", compute="cpu")
            tau = tjurina_number(0, variables=(x,), order="lex", compute="cpu")
        """)))
        values = session.variables
        empty = values["empty"]
        self.assertTrue(empty.finite)
        self.assertEqual(empty.dimension, 0)
        self.assertEqual(len(empty.basis), 0)
        for name in ("infinite", "loc", "A"):
            result = values[name]
            self.assertFalse(result.finite)
            self.assertIsNone(result.dimension)
            self.assertIsNone(result.basis)
        self.assertEqual(tuple(values["A"].generators), (0,))
        for name in ("A", "mu", "tau"):
            result = values[name]
            self.assertEqual(result.coordinates, (values["x"],))
            self.assertEqual(result.scope, "origin")
            self.assertEqual(result.order, "lex")
            self.assertEqual(result.execution["requested"], "cpu")
        for name in ("mu", "tau"):
            self.assertFalse(values[name].finite)
            self.assertIsNone(values[name].value)

    def test_tjurina_number_also_imposes_the_original_equation(self):
        session = new_session(None)
        list(session.run(dedent("""
            x, y = coordinates("x", "y")
            f = x^4 + y^5 + x^2*y^3
            mu = milnor_number(f)
            tau = tjurina_number(f)
        """)))
        self.assertEqual(session.variables["mu"].value, 12)
        self.assertEqual(session.variables["tau"].value, 11)

    def test_algebra_request_restores_basis_and_derivatives_in_its_ring(self):
        request = {"operation": "local-jacobian-algebra", "ring": {"variables": ["if"], "order": "lex"},
                   "operands": {"polynomial": "if^3"}, "compute": "cpu"}
        result = algebra_request(json.dumps(request))
        t, = result.coordinates
        self.assertEqual(tuple(result.generators), (3*t**2,))
        self.assertCountEqual(result.basis, (1, t))
        self.assertEqual(json.loads(result.json), result.record)
        replay = result.record["replay"]["script"]
        self.assertIn("algebra_request(", replay)
        script = run_script(replay)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(json.loads(script.stdout)["result"], result.result)
        again = algebra_request(json.dumps(request))
        with self.assertRaisesRegex(ValueError, "different coordinate spaces"):
            _ = again.coordinates[0] + result.basis[0]
