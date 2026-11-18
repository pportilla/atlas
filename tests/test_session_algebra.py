from contextlib import nullcontext
from fractions import Fraction
import json
import sys
import unittest
from textwrap import dedent
from unittest.mock import patch

from engine.session import new_session
from engine.session_polynomial import coordinates
from engine.session_algebra import (
    algebra_request, eliminate, gcd, groebner, ideal, intersection, membership, normal_form, polynomial,
)
from tests.cli_support import run_prompt, run_script


class AlgebraSessionTests(unittest.TestCase):
    def test_coordinate_declarations_are_local_to_one_session_and_ordered_ring(self):
        session = new_session(None)
        list(session.run("""
x, y = coordinates("x", "y")
f = y^2 + x
u = coordinates("x")
a, b = coordinates("y", "x")
x, y = coordinates("x", "y", coeffs=QQ)
"""))
        values = session.variables
        for name in ("u", "a"):
            self.assertIsNot(values[name].parent, values["f"].parent)
        self.assertIs(values["x"].parent, values["f"].parent)
        other = new_session(None)
        list(other.run('x, y = coordinates("x", "y")'))
        self.assertIsNot(other.variables["x"].parent, values["x"].parent)

    def test_fresh_coordinate_requests_and_invalid_declarations_do_not_change_the_default(self):
        session = new_session(None)
        list(session.run('a = coordinates("x", fresh=True)\nx = coordinates("x")\nb = coordinates("x", fresh=True)'))
        values = session.variables
        self.assertEqual(len({values[name].parent for name in ("a", "x", "b")}), 3)
        for invalid, message in (('coordinates("x", fresh=1)', "fresh"),
                                 ('coordinates("x", coeffs=[])', "coefficient field"),
                                 ('coordinates(["x"])', "coordinate names"),
                                 ('coordinates("x", "x")', "distinct coordinate names")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, message):
                list(session.run(invalid))
        list(session.run('same = coordinates("x")'))
        self.assertIs(values["same"].parent, values["x"].parent)

    def test_gcd_reuses_polynomial_space_and_normalization_order(self):
        x, y = coordinates("x", "y")
        f = x + 2*y**2
        lex = gcd(f, 0, order="lex")
        graded = gcd(f, 0)
        self.assertEqual(lex.polynomial, f)
        self.assertEqual(graded.polynomial, f/2)
        self.assertEqual(gcd(lex, x*f).order, "lex")
        with self.assertRaisesRegex(ValueError, "explicit monomial order"):
            gcd(lex, graded)
        self.assertEqual(gcd(lex, graded, order="lex"), f)
        other_x, _ = coordinates("x", "y")
        with self.assertRaisesRegex(ValueError, "different coordinate spaces"):
            gcd(other_x, f)
        zero = gcd(0, 0)
        self.assertEqual((zero, zero.variables), (0, ()))
        unit = gcd(0, Fraction(-3, 7), variables=[x, y])
        self.assertEqual(unit, 1)
        self.assertIs(unit.parent, x.parent)

    def test_gcd_uses_prepared_operands_and_preserves_records(self):
        from engine import session_ideal_native
        from engine.algebra import dispatch
        from engine.session_polynomial import Polynomial

        for binding in ("installed", "ctypes"):
            backend = (patch.object(session_ideal_native, "_compiled", return_value=None)
                       if binding == "ctypes" else nullcontext())
            with backend:
                z, a, b = coordinates("z", "a", "b")
                h = 2*z*b + 3*a**2 - 1
                left, right = -h*z/7, h*(z+1)/5
                for compute, order, expected in (("auto", "lex", h/2), ("cpu", "grevlex", h/3)):
                    with (self.subTest(binding=binding, compute=compute, order=order),
                          patch("engine.worker_process.run_worker", side_effect=AssertionError("GCD started a worker"))):
                        with patch.object(Polynomial, "__str__", side_effect=AssertionError("GCD serialized an operand")):
                            value = gcd(left, right, order=order, compute=compute)
                            chained = gcd(value, left)
                        self.assertIs(value.polynomial, value)
                        self.assertIs(value.parent, h.parent)
                        self.assertEqual((value, chained, chained.order), (expected, expected, order))
                        self.assertEqual((value.status, value.operation, value.type), ("ok", "gcd", "polynomial"))
                        recorded = dispatch({"operation": "gcd", "ring": value.ring, "compute": compute,
                                             "operands": {"polynomial": str(left), "other_polynomial": str(right)}})
                        self.assertEqual(value.result, recorded["result"])
                        self.assertEqual(value.record["replay"], recorded["replay"])
                        self.assertEqual(json.loads(value.json), value.record)
                        self.assertEqual((value.execution["requested"], value.execution["backend"]), (compute, "cpu"))
                        self.assertEqual(value.execution["stages"][0]["operation"], "polynomial_gcd")

    def test_wide_gcd_survives_failed_assignment_and_replays_after_rebinding(self):
        setup = dedent("""
            x, y, z = coordinates("x", "y", "z")
            h = x^(2^70+3) + (2^200+7)*y/3
            d = gcd(h*z, h*(z+1))
            x, y, z = coordinates("x", "y", "z", fresh=True)
        """).lstrip()
        bad = "d = gcd(d, x)\n"
        check = "print(d == h, d.parent == h.parent, d.parent == x.parent)\n"
        status, out, err = run_prompt(setup + bad + check + "quit\n")
        self.assertEqual(status, 0)
        self.assertIn("different coordinate spaces", err)
        self.assertIn("True True False", out)

        script = run_script(setup + check + "print(d.json)\n" + bad)
        self.assertEqual(script.returncode, 2)
        self.assertIn("different coordinate spaces", script.stderr)
        check_output, raw = script.stdout.splitlines()
        self.assertEqual(check_output, "True True False")
        record = json.loads(raw)
        replay = run_script(record["replay"]["script"])
        self.assertEqual((replay.returncode, replay.stderr), (0, ""))
        self.assertEqual(json.loads(replay.stdout)["result"], record["result"])

    def test_short_sparse_gcd_with_large_degree_or_coefficient_uses_the_worker(self):
        from engine import session_algebra

        x, y, z = coordinates("x", "y", "z")
        for h in (x**129 + y, x + (2**4100 + 7)*y):
            with self.subTest(common_factor=str(h)[:40]):
                left, right = h*z, h*(z+1)
                with patch.object(session_algebra, "_evaluate", wraps=session_algebra._evaluate) as worker:
                    value = gcd(left, right)
                    worker.assert_called_once()
                self.assertEqual(value, h)
                self.assertIs(value.parent, x.parent)

    def test_reuse_basis_and_polynomial_in_original_coordinates(self):
        from engine.algebra import calculate_algebra

        x, y = coordinates("x", "y")
        I = ideal(x**2-1, y-x, order="lex")
        basis = groebner(I, compute="cpu")
        # Even a Python caller editing the exported JSON must not be able to
        # substitute arbitrary generators into the trusted-basis worker path.
        basis.record["result"]["generators"][:] = ["1"]
        with patch("engine.algebra.calculate_algebra", wraps=calculate_algebra) as dispatch:
            remainder = normal_form(x**2 + y/2, basis, compute="cpu")
            self.assertTrue(dispatch.call_args.kwargs["known_basis"])
        self.assertEqual(remainder.polynomial, 1 + y/2)
        self.assertEqual(remainder + x, x + 1 + y/2)
        self.assertEqual(ideal(remainder).order, "lex")
        self.assertEqual(normal_form(remainder, ideal(y), compute="cpu"), 1)
        self.assertFalse(membership(remainder, basis, compute="cpu"))
        self.assertTrue(membership(y**2 - 1, basis, compute="cpu"))
        self.assertEqual(json.loads(remainder.json), remainder.record)

    def test_exact_large_result_reused_without_changing_the_session_literal_limit(self):
        x = coordinates("x")
        I = ideal(x - 2)
        r = normal_form(x**20000, I, compute="cpu")
        self.assertEqual(r, 2**20000)
        self.assertEqual(normal_form(r, I, compute="cpu"), r)

    def test_same_spelling_in_different_spaces_is_rejected(self):
        x, y = coordinates("x", "y")
        a, b = coordinates("x", "y")
        I = ideal(x + y)
        with self.assertRaisesRegex(ValueError, "different coordinate spaces"):
            normal_form(a + b, I)
        with self.assertRaisesRegex(ValueError, "different coordinate spaces"):
            ideal(x, b)
        with self.assertRaisesRegex(ValueError, "different monomial orders"):
            intersection(I, ideal(x, order="lex"))
        with self.assertRaisesRegex(ValueError, "declared order"):
            ideal(0, variables=(y, x))

    def test_elimination_supplies_surviving_coordinates_and_scalar_field(self):
        x, y = coordinates("x", "y")
        I = ideal(x**2 - 1, y - x, order="lex")
        projected = eliminate(I, x, compute="cpu")
        t, = projected.coordinates
        self.assertEqual(projected.variables, ("y",))
        self.assertTrue(membership(t**2 - 1, projected, compute="cpu"))
        again = eliminate(I, x, compute="cpu")
        self.assertEqual(projected.generators, again.generators)
        constants = eliminate(projected, t, compute="cpu")
        self.assertEqual(constants.variables, ())
        self.assertEqual(normal_form(3, constants, compute="cpu"), 3)

    def test_scalar_generators_keep_the_ambient_variables(self):
        session = new_session(None)
        list(session.run("""
x, y = coordinates("x", "y", coeffs=QQ)
Z = ideal(variables=(x,y))
U = ideal(1, variables=(x,y))
d = dimension(Z, compute="cpu")
r = normal_form(2/3, U, compute="cpu")
"""))
        self.assertEqual(session.variables["d"].value, 2)
        self.assertEqual(session.variables["r"], 0)
        self.assertEqual(session.variables["r"].variables, ("x", "y"))

    def test_replay_handles_variable_names_reserved_by_the_session(self):
        request = dict(operation="normal-form", ring=dict(variables=["if"], order="lex"),
                       operands=dict(ideal=[], polynomial="if+1/2"), compute="cpu")
        result = algebra_request(json.dumps(request))
        coordinate, = result.coordinates
        self.assertEqual(2 * (result - coordinate), 1)
        self.assertEqual(str(result), "if+1/2")
        self.assertEqual(result.execution["requested"], "cpu")

    def test_quoted_polynomial_uses_original_or_projected_coordinates(self):
        session = new_session(None)
        list(session.run("""
a, b = coordinates("if", "y", coeffs=QQ)
p = polynomial("if-y", variables=[a,b])
I = ideal(p, b^2-1)
J = eliminate(I, a)
t, = J.coordinates
q = polynomial("y/2", variables=t)
r = normal_form(q, J)
c = polynomial("2/3", variables=[])
"""))
        values = session.variables
        self.assertEqual(values["p"], values["a"] - values["b"])
        self.assertEqual(values["r"], values["t"] / 2)
        self.assertEqual(values["c"].variables, ())
        self.assertEqual(str(values["c"]), "2/3")
        with self.assertRaisesRegex(ValueError, "declared order"):
            list(session.run('p = polynomial("y", variables=[b,a])'))

    def test_distinct_ordinary_literals_reuse_coordinates_without_a_worker(self):
        x, y = coordinates("x", "y")
        with patch("engine.worker_process.run_worker", side_effect=AssertionError("unexpected worker")):
            for n in range(1, 9):
                self.assertEqual(polynomial(f"x+{n}*y", variables=(x, y)), x + n*y)

    def test_polynomial_requires_one_live_ring_or_its_coordinates(self):
        x = coordinates("x")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            polynomial("x", ring=x.parent, variables=(x,))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            polynomial("x")
        with self.assertRaisesRegex(TypeError, "parent ring"):
            polynomial("x", ring=ideal(x).ring)

    def test_worker_preserves_large_decimal_and_monomial_literals(self):
        from engine.worker_process import run_worker

        x = coordinates("x")
        get_limit = getattr(sys, "get_int_max_str_digits", lambda: None)
        previous_limit = get_limit()
        digits = "9" * 5001
        with patch("engine.worker_process.run_worker", wraps=run_worker) as isolated:
            value = polynomial(f"{digits}*x+1", variables=x)
            monomial = polynomial("x^129+1", variables=x)
        self.assertEqual(isolated.call_count, 2)
        self.assertEqual(value, (10**5001-1)*x + 1)
        self.assertEqual(monomial, x**129 + 1)
        self.assertEqual(get_limit(), previous_limit)

    def test_expansion_is_interruptible_and_another_literal_can_follow_it(self):
        x, y = coordinates("x", "y")
        with patch("engine.worker_process.run_worker", side_effect=KeyboardInterrupt) as isolated:
            with patch("engine.session_algebra.parse_expression", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    polynomial("x+y", variables=(x, y))
                isolated.assert_not_called()
            with self.assertRaises(KeyboardInterrupt):
                polynomial("(x+y)^10000", variables=(x, y))
            isolated.assert_called_once()
            self.assertEqual(polynomial("x-y", variables=(x, y)), x-y)

    def test_quoted_polynomial_rejects_invalid_token_boundaries_and_denominators(self):
        x, y = coordinates("x", "y")
        for source in ("x * * 2", "x^2^3", "x y", "x/y", "x/0", "x+z"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                polynomial(source, variables=(x, y))
