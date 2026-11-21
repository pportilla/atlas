"""Geometric factors, their common coefficient field, and the live input embedding."""
from fractions import Fraction
import unittest
from unittest.mock import patch

import sympy as sp

from engine.algebra import calculate_algebra
from engine.session import new_session
from engine.session_help import algebra_value_help
from tests.cli_support import ROOT, run_prompt, run_script


def compute(names, expression, order="lex"):
    request = {"operation": "absolute-factor", "ring": {"variables": list(names), "order": order},
               "operands": {"polynomial": expression}, "compute": "cpu"}
    status, record = calculate_algebra(request)
    if status != 200:
        raise AssertionError(record)
    return record["result"]


class AbsoluteFactorizationTests(unittest.TestCase):
    def check_product(self, names, expression, powers, field_divisor=1, order="lex"):
        result = compute(names, expression, order)
        self.assertEqual(result["type"], "absolute-factorization")
        ring = result["ring"]
        self.assertEqual((ring["variables"], ring["order"]), (list(names), order))
        symbols = {name: sp.Symbol(name) for name in names}
        coordinates = tuple(symbols.values())
        field = ring["field"]
        theta = modulus = None
        if field == "QQ":
            self.assertEqual(field_divisor, 1)
        else:
            name = field["generator"]
            self.assertNotIn(name, symbols)
            theta = symbols[name] = sp.Symbol(name)
            modulus = sp.Poly(sp.sympify(field["minimal_polynomial"], locals=symbols), theta, domain=sp.QQ)
            self.assertEqual(modulus.degree(), field["degree"])
            self.assertEqual(field["degree"] % field_divisor, 0)
            self.assertTrue(modulus.is_irreducible)
            self.assertIn(field["root_index"], range(1, field["degree"] + 1))
        original = sp.sympify(expression, locals=symbols)
        reconstructed = sp.Rational(result["unit"])
        factors = []
        self.assertCountEqual([entry["multiplicity"] for entry in result["factors"]], powers)
        for entry in result["factors"]:
            factor = sp.sympify(entry["polynomial"], locals=symbols)
            poly = sp.Poly(factor, *coordinates, domain="EX")
            self.assertEqual(poly.terms(order=order)[0][1], 1)
            factors.append(poly)
            reconstructed *= factor ** entry["multiplicity"]
        for _, coefficient in sp.Poly(reconstructed - original, *coordinates, domain="EX").terms():
            if theta is None:
                self.assertEqual(coefficient, 0)
            else:
                self.assertTrue(sp.Poly(coefficient, theta, domain=sp.QQ).rem(modulus).is_zero)
        return result, factors

    def test_univariate_cubic_and_two_independent_quadratic_extensions(self):
        _, factors = self.check_product(("x",), "x^3-2", [1, 1, 1], field_divisor=6)
        self.assertEqual([poly.total_degree() for poly in factors], [1]*3)
        result, factors = self.check_product(("a", "a1"),
            "-7/5*(a^2+a1^2)^3*(a^2-2*a1^2)^2", [2, 2, 3, 3], field_divisor=4)
        self.assertEqual(result["unit"], "-7/5")
        self.assertEqual([poly.total_degree() for poly in factors], [1]*4)

    def test_nonlinear_conjugate_factors_and_monic_order(self):
        expr = "(x^2+y^3+z)^2-2*(y*z+x)^2"
        for order in ("lex", "grevlex"):
            with self.subTest(order=order):
                _, factors = self.check_product(("x", "y", "z"), expr, [1, 1], field_divisor=2, order=order)
                # Both branches are primitive and linear in z over Qbar[x,y].
                self.assertEqual([poly.degree(sp.Symbol("z")) for poly in factors], [1, 1])
        f = "-1/7*(2*x-3*y^2)^2*(x^2+1)"
        for order, unit in (("lex", "-4/7"), ("grevlex", "-9/7")):
            result, _ = self.check_product(("x", "y"), f, [2, 1, 1], field_divisor=2, order=order)
            self.assertEqual(result["unit"], unit)

    def test_quartic_section_repeats_each_of_two_conjugate_components(self):
        # At y=0 the section is x^4-2. Its four embeddings give two conics,
        # both defined over the smaller field of the squared section roots.
        result, factors = self.check_product(("x", "y"), "(x^2+y^2)^2-2*(y^2+1)^2", [1, 1],
                                             field_divisor=2)
        self.assertEqual(result["ring"]["field"]["degree"], 2)
        self.assertEqual([p.total_degree() for p in factors], [2, 2])

    def test_sextic_changes_no_constant_field(self):
        D, A, B = "(y^3-x^5)", "(y^2-x^3)", "(x^4+y^3)"
        f = f"(z^2-{D})*((z-{A})^2-{D})*((z-{A}-{B})^2-{D})"
        result, factors = self.check_product(("x", "y", "z"), f, [1, 1, 1])
        self.assertEqual(result["ring"]["field"], "QQ")
        self.assertEqual([poly.degree(sp.Symbol("z")) for poly in factors], [2]*3)

    def test_constants_zero_and_wide_values(self):
        result = compute((), "-13/8")
        self.assertEqual((result["unit"], result["ring"]["field"], result["factors"]), ("-13/8", "QQ", []))
        status, result = calculate_algebra({"operation": "absolute-factor", "ring": {"variables": ["x"]},
                                            "operands": {"polynomial": "x-x"}})
        self.assertEqual(status, 400)
        self.assertIn("zero", result["error"]["message"])
        exponent = 2**64 + 9
        scalar = Fraction(-2**320, 3)
        result = compute(("x", "y"), f"({scalar})*x^{exponent}*(y^2+1)")
        self.assertEqual(result["unit"], str(scalar))
        self.assertIn({"polynomial": "x", "multiplicity": exponent}, result["factors"])
        self.assertCountEqual([r["multiplicity"] for r in result["factors"]], [exponent, 1, 1])
        result = compute(("x", "y"), f"x^{exponent}+y")
        self.assertEqual(result["ring"]["field"], "QQ")
        self.assertEqual(result["factors"], [{"polynomial": f"x^{exponent}+y", "multiplicity": 1}])
        result = compute(("x", "y"), f"-3*x^7*y^4*(x^{2*exponent}+y^2)")
        self.assertEqual((result["unit"], len(result["factors"])), ("-3", 4))
        self.assertIn({"polynomial": "x", "multiplicity": 7}, result["factors"])
        self.assertIn({"polynomial": "y", "multiplicity": 4}, result["factors"])
        field = result["ring"]["field"]
        x, y, a = sp.symbols("x y " + field["generator"])
        modulus = sp.sympify(field["minimal_polynomial"])
        coeffs = []
        for entry in result["factors"]:
            if entry["multiplicity"] == 1:
                h = sp.sympify(entry["polynomial"])
                coeffs.append(h.coeff(y))
                self.assertEqual(sp.expand(h - x**exponent - coeffs[-1]*y), 0)
        u, v = coeffs
        self.assertEqual(sp.rem(u+v, modulus, a), 0)
        self.assertEqual(sp.rem(u*v-1, modulus, a), 0)

    def test_live_parents_and_failure_restore(self):
        setup = ('x,y=coordinates("x","y",coeffs=QQ)\n'
                 'P=x.parent\nf=x^2+y^2\nF=absolute_factor(f)\nK=F.ambient\n'
                 'g=F.factors[0].polynomial\nphi=F.inclusion\n'
                 'x,y=coordinates("x","y",coeffs=QQ,fresh=True)\n'
                 'print(g.parent==K, phi.domain==P, phi.codomain==K, x.parent==P)\n'
                 'print(phi(f)==F.factors[0].polynomial*F.factors[1].polynomial)\n'
                 'print(absolute_factor(f,compute="gpu").ambient==K, absolute_factor(x^2+y^2).ambient==K)\n')
        script = run_script(setup)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), ["True True True False", "True", "True False"])
        code, output, errors = run_prompt(setup+'F=absolute_factor(f,variables=(x,y))\nprint(F.ambient==K)\nquit\n')
        self.assertEqual(code, 0)
        self.assertIn("True True True False", output)
        self.assertIn("True", output.splitlines()[-2])
        self.assertTrue(errors)

        session = new_session(None)
        list(session.run('x,y=coordinates("x","y")\nP=x.parent\nF=absolute_factor(x^2+y^2)\n'
                         'K=F.ambient\nphi=F.inclusion\ng=F.factors[0].polynomial\n'
                         'x,y=coordinates("x","y",fresh=True)\nremoved=delete_ring(K)'))
        self.assertTrue({"F", "K", "phi", "g"}.isdisjoint(session.variables))
        self.assertTrue({"x", "y", "P"}.issubset(session.variables))

    def test_help_uses_the_saved_name_and_has_executable_examples(self):
        session = new_session(None)
        list(session.run('x,y=coordinates("x","y")\nP=x.parent\nF=absolute_factor(x^2+y^2)\nK=F.ambient'))
        before = dict(session.variables)
        with (patch("engine.session_algebra._evaluate", side_effect=AssertionError("Help started a worker")),
              patch("engine.polynomial_factorization.kernel_json", side_effect=AssertionError("Help computed factors"))):
            card = algebra_value_help("F", before["F"], session.variables)
        self.assertEqual(before, session.variables)
        self.assertIn("F", card)
        self.assertIn("K", card)
        self.assertIn("examples/polynomial-ideals.md#factor-over-the-algebraic-closure", card)
        self.assertIn("## Factor over the algebraic closure", (ROOT/"examples/polynomial-ideals.md").read_text())
        self.assertIn("F.factors", card)
        for line in card.splitlines():
            if line.strip().startswith("F."):
                list(session.run(line.strip()))
        status, printed, errors = run_prompt('x,y=coordinates("x","y")\nF=absolute_factor(x^2+y^2)\nhelp F\nquit\n')
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("F.factors", printed)
