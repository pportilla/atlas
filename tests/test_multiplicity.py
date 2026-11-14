"""Global finite-colength multiplicity over QQ, including closed points of higher degree."""
import shutil
import unittest
from unittest.mock import patch

from engine import local_algebra
from engine.algebra import dispatch
from tests.test_local_algebra_singular import calculate, singular


class MultiplicityTests(unittest.TestCase):
    def check(self, ideal, expected, names=("x", "y"), order="grevlex"):
        result = calculate("multiplicity", ideal=ideal, variables=names, order=order)
        self.assertEqual(result, {
            "type": "integer", "scope": "global", "finite": expected is not None, "value": expected,
            "ring": {"field": "QQ", "variables": list(names), "order": order},
        })

    def test_nonreduced_rational_and_nonrational_support(self):
        # One triple rational point and a double point over QQ(i) give 3 + 2*2.
        for order in ("lex", "grevlex"):
            self.check(["(x-1/2)^3*(x^2+1)^2", "y-x^2"], 7, order=order)
            self.check(["x^2+y^2-1", "x*y"], 4, order=order)
            # An entire line at x=1 coexists with the isolated origin.
            self.check(["(x-1)*x^2", "(x-1)*y"], None, order=order)

    def test_zero_unit_and_scalar_ideals(self):
        self.check([], None)
        self.check(["x-y", "x-y+1"], 0)
        self.check([], 1, names=())
        self.check(["0", "-2/7"], 0, names=())

    def test_count_wide_staircases_without_visiting_the_basis(self):
        n = 2**64 + 17
        from engine.quotient_monomials import affine_request
        with patch("engine.quotient_monomials.affine_request", wraps=affine_request) as request:
            self.check([f"x^{n}", f"y^{n}"], n*n)
            self.check([f"x^{n}", f"y^{n}", "x^2*y^3"], 5*n-6)
        self.assertEqual({call.args[0][0] for call in request.call_args_list}, {"monomial_length"})

    def test_verified_basis_contributes_its_heads_directly(self):
        ring = {"variables": ["x", "y"], "order": "grevlex"}
        ideal = ["x^2+y^2-1", "x*y"]
        basis = calculate("groebner", ideal=ideal)["generators"]
        with patch.object(local_algebra, "reduced_basis", side_effect=AssertionError("recomputed saved basis")):
            record = dispatch({"operation": "multiplicity", "ring": ring, "operands": {"ideal": basis},
                               "compute": "cpu"}, known_basis=True)
        self.assertEqual(record["result"]["value"], 4)
        self.assertIn("restore_global_basis", [stage["operation"] for stage in record["execution"]["stages"]])

    @unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
    def test_global_colength_against_singular(self):
        ideals = ([], ["1"], ["x*y"], ["x-1/2", "y^2+y+1"],
                  ["(x-1/2)^3*(x^2+1)^2", "y-x^2"], ["x*y-y", "x^2-y^2+2", "x^3-x*y^2+2*x"])
        for order, singular_order in (("lex", "lp"), ("grevlex", "dp")):
            commands = [f'print("ANSWER "+string(vdim(std(ideal({",".join(ideal) or "0"})))));'
                        for ideal in ideals]
            dimensions = singular(singular_order, commands)
            self.assertEqual(len(dimensions), len(ideals))
            for ideal, dimension in zip(ideals, dimensions):
                with self.subTest(order=order, ideal=ideal):
                    self.check(ideal, None if dimension < 0 else dimension, order=order)
