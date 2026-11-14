from fractions import Fraction
import unittest
from unittest.mock import patch

from engine import algebra_groebner as algebra
from engine import native_polynomial as np
from engine.native_library import _kernel


@unittest.skipUnless(_kernel(), "the native polynomial kernel is not built")
class RationalBasisHandlers(unittest.TestCase):
    def setUp(self):
        self.x, self.y = np.Symbol("x"), np.Symbol("y")
        self.variables = (self.x, self.y)
        self.ring = {"field": "QQ", "variables": ["x", "y"], "order": "lex"}

    def test_one_basis_serves_several_queries(self):
        x, y = self.variables
        generators = (x**2 - 1, y - x)
        basis = algebra.reduced_basis(generators, self.variables, order="lex")
        self.assertEqual(basis.gens, self.variables)
        self.assertEqual(basis.order, "lex")
        self.assertEqual(basis.exprs, (x - y, y**2 - 1))

        # Every query below must use the saved kernel basis.
        with patch("engine.groebner.groebner", side_effect=AssertionError("recomputed the basis")):
            self.assertEqual(algebra.normal_form(x**2 + y/2, basis), 1 + y/2)
            self.assertEqual(algebra.normal_form(x*y + Fraction(2, 3), basis), Fraction(5, 3))
            self.assertIs(algebra.ideal_membership(y**2 - 1, basis), True)
            self.assertIs(algebra.ideal_membership(y, basis), False)
            for operation, polynomial, result in (
                ("groebner", None, {"type": "basis", "generators": ["x-y", "y^2-1"]}),
                ("normal-form", x**2 + y/2, {"type": "polynomial", "polynomial": "1/2*y+1"}),
                ("membership", y, {"type": "boolean", "value": False}),
            ):
                with self.subTest(operation=operation):
                    operands = {"ideal": basis}
                    if polynomial is not None:
                        operands["polynomial"] = polynomial
                    self.assertEqual(algebra.handle(operation, self.ring, operands),
                                     {**result, "ring": self.ring})

    def test_graded_reverse_lex_is_distinct_from_lex_and_graded_lex(self):
        x, y = self.variables
        z = np.Symbol("z")
        for order, generator, remainder in (
            ("lex", x*z - y**2, y**2), ("grevlex", y**2 - x*z, x*z),
        ):
            with self.subTest(order=order):
                basis = algebra.reduced_basis([x*z - y**2], (x, y, z), order=order)
                self.assertEqual(basis.order, order)
                self.assertEqual(basis.exprs, (generator,))
                self.assertEqual(algebra.normal_form(x*z, basis), remainder)

    def test_zero_and_unit_ideals(self):
        x, y = self.variables
        f = x**2 + y/2
        for order in ("lex", "grevlex"):
            for generators, expected, remainder, member in (
                ([], (), f, False), ([0, 0], (), f, False),
                ([Fraction(2, 7)], (1,), 0, True), ([x, x + 1], (1,), 0, True),
            ):
                with self.subTest(order=order, generators=generators):
                    basis = algebra.reduced_basis(generators, self.variables, order=order)
                    self.assertEqual(basis.exprs, expected)
                    self.assertEqual(algebra.normal_form(f, basis), remainder)
                    self.assertIs(algebra.ideal_membership(f, basis), member)
                    self.assertIs(algebra.ideal_membership(0, basis), True)

    def test_wire_inputs_use_the_declared_variable_precedence(self):
        x, y = self.variables
        ring = dict(self.ring, variables=["y", "x"])
        result = algebra.handle("normal-form", ring, {"ideal": [x - y], "polynomial": y})
        self.assertEqual(result, {"type": "polynomial", "polynomial": "x", "ring": ring})

    def test_prepared_basis_rejects_a_different_order_or_ring(self):
        x, y = self.variables
        basis = algebra.reduced_basis((x - y,), self.variables, order="lex")
        for ring in (dict(self.ring, order="grevlex"), dict(self.ring, variables=["y", "x"]),
                     dict(self.ring, variables=["x", "z"])):
            with self.subTest(ring=ring), self.assertRaises(ValueError):
                algebra.handle("membership", ring, {"ideal": basis, "polynomial": x})

    def test_same_named_coordinates_keep_their_identity(self):
        x, y = (np.Dummy("x"), np.Dummy("y"))
        other_x = np.Dummy("x")
        basis = algebra.reduced_basis([x - y], (x, y), order="lex")
        self.assertEqual(algebra.normal_form(x + y, basis), 2*y)
        with self.assertRaises(ValueError):
            algebra.normal_form(other_x, basis)
        with self.assertRaises(ValueError):
            algebra.reduced_basis([other_x], (x, y))

    def test_sparse_high_degree_remainder_on_a_saved_basis(self):
        x, y = self.variables
        basis = algebra.reduced_basis([y**2 - 2], self.variables)
        power = x**40000
        self.assertEqual(algebra.normal_form(power*(y**2 + y), basis), power*(2 + y))
        self.assertTrue(algebra.ideal_membership(power*(y**2 - 2), basis))

    @unittest.skipUnless(algebra.gb._MODE == "native" and hasattr(_kernel(), "atlas_wide_remainder_abi"),
                         "requires native arbitrary-precision sparse remainders")
    def test_wide_binomial_basis_and_trusted_restore(self):
        x, y = self.variables
        # The lex output exponent has 4,311 decimal digits and the quotient
        # expressing x^e as a multiple of x-y^e would have e terms.
        e = 10**2155
        generators = (x - y**e, x**e)
        basis = algebra.reduced_basis(generators, self.variables, order="lex")
        expected_power = np.from_terms(self.variables, {(0, e*e): 1})
        self.assertEqual(basis.exprs, (generators[0], expected_power))
        with patch("engine.groebner.groebner", side_effect=AssertionError("recomputed a wide basis")):
            restored = algebra.basis_from_reduced(basis.exprs, self.variables, order="lex")
            self.assertEqual(algebra.normal_form(generators[1] + y/2, restored), y/2)
            self.assertTrue(algebra.ideal_membership(generators[1], restored))
            result = algebra.handle("groebner", self.ring, {"ideal": restored})
            self.assertEqual(result["generators"][1], "y^1" + "0"*4310)
