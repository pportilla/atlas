"""Exact units, multiplicities and irreducibles in ordered QQ polynomial rings."""

from fractions import Fraction
import unittest

from engine import native_polynomial as np
from engine.polynomial import InputError
from engine.polynomial_factorization import factor_polynomial


class RationalFactorization(unittest.TestCase):
    def check_factors(self, variables, polynomial, order, expected_unit, expected_factors):
        unit, factors = factor_polynomial(variables, polynomial, order=order)
        self.assertEqual(unit, expected_unit)
        self.assertCountEqual(factors, expected_factors)
        product = np.coerce(unit)
        for factor, multiplicity in factors:
            product *= factor**multiplicity
        self.assertEqual(product, polynomial)

    def test_units_and_zero(self):
        for variables in ((), np.symbols("x y")):
            for value in (1, -1, Fraction(-21, 8)):
                self.check_factors(variables, value, "lex", value, ())
            with self.assertRaisesRegex(InputError, "zero polynomial"):
                factor_polynomial(variables, 0)

    def test_repeat_rational_irreducible_and_nonprimitive_binomial(self):
        x, y = variables = np.symbols("x y")
        p = Fraction(-8, 15) * (2*x + 1)**3 * (y**2 + 1)
        self.check_factors(variables, p, "lex", Fraction(-64, 15),
                           ((x + Fraction(1, 2), 3), (y**2 + 1, 1)))
        self.check_factors(variables, x**2 - y**2, "lex", 1, ((x-y, 1), (x+y, 1)))

    def test_monic_with_permuted_coordinates_and_monomial_order(self):
        x, y, z = variables = np.symbols("x y z")
        h = 2*x*z - 3*y**2
        p = Fraction(1, 7) * h**3 * (y-1)
        for coords, order, lead in ((variables, "lex", 2), (variables, "grevlex", -3),
                                    ((y, x, z), "lex", -3)):
            with self.subTest(order=order, coordinates=coords):
                self.check_factors(coords, p, order, Fraction(lead**3, 7),
                                   ((h/lead, 3), (y-1, 1)))

    def test_wide_multiplicities_and_large_rational_coefficient(self):
        x, y = variables = np.symbols("x y")
        power = 10**4500
        scalar = Fraction(-(2**18000 + 1), 2**17000 + 1)
        p = scalar * x**power * y**(2**64 + 17)
        self.check_factors(variables, p, "grevlex", scalar, ((x, power), (y, 2**64 + 17)))

    def test_wide_primitive_binomial_with_common_coordinate_factors(self):
        x, y = variables = np.symbols("x y")
        power = 2**64 + 17
        h = x**power - Fraction(3, 2)*y**2
        p = -2*x**3 * y**5 * h
        self.check_factors(variables, p, "lex", -2, ((x, 3), (y, 5), (h, 1)))
