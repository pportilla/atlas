"""Sparse tangent forms at rational points with dependent coordinates."""

from fractions import Fraction
from itertools import product
import unittest

from engine import native_polynomial as np
from engine.point_initial_form import principal_initial_form


class RationalInitialFormTests(unittest.TestCase):
    def test_mixed_directional_coefficients(self):
        variables = x, y, z = np.symbols("x y z")
        germ = (x - 2*y + z)**3 + x*y*z**2
        for shifts in product((Fraction(-2, 3), Fraction(0)), repeat=3):
            with self.subTest(shifts=shifts):
                original = germ.xreplace({v: v-q for v, q in zip(variables, shifts) if q})
                self.assertEqual(principal_initial_form(variables, shifts, original), (x - 2*y + z)**3)

    def test_common_huge_scalar_and_unshifted_power(self):
        variables = x, y, z = np.symbols("x y z")
        n = 2**64 + 1
        f = (x**(2*n) - y**n)**2 * z**n
        for a, b in ((2, 4), (-2, 4), (Fraction(-1, 2), Fraction(1, 4))):
            with self.subTest(point=(a, b)):
                shifts = tuple(map(Fraction, (a, b, 0)))
                form = principal_initial_form(variables, shifts, f)
                self.assertEqual(form, (x - y/(2*Fraction(a)))**2 * z**n)
