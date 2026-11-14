"""Sparse hypersurface orders and their effect on the Hilbert–Samuel function."""

from fractions import Fraction
from itertools import product
import unittest

from engine import native_polynomial as np
from engine.algebra import dispatch
from engine.point_order import order_at_rational_point


class SparsePointOrderTests(unittest.TestCase):
    def test_mixed_derivatives_match_exact_translation(self):
        x, y, z = variables = np.symbols("x y z")
        germs = [x**3 * y, (x-y)**4, x*y*z, x*y**2 + x**3 + z**5]
        for point in product((Fraction(0), Fraction(-1, 2)), repeat=3):
            shift = {v: v-q for v, q in zip(variables, point) if q}
            for germ in germs:
                with self.subTest(point=point, germ=germ):
                    original = germ.xreplace(shift)
                    self.assertEqual(order_at_rational_point(variables, point, original),
                                     min(map(sum, np.terms(variables, germ))))

    def test_wide_powers_on_nonzero_and_zero_axes(self):
        n = 2**64 + 7
        equation = f"(x^{n}-y^{n})^2*z^{n}"
        # At (2,2,0), x^n-y^n has order 1, while z has order 1.
        request = {"operation": "hilbert-samuel-multiplicity", "ring": {"variables": ["x", "y", "z"]},
                   "operands": {"ideal": [equation], "point": ["2", "2", 0]}}
        result = dispatch(request)["result"]
        self.assertEqual((result["dimension"], result["value"]), (2, n+2))

        # A double zero at 1 in a sparse univariate polynomial gives order 2
        # in the plane, even at degrees beyond the exponent word size.
        request = {"operation": "hilbert-samuel-function", "ring": {"variables": ["x", "y"]},
                   "operands": {"ideal": [f"(x^{n}-1)^2"], "point": [1, 0], "degrees": [0, 1, n]}}
        result = dispatch(request)["result"]
        self.assertEqual((result["dimension"], result["multiplicity"]), (1, 2))
        self.assertEqual(result["values"], [1, 3, 2*n+1])
        self.assertEqual(result["graded_values"], [1, 2, 2])
