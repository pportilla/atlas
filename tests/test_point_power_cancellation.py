"""Sparse point evaluation where different rational bases have equal powers."""
from fractions import Fraction
from unittest import TestCase
from unittest.mock import patch

from engine.algebra import dispatch
from engine.point_germs import nonzero_terms_at_point


class PointPowerCancellationTests(TestCase):
    def test_dependent_coordinates_in_sparse_hypersurface_multiplicity(self):
        n = 2**64 + 1
        power = Fraction.__pow__

        def bounded_power(base, exponent):
            if exponent > 64:
                raise AssertionError("Tried to expand a large rational point coordinate power")
            return power(base, exponent)

        for point, ideal in (
            ([2, 4], f"x^{2*n}-y^{n}"),
            (["-3/2", "9/4"], f"x^{2*n}-y^{n}"),
            (["-2/3", "-4/9"], f"x^{2*n}+y^{n}"),
            (["-6/5", "25/36"], f"x^{2*n}*y^{n}-1"),
        ):
            with self.subTest(point=point), patch.object(Fraction, "__pow__", bounded_power):
                record = dispatch({"operation": "hilbert-samuel-multiplicity",
                                   "ring": {"variables": ["x", "y"]},
                                   "operands": {"ideal": [ideal], "point": point}, "compute": "cpu"})
                result = record["result"]
                self.assertEqual((result["dimension"], result["value"]), (1, 1))

    def test_shared_factors_across_three_rationals_and_nonzero_modular_collision(self):
        n = 2**64 + 1
        point = tuple(map(Fraction, ("6/35", "14/15", "4/25")))
        # x*y=z while no two individual coordinate powers coincide.
        terms = {(n, n, 0): Fraction(3, 7), (0, 0, n): Fraction(-3, 7)}
        self.assertFalse(nonzero_terms_at_point(point, terms))
        # Each trial prime also divides the residual constant. Zero modular
        # evaluations must still recover this nonzero exact result.
        terms[0, 0, 0] = Fraction(1000000007 * 1000000009 * 998244353)
        self.assertTrue(nonzero_terms_at_point(point, terms))
