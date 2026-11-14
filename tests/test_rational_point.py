from fractions import Fraction
import unittest

from engine.polynomial import InputError
from engine.rational_point import coordinate_fraction, normalize_point


class RationalPointTests(unittest.TestCase):
    def test_signs_reduction_and_variable_order(self):
        point = [7, "-000", "+03/-06", "-10/-25", "4/+2"]
        self.assertEqual(normalize_point(point, 5), ["7", "0", "-1/2", "2/5", "2"])
        self.assertEqual(coordinate_fraction("-21/+14"), Fraction(-3, 2))
        self.assertEqual(normalize_point([], 0), [])

    def test_large_coefficients_round_trip_as_exact_strings(self):
        digits = "1" + "0" * 6000
        point = [10**6000, "2" + "0" * 6000 + "/2", "-1/-" + digits]
        self.assertEqual(normalize_point(point, 3), [digits, digits, "1/" + digits])

    def test_wrong_shape_and_missing_coordinates(self):
        for point in (None, 0, "0,1", (0, 1), {"x": 0, "y": 1}, [], [0], [0, 1, 2]):
            with self.subTest(point=point), self.assertRaises(InputError):
                normalize_point(point, 2)

    def test_only_integer_or_fraction_literals_are_accepted(self):
        for coordinate in (True, False, None, 0.5, [], {}, Fraction(1, 3), "", "2.5", " 2", "3/2 ",
                           "3 / 2", "1e2", "1+2", "x", "2/", "/2", "1/2/3", "1/+0",
                           "3/-00", "--2", "5/6\n", "１", "1_000"):
            with self.subTest(coordinate=coordinate), self.assertRaises(InputError):
                coordinate_fraction(coordinate)
