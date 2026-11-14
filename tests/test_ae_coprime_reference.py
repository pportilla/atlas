"""Independent rational ranks of the extended tangent space of three powers."""
from fractions import Fraction
from math import comb
import unittest

from tests.test_local_algebra_singular import calculate


def extend_basis(pivots, row):
    """Insert a sparse integer row into a rational row echelon basis."""
    while row:
        column = min(row)
        factor = row[column]
        if column not in pivots:
            pivots[column] = {j: Fraction(value, factor) for j, value in row.items()}
            return
        for j, value in pivots[column].items():
            remainder = row.get(j, 0) - factor * value
            if remainder:
                row[j] = remainder
            else:
                row.pop(j, None)


def graded_dimension(degrees, weight):
    # Component j has degree weight + d_j. In each binary form the power of
    # x specifies a monomial uniquely; the three blocks occupy disjoint columns.
    a, b, c = degrees
    widths = [max(0, weight + d + 1) for d in degrees]
    offsets = (0, widths[0], widths[0] + widths[1])
    pivots = {}

    # Pull back all weighted target monomials separately in each position.
    for offset, d in zip(offsets, degrees):
        degree = weight + d
        for z in range(degree // c + 1):
            for y in range((degree - c*z) // b + 1):
                x_power = degree - c*z - b*y
                if x_power % a == 0:
                    row = {offset + x_power + i: comb(c*z, i) for i in range(c*z + 1)}
                    extend_basis(pivots, row)

    # Multiply each column of df by all source monomials of degree weight+1.
    for x in range(weight + 2):
        common = {offsets[2] + x + i: c * comb(c-1, i) for i in range(c)}
        extend_basis(pivots, {**common, offsets[0] + x + a - 1: a})
        extend_basis(pivots, {**common, offsets[1] + x: b})
    return sum(widths) - len(pivots)


def tangent_codimension(degrees):
    a, b, _ = degrees
    # Q[x,y] is free over Q[x^a,y^b] on x^i y^j, i<a and j<b.
    # Both tangent actions are stable under this subring. Its generators
    # for the graded normal quotient N have weights at most the value below.
    # Above that bound N_k = x^a N_(k-a) + y^b N_(k-b). Thus max(a,b)
    # consecutive zero pieces certify every later piece by induction.
    largest_generator = a + b - 2 - min(degrees)
    total = 0
    zero_pieces = 0
    for weight in range(-max(degrees), 64):
        dimension = graded_dimension(degrees, weight)
        total += dimension
        zero_pieces = 0 if dimension else zero_pieces + 1
        if weight >= largest_generator and zero_pieces >= max(a, b):
            return total
    raise AssertionError(f"Missing graded tail certificate for {degrees} within the test budget")


class ThreePowersTangentRankTests(unittest.TestCase):
    def test_three_powers_against_extended_tangent_ranks(self):
        for degrees, expected in [((1, 2, 3), 1), ((2, 3, 5), 115), ((2, 3, 7), 242)]:
            with self.subTest(degrees=degrees):
                a, b, c = degrees
                dimension = tangent_codimension(degrees)
                self.assertEqual(dimension, expected)
                result = calculate("ae-codimension", components=[f"x^{a}", f"y^{b}", f"(x+y)^{c}"])
                self.assertIs(result["finite"], True)
                self.assertEqual(result["value"], dimension)
