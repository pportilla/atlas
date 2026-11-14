"""Independent length, multiplicity and vanishing checks for local Hilbert functions."""

from collections import Counter
from itertools import product
from math import comb, prod
from random import Random
import unittest
from unittest.mock import patch

from engine import hilbert_samuel as hs
from engine import native_polynomial as np
from engine.polynomial import InputError, parse_expression


def solve(ideal, *, names=("x", "y"), degrees=None, point=None, order="grevlex", multiplicity=False):
    names = list(names)
    coords = list(map(np.Symbol, names))
    rows = [np.from_terms(coords, parse_expression(f, names)) for f in ideal]
    operands = {"ideal": rows}
    if degrees is not None:
        operands["degrees"] = degrees
    if point is not None:
        operands["point"] = point
    operation = "hilbert-samuel-multiplicity" if multiplicity else "hilbert-samuel-function"
    return hs.handle(operation, {"field": "QQ", "variables": names, "order": order}, operands)


class HilbertSamuelKernelTests(unittest.TestCase):
    def test_coordinate_field_and_zero_ring(self):
        for names in ((), ("x", "y", "z")):
            width = len(names)
            zero = solve(["0"], names=names)
            self.assertEqual(zero["degrees"], list(range(6)))
            self.assertEqual(zero["dimension"], width)
            self.assertEqual(zero["multiplicity"], 1)
            self.assertEqual(zero["values"], [comb(n+width, width) for n in range(6)])
            self.assertEqual(zero["graded_values"], [a-b for a, b in zip(zero["values"], [0]+zero["values"])])
            unit = solve(["7"], names=names, degrees=[8, 1])
            self.assertEqual((unit["dimension"], unit["multiplicity"]), (-1, 0))
            self.assertEqual((unit["values"], unit["graded_values"]), ([0, 0], [0, 0]))

    def test_local_cancellation_and_translated_point(self):
        # At p the two rows generate ((y+2/3)^2,(x-1/2)^3).
        # At the origin the first row instead contains a unit.
        u, v = "(x-1/2)", "(y+2/3)"
        ideal = [f"{u}^3+{v}^2", f"2*{v}^2-{u}^3"]
        for order in ("lex", "grevlex"):
            result = solve(ideal, degrees=[3, 0, 2, 3, 6], point=["2/4", "-2/3"], order=order)
            self.assertEqual(result["point"], ["1/2", "-2/3"])
            self.assertEqual(result["scope"], "point")
            self.assertEqual((result["dimension"], result["multiplicity"]), (0, 6))
            self.assertEqual(result["graded_values"], [1, 1, 2, 1, 0])
            self.assertEqual(result["values"], [6, 1, 5, 6, 6])
            empty = solve(ideal, order=order)
            self.assertEqual((empty["scope"], empty["dimension"]), ("origin", -1))
            self.assertNotIn("point", empty)

    def test_lower_dimensional_parts_still_contribute_to_the_function(self):
        # QQ[x,y]/(x^3,x^2*y) is two free copies of QQ[y], shifted by
        # degrees 0 and 1, plus QQ*x^2. Its multiplicity is therefore 2.
        n = 10**70
        table = solve(["x^3", "x^2*y", "x^3*y^7"], degrees=[0, 2, n, n])
        self.assertEqual((table["dimension"], table["multiplicity"]), (1, 2))
        self.assertEqual(table["values"], [1, 6, 2*n+2, 2*n+2])
        self.assertEqual(table["graded_values"], [1, 3, 2, 2])

    def test_huge_sparse_degrees_bracket_the_hypersurface_order(self):
        k = 2**64+17
        degrees = [k+1, 0, k-1, 10**90]
        # Graded counts for a hypersurface of order k in two variables rise
        # by one up to k, then equal k in all subsequent degrees.
        result = solve([f"y^{k}"], point=["-3/2", 0], degrees=degrees)
        self.assertEqual((result["dimension"], result["multiplicity"]), (1, k))
        self.assertEqual(result["graded_values"], [min(d+1, k) for d in degrees])
        for d, value in zip(degrees, result["values"]):
            self.assertEqual(value, comb(d+2, 2) - (comb(d-k+2, 2) if d >= k else 0))

    def test_many_disconnected_large_powers_skip_the_subset_expansion(self):
        bounds = [2**(64+i) for i in range(45)]
        width = len(bounds) + 1
        heads = [tuple(k if j == i else 0 for j in range(width)) for i, k in enumerate(bounds)]
        n = sum(bounds)+10**90
        dimension, multiplicity, values = hs._invariants(width, heads, [0, 2, n])
        e = prod(bounds)
        self.assertEqual((dimension, multiplicity), (1, e))
        self.assertEqual(values[0], (1, 1))
        self.assertEqual(values[2], (comb(width+2, 2), comb(width+1, 2)))
        self.assertEqual(values[n], (e*(n+1) - e*sum(k-1 for k in bounds)//2, e))
        with patch.object(hs, "_numerator", side_effect=AssertionError("computed a coefficient")):
            self.assertEqual(hs._invariants(width, heads, []), (1, e, {}))

    def test_monomials_match_exhaustive_enumeration_in_small_rings(self):
        rng = Random(53108)
        width, max_degree = 3, 9
        monomials = [a for a in product(range(max_degree+1), repeat=width) if sum(a) <= max_degree]
        for _ in range(32):
            heads = [tuple(rng.randrange(4) for _ in range(width)) for _ in range(rng.randrange(8))]
            counter = Counter(sum(a) for a in monomials if not any(
                all(h <= e for h, e in zip(head, a)) for head in heads))
            with self.subTest(heads=heads):
                _, _, values = hs._invariants(width, heads, range(max_degree+1))
                self.assertEqual(values, {i: (sum(counter[n] for n in range(i+1)), counter[i])
                                          for i in range(max_degree+1)})

    def test_empty_query_multiplicity_and_invalid_degrees(self):
        result = solve(["x^2*y^3"], degrees=[])
        self.assertEqual(result["values"], [])
        self.assertEqual(result["graded_values"], [])
        scalar = solve(["x^2*y^3"], multiplicity=True)
        self.assertEqual((scalar["type"], scalar["dimension"], scalar["value"]), ("integer", 1, 5))
        self.assertNotIn("degrees", scalar)
        for bad in ([-1], [True], [2.0], ["2"], 3):
            with self.subTest(bad=bad), self.assertRaises(InputError):
                solve([], degrees=bad)
