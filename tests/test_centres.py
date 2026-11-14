"""The centre of blowing up: the invariant of Bierstone and Milman, and the blowups it directs."""
from fractions import Fraction
import unittest

import sympy as sp

from engine.centres import maximum
from engine.local_blowups import _point_components, acampo_over

INFINITE = float("inf")


class Invariant(unittest.TestCase):
    def test_cusp(self):
        x, y = sp.symbols("x y")
        value, key, centre = maximum((x, y), (), 2, (y**2 - x**3,), [], [(x, y)], {}, [])
        # Order 2; on the maximal contact line y = 0 the coefficient x^3 has weight 2.
        self.assertEqual(value, (2, 0, Fraction(3, 2), 0, INFINITE))
        self.assertEqual(set(centre), {x, y})

    def test_tangency_with_an_old_divisor(self):
        x, y = sp.symbols("x y")
        value, key, centre = maximum((x, y), (), 2, (y - x**2,), [("E", (y,))], [(x, y)], {"E": None}, ["E"])
        # Smooth, on one old divisor whose equation has order 2 on the curve.
        self.assertEqual(value, (1, 1, 2, 0, INFINITE))
        self.assertEqual(set(centre), {x, y})

    def test_new_divisor_is_factored_out(self):
        # After blowing up the point of tangency: the old divisor restricted to
        # the curve is the equation of the new one, so the residual order is 0.
        x, y = sp.symbols("x y")
        records = {"E": None, "B": (1, 1, 2, 0, INFINITE)}
        value, key, centre = maximum((x, y), (), 2, (y - x,), [("E", (y,)), ("B", (x,))], [(x, y)], records, ["E", "B"])
        self.assertEqual(value, (1, 1, 0))
        self.assertEqual(key, (0, 1))
        self.assertEqual(set(centre), {x, y})

    def test_node_needs_a_general_direction(self):
        # No pure second derivative of x*y is a unit: the coordinates are changed.
        x, y, z = sp.symbols("x y z")
        value, key, centre = maximum((x, y, z), (), 3, (x*y,), [], [(x, y, z)], {}, [])
        self.assertEqual(value, (2, 0, 1, 0, INFINITE))
        self.assertEqual(set(centre), {x, y})   # the double line

    def test_implicit_chart(self):
        # The same cusp on the graph w = x + y^2 of a coordinate change.
        x, y, w = sp.symbols("x y w")
        value, key, centre = maximum((x, y, w), (w - x - y**2,), 2, (y**2 - x**3,), [], [(x, y, w)], {}, [])
        self.assertEqual(value, (2, 0, Fraction(3, 2), 0, INFINITE))
        self.assertEqual(set(centre), {x, y, w})


class Blowups(unittest.TestCase):
    def test_both_rules_resolve_and_agree(self):
        x, y = sp.symbols("x y")
        for components, expected in (
                ([("E1", 3, 2, (y,)), ("strict", 1, 1, (y - x**2,))], {"B1": (4, 3), "B2": (8, 5)}),
                ([("strict", 1, 1, (y**2 - x**3,))], {"B1": (2, 2), "B2": (3, 3), "B3": (6, 5)})):
            for rule in ("direct", "complete"):
                with self.subTest(rule=rule, components=len(components)):
                    data, strata, charts, centres = acampo_over((x, y), components, (x, y), rules=(rule,))
                    self.assertEqual({label: numbers for label, numbers in data.items() if label.startswith("B")},
                                     expected)
                    allowed = {"direct": {"direct"}, "complete": {"invariant", "surface"}}[rule]
                    self.assertTrue(all(centre["rule"] in allowed for centre in centres))

    def test_invariant_decreases(self):
        x, y = sp.symbols("x y")
        _, _, _, centres = acampo_over((x, y), [("strict", 1, 1, (y**2 - x**5,))], (x, y), rules=("complete",))
        values = [tuple(INFINITE if entry == "oo" else Fraction(entry) for entry in centre["invariant"])
                  for centre in centres]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(values[0], (2, 0, Fraction(5, 2), 0, INFINITE))

    def test_components_of_a_finite_centre(self):
        x, y = sp.symbols("x y")
        parts = _point_components((x, y), (x**3 - x, y))    # 0 and the pair +-1... over Q: three rational points
        self.assertEqual(len(parts), 3)
        parts = _point_components((x, y), (x*(x**2 + 1), y - x))
        self.assertEqual(sorted(len(sp.Poly(part[-1], x, y).free_symbols) >= 1 for part in parts), [True, True])
        self.assertEqual(len(parts), 2)   # the origin and a pair of conjugate points


if __name__ == "__main__":
    unittest.main()
