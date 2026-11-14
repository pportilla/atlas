"""The centre of blowing up: the invariant of Bierstone and Milman, and the blowups it directs."""
from fractions import Fraction
import unittest

import sympy as sp

from engine.centres import Frame, maximum
from engine.local_blowups import _point_components, _radical_points, acampo_over

INFINITE = float("inf")


class OpenFrame(unittest.TestCase):
    def test_residual_removes_all_coordinate_powers(self):
        from engine.centres import _remove_coordinate_powers
        x, y, z = sp.symbols("x y z")
        frame = Frame((x, y, z), (x, y, z))
        self.assertEqual(_remove_coordinate_powers(frame, x**7*y**3*(z+x), [y, x]), ([3, 7], x+z))
    def test_jets_with_mixed_and_repeated_partials(self):
        x, y = sp.symbols("x y")
        frame = Frame((x, y), (x, y), units=(y,))
        self.assertEqual(frame.jet(x*y**2, 4), (x*y**2, y**2, 2*x*y, 2*y, 2*x, 2))
        frame = Frame((x, y), (x, y), units=(0,))
        self.assertEqual(frame.jet(x*y, 2), (x*y,))

    def test_units_modulo_the_locus_before_localization(self):
        x, y = sp.symbols("x y")
        # On xy=0, the first large unit is just y. Inverting y removes the
        # other axis. The second unit removes y=-1 as well.
        frame = Frame((x, y), (y,), (x*y,), (y + (x*y)**9, y + 1))
        self.assertTrue(frame.vanishes(x))
        self.assertFalse(frame.vanishes(y))
        self.assertTrue(frame.empty(x - 1))
        self.assertTrue(frame.empty(y + 1))
        self.assertFalse(frame.empty(y - 1))
        self.assertTrue(frame.inside(y - 1, (y**2 - 1,)))
        self.assertFalse(frame.inside(y + 1, (y**2 - 1,)))

    def test_contraction_preserves_multiplicities(self):
        x, y = sp.symbols("x y")
        frame = Frame((x, y), (y,), (x**2*y,), (y,))
        self.assertTrue(frame.basis().contains(x**2))
        self.assertFalse(frame.basis().contains(x))
        self.assertTrue(frame.inside(x, ()))

    def test_implicit_vector_fields_with_rational_rows(self):
        x, y, z, w = sp.symbols("x y z w")
        rows = ((w - x**2)/2, (z - y*w)/3)
        frame = Frame((x, y, w, z), (x, y), rows)
        # dw/dx=2x, dz/dx=2xy, dz/dy=w, scaled by the Jacobian minor 1/6.
        self.assertEqual(frame.field(x), {x: sp.Rational(1, 6), w: x/3, z: x*y/3})
        self.assertEqual(frame.field(y), {y: sp.Rational(1, 6), z: w/6})
        for row in rows:
            self.assertEqual(frame.derivative(row, x), 0)
            self.assertEqual(frame.derivative(row, y), 0)
        self.assertEqual(frame.derivative(z, y), w/6)

    def test_implicit_jets_and_divisor_orders_over_the_owned_open_set(self):
        from engine.centres import _remove_coordinate_powers
        x, y = sp.symbols("x y")
        frame = Frame((x,y), (x,), (y-x**2,), (x+1,))
        self.assertEqual(frame.jet(y**2,3), (y**2,4*x*y,12*y))
        self.assertEqual(_remove_coordinate_powers(frame, y**3*(1+x), (x,)), ([6], 1+x))
        # The other component of x*y=0 is excluded after localizing at y.
        frame = Frame((x,y), (y,), (x*y,), (y,))
        self.assertEqual(frame.jet(x*y + y**2,2), (x*y + y**2,2*y**2))
        self.assertEqual(_remove_coordinate_powers(frame, y**3, (y,)), ([0], y**3))

    def test_elimination_clears_the_linear_coefficient(self):
        x, y, z = sp.symbols("x y z")
        frame = Frame((x, y, z), (x, y, z))
        inside, move = frame.eliminated((x + 1)*z - y, z)
        self.assertEqual(inside.units, (x + 1,))
        self.assertEqual(move(z**2 + y*z), x*y**2 + 2*y**2)
        self.assertEqual(move((x + 1)**3*z**2), y**2)
        self.assertEqual(move((x + 1)*z - y), 0)
        self.assertEqual(move(y), y)

    def test_changed_frame_keeps_eliminated_variables_in_saved_rows(self):
        x, y, z = sp.symbols("x y z")
        frame, _ = Frame((x, y, z), (x, y, z)).eliminated(z - x*y, z)
        renamed = frame.changed({y: y+x})
        self.assertEqual(renamed.variables, (x, y))
        self.assertEqual(renamed.solved, (z-x*y-x**2,))

    def test_unit_vanishing_on_whole_frame(self):
        x, y = sp.symbols("x y")
        frame = Frame((x, y), (y,), (x,), (x*y,))
        self.assertTrue(frame.empty())
        self.assertTrue(frame.vanishes(1))


class Invariant(unittest.TestCase):
    def test_residual_thresholds_are_exact_with_fractional_exceptional_orders(self):
        from engine.centre_weights import Presentation
        labels = ("E", "F")
        collection = ((dict(E=Fraction(7,3), F=Fraction(5,4)), None, Fraction(3,2)),
                      (dict(E=Fraction(8,3), F=Fraction(3,2)), None, Fraction(2)))
        alpha, first, second = Presentation(labels, collection).residual()
        self.assertEqual(alpha, dict(E=Fraction(4,3), F=Fraction(3,4)))
        self.assertEqual(first, dict(E=Fraction(1,3), F=Fraction(1,8)))
        self.assertEqual(second, dict(E=0, F=0))
        table = Presentation(labels, ((first, None, Fraction(3,2)), (second, None, Fraction(2))))
        strata = ((), (0, 1))
        self.assertEqual(table.requirements(strata, Fraction(7,4)), [(3, 4), (3, 4)])
        self.assertEqual(table.requirements(strata, Fraction(1,4)), [(1, 1), (0, 1)])
        self.assertEqual(table.candidate_values(strata, [(0, 1), (1, 0)]),
                         [0, Fraction(11,36), Fraction(1,2), Fraction(35,36)])

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

    def test_point_radical_and_colliding_projection(self):
        x, y = sp.symbols("x y")
        # Three underlying points, including a conjugate pair. All three
        # initially project to x+y=0, so another form is needed.
        radical = _radical_points((x, y), ((x*(x**2 + 1))**2, (x+y)**3))
        self.assertEqual(_point_components((x, y), radical), [(x, y), (y**2 + 1, x+y)])
        self.assertEqual(_radical_points((x, y), (1,)), (sp.S.One,))
        self.assertEqual(_point_components((x, y), (1,)), [])

    def test_components_of_a_finite_centre(self):
        x, y = sp.symbols("x y")
        parts = _point_components((x, y), (x**3 - x, y))    # 0 and the pair +-1... over Q: three rational points
        self.assertEqual(len(parts), 3)
        parts = _point_components((x, y), (x*(x**2 + 1), y - x))
        self.assertEqual(sorted(len(sp.Poly(part[-1], x, y).free_symbols) >= 1 for part in parts), [True, True])
        self.assertEqual(len(parts), 2)   # the origin and a pair of conjugate points

    def test_squarefree_coordinate_eliminants(self):
        x, y = sp.symbols("x y")
        # Both projections are squarefree; the original relation still pairs the points.
        self.assertEqual(_radical_points((x, y), (x+y, 2*y**2-2*y)), (y**2-y, x+y))
        # Only x needs a new relation, and both distinct x-factors must survive.
        self.assertEqual(_radical_points((x, y), (x**2*(x-1), y**2-y)), (x**2-x, y**2-y))


if __name__ == "__main__":
    unittest.main()
