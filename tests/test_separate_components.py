"""Embedded blowups carry each rational factor as its own component.

The reference values are independent of Atlas: the topological zeta function
of a generic central arrangement is computed here from its intersection
lattice, and compared with the one Atlas reads off its certified charts.
"""
from fractions import Fraction
from math import comb
import unittest
from unittest.mock import patch

from engine.canonical_replay import canonical_outputs
from engine.origin_backend import RATIONAL_FACTORS, _resolve
from engine.polynomial import parse
from engine.resolve import resolve
from engine.strata import InvalidChart

FIVE = ["x", "y", "z", "t", "u"]
SIX = [*FIVE, "v"]
GENERIC = "x*y*z*t*u*(x+y+z+t+u)"


def generic_arrangement_zeta(planes, dimension, s):
    """Z_top,0 of `planes` generic hyperplanes through 0 in C^dimension, at s.

    One blowup of the origin resolves the arrangement. Its divisor P^(n-1)
    has (N, nu) = (planes, dimension). A flat of k hyperplanes is P^(n-1-k),
    which the other planes cut in generic position: j of them meet it in a
    P^(n-1-k-j) of Euler characteristic n-k-j, and inclusion-exclusion gives
    the Euler characteristic of the points on exactly those k hyperplanes.
    """
    total = Fraction(0)
    for k in range(dimension):
        chi = sum((-1) ** j * comb(planes - k, j) * (dimension - k - j) for j in range(dimension - k))
        total += comb(planes, k) * chi / (s + 1) ** k
    return total / (planes * s + dimension)


def value(expression, s):
    """An Atlas rational function of s at a rational point."""
    return eval(expression.replace("^", "**"), {"__builtins__": {}}, {"s": s})


class SeparateStrictComponents(unittest.TestCase):
    def blowups(self, equation, names, mode="embedded"):
        return _resolve(parse(equation, names), names, mode, 60, defer_outputs=True, skip_tangent=True)

    def test_generic_planes_need_only_the_origin_in_embedded_mode(self):
        cases = [(GENERIC, FIVE, 6), ("x*y*z*t*u*(x+y+z+t+u)*(x+2*y+3*z+4*t+5*u)", FIVE, 7),
                 ("x*y*z*t*u*v*(x+y+z+t+u+v)", SIX, 7)]
        for equation, names, planes in cases:
            with self.subTest(equation=equation):
                record = self.blowups(equation, names)
                self.assertEqual([(step["rule"], step["dimension"]) for step in record["steps"]],
                                 [("maximal order", 0)])
                self.assertEqual(record["certificate"]["strict_components"], RATIONAL_FACTORS)
                labels = {c["id"] for chart in record["atlas"] for c in chart["components"]}
                self.assertEqual(labels, {"B1", *(f"S{i}" for i in range(1, planes + 1))})
                # Each chart of the origin blowup misses the plane of its pivot.
                self.assertEqual(len(record["charts"]), len(names))

    def test_intrinsic_mode_keeps_the_whole_hypersurface_as_one_component(self):
        record = self.blowups("x*y", ["x", "y", "z", "t"], "intrinsic")
        self.assertNotIn("strict_components", record["certificate"])
        labels = {c["id"] for chart in record["atlas"] for c in chart["components"]}
        self.assertIn("strict", labels)
        self.assertFalse(any(label.startswith("S") for label in labels))
        # The two planes have to come apart: their line of intersection is blown up.
        self.assertEqual([step["dimension"] for step in record["steps"]], [2])

    def test_quick_resolution_and_zetas_agree_with_the_intersection_lattice(self):
        # (equation, variables, planes, Euler characteristic of the complement in the exceptional P^(n-1))
        cases = [(GENERIC, FIVE, 6, 1),
                 ("x*y*z*t*u*(x+y+z+t+u)*(x+2*y+3*z+4*t+5*u)", FIVE, 7, 5),
                 ("x*y*z*t*u*v*(x+y+z+t+u+v)*(x+2*y+3*z+4*t+5*u+6*v)", SIX, 8, -6)]
        # The fan of hyperplanes of the Newton method is the origin blowup
        # here. When that method declines, the smooth-centre rules blow up
        # the origin with every plane as a component. Both are compared with
        # the intersection lattice.
        routes = [("hyperplane-arrangement-fan", "E1", {}),
                  ("canonical-blowups", "B1", {"return_value": None})]
        for equation, names, planes, chi in cases:
            for engine, vertex, newton in routes:
              with self.subTest(planes=planes, variables=len(names), engine=engine):
                with patch("engine.resolve.QUICK_NEARBY_SECONDS", 1):
                    request = {"polynomial": equation, "variables": names, "mode": "embedded",
                               "search": "quick", "compute": "cpu", "monodromy": True}
                    if newton:
                        with patch("engine.newton_backend.resolve_newton", **newton):
                            record = resolve(request)
                    else:
                        record = resolve(request)
                self.assertEqual((record["status"], record["engine"]), ("resolved", engine))
                self.assertEqual(record["dual_complex"]["status"], "complete")
                self.assertEqual([v["id"] for v in record["dual_complex"]["vertices"]], [vertex])
                # A'Campo: the open part of the exceptional divisor alone meets the fibre.
                self.assertEqual(record["monodromy_zeta"]["factors"],
                                 [{"multiplicity": planes, "exponent": -chi}])
                expression = record["topological_zeta"]["expression"]
                for s in (Fraction(1), Fraction(2), Fraction(-1, 3), Fraction(5, 7), Fraction(3)):
                    self.assertEqual(value(expression, s), generic_arrangement_zeta(planes, len(names), s))
                self.assertEqual(record["monodromy_check"]["status"], "verified")
                # The optional search along the critical locus does not finish on these germs.
                self.assertEqual(record["monodromy_zeta"]["nearby_search"]["status"], "stopped")

    def test_strata_of_two_planes_that_are_not_coordinates_are_counted(self):
        # In each chart of the origin blowup two of the seven planes are not coordinates. The
        # kernel counts closed strata and takes their alternating sums: leaving the other
        # components out of each count did not end here.
        from engine.atlas import from_json
        from engine.chart_kernel import owned_strata
        record = self.blowups("x*y*z*t*u*(x+y+z+t+u)*(x+2*y+3*z+4*t+5*u)", FIVE)
        chart = from_json(record["atlas"], native=True)[0]
        labels = [label for label, *_ in chart.components]
        strata = {frozenset(labels[j] for j in subset): chi for subset, chi in owned_strata(chart).items()}
        # This chart owns the affine part A^4 of the exceptional P^4, with six planes in general position.
        open_part = sum((-1) ** j * comb(6, j) for j in range(5))  # A^4 less six generic hyperplanes
        self.assertEqual(strata[frozenset({"B1"})], open_part)
        # Four of the planes and B1 meet in one point; five planes do not meet in A^4.
        deepest = [key for key in strata if len(key) == 5]
        self.assertEqual(len(deepest), comb(6, 4))
        self.assertTrue(all(strata[key] == 1 for key in deepest))
        self.assertFalse(any(len(key) > 5 for key in strata))

    def test_replay_derives_the_same_components_from_the_equation(self):
        names = FIVE
        polynomial = parse(GENERIC, names)
        record = self.blowups(GENERIC, names)
        record.update(variables=names, mode="embedded", polynomial=GENERIC)
        atlas, output = canonical_outputs(record, polynomial, complex=True, intrinsic=False, integrals=True)
        self.assertEqual(atlas, record["atlas"])
        self.assertEqual(output["dual_complex"]["status"], "complete")
        strata = {tuple(row["components"]): row["euler_characteristic"] for row in output["strata"]}
        self.assertEqual(strata[("B1",)], 1)
        # Without the mark the replay carries one strict component and goes on
        # separating the planes: the saved single blowup is then not the end.
        legacy = {**record, "certificate": {k: v for k, v in record["certificate"].items()
                                              if k != "strict_components"}}
        with self.assertRaises(InvalidChart):
            canonical_outputs(legacy, polynomial, complex=True, intrinsic=False, integrals=True)


class ComponentsInAChartsOwnPart(unittest.TestCase):
    """Charts from the intrinsic resolution of x*(y^2+z^2+t^2+u^3), restricted to the strict transform.

    The strict transform has the branches a = 0 and g = 0 on these charts.
    B1 is y = 0, so B1 has the two branches y = a = 0 and y = g = 0.
    """

    def setUp(self):
        from engine import native_polynomial as np
        self.symbols = a, y, b, c, d = tuple(map(np.Symbol, ["a", "y", "b", "c", "d"]))
        self.g = y*d**3 + b**2 + c**2 + 1
        self.fibre = (a*y, y, b*y, c*y, d*y)

    def count(self, ambient, region, removed=(), fibre=None):
        from engine.local_blowups import Chart, cells
        y = self.symbols[1]
        chart = Chart(self.symbols, ambient, (("B1", 3, 4, (y,)),), self.fibre if fibre is None else fibre,
                      region, (), (), removed=removed)
        return cells([chart], 4)[frozenset({"B1"})]

    def test_branches_meeting_in_a_removed_centre_are_counted(self):
        a, y, b, c, d = self.symbols
        # The chart was blown up along a = g = 0 and stays for a = y = 0 off that centre:
        # the branch y = a = 0 is its own, the branch y = g = 0 is not.
        self.assertEqual(self.count((a*self.g,), (a, y), removed=((self.g, a),)), 1)

    def test_branches_meeting_at_points_of_other_charts_are_counted(self):
        a, y, b, c, d = self.symbols
        # The chart owns the line a = b = c = y = 0, where g = 1: no branch of B1 lies in it.
        self.assertEqual(self.count((a*self.g,), (a, b, c, y)), 0)

    def test_branches_meeting_at_points_of_the_chart_are_not_counted(self):
        a, y, b, c, d = self.symbols
        # On a*(b^2+c^2+d^2)+y = 0 the divisor y = 0 has the branch a = 0, which the chart
        # owns, and the cone b^2+c^2+d^2 = 0, which meets it there: the count stays open.
        self.assertIsNone(self.count((a*(b**2 + c**2 + d**2) + y,), (a,), fibre=(y*a,)))


if __name__ == "__main__":
    unittest.main()
