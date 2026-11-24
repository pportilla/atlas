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
from engine.resolve import (QUICK_CANONICAL_SECONDS, QUICK_NEARBY_SECONDS, quick_canonical_seconds,
                            quick_nearby_seconds, resolve)
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
        with patch("engine.resolve.QUICK_NEARBY_SECONDS", 1):
            record = resolve({"polynomial": GENERIC, "variables": FIVE, "mode": "embedded", "search": "quick",
                              "compute": "cpu", "monodromy": True})
        self.assertEqual((record["status"], record["engine"]), ("resolved", "canonical-blowups"))
        self.assertEqual(record["dual_complex"]["status"], "complete")
        self.assertEqual([v["id"] for v in record["dual_complex"]["vertices"]], ["B1"])
        # P^4 less six generic hyperplanes has Euler characteristic 1.
        self.assertEqual(record["monodromy_zeta"]["expression"], "1/(1-T^6)")
        expression = record["topological_zeta"]["expression"]
        for s in (Fraction(1), Fraction(2), Fraction(-1, 3), Fraction(5, 7), Fraction(3)):
            self.assertEqual(value(expression, s), generic_arrangement_zeta(6, 5, s))
        self.assertEqual(record["monodromy_check"]["status"], "verified")
        # The optional search along the critical locus does not finish on this germ.
        self.assertEqual(record["monodromy_zeta"]["nearby_search"]["status"], "stopped")

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


class QuickTimeForSmoothCentres(unittest.TestCase):
    def test_the_rules_get_the_rest_of_the_workers_limit(self):
        from engine import resolve as module
        from engine.resolve_cli import MAX_SECONDS
        token = module._REQUEST_START.set(1000.0)
        try:
            with patch("engine.resolve.perf_counter", return_value=1003.0):
                self.assertEqual(quick_canonical_seconds(), MAX_SECONDS - module.QUICK_FINAL_RESERVE - 3)
                self.assertEqual(quick_nearby_seconds(), QUICK_NEARBY_SECONDS)
            # Late in the request the rules keep their former share.
            with patch("engine.resolve.perf_counter", return_value=1048.0):
                self.assertEqual(quick_canonical_seconds(), QUICK_CANONICAL_SECONDS)
                self.assertEqual(quick_nearby_seconds(), MAX_SECONDS - module.QUICK_FINAL_RESERVE - 48)
            with patch("engine.resolve.perf_counter", return_value=1060.0):
                self.assertEqual(quick_nearby_seconds(), 0)
        finally:
            module._REQUEST_START.reset(token)

    def test_quick_search_passes_that_time_to_the_canonical_rules(self):
        from engine.resolve_cli import MAX_SECONDS
        with patch("engine.origin_backend.resolve_origin", return_value=None) as origin, \
             patch("engine.newton_backend.resolve_newton", return_value=None), \
             patch("engine.general_resolution.search", return_value=None):
            record = resolve({"polynomial": "x^2*y^2+z^3*t^2+u^5*x+y^4*t^3", "variables": FIVE,
                              "mode": "embedded", "search": "quick", "compute": "cpu"})
        self.assertEqual(record["status"], "inconclusive")
        canonical = [call for call in origin.call_args_list if call.kwargs.get("skip_tangent")]
        self.assertEqual(len(canonical), 1)
        seconds = canonical[0].kwargs["seconds"]
        self.assertGreater(seconds, QUICK_CANONICAL_SECONDS)
        self.assertLessEqual(seconds, MAX_SECONDS)
        self.assertEqual(record["diagnostic"]["canonical_seconds"], round(seconds, 1))


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


class RulesContinueWithLeftoverTime(unittest.TestCase):
    def run_rules(self, behaviour):
        """Run three rules whose blowups are replaced by `behaviour(name, visit, limit)`."""
        from time import monotonic
        from engine import local_blowups as lb
        from engine import native_polynomial as np
        x, y, z = symbols = tuple(map(np.Symbol, "xyz"))
        calls = []

        def fake(rule, variables, components, fibre, budget, limit, *rest, state=None, **options):
            visit = sum(1 for name, *_ in calls if name == rule.name)
            calls.append((rule.name, limit, state))
            return behaviour(rule.name, visit, limit, state)

        deadline = monotonic() + 0.6
        with patch.object(lb, "_run", side_effect=fake):
            try:
                answer = lb.acampo_over(symbols, [("strict", 1, 1, (x*y - z**2,))], symbols, dimension=3,
                                        rules=("order", "complete", "direct"), deadline=deadline)
            except lb.GiveUp as error:
                answer = error
        return answer, calls, deadline

    def test_a_rule_stopped_at_its_share_continues_when_the_others_give_up(self):
        from time import monotonic, sleep
        from engine.local_blowups import GiveUp

        def behaviour(name, visit, limit, state):
            if name == "order" and visit == 0:
                state["progress"] = monotonic()
                sleep(max(0.0, limit - monotonic()) + 0.02)
                raise GiveUp("The time allowed for the blowups is over.")
            if name == "order":
                return "continued"
            raise GiveUp(f"The {name} rule has nothing to do.")

        answer, calls, deadline = self.run_rules(behaviour)
        self.assertEqual(answer, "continued")
        self.assertEqual([name for name, *_ in calls], ["order", "complete", "direct", "order"])
        first, last = calls[0], calls[-1]
        self.assertIs(first[2], last[2])       # the same charts and centres
        self.assertLess(first[1], deadline)    # a share the first time
        self.assertEqual(last[1], deadline)    # everything that is left afterwards

    def test_a_rule_that_gave_up_for_its_own_reason_is_not_run_again(self):
        from engine.local_blowups import GiveUp

        def behaviour(name, visit, limit, state):
            raise GiveUp(f"The {name} rule has nothing to do.")

        answer, calls, _ = self.run_rules(behaviour)
        self.assertIsInstance(answer, GiveUp)
        self.assertEqual([name for name, *_ in calls], ["order", "complete", "direct"])


if __name__ == "__main__":
    unittest.main()
