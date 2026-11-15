"""Certificates in the graph blowup of a centre and its retained divisor charts."""
import gzip
import json
from pathlib import Path
import unittest

from engine import kernel, native_polynomial as np
from engine.blowup_ideal import blowup_charts, ideal_subset
from engine.groebner import groebner
from engine.local_blowups import Chart, _blow_up, _generators, _graph_divisor_present, _strict_ideal

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/"tests/fixtures/b8-centre-and-incidence.json.gz"


class CentreDivisionIncidence(unittest.TestCase):
    def test_same_division_over_all_graph_charts(self):
        x, y, z = variables = np.symbols("x y z")
        ambient = (z-y,)
        centre = (x*x+1, y)
        coeff = 2**154+23
        function = coeff*z**2 + (x*x+1)*y
        charts = blowup_charts(variables, ambient, None, centre)
        result = kernel.centre_power_transforms(variables, ambient, centre, function, charts)
        self.assertIsNotNone(result)
        order, quotients = result
        self.assertEqual(order, 2)
        self.assertEqual(len(quotients), 2)
        for child, quotient in zip(charts, quotients):
            basis = groebner(child.ambient, child.variables)
            self.assertTrue(basis.contains(function - child.pivot**order*quotient))
            self.assertFalse(groebner((*child.ambient, child.pivot), child.variables).contains(quotient))

    def test_fallback_for_large_order_and_sparse_wide_exponents(self):
        x, y, z = variables = np.symbols("x y z")
        centre = (x*x + 1, y)
        f = y**9
        parent = Chart(variables, (), (("strict", 1, 1, (f,)),), variables, (),
                       tuple((v.name, v) for v in variables), ())
        children = _blow_up(parent, centre, "E", 1, 3, 2)
        self.assertEqual(len(children), 2)
        for chart in children:
            exceptional = next(c for c in chart.components if c[0] == "E")
            self.assertEqual(exceptional[1], 9)

        # The normal strict-transform route also handles exponents outside the
        # packed range, even when a witness or a parent division is unavailable.
        ambient = (z-y,)
        residual = (2**140+7)*z**(2**32 + 1)+1
        order, quotient = _strict_ideal(variables, ambient, x, (x*residual,))
        self.assertEqual(order, 1)
        self.assertTrue(ideal_subset(variables, quotient, (*ambient, residual)))
        self.assertTrue(ideal_subset(variables, (residual,), (*ambient, *quotient)))

    def test_actual_b8_quotients_reconstruct_in_all_three_charts(self):
        with gzip.open(FIXTURE, "rt") as stream:
            captured = json.load(stream)
        parent = captured["parent"]
        variables = tuple(map(np.Symbol, parent["variables"]))
        centre = tuple(np.from_row(variables, row) for row in parent["chosen"])
        function = np.from_row(variables, captured["function"])
        charts = blowup_charts(variables, (), None, centre)
        result = kernel.centre_power_transforms(variables, (), centre, function, charts)
        self.assertIsNotNone(result)
        order, quotients = result
        self.assertEqual((order, len(quotients)), (2, 3))
        for chart, quotient in zip(charts, quotients):
            self.assertTrue(groebner(chart.ambient, chart.variables).contains(
                function - chart.pivot**order * quotient))
            # All three restrictions were independently computed in Singular.
            # The last ratio is z/p on the first two charts; z is the third pivot.
            expected = -np.Rational(1, 16)
            if chart.pivot_index != 2:
                expected *= chart.variables[-1]**2
            self.assertTrue(groebner((*chart.ambient, chart.pivot), chart.variables).contains(quotient-expected))

    def test_rational_centre_points_and_exact_graph_incidence(self):
        x, y, t, u = variables = np.symbols("x y t u")
        points = kernel.graph_centre_points((x, y), ((x-2)*(x+3)*(x*x+1), y-x))
        self.assertCountEqual(points, (("2", "2"), ("-3", "-3")))
        self.assertTrue(kernel.graph_centre_incidence(variables, (x+3, y+3, t, u), points))
        self.assertFalse(kernel.graph_centre_incidence(variables, (x+3, y+3, t-1, u), points))
        for equations in ((x*x+1, y), (x-1, x-2, y), (x-y,)):
            with self.subTest(equations=equations):
                self.assertFalse(kernel.graph_centre_points((x, y), equations))
        # A chart with all zero-ratio lifts removed can still have a point.
        # The exact incidence computation sees the nonzero ratio.
        self.assertTrue(kernel.present_outside(variables, (x+3, y+3, t-1, u), ()))

    def test_parent_slices_disjoint_from_the_exceptional_pivot(self):
        x, y, z = variables = np.symbols("x y z")
        points = (("1", "0", "0"),)
        # On y=0 the cubic has a point at x=2 with pivot x-1 invertible.
        self.assertTrue(kernel.graph_parent_slice(variables, (z-y,), ((x-1)*(x-2)*(x+1)+y,),
                                                  x-1, points))
        # A divisor contained entirely in x=1 cannot give any witness off that pivot.
        self.assertFalse(kernel.graph_parent_slice(variables, (z-y,), ((x-1)**2,), x-1, points))

        # The bounded parent calculation declines this degree. Ordinary chart
        # incidence still sees the point (2,0,0) on the complement of x=1.
        high_degree = (x-2)**129 + y
        self.assertIsNone(kernel.graph_parent_slice(variables, (), (high_degree,), x-1, points))
        self.assertTrue(kernel.present_outside(variables, (high_degree,), ((x-1,),)))

    def test_third_graph_chart_keeps_a_strict_divisor_off_its_pivot(self):
        with gzip.open(FIXTURE, "rt") as stream:
            captured = json.load(stream)
        record = captured["second_parent"]
        variables = tuple(map(np.Symbol, record["variables"]))
        def read(rows):
            return tuple(np.from_row(variables, r) for r in rows)
        ambient, chosen, strict = map(read, (record["ambient"], record["chosen"], record["strict"]))
        parent = Chart(variables, ambient, (("strict", 1, 1, strict),), (), (), (), ())
        points = kernel.graph_centre_points(variables, (*ambient, *chosen))
        centre = _generators(variables, ambient, chosen)
        children = blowup_charts(variables, ambient, None, centre)
        child = children[2]
        _, *rows = captured["third_pivot_input"].splitlines()
        equations = tuple(np.from_row(child.variables, row) for row in rows)
        self.assertEqual(child.ambient, equations[:-1])
        self.assertFalse(kernel.graph_centre_incidence(child.variables, equations, points))
        self.assertTrue(kernel.graph_parent_slice(variables, ambient, strict, child.pivot, points))
        self.assertTrue(_graph_divisor_present(child, equations[-1:], parent, strict, points))

    def test_coordinate_lifts_with_wide_terms(self):
        x, y = variables = np.symbols("x y")
        denominator = 2**132+3
        points = kernel.graph_centre_points((x,), (denominator*x-1,))
        self.assertEqual(points, ((f"1/{denominator}",),))
        # Zero affine ratios kill even a term beyond the machine-word exponent range.
        self.assertTrue(kernel.graph_centre_incidence(variables, (denominator*x-1, y**(2**72)), points))
        self.assertFalse(kernel.graph_centre_incidence(variables, (y**(2**72) + 1,), points))
        # An inconclusive large evaluation at the first point must allow a
        # subsequent exact witness at the origin.
        high = 2**72
        residual = x**(high+1)-2*x**high
        self.assertTrue(kernel.graph_centre_incidence(variables, (residual,), (("2",), ("0",))))

    def test_replays_the_owned_second_b8_parent(self):
        with gzip.open(FIXTURE, "rt") as stream:
            captured = json.load(stream)
        parent = captured["second_parent"]
        variables = tuple(map(np.Symbol, parent["variables"]))
        eqs = tuple(np.from_row(variables, r) for r in [*parent["ambient"], *parent["chosen"]])
        points = kernel.graph_centre_points(variables, eqs)
        self.assertCountEqual(points, (("0", "-1/4", "15/26", "0"), ("-2", "-1/4", "-15/26", "0")))
        graph = (*variables, *np.symbols("u v"))
        header, *rows = captured["second_parent_input"].splitlines()
        self.assertEqual(header, "open_locus 6 14 0")
        strict = tuple(np.from_row(graph, row) for row in rows)
        self.assertTrue(kernel.graph_centre_incidence(graph, strict, points))
        self.assertFalse(kernel.graph_centre_incidence(graph, (*strict, graph[-1]-1), points))
