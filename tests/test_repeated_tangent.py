import unittest
from dataclasses import replace

from engine import native_polynomial as polynomial
from engine.centres import GiveUp
from engine.local_blowups import Chart, _run, acampo_over
from engine.repeated_tangent import RepeatedTangent, certify_fibre


def coordinates(names):
    Symbol = polynomial.Symbol
    return tuple(map(Symbol, names))


class RepeatedProjectiveTangent(unittest.TestCase):
    def test_smooth_repeated_quadric_produces_a_certified_fourfold_atlas(self):
        x, y, z, t, u = variables = coordinates("xyztu")
        quadric = x*y - z*t + u**2
        quintic = sum(i*v**5 for i, v in enumerate(variables, 1))
        _, _, charts, centres = acampo_over(variables, [("strict", 1, 1, (quadric**2 + quintic,))],
                                            variables, rules=("tangent",), count_strata=False)
        self.assertEqual([c["dimension"] for c in centres], [0, 3, 3])
        data = {label: (n, nu) for chart in charts for label, n, nu, _ in chart.components}
        self.assertEqual(data, {"strict": (1, 1), "B1": (4, 5), "B2": (5, 6), "B3": (10, 11)})

    def test_singular_support_is_rejected_before_blowup(self):
        x, y, z, t = variables = coordinates("xyzt")
        # The projective cone xy-z^2=0 is singular at [0:0:0:1].
        f = (x*y - z**2)**2 + t**5
        with self.assertRaisesRegex(GiveUp, "centre is not smooth"):
            acampo_over(variables, [("strict", 1, 1, (f,))], variables, rules=("tangent",),
                        count_strata=False)

    def test_node_family_at_a_simple_root_of_the_normal_quadratic(self):
        # On the smooth curve of nontransverse points, the strict transform
        # is singular exactly where the coefficient of the radial term vanishes.
        r, v, x, y, z = variables = coordinates("rvxyz")
        parts = [("strict", 1, 1, (x**2 + y**2 + z**2 + r*v,)), ("B", 5, 6, (r,))]
        rule = RepeatedTangent()
        rule.exceptional = ["E", "B", "C"]
        # Resume the centre rule on this transversal model. The first
        # and third exceptional divisors miss this chart.
        _, _, charts, centres = _run(rule, variables, parts, (r,), 400, None, count_strata=False)
        self.assertEqual([c["dimension"] for c in centres], [1, 3])
        self.assertTrue(charts)

    def test_a_crossing_failure_on_one_fibre_branch(self):
        x, y = coordinates("xy")
        for variables in ((x, y), (y, x)):
            chart = Chart(variables, (), (("E", 1, 1, (x,)), ("strict", 1, 1, (x + y**2,))),
                          (x*(x-1),), (), (), ())
            with self.assertRaisesRegex(GiveUp, "fails normal crossings"):
                certify_fibre(chart, 2)
            certify_fibre(replace(chart, removed=((x, y),)), 2)
            certify_fibre(replace(chart, region=(x-1,)), 2)

    def test_sparse_replay_beyond_packed_exponents(self):
        x, y = variables = coordinates("xy")
        exponent = 2**67 + 1
        f = x**exponent + (2**140 + 17)*y
        chart = Chart(variables, (), (("strict", 1, 1, (f,)),), (x,), (), (), ())
        certify_fibre(chart, 2)
        with self.assertRaisesRegex(GiveUp, "fails normal crossings"):
            certify_fibre(replace(chart, components=(*chart.components, ("E", 1, 1, (y,)))), 2)

    def test_sparse_replay_of_open_region_in_33_coordinates(self):
        variables = coordinates(tuple("x" + str(j) for j in range(33)))
        x, y = variables[0], variables[-1]
        chart = Chart(variables, (), (("strict", 1, 1, (x+y**2,)), ("E", 1, 1, (x,))),
                      (x,), (), (), (), removed=((x, y),))
        certify_fibre(chart, 33)
        with self.assertRaisesRegex(GiveUp, "fails normal crossings"):
            certify_fibre(replace(chart, removed=()), 33)
