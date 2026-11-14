import unittest

import sympy as s

from engine.blowup_ideal import (
    blowup_charts, empty_locus, ideal_subset, radical_subset,
    saturate, smooth_locus, verify_snc,
)


class IdealBlowupTests(unittest.TestCase):
    def test_distinct_internal_symbols_with_the_same_print_name(self):
        # Absolute factorization uses one Dummy("coefficient") per unknown.
        a,b=s.Dummy("coefficient"),s.Dummy("coefficient")
        self.assertTrue(empty_locus((a,b), (a,b,1-a-b)))
        self.assertFalse(empty_locus((a,b), (a-b-1,)))

    def test_curved_center_and_chart_projection(self):
        x, y, z = s.symbols("x y z")
        variables = (x, y, z)
        h = y-x**2
        charts = blowup_charts(variables, (), (z-h**2,), (h, z))
        a, b = charts
        self.assertEqual(a.ratios[0], 1)
        self.assertEqual(b.projection, variables)
        self.assertTrue(ideal_subset(a.variables, (z-h*a.ratios[1],), a.ambient))
        self.assertTrue(ideal_subset(a.variables, (a.ratios[1]-h,), a.strict))
        for c in charts:
            self.assertTrue(verify_snc(c.variables, c.ambient, c.strict, c.exceptional, 3)["ok"])

    def test_saturation_removes_components_in_center(self):
        x,y = s.symbols("x y")
        self.assertEqual(saturate((x,y), (x*y,), x), (y,))
        c = blowup_charts((x,y), (), (x,), (x,y))
        self.assertTrue(empty_locus(c[0].variables, c[0].strict))
        self.assertFalse(empty_locus(c[1].variables, c[1].strict))
        empty = blowup_charts((x,y), (x,), (), (x,y))[0]
        self.assertTrue(empty_locus(empty.variables, empty.ambient))

    def test_first_cusp_blowup_has_one_tangency(self):
        x,y = s.symbols("x y")
        charts = blowup_charts((x,y), (), (y**2-x**3,), (x,y))
        result = [verify_snc(c.variables, c.ambient, c.strict, c.exceptional, 2) for c in charts]
        self.assertEqual([r["ok"] for r in result], [False,True])
        bad = result[0]["bad_strata"]
        self.assertEqual([s["components"] for s in bad], [["strict","E1"]])
        self.assertTrue(radical_subset(charts[0].variables, charts[0].variables, bad[0]["bad_ideal"]))
        c = charts[0]
        children = blowup_charts(c.variables, c.ambient, c.strict, (x,c.ratios[1]), c.exceptional)
        first, second = children
        self.assertTrue(verify_snc(first.variables,first.ambient,first.strict,first.exceptional,2)["ok"])
        triple = verify_snc(second.variables,second.ambient,second.strict,second.exceptional,2)
        self.assertEqual([z["components"] for z in triple["bad_strata"]], [["strict","E1","E2"]])
        final = blowup_charts(second.variables,second.ambient,second.strict,
                              (c.ratios[1],second.ratios[0]),second.exceptional)
        self.assertTrue(all(verify_snc(z.variables,z.ambient,z.strict,z.exceptional,2)["ok"] for z in final))

    def test_exact_standard_smooth_cover_and_wrong_dimension(self):
        x,y = s.symbols("x y")
        # At (1,0) the first derivative has rank one, as on the line x=0,
        # but the disjoint isolated point is thickened in the y direction.
        result = smooth_locus((x,y), (x*(x-1), x*y**2), 1)
        self.assertFalse(result["smooth"])
        self.assertTrue(radical_subset((x,y), (x-1,y), result["bad_ideal"]))
        self.assertTrue(radical_subset((x,y), result["bad_ideal"], (x-1,y)))
        # Three generators need codimension-two local presentations.
        z=s.Symbol("z")
        disjoint_lines=(x*(x-1), x*y, (x-1)*z, y*z)
        self.assertTrue(smooth_locus((x,y,z), disjoint_lines, 1)["smooth"])

    def test_algebraic_field_and_high_codimension_empty_strata(self):
        x,y = s.symbols("x y")
        a=s.sqrt(2); K=s.QQ.algebraic_field(a)
        self.assertTrue(radical_subset((x,y), (y-a*x,), ((y-a*x)**2,), domain=K))
        self.assertFalse(ideal_subset((x,y), (y-a*x,), ((y-a*x)**2,), domain=K))
        c = blowup_charts((x,y), (), (y-a*x,), (x,y), domain=K)[0]
        self.assertTrue(verify_snc(c.variables, c.ambient, c.strict, c.exceptional, 2, domain=K)["ok"])
        self.assertTrue(verify_snc((x,y), (), None, ((x,),(x-1,),(y,)), 2)["ok"])
        self.assertFalse(verify_snc((x,y), (), None, ((x,),(y,),(x+y,)), 2)["ok"])


if __name__ == "__main__":
    unittest.main()
