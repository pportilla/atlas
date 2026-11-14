"""Centre transport between old affine parents without saved torus bases."""
import unittest
from dataclasses import replace

from engine.blowup_ideal import radical_subset
from engine.chart_expressions import runtime_expressions
from engine.local_blowups import Chart, _present
from engine.repair_gluing import repair_charts


class ArchivedParentOverlaps(unittest.TestCase):
    def test_rational_transition_cofactors_and_units(self):
        from engine.affine_kernel import sparse_terms
        from engine.groebner import _MODE
        from engine.repair_gluing import _reduced_fraction, transition_on_field_sheet
        api = runtime_expressions()
        x, y = coords = api.symbols("x y")
        chart = Chart(coords, (), (), (), (), (), ())
        zero, one = api.S.Zero, api.S.One
        self.assertEqual(_reduced_fraction(chart, zero, x), (zero, one))
        with self.assertRaisesRegex(ArithmeticError, "denominator"):
            _reduced_fraction(chart, zero, zero)
        self.assertEqual(_reduced_fraction(chart, x, 3*one), (x, 3*one))
        # A nonmonic primitive gcd and an enormous common monomial. Clearing
        # the monic gcd's denominators keeps both original rational units.
        factor = x**(2**64+7 if _MODE == "native" else 70) * (6*x+4*y+3)
        a, b = api.Rational(2,3)*(y+1), -api.Rational(5,7)*(y-1)
        for common in (one, factor):
            numerator, denominator = _reduced_fraction(chart, a*common, b*common)
            if _MODE == "sympy":
                self.assertEqual(api.expand(numerator*b-denominator*a), 0)
            else:
                self.assertEqual(sparse_terms(coords, numerator-a), ())
                self.assertEqual(sparse_terms(coords, denominator-b), ())
        # The exceptional factor cancels in v=B/A. The two charts share
        # their entire fibre under (x,y) -> (x,y+x).
        move, holes = transition_on_field_sheet(chart, chart, (x,x*(y+x)), (x,x*y))
        self.assertEqual(holes, ())
        self.assertEqual(sparse_terms(coords, move(y)-y-x), ())

    def test_projective_pivots_have_complete_disjoint_ownership(self):
        from engine.repair_cover import verify_repair_cover
        api = runtime_expressions()
        x,y,a,b = api.symbols("x y a b")
        # A triple point of the boundary is resolved by one blowup of the
        # plane. Its two affine charts cover the exceptional projective line.
        # In the first chart use the nonlinear shear (x,b_old)=(a,b+a²).
        parent = Chart((x,y), (), (("E",1,1,(x,)),("strict",1,1,(y*y-x*x,))),
                       (x,y), (), (("X",x),("Y",y)), (), removed=((x,y),))
        slope = b+a*a
        mapping = a,a*slope
        first = Chart((a,b), (), (("strict",1,1,(slope*slope-1,)),("new",3,2,(a,))),
                      mapping, (a,), tuple(zip(("X","Y"),mapping)), (), repair_projection=(0,mapping))
        mapping = a*b,b
        second = Chart((a,b), (), (("E",1,1,(a,)),("strict",1,1,(a*a-1,)),("new",3,2,(b,))),
                       mapping, (a,b), tuple(zip(("X","Y"),mapping)), (), repair_projection=(0,mapping))
        self.assertTrue(verify_repair_cover([parent,first,second]))
        at_zero = replace(first, region=(*first.region,b))
        away = replace(first, removed=((b,),))
        # The same affine chart can have two complementary owners.
        self.assertTrue(verify_repair_cover([parent,second,away,at_zero]))
        self.assertFalse(verify_repair_cover([parent,second,away]))
        self.assertFalse(verify_repair_cover([parent,second,first,at_zero]))
        wrong = replace(first, components=(("strict",1,1,(slope*slope-4,)),first.components[1]))
        self.assertFalse(verify_repair_cover([parent,wrong,second]))

    def test_deleted_centres_follow_the_regular_parent_overlap(self):
        api = runtime_expressions()
        x, y, u, v, t = api.symbols("x y u v theta")
        # Two charts of the blowup of a plane at the origin. In the first
        # we repair the conjugate pair of slopes ±sqrt(2) and the slope 0.
        # Only the pair lies in the second chart (there the slope is 1/u).
        parent = Chart((x,y), (), (), (x,), (), (("X",x),("Y",x*y)), (),
                       removed=((x,y*y-2),(x,y)))
        # Use the opposite serialization order of the original coordinates.
        peer = Chart((u,v), (), (), (v,), (u,), (("Y",v),("X",u*v)), ())
        children = []
        for relation, point in (((),api.S.Zero), ((t*t-2,),t)):
            coords = (x,y,t) if relation else (x,y)
            for projection, region in (((x,point+x*y),(x,)), ((x*y,point+y),(x,y))):
                a,b = projection
                children.append(Chart(coords, relation, (), (a,), region, (("X",a),("Y",a*b)), (),
                                      repair_projection=(0,projection)))
        result = repair_charts([parent, peer, *children])
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 8)
        transported = result[1]
        minimal = transported.coefficient_field[1]
        # FLINT may mark either conjugate. Together the two point ideals
        # on the second open must be exactly v=0, u=±1/sqrt(2).
        self.assertEqual(len(transported.removed),2)
        locus = (minimal,v,2*u*u-1)
        for hole in transported.removed:
            self.assertTrue(radical_subset(transported.variables, locus, hole))
        self.assertFalse(_present(transported.variables, locus, transported.removed))
        broken = replace(peer, coordinates=(("Y",v),("X",u*u*v)))
        self.assertIsNone(repair_charts([parent, broken, *children]))


if __name__ == "__main__":
    unittest.main()
