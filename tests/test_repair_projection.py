"""Exact lifting of saved chart maps, including affine quotients and deleted sheets."""
import unittest

from engine.chart_expressions import runtime_expressions
from engine.local_blowups import Chart
from engine.repair_projection import recover_projection


class ArchivedProjectionTests(unittest.TestCase):
    def test_child_ring_changes_the_quotient(self):
        api = runtime_expressions()
        u, v, r, s, t = api.symbols("u v r s t")
        parent = Chart((u,v), (), (), (), (), (("A",u),("B",u*v)), ())
        # B/A has no polynomial representative before imposing 2t²+t−3=0.
        # Its two separate components t=1 and t=−3/2 both see a regular
        # function. The test works directly in this reducible coordinate ring.
        relation = 2*t*t+t-3
        child = Chart((r,s,t), (relation,), (), (), (),
                      (("B",r*s*t*t + s*relation),("A",r*t)), ())
        self.assertEqual(recover_projection(child,parent), (r*t,s*t))

    def test_rejects_a_pole_and_a_denominator_zero_on_one_component(self):
        api = runtime_expressions()
        u, v, r, s, t = api.symbols("u v r s t")
        parent = Chart((u,v), (), (), (), (), (("A",u),("B",u*v)), ())
        pole = Chart((r,s), (), (), (), (), (("A",r),("B",s)), ())
        self.assertIsNone(recover_projection(pole,parent))
        # B/A = s on t=1 but every value of the parent v composes on t=0.
        ambiguous = Chart((r,s,t), (t*(t-1),), (), (), (),
                          (("A",r*t),("B",r*s*t)), ())
        self.assertIsNone(recover_projection(ambiguous,parent))

    def test_an_algebraic_unit_in_the_inverse_denominator(self):
        api = runtime_expressions()
        u, v, s, t = coords = api.symbols("u v s t")
        parent = Chart((u,v), (), (), (), (), (("X",u),("Y",u*v)), ())
        # The chosen parameter has two conjugates. 1/t = -t-1 holds on
        # the whole unmarked cover, including at s=0.
        child = Chart(coords[2:], (t**2+t+1,), (), (), (),
                      (("X",t*s),("Y",s**2)), ())
        self.assertEqual(recover_projection(child,parent), (t*s, -s*t-s))

    def test_nonmonic_base_map_and_implicit_child(self):
        api = runtime_expressions()
        u, v, r, s, h = api.symbols("u v r s h")
        parent = Chart((u,v), (), (), (), (), (("A",2*u),("B",3*u*v)), ())
        child = Chart((r,s,h), (h-r*s,), (), (), (), (("A",2*r),("B",3*h)), ())
        self.assertEqual(recover_projection(child,parent), (r,s))
        double_cover = Chart((u,v), (), (), (), (), (("A",u*u),("B",v)), ())
        self.assertIsNone(recover_projection(child,double_cover))


class ArchivedParentCoverTests(unittest.TestCase):
    def test_recover_sheared_parent_fan_and_certify_saved_ownership(self):
        from dataclasses import replace
        from engine.repair_parent_cover import certified_parent_cover
        api = runtime_expressions()
        u, v = api.symbols("u v")
        a = Chart((u,v), (), (), (u,), (), (("X",u+u*v),("Y",u*v)), ())
        b = Chart((u,v), (), (), (v,), (u,), (("Y",v),("X",u*v+v)), ())
        self.assertTrue(certified_parent_cover([b,a]))
        self.assertFalse(certified_parent_cover([a]))
        self.assertFalse(certified_parent_cover([a,b,a]))
        self.assertFalse(certified_parent_cover([a,replace(b,region=())]))
        self.assertFalse(certified_parent_cover([a,replace(b,region=(api.S.One,))]))
        self.assertFalse(certified_parent_cover([a,replace(b,fibre=(u,))]))
        # The second chart contracts its torus along a different base map.
        self.assertFalse(certified_parent_cover([a,replace(b,coordinates=(("X",u*v+2*v),("Y",v)))]))
