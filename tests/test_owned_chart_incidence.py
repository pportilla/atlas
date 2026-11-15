import unittest
from dataclasses import replace
from unittest.mock import patch

from engine import kernel_json, native_library
from engine import native_polynomial as np
from engine.local_blowups import Chart, GiveUp, cells


@unittest.skipUnless(native_library._MODE == "native" and
                     hasattr(native_library._kernel(), "atlas_owned_chart_incidence"),
                     "Requires native owned chart incidence.")
class OwnedChartIncidenceTests(unittest.TestCase):
    def compare(self, chart, dimension):
        def run():
            rational = {}
            return cells((chart,), dimension, rational_counts=rational), rational

        actual = run()
        native = kernel_json.kernel_json
        def reference(symbol, operation, **data):
            return None if symbol == "atlas_owned_chart_incidence" else native(symbol, operation, **data)
        with patch.object(kernel_json, "kernel_json", reference):
            self.assertEqual(actual, run())
        return actual

    @staticmethod
    def chart(axes, components, fibre, **kwargs):
        return Chart(axes, (), tuple((label,1,1,ideal) for label,ideal in components),
                     fibre, (), tuple((str(v),v) for v in axes), (), **kwargs)

    def test_open_fibre_holes_and_noncontained_components(self):
        axes = x,y = np.symbols("x y")
        source = self.chart(axes, (("E",(x,)),("F",(y,))), (x*y,))
        E,F,EF = map(frozenset, ("E","F","EF"))
        self.assertEqual(self.compare(source,2), ({E:1,F:1,EF:1},{E:1,F:1}))
        self.assertEqual(self.compare(replace(source,removed=((x,y),)),2),
                         ({E:1,F:1},{E:1,F:1}))
        self.assertEqual(self.compare(replace(source,fibre=(y,)),2), ({E:None,F:1,EF:1},{E:None,F:1}))
        self.assertEqual(self.compare(replace(source,region=(np.coerce(1),)),2), ({},{}))

    def test_marked_counts_descend_by_the_field_degree(self):
        x,y,t = np.symbols("x y t")
        chart = self.chart((x,y,t), (("E",(x,)),("F",(y,))), (x*y,))
        chart = replace(chart,ambient=(t*t-2,), coefficient_field=(t,t*t-2,2,"sqrt(2)"))
        E,F,EF = map(frozenset, ("E","F","EF"))
        self.assertEqual(self.compare(chart,2), ({E:1,F:1,EF:1},{E:1,F:1}))
        row = lambda f: np.row((x,y,t), f)
        raw = kernel_json.kernel_json("atlas_owned_chart_incidence", "cells", width=3, dimension=2,
                                      ambient=[row(t*t-2)], relation=row(t*t-2), degree="2",
                                      labels=[[row(x)],[row(y)]], fibre=[row(x*y)], region=[], holes=[])
        self.assertEqual([(r["indices"],r["here"],r["rational"]) for r in raw],
                         [([0],"2","2"),([1],"2","2"),([0,1],"2",None)])
        with self.assertRaises(GiveUp):
            cells((replace(chart,coefficient_field=(t,t*t-2,3,"sqrt(2)")),),2)

    def test_split_factors_and_an_unresolved_geometric_orbit(self):
        axes = x,y,z = np.symbols("x y z")
        D = frozenset(("D",))
        h = x*(x-1)
        chart = self.chart(axes, (("D",(h,)),), (h,))
        self.assertEqual(self.compare(chart,3), ({D:2},{D:2}))
        q = x*x+1
        chart = self.chart(axes, (("D",(q,)),), (q,))
        self.assertEqual(self.compare(chart,3), ({D:None},{D:None}))

    def test_wide_rows_and_large_hole_covers_keep_the_general_verifier(self):
        axes = np.symbols("w0:35")
        x,y = axes[:2]
        D = frozenset(("D",))
        chart = self.chart(axes, (("D",(x,)),), ((2**197+9)*x**(2**65+7),))
        self.assertEqual(self.compare(chart,35), ({D:1},{D:1}))
        chart = self.chart((x,y), (("D",(x,)),), (x,), removed=tuple((y-i,) for i in range(17)))
        self.assertEqual(self.compare(chart,2), ({D:1},{D:1}))

    def test_factor_budget_declines_before_replaying_the_general_count(self):
        axes = x,y,z = np.symbols("x y z")
        q = x**37-1
        row = np.row(axes, q)
        self.assertIsNone(kernel_json.kernel_json("atlas_owned_chart_incidence", "cells", width=3,
                            dimension=3, ambient=[], relation=None, degree=None, labels=[[row]],
                            fibre=[row], region=[], holes=[]))
        chart = self.chart(axes, (("D",(q,)),), (q,))
        self.assertEqual(self.compare(chart,3), ({frozenset("D"):None},{frozenset("D"):None}))
