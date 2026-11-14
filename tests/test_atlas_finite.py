"""Partial finite-centre identifications must not certify a global divisor."""
import unittest
from dataclasses import replace

import sympy as sp

from engine.atlas_finite import finite_complex
from engine.local_blowups import Chart


class FiniteCentre(unittest.TestCase):
    def test_ownership_is_checked_independently_on_rational_orbits(self):
        t, v, w = sp.symbols("t v w")
        # On this affine modification, y/x=t extends regularly across
        # D=(t³-1). One open omits its rational component t=1.
        p = t**3 - 1
        mapping = (("x", v*p), ("y", t*v*p), ("z", w*p))
        c = Chart((t, v, w), (), (("D", 1, 1, (p,)),), tuple(f for _, f in mapping), (), mapping, (),
                  removed=((t-1,),))
        counts = {frozenset(("D",)): None}
        self.assertIsNone(finite_complex([c], 3, counts))
        other = replace(c, removed=((t**2+t+1,),))
        dual = finite_complex([c, other], 3, counts)
        self.assertEqual(len(dual["vertices"]), 3)
        domains = dual["finite_sheet_certificate"]["chart_sections"]
        self.assertEqual([row["id"] for row in domains[0]["divisors"]["D"]], ["D.1", "D.2"])
        self.assertEqual([row["id"] for row in domains[1]["divisors"]["D"]], ["D.3"])
        self.assertIsNone(finite_complex([c, other, c], 3, counts))
        away = (("x", sp.S.One), ("y", t), ("z", v))
        self.assertIsNone(finite_complex([replace(c, coordinates=away, fibre=(sp.S.One, t, v), removed=())],
                                         3, counts))

    def test_two_independent_sheet_parameters_and_wide_graph_equation(self):
        from engine import atlas
        from engine.chart_expressions import runtime_expressions
        api = runtime_expressions()
        a, b, v, w, h = api.symbols("a b v w h")
        high = 2**72
        coefficient = 2**160 + 7
        p, q = a**2-2, b**3-3
        images = (v*p*q, a*v*p*q, b*v*p*q, w*p*q)
        mapping = tuple(zip(("x", "y", "z", "t"), images))
        chart = Chart((h,a,b,v,w), (h-coefficient*w**high,), (("D",1,1,(p,)),("E",1,1,(q,))),
                      images, (), mapping, ())
        dual = atlas.boundary_complex([chart], 4)
        self.assertEqual(dual["status"], "complete")
        self.assertEqual(len(dual["vertices"]), 5)
        self.assertEqual({tuple(cell["vertices"]) for cell in dual["cells"]},
                         {(f"D.{i}", f"E.{j}") for i in (1,2) for j in (1,2,3)})
        self.assertTrue(all(set(c["faces"]) == set(c["vertices"]) for c in dual["cells"]))
        self.assertEqual({tuple(s["intersection"]):s["degree"] for s in dual["finite_sheet_certificate"]["bases"]},
                         {("D",):2, ("E",):3, ("D","E"):6})
        from engine.atlas_finite import geometric_chart_records
        from engine.session_values import Resolution
        public = dict(id="C1", coordinates=[str(x) for x in chart.variables], ambient_ideal=[str(chart.ambient[0])],
                      map={k:str(f) for k,f in mapping}, strict_transform="1", exceptional=[
                          {"id": "D", "ideal": ["a^2-2"]}, {"id": "E", "ideal": ["b^3-3"]}])
        record = {"variables": list(dict(mapping)), "charts": [public], "dual_complex": dual}
        projected = geometric_chart_records(record)
        self.assertEqual(geometric_chart_records({**record, "charts": projected}), projected)
        linked = Resolution(record)
        self.assertEqual({v.id for v in linked.chart("C1").divisors}, {v["id"] for v in dual["vertices"]})
        left = linked.divisor("D.1").ideal_in(linked.chart("C1"))
        right = linked.divisor("D.2").ideal_in(linked.chart("C1"))
        self.assertEqual(left[0], right[0])
        self.assertNotEqual(left[1], right[1])

    def test_independent_roots_and_relative_field_sheet(self):
        from engine.atlas_finite import marked_primes, splitting_charts
        from engine.chart_expressions import runtime_expressions
        api = runtime_expressions()
        x,y,z,u,v,t = api.symbols("x y z u v t")
        cases = (
            ((x,y,z,u,v), (), (x*x-2,y*y-3), 3, 4, 4),
            ((x,y,z,u,v,t), (t,t*t-2,2,"root 1 (FLINT order)"), (x*x-t,z), 3, 2, 4),
            ((x,y,z,u,v), (), (z,x*x-2*(1+y*y)**2), 3, 2, 2),
        )
        for coords, field, section, dim, n, degree in cases:
            with self.subTest(section=section):
                c = Chart(coords, (field[1],) if field else (), (("D",1,1,section),),
                          (), (), (), (), coefficient_field=field)
                grown = splitting_charts([c], (0,(*c.ambient,*section),dim,True))
                self.assertIsNotNone(grown)
                chart, = grown
                self.assertEqual(chart.coefficient_field[2], degree)
                result = marked_primes(chart, (*chart.ambient,*chart.components[0][3]), dim, branches=True)
                self.assertEqual(len(result), n)

    def test_cubic_cylinder_triangle_faces_on_reordered_opens(self):
        from itertools import combinations, product
        from engine import atlas
        from engine.chart_expressions import runtime_expressions
        api = runtime_expressions()
        local = api.symbols("u v a b c d")
        base = api.symbols("x y z w p q")
        x,y = base[:2]
        # Blow up three independent pairs of coordinate axes. The first
        # modification separates the three tangent directions over Qbar.
        # The other two divisors meet each of those sheets in an edge,
        # and their common face belongs to all three triangles.
        f = y**3-2*x**3+x**4
        charts, earlier = [], []
        for pivots in product((0,1), repeat=3):
            images, divisors = [], []
            for (a,b), pivot, label in zip(zip(local[::2],local[1::2]),pivots,"DFG"):
                images.extend((a,a*b) if pivot == 0 else (a*b,b))
                divisors.append((label,(a,b)[pivot]))
            chart,inverse = atlas.chart(local,tuple(images),divisors,f,base,earlier)
            charts.append(chart)
            earlier.append(inverse)
        for saved in (charts,list(reversed(charts))):
            copies = atlas.from_json(atlas.to_json(saved))
            dual = atlas.boundary_complex(copies,6,True)
            self.assertEqual(dual["status"],"complete")
            roots = {"D.1","D.2","D.3"}
            self.assertEqual({v["id"] for v in dual["vertices"]},roots|{"F","G"})
            self.assertEqual({v["dimension"] for v in dual["vertices"]},{4})
            edges = {c["id"]:frozenset(c["vertices"]) for c in dual["cells"] if c["dimension"] == 1}
            self.assertEqual(len(edges),7)
            triangles = [c for c in dual["cells"] if c["dimension"] == 2]
            self.assertEqual(len(triangles),3)
            self.assertEqual({frozenset(c["vertices"]) for c in triangles},
                             {frozenset((root,"F","G")) for root in roots})
            for cell in triangles:
                self.assertEqual({edges[k] for k in cell["faces"]},
                                 {frozenset(p) for p in combinations(cell["vertices"],2)})
