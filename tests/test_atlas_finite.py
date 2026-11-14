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
        self.assertEqual(len(finite_complex([c, other], 3, counts)["vertices"]), 3)
        self.assertIsNone(finite_complex([c, other, c], 3, counts))
        away = (("x", sp.S.One), ("y", t), ("z", v))
        self.assertIsNone(finite_complex([replace(c, coordinates=away, fibre=(sp.S.One, t, v), removed=())],
                                         3, counts))

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
