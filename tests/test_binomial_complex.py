import unittest

import sympy as sp

from engine.binomial_complex import binomial_complex, binomial_branch_equations
from engine.binomial_fan import build_binomial_fan
from engine.blowup_ideal import empty_locus, verify_snc


class BinomialComplexTests(unittest.TestCase):
    def test_disconnected_components_over_the_same_ambient_ray(self):
        # Three-dimensional point, point, and shared-face blowups for xy-z^3.
        # This nonminimal terminal model has two distinct strict components
        # inside ray 3, which appear in overlapping pairs of affine charts.
        rays=[(1,0,0),(0,1,0),(0,0,1),(1,1,1),(2,2,1),(3,3,2)]
        cones=[(3,1,2),(0,3,2),(0,1,4),(5,1,3),(4,1,5),(0,5,3),(0,4,5)]
        delta=(1,1,-3)
        dual, charts=binomial_complex(rays,cones,delta,-2,"intrinsic")
        split=[v for v in dual["vertices"] if v["ray_index"]==3]
        self.assertEqual({tuple(v["generic_zero_rays"]) for v in split},{(0,3),(1,3)})
        ids={v["id"] for v in split}
        middle,=[v["id"] for v in dual["vertices"] if v["id"] not in ids]
        edges={frozenset(c["vertices"]) for c in dual["cells"]}
        self.assertEqual(edges,{frozenset((end,middle)) for end in ids})
        at=lambda i,ray:next(a for a in charts[i] if a["ray_index"]==ray)
        self.assertEqual(at(0,3)["id"],at(3,3)["id"])
        self.assertEqual(at(1,3)["id"],at(5,3)["id"])
        self.assertNotEqual(at(0,3)["id"],at(1,3)["id"])
        self.assertEqual(at(0,3)["ideal"],["u1","u2"])

        u=sp.symbols("u1:4")
        for cone in cones:
            orders=[sum(a*b for a,b in zip(delta,rays[r])) for r in cone]
            f=sp.prod(v**max(0,d) for v,d in zip(u,orders)) + 2*sp.prod(v**max(0,-d) for v,d in zip(u,orders))
            boundary=[(u[j],) for j,r in enumerate(cone) if r>=3]
            self.assertTrue(verify_snc(u,(),(f,),boundary,3)["ok"])
        ambient,_=binomial_complex(rays,cones,delta,-2,"embedded")
        self.assertEqual([v["ray_index"] for v in ambient["vertices"]],[3,4,5])

    def test_piece_in_chart_missing_fiber_glues_before_filtering(self):
        rays=[(1,0,0),(0,1,0),(0,0,1),(0,1,1),(0,2,1),(0,3,2)]
        cones=[(0,3,2),(0,4,5),(0,5,3),(0,1,4)]
        # x*y^2-z^3, blown up along smooth coordinate centers over its singular locus.
        d,charts=binomial_complex(rays,cones,(1,2,-3),1,"intrinsic")
        piece=lambda i:next(q for q in charts[i] if q["ray_index"]==5)
        self.assertEqual(piece(1)["id"],piece(2)["id"])
        self.assertIn(piece(1)["id"],[v["id"] for v in d["vertices"]])
        u=sp.symbols("u1:4")
        for i,misses in ((1,True),(2,False)):
            cone=cones[i]
            mapping=[sp.prod(v**rays[r][a] for v,r in zip(u,cone)) for a in range(3)]
            equations=[sp.sympify(t.replace("^","**"),locals={str(v):v for v in u})
                       for t in piece(i)["ambient_ideal"]]
            self.assertEqual(empty_locus(u,(*mapping,*equations)),misses)

    def test_geometric_roots_give_disjoint_copies_on_every_chart(self):
        # x^4*y^2-t^6, and the same exponents with an irrational splitting
        # field, exercise both unit choices on the final affine charts.
        fan=build_binomial_fan([(4,2,0),(0,0,6)])
        rays,cones=fan["rays"],fan["cones"]
        self.assertEqual(len(cones),9)
        delta=(4,2,-6)
        for coefficient in (1,2):
            with self.subTest(coefficient=coefficient):
                dual,charts=binomial_complex(rays,cones,delta,coefficient,"intrinsic")
                family=dual["root_families"]["alpha1"]
                self.assertEqual(family["degree"],2)
                self.assertEqual(family["direction"],[2,1,-3])
                self.assertEqual(family["power_value"],str(coefficient))
                vertices={v["id"]:v["root"] for v in dual["vertices"]}
                self.assertCountEqual(vertices.values(),["alpha1_1"]*3+["alpha1_2"]*3)
                self.assertEqual(len(dual["cells"]),4)
                for cell in dual["cells"]:
                    self.assertEqual({vertices[v] for v in cell["vertices"]},{cell["root"]})
                    self.assertEqual(set(cell["faces"]),set(cell["vertices"]))
                u=sp.symbols("u1:4")
                env={str(v):v for v in u}
                env.update(alpha1_1=-sp.sqrt(coefficient),alpha1_2=sp.sqrt(coefficient))
                field=sp.QQ if coefficient==1 else sp.QQ.algebraic_field(sp.sqrt(coefficient))
                for cone,divisors in zip(cones,charts):
                    branches=binomial_branch_equations(rays,cone,delta,coefficient)
                    by_root={b["root"]:b["equation"] for b in branches}
                    equations=[sp.sympify(b["equation"].replace("^","**"),locals=env) for b in branches]
                    powers=[sum(a*b for a,b in zip(rays[i],delta)) for i in cone]
                    strict=(sp.prod(v**max(0,k) for v,k in zip(u,powers))
                            -coefficient*sp.prod(v**max(0,-k) for v,k in zip(u,powers)))
                    self.assertEqual(sp.expand(sp.prod(equations)-strict),0)
                    self.assertTrue(empty_locus(u,equations,domain=field))
                    for d in divisors:
                        self.assertEqual(d["root_equation"],by_root[d["root"]])
                        self.assertIn(d["root_equation"],d["ideal"])
                        self.assertEqual(d["root"],vertices[d["id"]])
                ambient,_=binomial_complex(rays,cones,delta,coefficient,"embedded")
                self.assertEqual(len(ambient["vertices"]),7)
                self.assertTrue(all("root" not in v for v in ambient["vertices"]))

    def test_mixed_chart_cannot_claim_disjoint_strict_branches(self):
        rays=[(1,0),(0,1)]
        with self.assertRaisesRegex(ValueError,"both monomials"):
            binomial_complex(rays,[(0,1)],(2,-2),1,"intrinsic")

    def test_rejects_a_false_terminal_stratum(self):
        # On the central chart after one blowup of the cusp surface, E cuts
        # out u1*u2=0 and still requires refinement in this ambient model.
        r=[(1,0,0),(0,1,0),(0,0,1),(1,1,1)]
        c=[(3,1,2),(0,3,2),(0,1,3)]
        with self.assertRaisesRegex(ValueError,"Nonterminal"):
            binomial_complex(r,c,(1,1,-3),1,"intrinsic")


if __name__=="__main__":
    unittest.main()
