"""Independent Euler counts from toric orbits of the actual A-series charts."""
from collections import defaultdict
from itertools import combinations
import unittest

import sympy as sp

from engine.chain_backend import resolve_chain
from tests.special_monodromy_oracle import special_contributions


def chain_result(k,mode):
    p={(1,1,0):1,(0,0,k):1}
    result=resolve_chain(p,list("xyz"),mode)
    result.update(variables=list("xyz"),polynomial=f"x*y+z^{k}",mode=mode)
    return result


def toric_euler_counts(result,k):
    # x,y,z are monomials of the normalized model. Each open toric orbit is
    # determined by the rays of its zero coordinates, including original rays.
    counts=defaultdict(int)
    seen=set()
    for chart in result["charts"]:
        coords=sp.symbols("a b c")
        # These test-only expressions are produced by the in-process native engine.
        ring=dict(zip(chart["coordinates"],coords))
        exps=[sp.Poly(sp.sympify(chart["map"][name].replace("^","**"),locals=ring),*coords).monoms()[0]
              for name in "xyz"]
        rays=list(zip(*exps))
        exceptional={chart["coordinates"].index(d["equation"]):d["id"] for d in chart["exceptional"]}
        for size in (1,2,3):
            for zeros in combinations(range(3),size):
                rayset=frozenset(rays[j] for j in zeros)
                components=[exceptional[j] for j in zeros if j in exceptional]
                if rayset in seen or len(components)!=1:
                    continue
                seen.add(rayset)
                signs=[r[0]+r[1]-k*r[2] for r in rayset]
                orbit_dim=3-size
                # Both killed: entire orbit lies on strict transform. When
                # neither is killed, its primitive binomial cuts one point in
                # a 1-torus and has Euler zero in all higher-dimensional tori.
                if min(signs)<0<max(signs):
                    euler=0
                elif all(q==0 for q in signs):
                    euler=-int(orbit_dim==1)
                else:
                    euler=int(orbit_dim==0)
                counts[components[0]]+=euler
    return dict(counts)


class SpecialMonodromyTests(unittest.TestCase):
    def test_chain_point_and_line_strata(self):
        for k in (2,3,4,5,7,8):
            with self.subTest(k=k):
                r=chain_result(k,"embedded")
                terms=special_contributions(r)
                chi={p["component"]:p["euler_characteristic"] for p in terms}
                self.assertEqual(chi,toric_euler_counts(r,k))
                reported={d["id"]:d["multiplicity"] for c in r["charts"] for d in c["exceptional"]}
                self.assertEqual({p["component"]:p["multiplicity"] for p in terms},reported)
                self.assertEqual([(z["multiplicity"],z["euler_characteristic"])
                                 for z in terms if z["euler_characteristic"]],[(k,1)])
                intrinsic=special_contributions(chain_result(k,"intrinsic"))
                self.assertEqual([(z["multiplicity"],z["euler_characteristic"]) for z in terms],
                                 [(z["multiplicity"],z["euler_characteristic"]) for z in intrinsic])
                self.assertEqual("auxiliary" in intrinsic[-1]["component"],bool(k%2))

    def test_function_powers_and_conflicting_mixed_metadata(self):
        r=chain_result(5,"embedded")
        r["input_divisor"]={"factors":[{"equation":"x*y+z^5","multiplicity":3}]}
        r["polynomial"]="(1+x)*(x*y+z^5)^3"
        terms=special_contributions(r)
        self.assertEqual([(p["multiplicity"],p["euler_characteristic"]) for p in terms],
                         [(6,0),(12,0),(15,1)])
        r["input_divisor"]["factors"].append({"equation":"x","multiplicity":2})
        self.assertIsNone(special_contributions(r))

    def test_snc_at_origin_in_original_coordinates(self):
        def contributions(f, **kw):
            return special_contributions(dict(status="resolved", variables=list("xyz"),polynomial=f,**kw))
        smooth=contributions("z+x^8")
        self.assertEqual([(q["multiplicity"],q["euler_characteristic"]) for q in smooth],[(1,1)])
        power=contributions("(1+x)*(y-z^2)^4")
        self.assertEqual([(q["multiplicity"],q["euler_characteristic"]) for q in power],[(4,1)])
        crossing=contributions("(1+x)*x^2*y^3")
        self.assertCountEqual([(q["multiplicity"],q["euler_characteristic"]) for q in crossing],[(2,0),(3,0)])
        curved=contributions("(x-y^2)^42*(y+z)^35",input_divisor={"factors":[
            {"equation":"x-y^2","multiplicity":42},{"equation":"y+z","multiplicity":35}]})
        self.assertEqual([(q["multiplicity"],q["euler_characteristic"]) for q in curved],[(42,0),(35,0)])
        self.assertIsNone(contributions("(x-y^2)*(x+y^2)"))

    def test_invalid_saved_equations_are_never_evaluated(self):
        r=chain_result(3,"embedded")
        r["certificate"]["normalized_chart"]["equation"]="a*b+c^__import__('os')"
        self.assertIsNone(special_contributions(r))
        self.assertIsNone(special_contributions(dict(status="resolved",variables=["x","y"],
                                                      polynomial="lambda: 1")))


if __name__=="__main__":
    unittest.main()
