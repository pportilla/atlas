"""Independent affine-chart and fan checks for binomials with a smooth coordinate stratum."""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
from math import gcd, prod
import unittest

import sympy as sp

from engine.resolve import resolve


def subsets(items):
    for size in range(1, len(items)+1):
        yield from combinations(items, size)


class BinomialResolution(unittest.TestCase):
    def check_resolution(self, polynomial, names):
        embedded, intrinsic = [
            resolve({"polynomial":polynomial, "variables":list(names), "mode":mode})
            for mode in ("embedded", "intrinsic")]
        self.assertEqual([r["status"] for r in (embedded,intrinsic)], ["resolved"]*2)
        self.assertEqual(embedded["fan"], intrinsic["fan"])
        fan = embedded["fan"]
        rays = list(map(tuple, fan["rays"]))
        n = len(names)
        self.assertEqual(rays[:n], [tuple(int(i==j) for j in range(n)) for i in range(n)])
        self.assertTrue(all(min(r)>=0 and gcd(*r)==1 for r in rays))

        # Replay each smooth invariant center on its entire star. Its image
        # must lie in t=0 and one of the repeated positive-coordinate factors.
        cones = {frozenset(range(n))}
        for k, step in enumerate(embedded["steps"], n):
            self.assertEqual(step["new_ray_index"], k)
            center = frozenset(step["center_ray_indices"])
            self.assertGreaterEqual(len(center),2)
            self.assertTrue(all(i<k for i in center))
            ray = tuple(map(sum, zip(*(rays[i] for i in center))))
            self.assertEqual(ray,rays[k])
            image = {names[j] for j,a in enumerate(ray) if a}
            self.assertEqual(set(step["image_zero_coordinates"]),image)
            self.assertIn("t",image)
            self.assertTrue(image.intersection(set(names)-{"x","t"}))
            star = {c for c in cones if center<=c}
            self.assertTrue(star)
            cones -= star
            cones.update(c-{i}|{k} for c in star for i in center)
        self.assertEqual(cones,set(map(frozenset,fan["cones"])))
        self.assertTrue(any({names.index("x"),names.index("t")}<=c for c in cones))
        self.assertEqual(sum(Fraction(1,prod(sum(rays[i]) for i in c)) for c in cones),1)

        x = sp.symbols(names)
        f = sp.sympify(polynomial.replace("^","**"),locals=dict(zip(names,x)))
        expected_ambient, components = set(), defaultdict(set)
        charts = {c["id"]:c for c in intrinsic["charts"]}
        for chart in embedded["charts"]:
            intrinsic_chart = charts[chart["id"]]
            self.assertEqual(intrinsic_chart["map"],chart["map"])
            self.assertEqual(intrinsic_chart["strict_transform"],chart["strict_transform"])
            ids = chart["ray_indices"]
            u = sp.symbols(chart["coordinates"])
            symbols = dict(zip(chart["coordinates"],u))
            def read(s):
                return sp.sympify(s.replace("^","**"),locals=symbols)
            mapped = [read(chart["map"][var]) for var in names]
            for j, expr in enumerate(mapped):
                self.assertEqual(sp.Poly(expr,*u).terms(),[(tuple(rays[i][j] for i in ids),1)])
            self.assertEqual(abs(sp.Matrix([rays[i] for i in ids]).det()),1)
            strict = read(chart["strict_transform"])
            pullback = f.xreplace(dict(zip(x,mapped)))
            self.assertEqual(sp.expand(pullback-read(chart["pullback_factor"])*strict),0)

            # Independently check the Jacobian on every orbit. A two-term
            # nonconstant torus equation has a nonvanishing logarithmic
            # derivative at every root. If the equation vanishes identically,
            # an ordinary derivative outside the exceptional directions must
            # be an invertible monomial on the orbit.
            deriv = [strict.diff(v) for v in u]
            for subset in [(),*subsets(range(n))]:
                zero = {u[i]:0 for i in subset}
                restricted = sp.Poly(strict.xreplace(zero),*u)
                if restricted.is_zero:
                    normal = [sp.Poly(deriv[i].xreplace(zero),*u) for i in subset if ids[i]<n]
                    self.assertTrue(any(not p.is_zero and len(p.terms())==1 for p in normal))
                elif len(restricted.terms())>1:
                    a,b = restricted.monoms()
                    self.assertEqual(gcd(*(v-w for v,w in zip(a,b))),1)

            # Classify the restriction to every intersection of exceptional
            # axes as a polynomial on the full affine stratum. This detects
            # components living entirely in an original coordinate divisor.
            exceptional_positions = [j for j,i in enumerate(ids) if i>=n]
            present_here = set()
            for subset in subsets(exceptional_positions):
                face = frozenset(ids[j] for j in subset)
                expected_ambient.add(face)
                g = sp.Poly(strict.xreplace({u[j]:0 for j in subset}),*u)
                self.assertFalse(g.is_zero)
                if len(g.terms())==2:
                    components[face].add("torus")
                    present_here.add(face)
                else:
                    exps, = g.monoms()
                    for j,e in enumerate(exps):
                        if e:
                            self.assertEqual(e,1)
                            self.assertEqual(ids[j],names.index("x"))
                            components[face].add("coordinate_x")
                            present_here.add(face)
            intrinsic_vertices = {v["id"]:v["ray_index"] for v in intrinsic["dual_complex"]["vertices"]}
            self.assertEqual({intrinsic_vertices[e["id"]] for e in intrinsic_chart["exceptional"]},
                             {next(iter(s)) for s in present_here if len(s)==1})
            for entry in intrinsic_chart["exceptional"]:
                axis = u[ids.index(intrinsic_vertices[entry["id"]])]
                expected_ideal = sp.groebner([strict,axis],*u,domain=sp.QQ)
                reported = [read(eq) for eq in entry["ambient_ideal"]]
                self.assertEqual(sp.groebner(reported,*u,domain=sp.QQ),expected_ideal)

        self.assertTrue(any("coordinate_x" in tags for tags in components.values()))
        self.assertTrue(all(len(tags)==1 for tags in components.values()))
        for output, expected in [(embedded,expected_ambient),(intrinsic,set(components))]:
            dual = output["dual_complex"]
            self.assertEqual(dual["status"],"complete")
            vertices = {v["id"]:v["ray_index"] for v in dual["vertices"]}
            covered = {key:frozenset([i]) for key,i in vertices.items()}
            for cell in dual["cells"]:
                face = frozenset(vertices[i] for i in cell["vertices"])
                self.assertEqual(cell["dimension"],len(face)-1)
                covered[cell["id"]] = face
            self.assertEqual(Counter(covered.values()),Counter({s:1 for s in expected}))
            for cell in dual["cells"]:
                face = covered[cell["id"]]
                self.assertEqual({covered[k] for k in cell["faces"]},{face-{i} for i in face})

    def test_published_four_variable_binomial(self):
        self.check_resolution("x*y^2*z^3-t^6",list("xyzt"))

    def test_scaled_three_variable_binomial(self):
        self.check_resolution("2*x*y^2-3*t^3",list("xyt"))


if __name__=="__main__":
    unittest.main()
