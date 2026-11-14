"""Verify complete, compatible smooth Newton fans using independent arithmetic."""
from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import gcd, prod
import unittest

from engine.newton_fan import build_fan
from engine.polynomial import LimitError, parse


def determinant(matrix):
    # Fraction-free elimination with row pivoting.
    a = [list(row) for row in matrix]
    last, sign = 1, 1
    for j in range(len(a)-1):
        pivot = next((i for i in range(j,len(a)) if a[i][j]), None)
        if pivot is None:
            return 0
        if j != pivot:
            a[j],a[pivot] = a[pivot],a[j]
            sign *= -1
        for i in range(j+1,len(a)):
            for k in range(j+1,len(a)):
                a[i][k] = (a[i][k]*a[j][j]-a[i][j]*a[j][k])//last
        last = a[j][j]
    return sign*a[-1][-1]


def check_fan(test, fan, support):
    rays, cones = fan["rays"], fan["cones"]
    dim = len(support[0])
    test.assertEqual(rays[:dim], [[int(i==j) for j in range(dim)] for i in range(dim)])
    test.assertEqual(len({tuple(r) for r in rays}),len(rays))
    test.assertTrue(all(min(r)>=0 and gcd(*r)==1 for r in rays))
    test.assertEqual(fan["boundary_rays"], [i for i in range(dim,len(rays)) if 0 in rays[i]])
    test.assertEqual(fan["isomorphism_off_origin"], not fan["boundary_rays"])
    minima = []
    for ray in rays:
        weights = [sum(a*b for a,b in zip(ray,e)) for e in support]
        minima.append({j for j,v in enumerate(weights) if v==min(weights)})
    volume = Fraction(0)
    facets = Counter()
    for cone in cones:
        rr = [rays[i] for i in cone]
        test.assertEqual(determinant(rr),1)
        # The same source monomial achieves the minimum on *every* ray in a
        # cone. Therefore its linear form equals the lower envelope throughout.
        test.assertTrue(set.intersection(*(minima[i] for i in cone)))
        volume += Fraction(1,prod(sum(r) for r in rr))
        for f in combinations(cone,dim-1):
            facets[tuple(sorted(f))] += 1
    # All nonboundary facets have precisely one matching chart across them.
    for f, count in facets.items():
        boundary = any(all(rays[i][j]==0 for i in f) for j in range(dim))
        test.assertEqual(count,1 if boundary else 2,f)
    # The normalized volumes of the cone sections by sum(x)=1 exactly add up
    # to the volume of the full simplex.
    test.assertEqual(volume,1)


class NewtonFan(unittest.TestCase):
    def test_exhaustive_resolution_certifies_large_compact_face(self):
        from engine.resolve import resolve

        expression = "x^4098+y^4098+z^4098+x^2049*y^2049+x^2049*z^2049+y^2049*z^2049"
        request = {"polynomial": expression, "variables": list("xyz"), "mode": "embedded"}
        self.assertEqual(resolve(dict(request, search="quick"))["status"], "inconclusive")
        resolution = resolve(dict(request, search="exhaustive"))
        self.assertEqual((resolution["status"], resolution["engine"]), ("resolved", "newton-polyhedron"))
        face, = (f for f in resolution["certificate"]["compact_faces"] if f["face_dimension"] == 2)
        self.assertEqual(face["status"], "nondegenerate")
        self.assertTrue(face["unit_ideal"])
        self.assertEqual(len(resolution["charts"]), 3)

    def test_exhaustive_fan_is_invariant_under_dilation_and_translation(self):
        standard = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
        diagonal_fan = build_fan(standard)
        for degree in (130, 4097, 2**300):
            support = [tuple(degree if i == j else 0 for j in range(3)) for i in range(3)]
            with self.subTest(degree=degree):
                self.assertEqual(build_fan(support, exhaustive=True), diagonal_fan)
                with self.assertRaises(LimitError):
                    build_fan(support)
        base = [(2, 0, 0), (0, 3, 0), (0, 0, 5), (1, 1, 1)]
        offset = (2**129, 4, 2**143)
        support = [tuple(c+t for c, t in zip(row, offset)) for row in base]
        fan = build_fan(support, exhaustive=True)
        self.assertEqual(fan, build_fan(base))
        check_fan(self, fan, support)

    def test_nonconvenient_four_variable_germ(self):
        p = parse("x^5+z*y^3+z^15+t^4*x+t^8", ["x","y","z","t"])
        fan = build_fan(p)
        self.assertEqual(fan["newton_vertex_count"],5)
        self.assertTrue(fan["boundary_subdivided"])
        self.assertGreater(fan["newton_ray_count"], 5)
        check_fan(self,fan,list(p))
        self.assertEqual(fan,build_fan(reversed(list(p))))

    def test_convenient_and_nonvertex_support(self):
        p = parse("x^2+y^3+z^5+x*y*z+x^11", ["x","y","z"])
        f = build_fan(p)
        self.assertEqual(f["newton_vertex_count"],3)
        self.assertTrue(f["isomorphism_off_origin"])
        check_fan(self,f,list(p))

    def test_shared_nonsimplicial_faces(self):
        # Hyperplane cuts the 5-orthant into two nonsimplicial normal cones.
        # The boundary between them needs exactly the same triangulation from
        # each side; independent default triangulations can disagree here.
        p = parse("a*b+c*d*e",list("abcde"))
        fan = build_fan(p)
        self.assertEqual(fan["newton_vertex_count"],2)
        self.assertTrue(fan["boundary_subdivided"])
        check_fan(self,fan,list(p))

    def test_high_dimension_and_single_vertex(self):
        for dim in (2,4,8):
            for support in [[tuple(int(j==i) for j in range(dim)) for i in range(dim)],
                            [(1,)*dim, (3,)*dim]]:
                with self.subTest(dim=dim, count=len(support)):
                    fan=build_fan(support)
                    self.assertTrue(fan["isomorphism_off_origin"])
                    check_fan(self,fan,support)


if __name__ == "__main__":
    unittest.main()
