import unittest
from itertools import islice

import sympy as s

from engine.blowup_ideal import blowup_charts, verify_snc
from engine.general_resolution import verify_candidate
from engine.resolution_search import decode_candidate, priority_candidates, scheduled_candidates


def natural(n):
    binary = bin(n+1)[2:]
    return "0"*(len(binary)-1) + binary


def integer(n):
    return natural(2*n-1 if n > 0 else -2*n)


def sequence(items, minimum=0):
    return natural(len(items)-minimum) + "".join(items)


def term(n, exponents, basis=None):
    return integer(n) + ("" if basis is None else natural(basis)) + "".join(map(natural, exponents))


def as_code(bits):
    return int("1"+bits, 2)-1


class SearchTests(unittest.TestCase):
    def test_early_center_resolves_whitney_umbrella(self):
        vars = x,y,z = s.symbols("x y z")
        f = x*x-y*y*z
        candidate = next(c for c in priority_candidates(vars,f) if c.center == (x,y))
        charts = blowup_charts(vars, (), (f,), candidate.center)
        self.assertTrue(all(verify_snc(c.variables,c.ambient,c.strict,c.exceptional,3)["ok"] for c in charts))

    def test_nonlinear_singular_locus(self):
        vars = x,y,z = s.symbols("x y z")
        f=(y-x*x)**2-z**3
        centers=[set(c.center) for c in priority_candidates(vars,f)]
        self.assertIn({x*x-y,z}, centers)

    def test_tacnode_requires_the_two_blowup_candidate(self):
        x,y = s.symbols("x y")
        f = y**2-x**4
        seq = list(priority_candidates((x,y),f))
        two = seq[-1]
        self.assertEqual(two.center, (x**3,x*y,y**2))
        self.assertTrue(all(verify_candidate((x,y),f,c,"embedded") is None for c in seq[:-1]))
        result = verify_candidate((x,y),f,two,"embedded")
        self.assertEqual(len(result["charts"]),3)
        dual = result["dual_complex"]
        self.assertEqual((len(dual["vertices"]),len(dual["cells"])),(2,1))
        # The repeated direction is only a hypothesis; two centers may
        # still leave a nontransverse intersection on a higher-order germ.
        self.assertIsNone(verify_candidate((x,y),y**2-x**5,two,"embedded"))

    def test_repeated_rational_section_over_coordinate_center(self):
        x,y,z = s.symbols("x y z")
        h = y-3*x/2
        f = 4*h**2-(1+z)*x**4
        two, = (c for c in priority_candidates((x,y,z),f) if len(c.boundary)==2)
        self.assertEqual(two.center,(x**3,x*h,h**2))
        result = verify_candidate((x,y,z),f,two,"embedded")
        self.assertEqual(result["certificate"]["center_source"],two.source)
        self.assertEqual(len(result["dual_complex"]["vertices"]),2)

    def test_vertical_and_special_fiber_tangent(self):
        x,y,z = s.symbols("x y z")
        vertical = [c for c in priority_candidates((x,y),x**2-y**4) if len(c.boundary)==2]
        self.assertEqual([c.center for c in vertical],[(y**3,x*y,x**2)])
        # Tangency only on z=0 does not give a section on the full z-axis.
        self.assertFalse(any(len(c.boundary)==2 for c in
                             priority_candidates((x,y,z), y**2+z*x**2+x**4)))

    def test_common_repeated_slopes_have_numerical_order(self):
        from engine.resolution_search import _two_blowup_candidates
        x,y,z = s.symbols("x y z")
        repeated = (y-x)**2*(2*y+x)**2
        f = repeated*((y-2*x)**2 + z*(x**2+y**2)) + x**7
        candidates = _two_blowup_candidates((x,y,z), s.Poly(f, x,y,z, domain=s.QQ))
        self.assertEqual([c.center for c in candidates],
                         [(x**3, x*h, h**2) for h in (y+x/2, y-x)])

    def test_domain_and_homogeneous_components_in_unbounded_code(self):
        x,y = s.symbols("x y")
        # K=Q(alpha), alpha^2-2=0, root index 1.
        field=natural(1)+integer(-2)+integer(0)+natural(1)
        center=sequence([sequence([term(1,(1,0))],1),
                         sequence([term(1,(0,2)),term(-1,(1,0))],1)],1)
        boundary=sequence([sequence([sequence([term(1,(0,1,2,0),0),
                                               term(1,(1,0,0,2),1)],1)],1)])
        candidate=decode_candidate((x,y), as_code(field+center+boundary))
        self.assertEqual(candidate.center,(x,y*y-x))
        self.assertEqual(candidate.domain.ext.minpoly.degree(),2)
        u,v=candidate.projective
        self.assertEqual(candidate.boundary,((y*u*u+candidate.primitive*x*v*v,),))
        # Failure of homogeneity is an invalid projective certificate.
        mixed=sequence([sequence([sequence([term(1,(0,1,2,0),0),
                                            term(1,(1,0,0,1),1)],1)],1)])
        self.assertIsNone(decode_candidate((x,y),as_code(field+center+mixed)))
        self.assertIsNone(decode_candidate((x,y),0))

    def test_scheduling_revisits_even_when_new_codes_are_invalid(self):
        x,y=s.symbols("x y")
        first=list(islice(scheduled_candidates((x,y), x*y), 15))
        budgets=[b for c,b in first if c.source == "origin"]
        self.assertGreater(len(budgets),1)
        self.assertGreater(max(budgets),min(budgets))
        self.assertTrue(all(isinstance(b,int) and b>=1 for _,b in first))


if __name__ == "__main__":
    unittest.main()
