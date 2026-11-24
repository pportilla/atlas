"""Verify complete, compatible smooth Newton fans using independent arithmetic."""
from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import gcd, prod
import unittest

from engine.newton_fan import build_fan, trial_supports
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
    def test_bounded_native_fans_match_isolated_normal_fans(self):
        from unittest.mock import patch
        from engine import newton_fan

        supports = [[(2, 0, 0, 0), (0, 2, 1, 0), (0, 2, 0, 1)],
                    [(0, 1, 0, 0), (2, 0, 0, 0)],
                    [(0, 2), (3, 0), (0, 2)], [(2, 2), (0, 0), (1, 1)]]
        for support in supports:
            with self.subTest(support=support):
                with patch.object(newton_fan, "_small_fan", return_value=None):
                    expected = build_fan(support)
                    proposals = trial_supports(support)
                with patch("engine.native_process.run", side_effect=AssertionError):
                    fan = build_fan(support)
                    self.assertEqual(trial_supports(support), proposals)
                self.assertEqual(fan, expected)
                check_fan(self, fan, support)

        support = [(2, 3, 0), (1, 1, 3)]
        from engine import native_process
        with patch.object(native_process, "run", wraps=native_process.run) as run:
            fan = build_fan(support)
        run.assert_called_once()
        check_fan(self, fan, support)

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

    def test_axial_dominance_preserves_mixed_vertices_and_boundary_faces(self):
        vertices = [(3, 0, 0), (0, 2, 1), (1, 0, 1)]
        support = [(5, 0, 0), (3, 1, 0), *vertices, (3, 0, 0), (3, 0, 2)]
        fan = build_fan(support)
        self.assertEqual(fan, build_fan(vertices))
        self.assertEqual(fan["newton_vertex_count"], len(vertices))
        check_fan(self, fan, support)

    def test_shared_nonsimplicial_faces(self):
        # Hyperplane cuts the 5-orthant into two nonsimplicial normal cones.
        # The boundary between them needs exactly the same triangulation from
        # each side; independent default triangulations can disagree here.
        p = parse("a*b+c*d*e",list("abcde"))
        fan = build_fan(p)
        self.assertEqual(fan["newton_vertex_count"],2)
        self.assertTrue(fan["boundary_subdivided"])
        check_fan(self,fan,list(p))

    def test_five_and_six_variable_germs_have_regular_fans_within_memory(self):
        # The sampled subdivision points gave 339,349 cones for the first
        # germ, past the former budget of 120,000 cone records. The second
        # germ stopped at the same budget.
        cases = [("x^3+y^5+z^8+t^9+u^9+x*y*z*t*u", "xyztu", 30_000),
                 ("x^3+y^5+z^7+t^9+u^10+v^10+x*y*z*t*u*v", "xyztuv", 50_000)]
        for expression, names, bound in cases:
            with self.subTest(germ=expression):
                p = parse(expression, list(names))
                fan = build_fan(p, seconds=None)
                self.assertTrue(fan["isomorphism_off_origin"])
                self.assertLess(len(fan["cones"]), bound)
                check_fan(self, fan, list(p))
                self.assertEqual(fan, build_fan(reversed(list(p)), seconds=None))

    def test_exact_subdivision_point_gives_fewer_cones_than_sampled_points(self):
        # Sampling 96 multiples of each coordinate residue gave 61 rays and
        # 117 cones here. A point of smaller total child index exists in a
        # parallelepiped the samples did not cover.
        support = [(7, 0, 0), (0, 5, 0), (0, 0, 3), (1, 1, 1)]
        fan = build_fan(support)
        check_fan(self, fan, support)
        self.assertLessEqual(len(fan["rays"]), 45)
        self.assertLessEqual(len(fan["cones"]), 85)

    def test_chart_memory_bound_follows_the_available_memory(self):
        from unittest.mock import patch
        from engine import newton_fan

        axes = [[int(i == j) for j in range(5)] for i in range(5)]
        fan = {"rays": axes, "cones": [list(range(5))] * 30_000}
        for available in (64 << 30, None):
            with patch.object(newton_fan, "available_memory", return_value=available):
                newton_fan.require_chart_memory(fan, 6)
        with patch.object(newton_fan, "available_memory", return_value=1 << 30):
            with self.assertRaisesRegex(LimitError, "30000 charts.*memory.*1024 MB are available"):
                newton_fan.require_chart_memory(fan, 6)
        self.assertGreater(newton_fan.available_memory(), 0)

    def test_memory_bound_hands_over_to_the_smooth_centre_rules(self):
        from unittest.mock import patch
        from engine import newton_fan
        from engine.resolve import resolve

        # With too little memory for its charts the Newton method stops and
        # the quick search continues with the smooth-centre rules.
        calls = []

        def smooth_centres(p, variables, mode, **options):
            calls.append("repeated tangent" if options.get("tangent_only") else "canonical")
            return None

        request = {"polynomial": "x^3+y^6+z^7+t^7+u^8+x*y*z*t*u", "variables": list("xyztu"),
                   "mode": "embedded", "search": "quick", "compute": "cpu"}
        with patch.object(newton_fan, "available_memory", return_value=1 << 20), \
                patch("engine.origin_backend.resolve_origin", side_effect=smooth_centres), \
                patch("engine.general_resolution.search", return_value=None):
            record = resolve(request)
        self.assertEqual(calls, ["repeated tangent", "canonical"])
        self.assertEqual(record["status"], "inconclusive")
        # The fan program stopped at the number of cones whose charts fit.
        self.assertRegex(record["diagnostic"]["fast_path_limit"],
                         "more than 12 cones, the number whose charts fit in the available memory")

    def test_fan_stops_at_the_cones_whose_charts_fit_in_memory(self):
        from unittest.mock import patch
        from engine import newton_fan

        p = parse("x^3+y^6+z^7+t^7+u^8+x*y*z*t*u", list("xyztu"))
        free = build_fan(p, seconds=None)
        size = len(free["cones"])
        need = newton_fan._chart_bytes(5, len(p))
        # Room for exactly this fan: the first scoring rule passes the limit,
        # the second stays within it, and the answer is the unlimited one.
        with patch.object(newton_fan, "available_memory", return_value=2 * need * size):
            self.assertEqual(newton_fan.chart_cone_limit(5, len(p)), size)
            self.assertEqual(build_fan(p, seconds=None, chart_terms=len(p)), free)
            newton_fan.require_chart_memory(free, len(p))
        with patch.object(newton_fan, "available_memory", return_value=2 * need * (size - 1)):
            with self.assertRaisesRegex(LimitError, f"more than {size - 1} cones.*memory"):
                build_fan(p, seconds=None, chart_terms=len(p))
            with self.assertRaisesRegex(LimitError, f"{size} charts"):
                newton_fan.require_chart_memory(free, len(p))
            # A caller that builds no charts on the fan has no such limit.
            self.assertEqual(build_fan(p, seconds=None), free)

    def test_trial_fans_are_tried_when_the_fan_of_the_germ_does_not_fit(self):
        from unittest.mock import patch
        from engine import newton_backend
        from engine.newton_backend import resolve_newton

        names = list("xyzt")
        p = parse("x^2*y+y^3*z+z^4+t^5", names)
        expected = resolve_newton(p, names, "embedded", verified=[], integrals=True)
        self.assertEqual(expected["engine"], "verified-monomial-modification")
        real, calls = newton_backend.build_fan, []

        def first_fan_too_large(exponents, **options):
            calls.append(options)
            if len(calls) == 1:
                raise LimitError("The regular Newton fan has more than 3 cones.")
            return real(exponents, **options)

        with patch.object(newton_backend, "build_fan", side_effect=first_fan_too_large):
            record = resolve_newton(p, names, "embedded", verified=[], integrals=True)
        self.assertEqual((record["engine"], len(record["charts"])), (expected["engine"], len(expected["charts"])))
        self.assertEqual(calls[0], {"exhaustive": False, "seconds": None, "chart_terms": len(p)})

    def test_large_trial_fans_are_deferred_and_then_tried_within_memory(self):
        from unittest.mock import patch
        from engine import newton_fan, newton_trials

        names = list("xyzt")
        p = parse("x^2*y+y^3*z+z^4+t^5", names)
        support = next(s for s in trial_supports(p)
                       if newton_trials._try(p, names, s, verified=[], integrals=True) is not None)
        direct = newton_trials._try(p, names, support, verified=[], integrals=True)
        self.assertEqual(direct["engine"], "verified-monomial-modification")
        size = len(direct["charts"])
        self.assertGreater(size, 100)
        with patch.object(newton_trials, "DEFERRED_CONES", size - 1):
            # Above the threshold the trial is set aside, or dropped by a
            # caller that keeps no list. It is never an error.
            self.assertIsNone(newton_trials._try(p, names, support, verified=[], integrals=True))
            deferred = []
            self.assertIsNone(newton_trials._try(p, names, support, verified=[], integrals=True, deferred=deferred))
            self.assertEqual([len(fan["cones"]) for _, fan, _ in deferred], [size])
            small = build_fan([(1, 1, 1, 1)])
            self.assertEqual(len(small["cones"]), 1)
            order = []
            real = newton_trials._try

            def watch(poly, variables, trial, **options):
                order.append(len(options["fan"]["cones"]))
                self.assertFalse(options["defer"])
                return real(poly, variables, trial, **options)

            # Smallest fan first. The one-cone fan is not a resolution.
            listed = [*deferred, (p, small, deferred[0][2])]
            with patch.object(newton_trials, "_try", side_effect=watch):
                late = newton_trials.resolve_deferred_trials(p, names, listed)
            self.assertEqual(order, [1, size])
            self.assertEqual((late["engine"], len(late["charts"])), (direct["engine"], size))
            # Without memory for its charts the deferred fan stops the Newton
            # method with a limit that its caller hands over.
            with patch.object(newton_fan, "available_memory", return_value=1 << 20):
                with self.assertRaisesRegex(LimitError, f"{size} charts.*memory"):
                    newton_trials.resolve_deferred_trials(p, names, deferred)
            self.assertIsNone(newton_trials.resolve_deferred_trials(p, names, [(p, small, deferred[0][2])]))

    def test_five_variable_cusp_resolves_with_both_zetas(self):
        from itertools import combinations as subsets
        from engine.resolve import resolve

        # Its fan has 26,045 cones with the sampled subdivision points: more
        # strata than the former budget of 750,000 allowed in five variables.
        powers = (3, 6, 7, 7, 8)
        expression = "x^3+y^6+z^7+t^7+u^8+x*y*z*t*u"
        record = resolve({"polynomial": expression, "variables": list("xyztu"), "mode": "embedded",
                          "search": "quick", "compute": "cpu", "monodromy": True})
        self.assertEqual((record["status"], record["engine"]), ("resolved", "newton-polyhedron"))
        self.assertEqual(len(record["charts"]), len(record["fan"]["cones"]))
        self.assertLess(len(record["charts"]), 10_000)
        self.assertEqual(len(record["atlas"]), len(record["charts"]))
        self.assertEqual(record["dual_complex"]["status"], "complete")
        self.assertEqual(record["topological_zeta"]["status"], "computed")
        self.assertEqual(record["monodromy_zeta"]["status"], "computed")
        # Kouchnirenko: on the Newton boundary of sum x_i^a_i + x_1...x_n the
        # two top volumes cancel, leaving the alternating sum of the
        # elementary symmetric functions of the a_i of degree at most n-2.
        n = len(powers)
        milnor = sum((-1) ** (n - k) * sum(prod(choice) for choice in subsets(powers, k)) for k in range(n - 1))
        self.assertEqual(milnor, 1890)
        self.assertEqual(record["monodromy_zeta"]["milnor_fiber_euler_characteristic"], 1 + (-1) ** (n - 1) * milnor)

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
