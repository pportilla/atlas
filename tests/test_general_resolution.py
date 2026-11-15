import unittest

import sympy as sp

from engine.general_resolution import verify_candidate, _attempt, search
from engine.polynomial import parse
from concurrent.futures import ThreadPoolExecutor
from engine.resolution_search import ResolutionCandidate


def candidate(center, coords, boundary):
    return ResolutionCandidate(center,coords,boundary,sp.QQ,None,"test certificate")


class GeneralResolutionTests(unittest.TestCase):
    def test_polynomial_graph_centre_uses_free_coordinates_in_any_order(self):
        from unittest.mock import patch
        from engine import general_resolution, native_library
        if native_library._MODE != "native":
            self.skipTest("The bounded polynomial graph uses the native solver.")
        names = ["x", "y", "z"]
        for source in ("(y-x^2)^2-x*z^2", "(2*x^2-3*y)^2-x*(7*x+5*z)^2"):
            with self.subTest(source=source), patch.object(general_resolution, "_run_attempt", side_effect=AssertionError):
                record = general_resolution.smooth_reduced_graph(parse(source, names), names, "embedded", seconds=2)
                self.assertEqual(len(record["charts"]), 2)
                self.assertEqual(record["dual_complex"]["vertices"], [{"id": "E1", "dimension": 2}])

    def test_bounded_smooth_centre_declines_invalid_native_reply_and_wide_transport(self):
        from subprocess import CompletedProcess, TimeoutExpired
        from unittest.mock import patch
        from fractions import Fraction
        from engine import general_resolution
        from engine.native_library import _MODE, _kernel
        if _MODE != "native" or not hasattr(_kernel(), "atlas_smooth_reduced_centre_abi"):
            self.skipTest("Native reduced centre unavailable.")
        names = list("xyz")
        poly = parse("(x+y+z)^2-z*(x-y)^2", names)
        for reply in ("", "not a proof", "unknown 2\n", "centre 2\n1 1 0 0\n1 0 1 0\npass 1\n-1 0\n", "centre 2\n1 1 0 0\n1 0 1 0\npass 3\n1 1\n"):
            with patch("engine.native_process.run",return_value=CompletedProcess([],0,reply,"")):
                self.assertIsNone(general_resolution.smooth_reduced_graph(poly,names,"embedded"))
        for exc in (OSError("native binary unavailable"), TimeoutExpired("graph_verify",.1)):
            with patch("engine.native_process.run",side_effect=exc) as native:
                self.assertIsNone(general_resolution.smooth_reduced_graph(poly,names,"embedded"))
                self.assertLessEqual(native.call_args.kwargs["timeout"],.1)
        with patch("engine.native_process.run",side_effect=AssertionError):
            for exponent,coefficient in [((2**100,0,0),Fraction(1)),((4,0,0),Fraction(1,2**300))]:
                self.assertIsNone(general_resolution.smooth_reduced_graph(
                    dict(poly)|{exponent:coefficient}, names, "embedded"))

    def test_reduced_centre_decline_only_needs_parsed_sparse_terms(self):
        import sys
        from fractions import Fraction
        from subprocess import CompletedProcess
        from unittest.mock import patch
        from engine import native_library, reduced_centre
        from engine.polynomial import sparse_row

        if native_library._MODE != "native" or not hasattr(native_library._kernel(), "atlas_smooth_reduced_centre_abi"):
            self.skipTest("Native reduced centre unavailable.")
        names = ["z", "a", "x"]
        terms = {(1, 0, 2): Fraction(-2, 7), (0, 3, 1): Fraction(5, 9), (1, 2, 0): 0}
        with patch.dict(sys.modules, {"engine.general_resolution": None, "engine.native_polynomial": None}), \
                patch("engine.native_process.run", return_value=CompletedProcess([], 0, "none", "")) as run:
            self.assertIsNone(reduced_centre.smooth_reduced_graph(terms, names, "intrinsic"))
            header, equation = run.call_args.kwargs["input"].splitlines()
            self.assertEqual(header, "smooth_reduced_graph 3 0")
            self.assertEqual(sparse_row(equation), {e: c for e, c in terms.items() if c})
            with self.assertRaisesRegex(ValueError, "distinct polynomial coordinates"):
                reduced_centre.smooth_reduced_graph(terms, ["z", "z", "x"], "intrinsic")
            with self.assertRaisesRegex(ValueError, "Invalid polynomial exponent vector"):
                reduced_centre.smooth_reduced_graph({(1, 2): 1}, names, "embedded")
            run.assert_called_once()

    def test_curved_center_in_both_modes(self):
        x,y,z=sp.symbols("x y z")
        u,v=sp.symbols("U V")
        # Whitney umbrella with its double curve moved to a parabola.
        h=y-x**2
        f=h*h - x*z*z
        c=candidate((h,z),(u,v),((h,z),))
        for mode, dimension in (("embedded",2),("intrinsic",1)):
            with self.subTest(mode=mode):
                r=verify_candidate((x,y,z),f,c,mode)
                self.assertEqual(r["status"],"resolved")
                self.assertEqual(len(r["charts"]),2)
                self.assertEqual(r["dual_complex"]["vertices"],[{"id":"E1","dimension":dimension}])
                self.assertEqual(r["dual_complex"]["cells"],[])
                self.assertTrue(all([p["id"] for p in chart["exceptional"]]==["E1"] for chart in r["charts"]))

    def test_nonradical_ideal_encodes_two_point_blowups(self):
        x,y=sp.symbols("x y")
        u,v,w=sp.symbols("U V W")
        J=(x**3,x*y,y**2)  # (x,y)*(x^2,y)
        # On the exceptional fiber the directions (U:V:W) are two lines.
        c=candidate(J,(u,v,w),((x,y,u),(x,y,w)))
        f=y**2-x**4
        embedded=verify_candidate((x,y),f,c,"embedded")
        d=embedded["dual_complex"]
        self.assertEqual(len(embedded["charts"]),3)
        self.assertEqual(len(d["vertices"]),2)
        self.assertEqual(len(d["cells"]),1)
        self.assertEqual(set(d["cells"][0]["vertices"]),{"E1","E2"})
        support=[{q["id"] for q in chart["exceptional"]} for chart in embedded["charts"]]
        self.assertEqual([len(t) for t in support],[1,2,1])
        self.assertEqual(support[0] | support[2],support[1])
        self.assertFalse(support[0] & support[2])

        intrinsic=verify_candidate((x,y),f,c,"intrinsic")
        self.assertEqual(len(intrinsic["dual_complex"]["vertices"]),2)
        self.assertEqual(intrinsic["dual_complex"]["cells"],[])
        self.assertEqual([[q["id"] for q in chart["exceptional"]] for chart in intrinsic["charts"]],
                         [["E1","E2"],["E1","E2"],[]])

    def test_filters_secondary_singular_fiber_after_gluing(self):
        x,y=sp.symbols("x y")
        u,v=sp.symbols("U V")
        h=x*(x-1)
        f=y*y-h*h
        c=candidate((y,h),(u,v),((y,h),))
        # Two distinct singular fibers in the global model. Only components
        # meeting the x=y=0 fiber survive localization at the chosen origin.
        for mode, count in (("embedded",1),("intrinsic",2)):
            with self.subTest(mode=mode):
                r=verify_candidate((x,y),f,c,mode,effort=4)
                self.assertEqual(len(r["dual_complex"]["vertices"]),count)
                self.assertTrue(all("x - 1" not in d["ideal"] for chart in r["charts"]
                                    for d in chart["exceptional"]))

    def test_independent_thread_deadlines_and_unambiguous_field_name(self):
        x,y,z=sp.symbols("field_generator y z")
        u,v=sp.symbols("U V")
        theta=sp.sqrt(2)
        J=x,y
        c=ResolutionCandidate(J,(u,v),(J,),sp.QQ.algebraic_field(theta),theta,"thread test")
        f=x*x-y*y*z
        with ThreadPoolExecutor(max_workers=2) as executor:
            expired=executor.submit(_attempt,(x,y,z),f,c,"embedded",1,0.00001)
            good=executor.submit(_attempt,(x,y,z),f,c,"embedded",1,10)
            self.assertEqual(expired.result(timeout=5),("undecided",None))
            state,record=good.result(timeout=15)
        self.assertEqual(state,"certified")
        self.assertEqual(record["coefficient_field"]["generator"],"field_generator_")
        self.assertIn("field_generator",record["charts"][0]["map"])

    def test_graph_coefficients_written_in_the_marked_primitive_basis(self):
        from engine.affine_kernel import decode
        from engine.general_resolution import _text
        x,t = sp.symbols("x field_generator")
        field = sp.QQ.algebraic_field(sp.sqrt(2),sp.sqrt(3))
        f = sp.sqrt(3)*x + sp.sqrt(2)
        reported = decode((x,t), _text(f, field))
        self.assertEqual(sp.Poly(reported.subs(t, field.ext.as_expr())-f, x, domain=field).as_expr(), 0)

    def test_candidate_field_and_sparse_coefficients_survive_worker_transport(self):
        from engine.field_coefficient import indexed_field, root_position
        from engine.general_resolution import _read_candidate, _wire_candidate
        x,u = sp.symbols("x U")
        alpha = sp.sqrt(2) + sp.sqrt(3)
        field = sp.QQ.algebraic_field(alpha)
        # The coefficient sqrt(3) is not a literal power of the generator.
        boundary = ((u * (x + sp.sqrt(3))**3, sp.S.Zero),)
        original = ResolutionCandidate((x,), (u,), boundary, field, alpha, "round trip")
        wire = _wire_candidate((x,), x**2, original, "embedded", 1)
        variables, f, restored, mode, effort = _read_candidate(wire)
        self.assertEqual(_wire_candidate(variables, f, restored, mode, effort), wire)
        self.assertIs(_read_candidate(wire)[2].domain, restored.domain)
        self.assertEqual(wire["field"][1], 3)
        self.assertEqual(wire["boundary"][0][1], [])

        # Historical ordering puts the lower member of each conjugate pair first.
        for generator, index in ((-sp.sqrt(2),0), (1+sp.I,1),
                                 (sp.exp(sp.I*sp.pi/4),3), (-sp.exp(sp.I*sp.pi/4),0)):
            with self.subTest(generator=generator):
                poly = sp.QQ.algebraic_field(generator).ext.minpoly
                self.assertEqual(root_position(poly, generator), index)
        # CRootOf normalizes some input polynomials by rescaling the variable.
        coeffs = (1,0,0,0,512,1024)
        root = sp.CRootOf(sp.Poly.from_list(coeffs, x).as_expr(), 2)
        rebuilt, actual = indexed_field(coeffs, x, 2)
        self.assertEqual(actual, root)
        self.assertEqual(rebuilt, sp.QQ.algebraic_field(root))

    def test_removed_unit_exposes_open_domain_in_each_chart(self):
        names=list("xyz")
        poly=parse("(1+z)*(x^2-y^2*z)",names)
        r=search(poly,names,"embedded",seconds=10)
        self.assertEqual(r["certificate"]["unit_removed_at_origin"],"z + 1")
        self.assertEqual(r["base_domain"]["invert"],["z + 1"])
        self.assertTrue(all(c["base_domain"]==r["base_domain"] and c["invert"]==["z + 1"]
                            for c in r["charts"]))

    def test_nonconstant_unit_keeps_its_scalar_and_power(self):
        names = list("xyz")
        record = search(parse("-(3/2)*(1+z)^2*(x^2-y^2*z)", names), names, "embedded", seconds=10)
        x, y, z = sp.symbols("x y z")
        cert = record["certificate"]
        unit = sp.sympify(cert["unit_removed_at_origin"].replace("^", "**"))
        active = sp.sympify(cert["resolved_equation"].replace("^", "**"))
        self.assertEqual(unit, -3*z**2/2 - 3*z - sp.Rational(3, 2))
        self.assertEqual(active, x**2-y**2*z)

    def test_a_chart_cover_or_a_smooth_strict_transform_is_insufficient(self):
        x,y=sp.symbols("x y")
        u,v,w=sp.symbols("U V W")
        J=(x**3,x*y,y**2)
        for pieces in ((), ((x,y,u),), ((x,y,u),(x,y,u)), ((x,y),)):
            with self.subTest(pieces=pieces):
                c=candidate(J,(u,v,w),pieces)
                self.assertIsNone(verify_candidate((x,y),y*y-x**4,c,"embedded"))
        # A chart-wise expression must actually descend from a homogeneous
        # projective equation. The redundant generator x+U*x still vanishes
        # on this exceptional set, but cannot define a global graded ideal.
        c=candidate((x,y),(u,v),((x,y,x+u*x),))
        self.assertIsNone(verify_candidate((x,y),x*y,c,"embedded"))
        # After one blowup a cusp is smooth but tangent to E.
        c=candidate((x,y),(u,v),((x,y),))
        self.assertIsNone(verify_candidate((x,y),y*y-x**3,c,"embedded"))
        # A blowup of a smooth point on the original hypersurface is forbidden.
        self.assertIsNone(verify_candidate((x,y),x,c,"embedded"))


if __name__=="__main__":
    unittest.main()
