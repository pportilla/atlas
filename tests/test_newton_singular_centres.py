import unittest
from unittest.mock import patch

from engine import native_library

from engine.app import calculate_resolution
from engine.monodromy import monodromy_zeta
from engine.newton_backend import _boundary_centres
from engine.newton_fan import build_fan, trial_supports
from engine.polynomial import parse


class NewtonSingularCentres(unittest.TestCase):
    def test_transverse_homogeneous_pencil_preserves_newton_and_graph_prefix(self):
        from engine.kernel_json import kernel_json
        from engine.newton_backend import coordinate_choices, resolve_newton
        from engine.newton_faces import face_search
        from engine.point_geometry import encode_terms
        from engine.resolve import resolve
        from engine.general_resolution import search

        if native_library._MODE != "native" or not hasattr(native_library._library(), "atlas_transverse_pencil_abi"):
            self.skipTest("Native transverse pencil operation unavailable.")
        def recognise(p):
            return kernel_json("atlas_resolution_candidate", "transverse_pencil", width=3, terms=encode_terms(p))

        names = list("xyz")
        # All three parameter positions. Include a nonpower higher coefficient
        # and a rational leading scalar to exercise the exact factor convention.
        for source in ("(x+y)^2-(x-y)^2*z", "(z+2*x)^4+y*(3*z^4-x^4)", "(y+2*z)^3/7-x*(y^3+z^3)"):
            polynomial = parse(source,names)
            axes = recognise(polynomial)
            self.assertIsNotNone(axes)
            parameter = next(i for i in range(3) if i not in axes)
            with face_search(True):
                # The obstruction relies on precisely these shears. A future
                # Newton choice/repair policy must revisit the shortcut.
                choices = list(coordinate_choices(polynomial,3))
                self.assertEqual({tuple(trace) for _,_,trace in choices}, {((i,parameter),) for i in axes})
                self.assertIsNone(resolve_newton(polynomial,names,"embedded",compute="cpu",defer_outputs=True))
            reference = search(polynomial,names,"embedded",exhaustive=True)
            with patch("engine.exhaustive_resolution.search", side_effect=AssertionError):
                record = resolve(dict(polynomial=source,variables=names,monodromy=True,compute="cpu"))
            self.assertEqual((record["charts"],record["steps"]), (reference["charts"],reference["steps"]))
            self.assertEqual(record["certificate"]["candidates_examined"], 3)
        for source in ("x^2-(x-y)^2*z", "(x+y)^2-z*(x^2-y^2)", "(x+y)^2-z*(x-y)^3",
                       "(x+y)^2-z^2*(x-y)^2", "x^2+y^2-z*(x-y)^2"):
            self.assertIsNone(recognise(parse(source,names)))
        zero_terms = [["1",["1","1","0"]],["-1",["1","1","0"]]] * 2
        self.assertIsNone(kernel_json("atlas_resolution_candidate", "transverse_pencil", width=3, terms=zero_terms))

    def test_pencil_decline_preserves_search_and_quick_route(self):
        from engine import general_resolution, reduced_centre
        from engine.resolve import resolve
        if native_library._MODE != "native" or not hasattr(native_library._library(), "atlas_transverse_pencil_abi"):
            self.skipTest("Native transverse pencil operation unavailable.")
        request = dict(polynomial="(x+y)^2-(x-y)^2*z",variables=list("xyz"),compute="cpu")
        stop = RuntimeError("exhaustive worker started")
        with patch.object(general_resolution,"_isolated_graph_prefix",return_value=None), \
                patch.object(reduced_centre,"smooth_reduced_graph",return_value=None), \
                patch("engine.exhaustive_resolution.search",side_effect=stop):
            with self.assertRaisesRegex(RuntimeError,"exhaustive worker started"):
                resolve(request)
        with patch.object(general_resolution,"_isolated_graph_prefix",side_effect=AssertionError):
            result = resolve(dict(request,search="quick"))
        self.assertEqual(result["engine"], "verified-ideal-blowup")
        self.assertEqual(result["certificate"]["center_source"], "smooth reduced singular locus")
        with patch.object(reduced_centre,"smooth_reduced_graph",return_value=None):
            fallback = resolve(dict(request,search="quick"))
        self.assertEqual(fallback["engine"], "canonical-blowups")

    def test_nonconvenient_fan_preserves_the_smooth_locus(self):
        names = list("xyzt")
        p = "x^2*y^3-z^4*t^6+t^9*z^3"
        status, solved = calculate_resolution(p, names, mode="embedded", search="exhaustive",
                                              compute="cpu", monodromy=True)
        self.assertEqual(status, 200)
        self.assertTrue(solved["certificate"]["boundary_image_coordinate_ideals"])
        self.assertEqual(solved["certificate"]["atlas_verification"], "Embedded normal crossings on the owned origin fibre")
        self.assertEqual(solved["monodromy_zeta"]["expression"], monodromy_zeta(solved)["expression"])

    def test_auxiliary_fan_checks_original_polynomial(self):
        names = list("xyzt")
        expr = "t*x^4+x^2*y*z+y^2*z"
        p = parse(expr, names)
        self.assertIsNone(_boundary_centres(p, names, build_fan(p)))
        candidates = trial_supports(p)
        self.assertIn(((0, 1, 0, 0), (2, 0, 0, 0)), candidates)
        self.assertFalse(trial_supports({(2**70, 0, 0, 0): 1, (0, 0, 1, 0): 1}))
        status, solved = calculate_resolution(expr, names, mode="embedded", search="exhaustive",
                                              compute="cpu", monodromy=True)
        self.assertEqual(status, 200)
        self.assertEqual(solved["topological_zeta"]["expression"], "3/((s + 1)*(4*s + 3))")
        self.assertEqual(solved["monodromy_zeta"]["expression"], "1")
        self.assertEqual(len(solved["atlas"]), 3)

    @unittest.skipUnless(native_library._MODE != "sympy", "the native fan verifier is not in use")
    def test_auxiliary_fan_euler_on_chart_orbits_and_fallback(self):
        from engine import monomial_atlas
        from engine.newton_trials import _try
        from engine.strata import resolution_strata
        from engine.topological import topological_zeta

        names = list("xyzt")
        support = {(2, 0, 0, 0), (0, 1, 0, 0)}
        for expression, fan_calls, general_calls, expected in (
            ("x^2-y^2*(z+t)", 0, 0, "(s + 2)/(2*(s + 1)^2)"),
            ("(x-y)*(x-2*y)-y^2*(z+t)", 1, 0, "(s + 1)^(-2)"),
            ("(x-y)^2-y^2*(z+t)", 1, 2, "(s + 2)/(2*(s + 1)^2)"),
        ):
            with self.subTest(expression=expression):
                p = parse(expression, names)
                record = _try(p, names, support)
                record["polynomial"] = expression
                with patch.object(monomial_atlas, "_fan_orbit_strata", wraps=monomial_atlas._fan_orbit_strata) as fan:
                    with patch.object(monomial_atlas, "strata_rows", wraps=monomial_atlas.strata_rows) as rows:
                        strata = resolution_strata(record, p)
                self.assertEqual(fan.call_count, fan_calls)
                self.assertEqual(sum(not call.kwargs.get("monomial_only") for call in rows.call_args_list), general_calls)
                self.assertEqual(topological_zeta(record, strata=strata)["expression"], expected)
