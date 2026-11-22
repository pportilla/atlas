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

        if not hasattr(native_library._library(), "atlas_transverse_pencil_abi"):
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
        if not hasattr(native_library._library(), "atlas_transverse_pencil_abi"):
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

    def test_intrinsic_chain_and_loop_sections_replay_on_their_own_charts(self):
        from engine.resolution_complex import dual_complex
        from engine.resolve import resolve
        from engine.session_values import Resolution

        names = list("xyz")
        for source, count in (("x^3*y+y^3+z^4", 12), ("x^3*y+y^2*z+z^4", 13),
                              ("x^2*y+x*y^4+z^4", 10), ("x^2*y+y^3*z+z^3*x", 12),
                              ("x^2*y+y^3*z+z^4*x", 10)):
            with self.subTest(source=source):
                record = resolve(dict(polynomial=source, variables=names, mode="intrinsic",
                                      search="quick", compute="cpu"))
                self.assertEqual(record["status"], "resolved")
                boundary = dual_complex(record)
                self.assertEqual(boundary, record["dual_complex"])
                self.assertEqual((len(boundary["vertices"]), len(boundary["cells"])), (count, count-1))
                resolution = Resolution(record)
                divisors = {d.id for chart in resolution.charts for d in chart.divisors}
                self.assertEqual(divisors, {v["id"] for v in boundary["vertices"]})
                for chart in resolution.charts:
                    for divisor in chart.divisors:
                        self.assertTrue(divisor.ideal_in(chart))

        # This chart has two disjoint curves on one exceptional plane.
        # Setting u3=0 in u1^2+u1+u3 gives the factors u1 and u1+1.
        record = resolve(dict(polynomial="x^2*y+x*y^4+z^4", variables=names, mode="intrinsic",
                              search="quick", compute="cpu"))
        chart = next(c for c in record["charts"] if c["strict_transform"] == "u1^2+u1+u3")
        curves = [d for d in chart["exceptional"] if d["ambient_divisor"] == "E4"]
        self.assertEqual({d["id"] for d in curves}, {"E4.1", "E4.2"})
        self.assertEqual({tuple(d["ideal"]) for d in curves}, {("u3", "u1"), ("u3", "u1+1")})

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

    def test_intrinsic_fan_replay_checks_a_singular_exceptional_section(self):
        from engine.newton_trials import _try
        from engine.resolution_complex import dual_complex

        # The ordinary blowup gives a smooth strict surface. In its z chart
        # the exceptional plane cuts out two lines that meet, so a rule for
        # smooth, disjoint components of a single section cannot apply.
        names = list("xyz")
        support = [(1,0,0), (0,1,0), (0,0,1)]
        source = "x*y+z^3"
        verified = []
        record = _try(parse(source, names), names, support, mode="intrinsic", verified=verified)
        record.update(polynomial=source, atlas=verified[0][1][0])
        record["certificate"]["atlas_verification"] = "Embedded normal crossings on the owned origin fibre"
        replay = dual_complex(record)
        self.assertEqual(replay, record["dual_complex"])
        self.assertEqual([cell["vertices"] for cell in replay["cells"]], [["E1.1", "E1.2"]])
        # For z^5 even the strict surface remains singular on this fan.
        self.assertIsNone(_try(parse("x*y+z^5", names), names, support, mode="intrinsic"))

    def test_intrinsic_fan_rejects_contradictory_displayed_chart_claims(self):
        import json
        from engine.resolution_complex import dual_complex
        from engine.resolve import resolve
        from engine.strata import InvalidChart

        record = resolve(dict(polynomial="x^2*y+x*y^4+z^4", variables=list("xyz"), mode="intrinsic",
                              search="quick", compute="cpu"))
        saved = json.loads(json.dumps(record))
        for component in saved["atlas"][0]["components"]:
            generator, = component["ideal"]
            component["ideal"] = [f"2*({generator})", f"3*({generator})"]
        self.assertEqual(dual_complex(saved), record["dual_complex"])
        saved["fan"]["rays"][0][0] = bool(saved["fan"]["rays"][0][0])
        with self.assertRaises(InvalidChart):
            dual_complex(saved)
        minimal = json.loads(json.dumps(record))
        del minimal["engine"], minimal["certificate"]
        self.assertEqual(dual_complex(minimal), record["dual_complex"])
        minimal["charts"][0]["exceptional"][0]["ideal"] = ["1"]
        with self.assertRaises(InvalidChart):
            dual_complex(minimal)
        for owned in (True, False):
            for field in ("map", "strict_transform", "exceptional"):
                with self.subTest(owned=owned, field=field):
                    saved = json.loads(json.dumps(record))
                    if not owned:
                        del saved["atlas"]
                    chart = saved["charts"][0]
                    if field == "map":
                        chart["map"]["x"] = "1"
                    elif field == "strict_transform":
                        chart[field] = "1"
                    else:
                        chart[field][0]["ideal"] = ["1"]
                    with self.assertRaises(InvalidChart):
                        dual_complex(saved)
