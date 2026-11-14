import unittest
from unittest.mock import patch

from engine import native_library

from engine.app import calculate_resolution
from engine.monodromy import monodromy_zeta
from engine.newton_backend import _boundary_centres
from engine.newton_fan import build_fan, trial_supports
from engine.polynomial import parse


class NewtonSingularCentres(unittest.TestCase):
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
