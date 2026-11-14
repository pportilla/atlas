import unittest
from unittest.mock import patch

from engine.polynomial import LimitError, parse
from engine.resolve import resolve


class ExhaustiveFallback(unittest.TestCase):
    def test_newton_budget_still_reaches_constructive_blowups(self):
        request = dict(polynomial="(x*y-z*t+u^2)^2 + x^5+2*y^5+3*z^5+4*t^5+5*u^5",
                       variables=list("xyztu"), mode="embedded", search="exhaustive")
        with patch("engine.newton_backend.resolve_newton", side_effect=LimitError("fan limit")):
            result = resolve(request)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual([step["rule"] for step in result["steps"]], ["repeated tangent"] * 3)
        self.assertEqual(len(result["atlas"]), len(result["charts"]))
        self.assertEqual(result["certificate"]["atlas_verification"], "Embedded normal crossings on the owned origin fibre")

    def test_certificate_resolution_when_geometric_rule_does_not_apply(self):
        from engine.exhaustive_resolution import search
        names = list("wxyz")
        # The singular locus is a plane. Blowing up (x,y) gives a certified
        # resolution, including away from the centre of the origin blowup.
        result = search(parse("x^2-y^2*z", names), names, "embedded")
        self.assertEqual(result["status"], "resolved")
        self.assertTrue(result["charts"])

    def test_quick_keeps_a_bounded_search(self):
        request = dict(polynomial="(x*y-z*t+u^2)^2 + x^5+2*y^5+3*z^5+4*t^5+5*u^5",
                       variables=list("xyztu"), mode="embedded", search="quick", monodromy=True)
        with patch("engine.newton_backend.resolve_newton", side_effect=LimitError("fan limit")), \
             patch("engine.origin_backend.resolve_origin", return_value=None) as origin, \
             patch("engine.general_resolution.search", return_value=None) as quick, \
             patch("engine.exhaustive_resolution.search") as exhaustive:
            result = resolve(request)
        self.assertFalse(quick.call_args.kwargs["exhaustive"])
        self.assertTrue(origin.call_args.kwargs["tangent_only"])
        self.assertGreater(origin.call_args.kwargs["seconds"], 0)
        exhaustive.assert_not_called()
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["monodromy_zeta"]["status"], "unavailable")
