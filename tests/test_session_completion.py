import unittest
from unittest.mock import Mock, patch

from engine.session import new_session
from engine.session_completion import candidates
from engine.session_values import Chart, Resolution


class SessionCompletionTests(unittest.TestCase):
    def setUp(self):
        self.session = new_session(None)

    def test_top_level_completion_uses_live_bindings(self):
        commands = (":save", "dual-complex", "help", "recompute-dual-complex", "resolve")
        self.assertEqual(candidates("reso", self.session, commands), ["resolve("])
        self.assertEqual(candidates("Q", self.session, commands), ["QQ"])
        self.assertEqual(candidates(":", self.session, commands), [":save"])
        self.assertEqual(candidates("dual", self.session, commands), ["dual-complex", "dual_complex("])
        self.assertEqual(candidates("recomp", self.session, commands),
                         ["recompute-dual-complex", "recompute_dual_complex("])
        list(self.session.run("roots = 42"))
        self.assertEqual(candidates("roo", self.session), ["roots"])
        self.assertEqual(candidates("roots.", self.session), [])
        list(self.session.run('roots = coordinates("x", fresh=True)'))
        self.assertEqual(candidates("roots.", self.session),
                         ["roots.coeffs", "roots.multiplicity(", "roots.parent", "roots.variables"])

    def test_resolution_lists_methods_and_only_its_recorded_fields(self):
        self.session.variables["R"] = Resolution({
            "variables": ["x", "y"], "polynomial": "y^2-x^3", "charts": [{"id": "C"}],
            "dual_complex": {"vertices": [{"id": "E1"}]},
            "mode": "embedded", "status": "resolved", "_private": True,
        })
        self.assertEqual(candidates("R.", self.session), [
            "R.branches", "R.chart(", "R.charts", "R.coordinates", "R.divisor(",
            "R.divisors", "R.dual_complex", "R.mode", "R.polynomial", "R.record", "R.status", "R.variables",
        ])
        self.assertEqual(candidates("R.dual_complex.v", self.session), ["R.dual_complex.vertices"])
        list(self.session.run('C = R.chart("C")\nE = R.divisor("E1")'))
        self.assertEqual(candidates("E.e", self.session), ["E.equation_in("])
        getter = Mock(side_effect=AssertionError("Tab evaluated the strict transform"))
        with patch.object(Chart, "strict_transform", property(getter)):
            self.assertIn("C.strict_transform", candidates("C.", self.session))
            self.assertEqual(candidates("C.strict_transform.", self.session), [])
            getter.assert_not_called()

    def test_structured_algebra_results_share_their_declared_fields(self):
        list(self.session.run('x = coordinates("x")\nI = ideal(x)\ng = gcd(x, x)'))
        self.assertEqual(candidates("I.g", self.session), ["I.generators"])
        self.assertEqual(candidates("g.m", self.session), ["g.multiplicity("])
        self.assertEqual(candidates("g.ord", self.session), ["g.order"])
        self.assertEqual(candidates("g.pol", self.session), ["g.polynomial"])
        list(self.session.run("n = g.multiplicity(x)"))
        self.assertEqual(self.session.variables["n"], 1)

    def test_completion_filters_invalid_keys_and_never_runs_an_expression(self):
        self.session.variables["D"] = {"vertices": [], "nested": {"valid": 3}, "bad-key": 1,
                                       "items": 2, "if": 0, 0: True, "_private": 3}
        self.assertEqual(candidates("D.", self.session), ["D.items", "D.nested", "D.vertices"])
        self.assertEqual(candidates("D.nested.", self.session), ["D.nested.valid"])
        for token in ("resolve().", "D['nested'].", "D..", "D._private.", "D.vertices.", "absent."):
            with self.subTest(token=token):
                self.assertEqual(candidates(token, self.session), [])
