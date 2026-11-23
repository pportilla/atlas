"""The maximal order rule races in its own lane of the exhaustive search.

At an isolated singular point the canonical lane tries the complete rule
first, without a time limit. These tests check that the order lane resolves
a germ on which the complete rule stalls, and that the canonical lane keeps
its record whenever it answers in time.
"""
import json
import subprocess
import time
import unittest
from subprocess import Popen
from unittest.mock import Mock, patch

from engine import exhaustive_resolution as exhaustive
from tests.cli_support import ROOT


# A mu-constant family x^9 + x^4*y*z + (T+1/8)*x^3*y^3 + ... with Milnor number 58.
# At T = 0 the complete rule stalls in its first centre choice.
T0 = "x^9+x^4*y*z+1/8*x^3*y^3-3/2*x^3*z^2-1/32*x*y^4+1/2*x*y*z^2+y^2*z^2-z^3"
T1 = "x^9+x^4*y*z+9/8*x^3*y^3+5/2*x^3*z^2-25/32*x*y^4-23/2*x*y*z^2-2*y^3*z+y^2*z^2-9*z^3"
MONODROMY_ZETA = "(1-T^20)*(1-T^30)/((1-T^3)*(1-T^9)^4*(1-T^10)*(1-T^60))"
TOPOLOGICAL_ZETA = "(146*s^2 + 219*s + 82)/((s + 1)*(3*s + 2)*(60*s + 41))"

# A lane that answers after a delay: with its own record, with none, or not at all.
LANE = """
import json, sys, time
method, plan = sys.argv[1], json.loads(sys.argv[2])
json.load(sys.stdin)
delay, outcome = plan.get(method, [0, "none"])
time.sleep(delay)
if outcome == "exit":
    sys.exit(3)
record = {"engine": method} if outcome == "record" else None
print(json.dumps({"record": record, "certified": [[], {"dual_complex": {"status": "complete"}}],
                  "replay": {"winner": method}}))
"""


class OrderLaneSchedule(unittest.TestCase):
    def race(self, plan, *, canonical_first=True, patience=(3, 0.6)):
        started = {}

        def launch(argv, **kwargs):
            method = argv[-2]
            started[method] = time.monotonic()
            return Popen([argv[0], "-c", LANE, method, json.dumps(plan)], **kwargs)

        scope = Mock()
        scope.settings.return_value = {"compute": "cpu"}
        begin = time.monotonic()
        with patch.object(exhaustive, "_canonical_head_start", return_value=canonical_first), \
             patch.object(exhaustive, "_forkable", return_value=False), \
             patch.object(exhaustive.fork_server, "available", return_value=False), \
             patch.object(exhaustive, "ORIGIN_GRACE_SECONDS", .05), \
             patch.object(exhaustive, "NEWTON_GRACE_SECONDS", .05), \
             patch.object(exhaustive, "ORDER_PATIENCE", patience[0]), \
             patch.object(exhaustive, "ORDER_PATIENCE_SECONDS", patience[1]), \
             patch.object(exhaustive.subprocess, "Popen", side_effect=launch), \
             patch.object(exhaustive, "current_replay_scope", return_value=scope):
            record = exhaustive.search({(2, 0, 0): 1}, list("xyz"), "embedded", compute="cpu")
        winner = None if record is None else record["engine"]
        if winner is not None:
            scope.merge.assert_called_once_with({"winner": winner})
        return winner, time.monotonic() - begin, started

    def test_the_canonical_lane_keeps_an_answer_that_arrives_in_time(self):
        winner, _, started = self.race({"origin": [.3, "record"], "order": [0, "record"],
                                        "certificates": [30, "record"]})
        self.assertEqual(winner, "origin")
        self.assertEqual(set(started), {"origin", "order", "certificates"})

    def test_the_order_record_is_taken_once_the_canonical_lane_has_had_its_time(self):
        winner, elapsed, _ = self.race({"origin": [30, "record"], "order": [0, "record"],
                                        "certificates": [30, "record"]})
        self.assertEqual(winner, "order")
        self.assertGreaterEqual(elapsed, .6)
        self.assertLess(elapsed, 10)

    def test_the_wait_grows_with_the_time_the_order_lane_took(self):
        winner, elapsed, _ = self.race({"origin": [30, "record"], "order": [.4, "record"],
                                        "certificates": [30, "record"]}, patience=(3, .05))
        self.assertEqual(winner, "order")
        self.assertGreaterEqual(elapsed, 1.2)
        self.assertLess(elapsed, 10)

    def test_a_canonical_lane_without_a_record_releases_the_order_record(self):
        for outcome in ("none", "exit"):
            with self.subTest(outcome=outcome):
                winner, elapsed, _ = self.race({"origin": [.2, outcome], "order": [0, "record"],
                                                "certificates": [30, "record"]}, patience=(3, 20))
                self.assertEqual(winner, "order")
                self.assertLess(elapsed, 10)

    def test_other_lanes_answer_as_before_while_the_order_record_waits(self):
        winner, elapsed, _ = self.race({"origin": [30, "record"], "order": [0, "record"],
                                        "certificates": [.2, "record"]}, patience=(3, 20))
        self.assertEqual(winner, "certificates")
        self.assertLess(elapsed, 10)

    def test_the_order_lane_also_races_after_a_newton_head_start(self):
        winner, _, started = self.race({"newton": [0, "none"], "origin": [30, "record"],
                                        "order": [0, "record"], "certificates": [30, "record"]},
                                       canonical_first=False)
        self.assertEqual(winner, "order")
        self.assertEqual(set(started), {"newton", "origin", "order", "certificates"})

    def test_without_any_record_the_search_ends_when_every_lane_has_answered(self):
        winner, elapsed, _ = self.race({"origin": [.1, "none"], "order": [0, "none"],
                                        "certificates": [.1, "none"]}, patience=(3, 20))
        self.assertIsNone(winner)
        self.assertLess(elapsed, 10)


class OrderLaneRule(unittest.TestCase):
    def test_the_order_lane_leaves_non_isolated_points_to_the_canonical_lane(self):
        from engine.origin_backend import _resolve
        from engine.polynomial import parse_expression
        names = ["x", "y", "z"]
        self.assertIsNone(_resolve(parse_expression("x^2+y^3", names), names, "embedded", None,
                                   defer_outputs=True, order_only=True))


class OrderLaneResolution(unittest.TestCase):
    def resolve(self, polynomial, *options):
        result = subprocess.run([ROOT / "atlas", "resolve", polynomial, "--vars", "x,y,z", "--json", *options],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])
        return json.loads(result.stdout)

    def test_the_order_lane_resolves_a_germ_where_the_complete_rule_stalls(self):
        record = self.resolve(T0, "--monodromy")
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["dual_complex"]["status"], "complete")
        self.assertEqual(len(record["steps"]), 13)
        self.assertEqual(record["monodromy_zeta"]["expression"], MONODROMY_ZETA)
        self.assertEqual(record["topological_zeta"]["expression"], TOPOLOGICAL_ZETA)

    def test_intrinsic_mode_continues_with_the_order_rule(self):
        record = self.resolve(T0, "--mode", "intrinsic")
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["dual_complex"]["status"], "complete")

    def test_the_complete_rule_keeps_its_record_on_a_nearby_member(self):
        record = self.resolve(T1, "--monodromy")
        self.assertEqual(len(record["steps"]), 17)
        self.assertEqual(record["monodromy_zeta"]["expression"], MONODROMY_ZETA)
        self.assertEqual(record["topological_zeta"]["expression"], TOPOLOGICAL_ZETA)


if __name__ == "__main__":
    unittest.main()
