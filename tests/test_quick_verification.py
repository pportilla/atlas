"""A quick search bounds the verification of a proposal by the time it has left.

Verifying a Newton proposal replays its blowups. Without a deadline the
replay's complete rule could run until the quick worker's limit, so the
request ended inconclusive although the proposal was sound.
"""
import json
import subprocess
import unittest
from time import monotonic
from unittest.mock import patch

from engine import native_polynomial
from engine.local_blowups import GiveUp, acampo_over, time_limit, time_limit_passed
from tests.cli_support import ROOT


T0 = "x^9+x^4*y*z+1/8*x^3*y^3-3/2*x^3*z^2-1/32*x*y^4+1/2*x*y*z^2+y^2*z^2-z^3"
MONODROMY_ZETA = "(1-T^20)*(1-T^30)/((1-T^3)*(1-T^9)^4*(1-T^10)*(1-T^60))"


class TimeLimit(unittest.TestCase):
    def setUp(self):
        self.symbols = tuple(map(native_polynomial.Symbol, "xyz"))
        cusp = native_polynomial.from_terms(self.symbols, {(2, 0, 0): 1, (0, 3, 0): 1, (0, 0, 5): 1})
        self.components = [("strict", 1, 1, (cusp,))]

    def test_an_expired_limit_stops_the_blowups(self):
        with time_limit(monotonic() - 1):
            self.assertTrue(time_limit_passed())
            with self.assertRaisesRegex(GiveUp, "time"):
                acampo_over(self.symbols, self.components, self.symbols, dimension=3, count_strata=False)
        self.assertFalse(time_limit_passed())
        acampo_over(self.symbols, self.components, self.symbols, dimension=3, count_strata=False)

    def test_an_inner_limit_cannot_extend_an_outer_one(self):
        with time_limit(monotonic() - 1):
            with time_limit(monotonic() + 3600):
                self.assertTrue(time_limit_passed())

    def test_a_timed_out_cover_replay_is_not_reported_as_a_failure(self):
        from engine import repair_cover

        class Parent:
            variables = self.symbols
            components = self.components

        def stop(*args, **kwargs):
            raise GiveUp("The time allowed for the blowups is over.")

        with patch.object(repair_cover, "acampo_over", side_effect=stop):
            # Answers above this replay are cached, so a timeout must not become None.
            self.assertIsNone(repair_cover._local_pivots(0, Parent, self.symbols, ()))
            with time_limit(monotonic() - 1), self.assertRaises(GiveUp):
                repair_cover._local_pivots(0, Parent, self.symbols, ())


class QuickResolution(unittest.TestCase):
    def test_a_quick_search_verifies_the_newton_proposal_in_time(self):
        result = subprocess.run([ROOT / "atlas", "resolve", T0, "--vars", "x,y,z", "--quick", "--monodromy",
                                 "--json"], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])
        record = json.loads(result.stdout)
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["dual_complex"]["status"], "complete")
        self.assertEqual(record["monodromy_zeta"]["expression"], MONODROMY_ZETA)


if __name__ == "__main__":
    unittest.main()
