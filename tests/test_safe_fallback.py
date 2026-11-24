"""A quick search does not give up: the smooth-centre rules are its safe fallback.

After the fast methods the rules take turns without a deadline of their own,
and the projective search goes on after them. Only an attempt inside another
computation (`bounded`) keeps budgets and may return inconclusive.
"""
import unittest
from unittest.mock import patch

from engine.polynomial import parse
from engine.resolve import (QUICK_CANONICAL_SECONDS, QUICK_NEARBY_SECONDS, quick_canonical_seconds,
                            quick_nearby_seconds, resolve)

FIVE = ["x", "y", "z", "t", "u"]


class Reached(Exception):
    """The patched projective search was called."""


FAST_METHODS = ("engine.resolve.resolve_native", "engine.chain_backend.resolve_chain",
                "engine.toric_backend.resolve_toric", "engine.quadratic_fan.resolve_translated",
                "engine.plane_arrangement.resolve_arrangement", "engine.reduced_centre.smooth_reduced_graph",
                "engine.newton_backend.resolve_newton")


class QuickSearchHasNoEndOfItsOwn(unittest.TestCase):
    """After the fast methods a quick request runs the safe method, then the projective search."""

    CASES = ((FIVE, "x^2*y^2+z^3*t^2+u^5*x+y^4*t^3"), (["x", "y", "z"], "x^2*y^2+z^3*y^2+x^5*z+y^4*z^3"))

    def declined(self, request, certificates):
        """Resolve with every fast method and the smooth-centre rules declining."""
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name in FAST_METHODS:
                stack.enter_context(patch(name, return_value=None))
            origin = stack.enter_context(patch("engine.origin_backend.resolve_origin", return_value=None))
            search = stack.enter_context(patch("engine.general_resolution.search", **certificates))
            try:
                record = resolve(request)
            except Reached:
                record = None
        return record, origin, search

    def test_quick_search_gives_the_safe_method_no_deadline(self):
        for names, equation in self.CASES:
            with self.subTest(variables=len(names)):
                request = {"polynomial": equation, "variables": names, "mode": "embedded", "search": "quick",
                           "compute": "cpu"}
                record, origin, search = self.declined(request, dict(side_effect=Reached))
                # No inconclusive answer: the projective search, which ends only with a certificate, goes on.
                self.assertIsNone(record)
                self.assertTrue(search.call_args.kwargs["exhaustive"])
                safe = [call for call in origin.call_args_list if call.kwargs.get("safe")]
                self.assertEqual(len(safe), 1)
                self.assertIsNone(safe[0].kwargs["seconds"])
                self.assertFalse(any(call.kwargs.get("safe") for call in origin.call_args_list
                                     if call.kwargs.get("tangent_only")))

    def test_a_bounded_attempt_keeps_its_budgets_and_may_be_inconclusive(self):
        from engine.resolve_cli import MAX_SECONDS
        for names, equation in self.CASES:
            with self.subTest(variables=len(names)):
                request = {"polynomial": equation, "variables": names, "mode": "embedded", "search": "quick",
                           "compute": "cpu", "bounded": True}
                record, origin, search = self.declined(request, dict(return_value=None))
                self.assertEqual(record["status"], "inconclusive")
                self.assertFalse(search.call_args.kwargs["exhaustive"])
                rules = [call for call in origin.call_args_list if not call.kwargs.get("tangent_only")]
                self.assertEqual(len(rules), 1)
                self.assertFalse(rules[0].kwargs["safe"])
                self.assertTrue(0 < rules[0].kwargs["seconds"] <= MAX_SECONDS)

    def test_a_bounded_attempt_gets_the_rest_of_the_workers_limit(self):
        from engine import resolve as module
        from engine.resolve_cli import MAX_SECONDS
        token = module._REQUEST_START.set(1000.0)
        try:
            with patch("engine.resolve.perf_counter", return_value=1003.0):
                self.assertEqual(quick_canonical_seconds(), MAX_SECONDS - module.QUICK_FINAL_RESERVE - 3)
                self.assertEqual(quick_nearby_seconds(), QUICK_NEARBY_SECONDS)
            # Late in the request the rules keep their former share.
            with patch("engine.resolve.perf_counter", return_value=1048.0):
                self.assertEqual(quick_canonical_seconds(), QUICK_CANONICAL_SECONDS)
                self.assertEqual(quick_nearby_seconds(), MAX_SECONDS - module.QUICK_FINAL_RESERVE - 48)
            with patch("engine.resolve.perf_counter", return_value=1060.0):
                self.assertEqual(quick_nearby_seconds(), 0)
        finally:
            module._REQUEST_START.reset(token)

    def test_the_reason_of_the_smooth_centre_rules_is_kept(self):
        from engine.origin_backend import resolve_origin
        from subprocess import CompletedProcess
        polynomial = parse("x^2*y^2+z^3*t^2+u^5*x+y^4*t^3", FIVE)
        for done, expected in ((CompletedProcess([], 0, '{"gave_up": "The order rule: test reason."}', ""),
                                "The order rule: test reason."),
                               (CompletedProcess([], 1, "", "Traceback\nValueError: broken"),
                                "The process of the smooth-centre rules stopped: ValueError: broken")):
            why = {}
            with patch("engine.fork_server.run", return_value=done):
                self.assertIsNone(resolve_origin(polynomial, FIVE, "embedded", None, safe=True, diagnostics=why))
            self.assertEqual(why["reason"], expected)


class RulesContinueWithLeftoverTime(unittest.TestCase):
    def run_rules(self, behaviour):
        """Run three rules whose blowups are replaced by `behaviour(name, visit, limit)`."""
        from time import monotonic
        from engine import local_blowups as lb
        from engine import native_polynomial as np
        x, y, z = symbols = tuple(map(np.Symbol, "xyz"))
        calls = []

        def fake(rule, variables, components, fibre, budget, limit, *rest, state=None, **options):
            visit = sum(1 for name, *_ in calls if name == rule.name)
            calls.append((rule.name, limit, state))
            return behaviour(rule.name, visit, limit, state)

        deadline = monotonic() + 0.6
        with patch.object(lb, "_run", side_effect=fake):
            try:
                answer = lb.acampo_over(symbols, [("strict", 1, 1, (x*y - z**2,))], symbols, dimension=3,
                                        rules=("order", "complete", "direct"), deadline=deadline)
            except lb.GiveUp as error:
                answer = error
        return answer, calls, deadline

    def test_a_rule_stopped_at_its_share_continues_when_the_others_give_up(self):
        from time import monotonic, sleep
        from engine.local_blowups import GiveUp

        def behaviour(name, visit, limit, state):
            if name == "order" and visit == 0:
                state["progress"] = monotonic()
                sleep(max(0.0, limit - monotonic()) + 0.02)
                raise GiveUp("The time allowed for the blowups is over.")
            if name == "order":
                return "continued"
            raise GiveUp(f"The {name} rule has nothing to do.")

        answer, calls, deadline = self.run_rules(behaviour)
        self.assertEqual(answer, "continued")
        self.assertEqual([name for name, *_ in calls], ["order", "complete", "direct", "order"])
        first, last = calls[0], calls[-1]
        self.assertIs(first[2], last[2])       # the same charts and centres
        self.assertLess(first[1], deadline)    # a share the first time
        self.assertEqual(last[1], deadline)    # everything that is left afterwards

    def test_a_rule_that_gave_up_for_its_own_reason_is_not_run_again(self):
        from engine.local_blowups import GiveUp

        def behaviour(name, visit, limit, state):
            raise GiveUp(f"The {name} rule has nothing to do.")

        answer, calls, _ = self.run_rules(behaviour)
        self.assertIsInstance(answer, GiveUp)
        self.assertEqual([name for name, *_ in calls], ["order", "complete", "direct"])


class DecompositionsStopAtTheDeadline(unittest.TestCase):
    """The radical and the prime parts of a locus read the deadline of the rule."""

    REQUEST = ["ideal_radical", "x y", "grevlex", "1 2 0", "1 0 2"]

    def test_a_kernel_request_under_a_deadline_answers_or_expires(self):
        from engine.native_library import Expired, affine_request
        self.assertEqual(affine_request(self.REQUEST, 60.0), affine_request(self.REQUEST))
        with self.assertRaises(Expired):
            affine_request(self.REQUEST, 0.0)

    def test_an_expired_deadline_ends_the_turn_of_the_rule(self):
        from time import monotonic
        from engine import native_polynomial as np
        from engine.local_blowups import Chart, GiveUp, _prime_parts, _radical
        x, y, z = symbols = tuple(map(np.Symbol, "xyz"))
        chart = Chart(symbols, (), (), symbols, (), (), ())
        self.assertEqual(set(_radical(chart, (x**2, y**2), monotonic() + 60)), {x, y})
        self.assertEqual(len(_prime_parts(chart, (x*y, z), None, monotonic() + 60)), 2)
        for decompose in (lambda: _radical(chart, (x**2, y**2), monotonic() - 1),
                          lambda: _prime_parts(chart, (x*y, z), None, monotonic() - 1)):
            with self.assertRaisesRegex(GiveUp, "The time allowed for the blowups is over."):
                decompose()


class RulesTakeTurns(unittest.TestCase):
    """With `rounds` no rule is stopped for good by the time it has used."""

    def run_rules(self, behaviour, deadline=None):
        from engine import local_blowups as lb
        from engine import native_polynomial as np
        x, y, z = symbols = tuple(map(np.Symbol, "xyz"))
        calls = []

        def fake(rule, variables, components, fibre, budget, limit, *rest, state=None, **options):
            visit = sum(1 for name, *_ in calls if name == rule.name)
            calls.append((rule.name, limit, state, budget))
            return behaviour(rule.name, visit, limit, state)

        with patch.object(lb, "_run", side_effect=fake):
            try:
                answer = lb.acampo_over(symbols, [("strict", 1, 1, (x*y - z**2,))], symbols, dimension=3,
                                        rules=("order", "complete", "direct"), deadline=deadline, budget=None,
                                        rounds=0.05)
            except lb.GiveUp as error:
                answer = error
        return answer, calls

    @staticmethod
    def out_of_turn(limit):
        from time import monotonic, sleep
        from engine.local_blowups import GiveUp
        sleep(max(0.0, limit - monotonic()) + 0.01)
        raise GiveUp("The time allowed for the blowups is over.")

    def test_rules_continue_in_longer_turns_until_one_finishes(self):
        from engine.local_blowups import GiveUp

        def behaviour(name, visit, limit, state):
            if name == "direct":
                raise GiveUp("Normal crossings fail along a surface.")
            if name == "complete" and visit == 2:
                return "resolved"
            self.out_of_turn(limit)

        answer, calls = self.run_rules(behaviour)
        self.assertEqual(answer, "resolved")
        # The rule that gave up for its own reason leaves; the others come back.
        self.assertEqual([name for name, *_ in calls],
                         ["order", "complete", "direct", "order", "complete", "order", "complete"])
        states = {}
        for name, _, state, budget in calls:
            self.assertIs(states.setdefault(name, state), state)  # each continues from its own charts
            self.assertIsNone(budget)
        turns = [limit for name, limit, *_ in calls if name == "order"]
        self.assertTrue(all(limit is not None for limit in turns))

    def test_the_last_rule_left_runs_without_turns(self):
        from engine.local_blowups import GiveUp

        def behaviour(name, visit, limit, state):
            if name != "complete":
                raise GiveUp(f"The {name} rule has nothing to do.")
            return ("complete", limit)

        answer, calls = self.run_rules(behaviour)
        # complete had a turn while order was still there? order left first, direct was still waiting.
        self.assertEqual([name for name, *_ in calls][:2], ["order", "complete"])
        self.assertEqual(answer[0], "complete")

        def stubborn(name, visit, limit, state):
            if name != "complete":
                raise GiveUp(f"The {name} rule has nothing to do.")
            if visit == 0:
                self.out_of_turn(limit)
            return ("complete", limit)

        answer, calls = self.run_rules(stubborn)
        self.assertEqual(answer, ("complete", None))  # alone, and no deadline: nothing stops it

    def test_every_rule_giving_up_reports_each_reason(self):
        from engine.local_blowups import GiveUp

        def behaviour(name, visit, limit, state):
            raise GiveUp(f"The {name} rule has nothing to do.")

        answer, calls = self.run_rules(behaviour)
        self.assertIsInstance(answer, GiveUp)
        self.assertEqual(len(calls), 3)
        for name in ("order", "complete", "direct"):
            self.assertIn(f"The {name} rule has nothing to do.", str(answer))

    def test_only_the_callers_deadline_ends_the_turns(self):
        from time import monotonic
        from engine.local_blowups import GiveUp
        deadline = monotonic() + 0.4

        def behaviour(name, visit, limit, state):
            self.assertLessEqual(limit, deadline)
            self.out_of_turn(limit)

        answer, calls = self.run_rules(behaviour, deadline)
        self.assertIsInstance(answer, GiveUp)
        self.assertEqual(str(answer), "The time allowed for the blowups is over.")
        self.assertGreater(len(calls), 3)  # every rule came back at least once

if __name__ == "__main__":
    unittest.main()


class InternalErrorsHandOver(unittest.TestCase):
    """A fast method that stops on an internal error is replaced by the safe method in a quick search."""

    def test_a_defect_of_a_fast_method_is_recorded_and_the_safe_method_answers(self):
        # The A2 surface is resolved by the smooth-centre rules after the fast
        # search raises. The answer is certified and carries the defect.
        request = dict(polynomial="x^2+y^2+z^3", variables=["x", "y", "z"], mode="embedded", search="quick",
                       compute="cpu", zetas=["monodromy", "topological"])
        with patch("engine.resolve._resolve_polynomial", side_effect=RuntimeError("broken engine")):
            record = resolve(request)
        self.assertEqual((record["status"], record["engine"]), ("resolved", "canonical-blowups"))
        self.assertEqual(record["dual_complex"]["status"], "complete")
        self.assertEqual(record["topological_zeta"]["status"], "computed")
        self.assertEqual(record["defects"], ["fast search: RuntimeError: broken engine"])
        direct = resolve(request)
        self.assertNotIn("defects", direct)
        self.assertEqual(record["topological_zeta"]["expression"], direct["topological_zeta"]["expression"])

    def test_the_safe_method_gets_no_deadline_after_a_defect(self):
        calls = []

        def canonical(*args, **kwargs):
            calls.append((kwargs.get("safe"), kwargs.get("seconds")))
            raise Reached

        request = dict(polynomial="x^2*y^2+z^3*t^2+u^5*x+y^4*t^3", variables=FIVE, mode="embedded", search="quick",
                       compute="cpu")
        for error in (KeyError("ray"), ValueError("bad row"), OSError("pipe")):
            with self.subTest(error=type(error).__name__), \
                    patch("engine.resolve._resolve_polynomial", side_effect=error), \
                    patch("engine.resolve.canonical_record", side_effect=canonical):
                with self.assertRaises(Reached):
                    resolve(request)
        self.assertEqual(calls, [(True, None)] * 3)

    def test_other_searches_report_an_internal_error_as_before(self):
        base = dict(polynomial="x^2+y^2+z^3", variables=["x", "y", "z"], mode="embedded", compute="cpu")
        for extra in (dict(search="quick", bounded=True), dict(search="exhaustive")):
            with self.subTest(**extra), \
                    patch("engine.resolve._resolve_polynomial", side_effect=RuntimeError("broken engine")):
                with self.assertRaises(RuntimeError):
                    resolve(dict(base, **extra))

    def test_a_defect_of_the_smooth_centre_rules_reaches_the_projective_search(self):
        request = dict(polynomial="x^2*y^2+z^3*t^2+u^5*x+y^4*t^3", variables=FIVE, mode="embedded", search="quick",
                       compute="cpu")
        with patch("engine.resolve._resolve_polynomial", side_effect=KeyError("ray")), \
                patch("engine.resolve.canonical_record", side_effect=TypeError("bad chart")), \
                patch("engine.general_resolution.search", side_effect=Reached) as search:
            with self.assertRaises(Reached):
                resolve(request)
        self.assertTrue(search.call_args.kwargs.get("exhaustive"))
