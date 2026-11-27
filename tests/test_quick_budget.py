"""A quick search is the same search under a time budget, and the end of the budget is a normal answer.

The slow germ is the product (x^3-y^4)*(z*t-u*v) in six variables: no method
resolves it in two minutes. The fast germs are resolved in well under a second.
"""
from contextlib import ExitStack
import json
import subprocess
import threading
from time import monotonic
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from engine import quick_budget
from engine import resolve as resolver
from engine.quick_budget import (QUICK_SECONDS, checked_seconds, expired, expired_line, read_progress,
                                 request_seconds, with_budget)
from engine.resolve import resolve
from engine.resolve_cli import MAX_SECONDS, calculate_resolution, run_job
from tests.cli_support import ROOT

SLOW = "(x^3-y^4)*(z*t-u*v)"
SIX = ["x", "y", "z", "t", "u", "v"]
FIVE = SIX[:5]
# Seconds allowed on top of a budget for starting the worker and writing the answer.
OVERHEAD = 5


def atlas(*arguments, input=None, timeout=120):
    return subprocess.run([str(ROOT / "atlas"), *arguments], capture_output=True, text=True, input=input,
                          timeout=timeout, cwd=ROOT)


def without_times(value):
    """A record without its timings, which no two runs share."""
    if isinstance(value, dict):
        return {key: without_times(item) for key, item in value.items() if not key.endswith("elapsed_ms")}
    if isinstance(value, list):
        return [without_times(item) for item in value]
    return value


class TheBudgetOfARequest(unittest.TestCase):
    def test_a_quick_search_has_twenty_seconds_unless_the_request_names_a_budget(self):
        self.assertEqual(QUICK_SECONDS, 20)
        self.assertEqual(request_seconds({"search": "quick"}), QUICK_SECONDS)
        self.assertEqual(request_seconds({"search": "quick", "seconds": 7.5}), 7.5)
        self.assertIsNone(request_seconds({"search": "exhaustive"}))
        self.assertIsNone(request_seconds({}))

    def test_seconds_imply_a_quick_search_and_contradict_an_exhaustive_one(self):
        self.assertEqual(request_seconds({"seconds": 3}), 3)
        self.assertEqual(with_budget({"polynomial": "x", "seconds": 3})["search"], "quick")
        self.assertNotIn("search", with_budget({"polynomial": "x"}))
        with self.assertRaisesRegex(ValueError, "no time budget"):
            request_seconds({"search": "exhaustive", "seconds": 3})

    def test_a_budget_is_a_positive_number_of_seconds(self):
        for good in (1, 0.5, 20, 3600):
            self.assertEqual(checked_seconds(good), good)
        for bad in (0, -1, True, "20", None, float("inf"), float("nan"), [20]):
            with self.subTest(seconds=bad), self.assertRaisesRegex(ValueError, "positive number of seconds"):
                checked_seconds(bad)

    def test_every_entry_point_refuses_a_budget_that_is_not_one(self):
        request = {"polynomial": "y^2-x^3", "variables": ["x", "y"], "mode": "embedded"}
        self.assertEqual(run_job({**request, "seconds": -2})[0], 400)
        self.assertEqual(run_job({**request, "search": "exhaustive", "seconds": 2})[0], 400)
        status, record = calculate_resolution("y^2-x^3", ["x", "y"], mode="embedded", search="exhaustive", seconds="2")
        self.assertEqual(status, 400)
        self.assertIn("positive number of seconds", record["error"])
        with self.assertRaisesRegex(resolver.InputError, "no time budget"):
            resolve({**request, "search": "exhaustive", "seconds": 2})


class AnExhaustiveSearchHasNoBudget(unittest.TestCase):
    """Without --quick nothing changes: no time limit, no shares, no progress file."""

    REQUEST = {"polynomial": "y^2-x^3", "variables": ["x", "y"], "mode": "embedded", "search": "exhaustive"}

    def test_the_worker_of_an_exhaustive_request_has_no_deadline(self):
        with patch("engine.worker_process.run_worker", return_value=(200, {"status": "resolved"})) as worker:
            self.assertEqual(run_job(self.REQUEST), (200, {"status": "resolved"}))
        (module, data, timeout), options = worker.call_args
        self.assertEqual((module, data, timeout), (["engine.worker"], self.REQUEST, None))
        self.assertIsNone(options.get("env"))
        jobs = []
        calculate_resolution("y^2-x^3", ["x", "y"], mode="embedded", search="exhaustive",
                             job=lambda request, timeout: jobs.append((request, timeout)) or (200, {}))
        (request, timeout), = jobs
        self.assertEqual((request["search"], timeout), ("exhaustive", None))
        self.assertNotIn("seconds", request)

    def test_its_limits_are_not_shares_and_its_record_has_no_budget(self):
        self.assertIsNone(resolver._budget(self.REQUEST))
        self.assertIsNone(resolver._budget({"variables": ["x", "y"]}))
        record = resolve(self.REQUEST)
        self.assertEqual(record["status"], "resolved")
        self.assertNotIn("budget", record)
        self.assertIsNone(resolver._REQUEST_BUDGET.get())


class LimitsAreSharesOfTheBudget(unittest.TestCase):
    """The limits of the fast methods scale with the budget, so every budget reaches the safe method."""

    def test_the_shares_leave_most_of_the_budget_to_the_rest_of_the_search(self):
        for name in ("NEWTON_SHARE", "TANGENT_SHARE", "WEIGHTED_SHARE", "FALLBACK_SHARE", "NEARBY_SHARE", "TURN_SHARE"):
            self.assertTrue(0 < getattr(quick_budget, name) < 1, name)
        # After the point repairs of the Newton method there is time for a first turn of two rules.
        self.assertLessEqual(quick_budget.NEWTON_SHARE + 2 * quick_budget.TURN_SHARE, 1)
        self.assertLess(quick_budget.FINAL_SECONDS, QUICK_SECONDS * quick_budget.NEARBY_SHARE + 1)

    def test_each_limit_is_its_share_of_the_budget_of_the_request(self):
        for seconds in (QUICK_SECONDS, 8, 60):
            request = {"search": "quick"} if seconds == QUICK_SECONDS else {"search": "quick", "seconds": seconds}
            with self.subTest(seconds=seconds):
                self.assertAlmostEqual(resolver.quick_newton_seconds(request), quick_budget.NEWTON_SHARE * seconds)
                self.assertAlmostEqual(resolver.quick_tangent_seconds(request), quick_budget.TANGENT_SHARE * seconds)
                self.assertAlmostEqual(resolver.quick_turn_seconds(request), quick_budget.TURN_SHARE * seconds)
                self.assertAlmostEqual(resolver.quick_nearby_seconds(request), quick_budget.NEARBY_SHARE * seconds)

    def test_the_nearby_search_never_runs_into_the_end_of_the_budget(self):
        request = {"search": "quick", "seconds": 20}
        started = resolver._REQUEST_START.set(1000.0)
        budget = resolver._REQUEST_BUDGET.set(20)
        try:
            with patch("engine.resolve.perf_counter", return_value=1017.5):
                self.assertAlmostEqual(resolver.quick_nearby_seconds(request), 20 - quick_budget.FINAL_SECONDS - 17.5)
            with patch("engine.resolve.perf_counter", return_value=1019.5):
                self.assertEqual(resolver.quick_nearby_seconds(request), 0)
            # A solve inside the request has the budget of the request, whatever its own fields say.
            self.assertAlmostEqual(resolver.quick_newton_seconds({"search": "quick", "seconds": 300}),
                                   quick_budget.NEWTON_SHARE * 20)
        finally:
            resolver._REQUEST_BUDGET.reset(budget)
            resolver._REQUEST_START.reset(started)

    def test_a_bounded_attempt_keeps_its_fixed_limits(self):
        request = {"search": "quick", "bounded": True, "seconds": 5}
        self.assertEqual(resolver.quick_newton_seconds(request), resolver.QUICK_NEWTON_SECONDS)
        self.assertEqual(resolver.quick_tangent_seconds(request), resolver.QUICK_TANGENT_SECONDS)
        self.assertIsNone(resolver.quick_turn_seconds(request))
        self.assertEqual(resolver.quick_nearby_seconds(request), resolver.QUICK_NEARBY_SECONDS)

    def test_the_search_hands_its_shares_to_the_methods(self):
        fast = ("engine.resolve.resolve_native", "engine.quadratic_fan.resolve_translated",
                "engine.reduced_centre.smooth_reduced_graph")

        class Reached(Exception):
            pass

        request = {"polynomial": "x^2*y^2+z^3*t^2+u^5*x+y^4*t^3", "variables": FIVE, "mode": "embedded",
                   "seconds": 40, "compute": "cpu"}
        with ExitStack() as stack:
            for name in fast:
                stack.enter_context(patch(name, return_value=None))
            newton = stack.enter_context(patch("engine.newton_backend.resolve_newton", return_value=None))
            origin = stack.enter_context(patch("engine.origin_backend.resolve_origin", return_value=None))
            stack.enter_context(patch("engine.general_resolution.search", side_effect=Reached))
            with self.assertRaises(Reached):
                resolve(request)
        self.assertAlmostEqual(newton.call_args.kwargs["seconds"], quick_budget.NEWTON_SHARE * 40)
        tangent, safe = origin.call_args_list
        self.assertTrue(tangent.kwargs["tangent_only"])
        self.assertAlmostEqual(tangent.kwargs["seconds"], quick_budget.TANGENT_SHARE * 40)
        # The safe method has no deadline of its own: its turns are shares of the budget.
        self.assertIsNone(safe.kwargs["seconds"])
        self.assertTrue(safe.kwargs["safe"])
        self.assertAlmostEqual(safe.kwargs["turn"], quick_budget.TURN_SHARE * 40)


class TheNearbySearchEndsWithItsLimit(unittest.TestCase):
    """A transverse germ that is not of Brieskorn-Pham type is resolved by an attempt of its own."""

    ROW = {"equations": [[["1", ["1", "0", "0"]]], [["1", ["0", "1", "0"]]]],
           "germ": [["1", ["2", "1"]], ["1", ["0", "3"]]], "dimension": 1}

    def search(self, limit):
        from engine import monodromy
        from engine.polynomial import parse
        calls = []

        def factors(germ, names, seconds=None):
            calls.append(seconds)
            return [{"multiplicity": 1, "exponent": 1}]

        answer = {"rows": [self.ROW], "expired": False}
        with patch.object(monodromy, "_request", return_value=answer if limit is not None else [self.ROW]), \
             patch.object(monodromy, "_origin_factors", side_effect=factors):
            polynomial = parse("x^2*y+y^3", ["x", "y", "z"])
            if limit is None:
                found, stopped = monodromy._transverse_search(polynomial, ["x", "y", "z"])
            else:
                with monodromy.nearby_search_limit(limit):
                    found, stopped = monodromy._transverse_search(polynomial, ["x", "y", "z"])
        return found, stopped, calls

    def test_the_attempt_gets_the_time_that_the_search_has_left(self):
        found, stopped, calls = self.search(monotonic() + 30)
        self.assertEqual((len(found), stopped), (1, False))
        self.assertTrue(0 < calls[0] <= 30)

    def test_no_attempt_starts_after_the_limit(self):
        found, stopped, calls = self.search(monotonic() - 1)
        self.assertEqual((found, stopped, calls), ([], True, []))

    def test_without_a_limit_the_attempt_runs_as_before(self):
        found, stopped, calls = self.search(None)
        self.assertEqual((len(found), stopped, calls), (1, False, [None]))


class TheRecordOfAnExpiredBudget(unittest.TestCase):
    REQUEST = {"polynomial": SLOW, "variables": SIX, "mode": "embedded", "search": "quick", "monodromy": True}

    def test_the_record_says_how_long_the_search_ran_and_what_it_was_doing(self):
        progress = {"stage": "Blowing up centre B7 (order method)", "blowups": 6}
        status, record = expired(self.REQUEST, 20, 20.04, progress)
        self.assertEqual(status, 200)
        self.assertEqual((record["status"], record["can_search"]), ("inconclusive", True))
        self.assertEqual(record["summary"], "Not resolved within 20 seconds.")
        self.assertEqual(record["budget"], {"seconds": 20, "elapsed_seconds": 20.04, "expired": True, **progress})
        self.assertEqual(record["diagnostic"], {
            "code": "budget_expired",
            "description": "Not resolved within 20 seconds. Stopped while blowing up centre B7 (order method), "
                           "after 6 blowups."})
        self.assertEqual((record["charts"], record["steps"], record["dual_complex"]["status"]), ([], [], "unavailable"))
        for kind in ("monodromy_zeta", "topological_zeta"):
            self.assertEqual(record[kind], {"status": "inconclusive", "reason": record["summary"]})
        # Nothing in it speaks of a method that failed.
        text = json.dumps(record)
        for word in ("returned no", "gave up", "rejected", "failed", "could not"):
            self.assertNotIn(word, text)

    def test_one_line_for_a_person(self):
        _, record = expired(self.REQUEST, 20, 20.0, {"stage": "Blowing up centre B7 (order method)", "blowups": 6})
        self.assertEqual(expired_line(record), "Not resolved within 20 seconds. Stopped while blowing up centre B7 "
                         "(order method), after 6 blowups. Run without --quick to continue the search.")
        _, record = expired(self.REQUEST, 1, None, None)
        self.assertEqual(record["budget"], {"seconds": 1, "elapsed_seconds": 1, "expired": True})
        self.assertEqual(record["diagnostic"]["description"], record["summary"])
        self.assertEqual(expired_line(record), "Not resolved within 1 second. Run without --quick to continue the search.")

    def test_progress_is_read_from_the_file_of_the_search(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "progress")
            self.assertIsNone(read_progress(path))
            with open(path, "w") as stream:
                json.dump({"stage": "Checking charts", "blowups": 3, "method": "origin", "other": 1}, stream)
            self.assertEqual(read_progress(path), {"stage": "Checking charts", "method": "origin", "blowups": 3})
            with open(path, "w") as stream:
                stream.write("{not json")
            self.assertIsNone(read_progress(path))

    def test_a_job_ends_at_its_budget_with_that_record(self):
        started = monotonic()
        status, record = run_job({**self.REQUEST, "seconds": 3})
        elapsed = monotonic() - started
        self.assertEqual((status, record["status"]), (200, "inconclusive"))
        self.assertLess(elapsed, 3 + OVERHEAD)
        budget = record["budget"]
        self.assertEqual((budget["seconds"], budget["expired"]), (3, True))
        self.assertTrue(3 <= budget["elapsed_seconds"] < 3 + OVERHEAD)
        self.assertIsInstance(budget["stage"], str)
        self.assertEqual(record["summary"], "Not resolved within 3 seconds.")


class TheCommandLine(unittest.TestCase):
    def test_seconds_end_a_slow_search_with_a_calm_line_and_exit_code_three(self):
        started = monotonic()
        answer = atlas("resolve", SLOW, "--vars", ",".join(SIX), "--compute", "cpu", "--seconds", "3", "--monodromy")
        self.assertLess(monotonic() - started, 3 + OVERHEAD)
        self.assertEqual(answer.returncode, 3)
        self.assertEqual(answer.stderr, "")
        line, = answer.stdout.splitlines()
        self.assertTrue(line.startswith("Not resolved within 3 seconds."), line)
        self.assertTrue(line.endswith("Run without --quick to continue the search."), line)

    def test_the_record_of_the_command_line_names_the_last_stage(self):
        answer = atlas("resolve", SLOW, "--vars", ",".join(SIX), "--compute", "cpu", "--seconds", "4", "--json")
        self.assertEqual(answer.returncode, 3)
        record = json.loads(answer.stdout)
        self.assertEqual(record["budget"]["seconds"], 4)
        self.assertTrue(record["budget"]["expired"])
        self.assertIsInstance(record["budget"]["stage"], str)
        self.assertTrue(record["can_search"])

    def test_a_germ_that_resolves_gives_the_same_record_with_and_without_quick(self):
        cases = (("y^2-x^3", "x,y", "intrinsic", ()), ("x^2+y^3+z^5", "x,y,z", "embedded", ("--monodromy",)),
                 ("x^2+y^2+z^3+t^4+u^5", "x,y,z,t,u", "embedded", ("--monodromy",)))
        for equation, names, mode, zetas in cases:
            with self.subTest(equation=equation):
                flags = ("resolve", equation, "--vars", names, "--compute", "cpu", "--mode", mode, "--json", *zetas)
                full = atlas(*flags)
                records = [json.loads(full.stdout)]
                for policy in (("--quick",), ("--seconds", "30"), ("--quick", "--seconds", "30")):
                    quick = atlas(*flags, *policy)
                    self.assertEqual((quick.returncode, full.returncode), (0, 0))
                    records.append(json.loads(quick.stdout))
                self.assertEqual(records[0]["status"], "resolved")
                for record in records[1:]:
                    self.assertEqual(without_times(record), without_times(records[0]))
                    self.assertNotIn("budget", record)

    def test_seconds_do_not_go_with_an_exhaustive_search_and_must_be_a_budget(self):
        refused = atlas("resolve", "y^2-x^3", "--vars", "x,y", "--exhaustive", "--seconds", "5")
        self.assertEqual(refused.returncode, 2)
        self.assertIn("no time budget", refused.stderr)
        for bad in ("0", "-3", "soon", "inf"):
            with self.subTest(seconds=bad):
                self.assertEqual(atlas("resolve", "y^2-x^3", "--vars", "x,y", f"--seconds={bad}").returncode, 2)

    def test_help_names_the_budget_and_what_exit_code_three_means(self):
        text = " ".join(atlas("resolve", "--help").stdout.split())
        self.assertIn(f"time budget of {QUICK_SECONDS} seconds", text)
        self.assertIn("--seconds N", text)
        self.assertIn("not finished", text)


class TheSessionFunctions(unittest.TestCase):
    def test_a_script_gets_a_value_and_no_exception_when_the_budget_ends(self):
        source = f"""x, y, z, t, u, v = coordinates("x", "y", "z", "t", "u", "v")
R = resolve({SLOW}, seconds=3)
print(R.status, R.budget["seconds"], R.budget["expired"], R.summary)
C = monodromy_check({SLOW}, seconds=2)
print(C.status)
F = resolve(y^2 - x^3 + z^2 + t^2 + u^2 + v^2, seconds=30)
print(F.status, len(F.divisors))
"""
        started = monotonic()
        answer = atlas("run", "-", input=source)
        self.assertLess(monotonic() - started, 3 + 2 + 2 * OVERHEAD)
        self.assertEqual((answer.returncode, answer.stderr), (0, ""))
        self.assertEqual(answer.stdout.splitlines(), ["inconclusive 3 True Not resolved within 3 seconds.",
                                                      "inconclusive", "resolved 3"])

    def test_the_same_arguments_work_at_the_prompt_and_reject_a_bad_budget(self):
        from engine.app import calculate_resolution as calculate
        from engine.session import new_session
        calls = []

        def recorded(*args, **options):
            calls.append(options)
            return calculate(*args, **options)

        session = new_session(recorded)
        outputs = list(session.run('x, y = coordinates("x", "y")\nR = resolve(y^2 - x^3, seconds=30)\nR.status'))
        self.assertEqual(outputs[-1], (True, "resolved"))
        self.assertEqual((calls[0]["search"], calls[0]["seconds"]), ("quick", 30))
        for call in ("resolve(y^2 - x^3, seconds=0)", "monodromy_check(y^2 - x^3, seconds=-1)"):
            with self.subTest(call=call), self.assertRaisesRegex(Exception, "positive number of seconds"):
                list(session.run(call))


class TheServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from engine.app import AtlasServer
        from tests.http_support import QuietHandler
        cls.http = AtlasServer(("127.0.0.1", 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.http.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()

    def post(self, path, body):
        request = Request(self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return urlopen(request, timeout=60)

    def test_seconds_in_a_request_are_the_budget_of_a_quick_search(self):
        started = monotonic()
        with self.post("/api/resolve", {"polynomial": SLOW, "variables": SIX, "seconds": 2}) as response:
            record = json.load(response)
        self.assertLess(monotonic() - started, 2 + OVERHEAD)
        self.assertEqual((record["status"], record["budget"]["seconds"], record["budget"]["expired"]),
                         ("inconclusive", 2, True))
        self.assertTrue(record["can_search"])

    def test_a_request_that_holds_a_slot_has_a_bounded_budget(self):
        for body, message in (({"seconds": MAX_SECONDS + 1}, "/api/searches"), ({"seconds": 0}, "positive number"),
                              ({"seconds": 5, "search": "exhaustive"}, None)):
            with self.subTest(body=body):
                request = {"polynomial": "y^2-x^3", "variables": ["x", "y"], **body}
                if body.get("search") == "exhaustive":
                    # An exhaustive request becomes a search job, which refuses the contradiction.
                    with self.assertRaises(HTTPError) as refused:
                        self.post("/api/resolve", request)
                else:
                    with self.assertRaises(HTTPError) as refused:
                        self.post("/api/resolve", request)
                self.assertEqual(refused.exception.code, 400)
                text = json.load(refused.exception)["error"]
                refused.exception.close()
                if message:
                    self.assertIn(message, text)

    def test_a_search_job_takes_the_budget_too(self):
        from time import sleep
        with self.post("/api/searches", {"polynomial": SLOW, "variables": SIX, "seconds": 2}) as response:
            job = json.load(response)
        self.assertEqual(job["request"]["search"], "quick")
        for _ in range(100):
            with urlopen(self.url + "/api/searches/" + job["id"], timeout=10) as response:
                job = json.load(response)
            if job["state"] == "completed":
                break
            sleep(0.2)
        record = job["result"]
        self.assertEqual((record["status"], record["budget"]["seconds"], record["budget"]["expired"]),
                         ("inconclusive", 2, True))


class TheZooBuilder(unittest.TestCase):
    def test_a_step_that_ends_at_its_budget_raises_its_own_exception(self):
        from tools.build_resolution_zoo import BudgetExpired, atlas as command, build_record, compute
        flags = ("--vars", ",".join(SIX), "--compute", "cpu", "--quick", "--seconds", "2", "--mode", "embedded")
        with self.assertRaises(BudgetExpired) as stopped:
            command("resolve", SLOW, *flags, step="embedded resolution with zetas")
        error = stopped.exception
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual((error.step, error.budget["seconds"], error.budget["expired"]),
                         ("embedded resolution with zetas", 2, True))
        self.assertIn("not finished within 2 seconds", str(error))
        # A real failure stays an ordinary error.
        with self.assertRaises(RuntimeError) as failed:
            command("resolve", "x/0", "--vars", "x,y", "--quick")
        self.assertNotIsInstance(failed.exception, BudgetExpired)
        # The record builder passes the budget of the seed, or of the caller, to both resolutions.
        seed = {"id": "slow-product", "labels": [], "equation": SLOW}
        for label, arguments in (("caller", (seed, 5, SIX, 2)), ("seed", ({**seed, "seconds": 2}, 5, SIX))):
            with self.subTest(budget=label), self.assertRaises(BudgetExpired) as stopped:
                build_record(*arguments)
            self.assertEqual(stopped.exception.budget["seconds"], 2)
        with self.assertRaises(BudgetExpired) as named:
            compute((seed, 5, SIX), 2)
        self.assertEqual(named.exception.seed, "slow-product")
        self.assertTrue(str(named.exception).startswith("slow-product: the embedded resolution with zetas"))

    def test_the_budget_reaches_both_resolutions(self):
        from tools import build_resolution_zoo as builder
        calls = []

        def recorded(*arguments, step=None):
            calls.append((arguments, step))
            if arguments[0] == "algebra":
                return {"result": {"local_dimension": 1}}
            raise RuntimeError("stop after the first resolution")

        with patch.object(builder, "atlas", side_effect=recorded), self.assertRaisesRegex(RuntimeError, "stop after"):
            builder.build_record({"id": "cusp", "labels": [], "equation": "y^2-x^3"}, 1, ["x", "y"], 7)
        arguments, step = calls[-1]
        self.assertEqual(step, "embedded resolution with zetas")
        self.assertEqual(arguments[arguments.index("--seconds") + 1], "7")
        self.assertIn("--quick", arguments)
        with self.assertRaisesRegex(ValueError, "belongs to a quick search"):
            with patch.object(builder, "atlas", side_effect=recorded):
                builder.build_record({"id": "cusp", "labels": [], "equation": "y^2-x^3", "search": "exhaustive"},
                                     1, ["x", "y"], 7)


if __name__ == "__main__":
    unittest.main()
