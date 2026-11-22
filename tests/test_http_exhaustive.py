"""Exhaustive HTTP entry points and exact records across worker boundaries."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from engine.app import AtlasServer, MAX_REQUEST, MAX_SECONDS
from tests.cli_support import ROOT
from tests.http_support import QuietHandler


def post(url, data):
    request = Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    try:
        response = urlopen(request, timeout=15)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


class ExhaustiveHTTPTests(unittest.TestCase):
    def setUp(self):
        self.server = AtlasServer(("127.0.0.1", 0), QuietHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02})
        self.thread.start()
        self.addCleanup(self.thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def test_large_coefficient_through_both_routes(self):
        coefficient = "9" * (MAX_REQUEST + 1)
        polynomial = coefficient + "*x+y"
        algebra = {"operation": "ideal", "ring": {"variables": ["x", "y"]},
                   "operands": {"ideal": [polynomial]}, "compute": "cpu"}
        resolution = {"polynomial": polynomial, "variables": ["x", "y"], "compute": "cpu"}
        for path, data in [("/api/resolve", resolution), ("/api/algebra", algebra)]:
            status, _ = post(self.base + path, data)
            self.assertEqual(status, 413)
        status, result = post(self.base + "/api/algebra/exhaustive", algebra)
        self.assertEqual(status, 200, result.get("error"))
        self.assertEqual(result["result"]["generators"], [polynomial])
        status, job = post(self.base + "/api/searches", resolution)
        self.assertEqual(status, 202)
        end = time.monotonic() + 8
        while job["state"] == "running" and time.monotonic() < end:
            time.sleep(0.01)
            with urlopen(self.base + "/api/searches/" + job["id"], timeout=3) as response:
                job = json.load(response)
        self.assertEqual(job["state"], "completed", job.get("error"))
        self.assertEqual(job["result"]["status"], "resolved")
        self.assertIn(coefficient, job["result"]["polynomial"])

    def test_only_exhaustive_algebra_omits_deadline(self):
        with patch.object(self.server.algebra_bases, "calculate", return_value=(200, {})) as calculate:
            for path, timeout in [("/api/algebra", MAX_SECONDS), ("/api/algebra/exhaustive", None)]:
                status, _ = post(self.base + path, {})
                self.assertEqual(status, 200)
                self.assertEqual(calculate.call_args.kwargs["timeout"], timeout)

    def test_automatic_solve_finishes_a_missing_nearby_witness(self):
        request = {"polynomial": "z^3+x^4*y+x^5", "variables": list("xyz"), "monodromy": True}
        status, job = post(self.base + "/api/resolve", request)
        self.assertEqual(status, 202)
        end = time.monotonic() + 10
        while job["state"] in ("queued", "running") and time.monotonic() < end:
            time.sleep(0.02)
            with urlopen(self.base + "/api/searches/" + job["id"], timeout=3) as response:
                job = json.load(response)
        self.assertEqual(job["state"], "completed", job.get("error"))
        check = job["result"]["monodromy_check"]
        self.assertEqual(check["status"], "verified")
        self.assertEqual(check["local_bernstein_sato"]["by_pole"]["-7/12"]["root"], "-7/12")
        command = [str(ROOT / "atlas"), "resolve", request["polynomial"], "--vars", "x,y,z",
                   "--exhaustive", "--zetas-only", "--json"]
        cli = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
        self.assertEqual(json.loads(cli.stdout)["monodromy_check"], check)

    def test_server_close_reaps_long_running_algebra(self):
        started = threading.Event()
        children = []
        start = self.server.workers.start

        def wait_for_shutdown(*args, **options):
            child = start([sys.executable, "-c", "import signal\nsignal.pause()"], **options)
            children.append(child)
            started.set()
            return child

        with patch.object(self.server.workers, "start", side_effect=wait_for_shutdown), ThreadPoolExecutor(1) as pool:
            calculation = pool.submit(post, self.base + "/api/algebra/exhaustive", {})
            self.assertTrue(started.wait(timeout=3))
            self.server.server_close()
            self.assertEqual(calculation.result(timeout=3)[0], 503)
            self.assertIsNotNone(children[0].poll())

    def test_standalone_resolution_roundtrips_wide_numeric_fields(self):
        n = "9" * 4501
        # Exercise the production worker. Its dense diagnostic oracle cannot
        # materialize the input degree; the two integer identities below are exact.
        env = dict(os.environ, ATLAS_GROEBNER="native")
        env.pop("PYTHONINTMAXSTRDIGITS", None)
        process = subprocess.run([str(ROOT / "atlas"), "resolve", "x^" + n + "*y", "--vars", "x,y",
                                  "--exhaustive", "--json"], env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout, parse_int=str)
        self.assertEqual(result["multiplicity"], "1" + "0" * len(n))
        self.assertIn({"equation": "x", "multiplicity": n}, result["input_divisor"]["factors"])
