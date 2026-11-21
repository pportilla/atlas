"""The ordered components of a map germ survive every public algebra interface."""
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from engine import native_library
from engine.algebra import calculate_algebra, dispatch
from engine.polynomial import InputError
from engine.session_algebra import ae_codimension
from engine.session_polynomial import coordinates
from engine.worker_process import WorkerGroup
from tests.cli_support import ROOT, run_prompt, run_script, shell
from tests.http_support import endpoint, post


class AeCodimensionInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def test_cusp_ordered_components_and_replay(self):
        # (t²,t³) has one normal direction. Unimodular target reordering
        # and unit scaling retain it; replay should also retain the given map.
        components = ["-3/7*t^3", "t^2"]
        status, record = post(self.url, "ae_codimension", {"components": components},
                              {"variables": ["t"]}, compute="cpu")
        self.assertEqual(status, 200, record)
        self.assertEqual(record["operation"], "ae-codimension")
        result = record["result"]
        self.assertIs(result["finite"], True)
        self.assertEqual((result["scope"], result["value"]), ("origin", 1))
        self.assertTrue(result["method"])
        cmd = shlex.split(record["replay"]["command"])
        self.assertIn("--component=-3/7*t^3", cmd)
        for replay in (shell(cmd), run_script(record["replay"]["script"])):
            self.assertEqual(replay.returncode, 0, replay.stderr)
            self.assertEqual(json.loads(replay.stdout)["result"], result)

    def test_zero_components_and_large_exact_integer(self):
        # Embedding the smooth parameter line by (0,t,0,t) is still an immersion.
        for components, expected in [(["0", "t", "0", "t"], 0),
                                     (["(1234567890123456789012345678901/7)*t^18446744073709551621"], 2**64 + 3)]:
            with self.subTest(components=components):
                status, record = post(self.url, "ae-codimension", {"components": components}, {"variables": ["t"]})
                self.assertEqual(status, 200, record)
                self.assertEqual(record["result"]["value"], expected)
                args = shlex.split(record["replay"]["command"])
                direct = shell(args)
                self.assertEqual(direct.returncode, 0, direct.stderr)
                self.assertEqual(json.loads(direct.stdout)["result"], record["result"])
                self.assertEqual([args[i+1] for i, value in enumerate(args) if value == "--component"], components)
        status, record = post(self.url, "ae-codimension", {"components": ["0", "0"]}, {"variables": ["t"]})
        self.assertEqual(status, 200, record)
        self.assertIs(record["result"]["finite"], False)
        self.assertIsNone(record["result"]["value"])

    def test_help_coordinate_spaces_and_list_of_components(self):
        t = coordinates("t")
        t_elsewhere = coordinates("t")
        with self.assertRaisesRegex(ValueError, "different coordinate spaces"):
            ae_codimension(t, t_elsewhere)
        status, output, error = run_prompt('help ae_codimension\nt = coordinates("t")\n'
                                           'r = ae_codimension([t^2, t^3])\n'
                                           'print(r.finite, r.value, r.scope)\nquit\n')
        self.assertEqual((status, error), (0, ""))
        self.assertIn("ae_codimension(*components", output)
        self.assertIn("docs/tutorial/local-algebra.md#compare-the-cusp-and-its-parametrization", output)
        self.assertIn("True 1 origin", output)

    def test_command_help_example_and_weighted_result_field(self):
        help_text = shell(["./atlas", "algebra", "ae-codimension", "--help"])
        self.assertEqual(help_text.returncode, 0, help_text.stderr)
        self.assertIn("result.finite", help_text.stdout)
        example = shlex.split(help_text.stdout.split("Example\n", 1)[1])
        command = shell([*example, "--json"])
        self.assertEqual(command.returncode, 0, command.stderr)
        self.assertEqual(json.loads(command.stdout)["result"]["value"], 1)
        script = run_script("""
            x, y, z = coordinates("x", "y", "z")
            r = ae_codimension(x, y, z^3+x*z, z^5+y*z)
            print(r.value, r.last_weight)
        """)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.strip(), "2 6")

    def test_inconclusive_record_preserves_last_proved_lower_bound(self):
        names = [f"x{i}" for i in range(20)]
        components = ["+".join(f"{x}^{power}" for x in names) for power in (2, 3)]
        status, record = post(self.url + "/exhaustive", "ae-codimension",
                              {"components": components}, {"variables": names})
        self.assertEqual(status, 200, record)
        result = record["result"]
        self.assertIsNone(result["finite"])
        self.assertIsNone(result["value"])
        self.assertTrue(result["reason"])
        self.assertGreater(result["lower_bound"], 0)
        self.assertGreater(result["jet_degree"], 0)
        script = record["replay"]["script"].replace("print(result.json)", "print(result)")
        replay = run_script(script)
        self.assertEqual((replay.returncode, replay.stdout.strip()), (0, "inconclusive"), replay.stderr)
        fields = record["replay"]["script"].replace("print(result.json)", "print(result.lower_bound, result.jet_degree)")
        measured = run_script(fields)
        self.assertEqual((measured.returncode, measured.stdout.strip()),
                         (0, f"{result['lower_bound']} {result['jet_degree']}"), measured.stderr)
        command = shlex.split(record["replay"]["command"])
        command.remove("--json")
        direct = shell(command)
        self.assertEqual(direct.returncode, 0, direct.stderr)
        self.assertIn("Ae-codimension: inconclusive", direct.stdout)

    def test_expression_transport_preserves_whitespace_and_public_errors(self):
        status, record = post(self.url, "ae-codimension", {"components": ["t\u2003**\n2", "t^3"]},
                              {"variables": ["t"]})
        self.assertEqual(status, 200, record)
        self.assertEqual(record["result"]["value"], 1)
        for component, message in [
            ("t * * 2", "Expected a number"), ("z", "Unknown chart variable"),
            ("t/0", "Division needs a nonzero rational constant"), ("", "Enter a nonempty polynomial"),
        ]:
            with self.subTest(component=component):
                status, record = post(self.url, "ae-codimension", {"components": [component]}, {"variables": ["t"]})
                self.assertEqual(status, 400, record)
                self.assertEqual(record["error"]["code"], "invalid_request")
                self.assertIn(message, record["error"]["message"])
        with self.assertRaisesRegex(InputError, "valid Unicode"):
            dispatch({"operation": "ae-codimension", "ring": {"variables": ["t"]},
                      "operands": {"components": ["\ud800"]}})
        status, invalid = post(self.url, "ae-codimension", {"components": ["\ud800"]}, {"variables": ["t"]})
        self.assertEqual(status, 400, invalid)
        self.assertEqual(invalid["error"]["code"], "invalid_request")

    @unittest.skipUnless(hasattr(sys, "set_int_max_str_digits"), "host has no decimal-literal guard")
    def test_input_literal_limit_and_wide_computed_answers(self):
        saved = sys.get_int_max_str_digits()
        try:
            for limit in (640, 4300):
                sys.set_int_max_str_digits(limit)
                digits = "9" * (limit + 1)
                for source in ("t^" + digits, digits + "*t^2"):
                    operands = {"components": [source]}
                    request = {"operation": "ae-codimension", "ring": {"variables": ["t"]}, "operands": operands}
                    with self.subTest(limit=limit, exponent=source.startswith("t")):
                        with self.assertRaisesRegex(InputError, f"{limit} digits"):
                            dispatch(request)
                        for compute in ("cpu", "gpu"):
                            status, record = post(self.url, "ae-codimension", operands, request["ring"], compute=compute)
                            self.assertEqual(status, 400, record)
                            self.assertEqual(record["error"]["code"], "invalid_request")
                            self.assertIn(f"{limit} digits", record["error"]["message"])
                # Both input literals fit. Their exact product in the answer exceeds the guard.
                legal = "9" * limit
                request = {"operation": "ae-codimension", "ring": {"variables": ["x", "y"]},
                           "operands": {"components": [f"x^{legal}+y^{legal}"]}, "compute": "cpu"}
                status, record = calculate_algebra(request)
                self.assertEqual(status, 200)
                self.assertEqual(record["result"]["value"], (10**limit - 2)**2 - 1)
                t = coordinates("t")
                n = 10**limit + 7
                self.assertEqual(ae_codimension(t**n, compute="cpu").value, n - 2)
        finally:
            sys.set_int_max_str_digits(saved)

        env = {**os.environ, "PYTHONINTMAXSTRDIGITS": "640"}
        command = ["./atlas", "algebra", "ae-codimension", "--vars", "t", "--compute", "cpu", "--json",
                   "--component", "t^" + "9"*641]
        limited = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(limited.returncode, 2, limited.stdout)
        self.assertIn("640 digits", json.loads(limited.stdout)["error"]["message"])
        del env["PYTHONINTMAXSTRDIGITS"]
        command[-1] = "t^" + "9"*4301
        unlimited = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(unlimited.returncode, 0, unlimited.stderr)
        self.assertEqual(json.loads(unlimited.stdout, parse_int=str)["result"]["value"], "9"*4300 + "7")

    def test_worker_routing_and_gpu_policy(self):
        request = {"operation": "ae-codimension", "ring": {"variables": ["t"]},
                   "operands": {"components": ["t^2", "t^3"]}}
        with tempfile.TemporaryDirectory() as cache, patch.dict(os.environ, {"XDG_CACHE_HOME": cache}):
            for variable in ("RESOLUTION_GPU_BACKEND", "RESOLUTION_GPU_INDEX", "ATLAS_TRACE_PROFILE",
                             "ATLAS_GROEBNER_GPU"):
                os.environ.pop(variable, None)
            for compute, native in (("cpu", True), ("auto", True), ("gpu", False)):
                with self.subTest(compute=compute):
                    status, record = calculate_algebra(dict(request, compute=compute))
                    self.assertEqual(status, 200, record)
                    self.assertEqual(record["result"]["value"], 1)
                    self.assertEqual(record["execution"]["requested"], compute)
                    self.assertEqual(record["execution"].get("implementation") == "native-cli", native)
            with patch.dict(os.environ, {"RESOLUTION_GPU_BACKEND": "opencl"}):
                status, configured = calculate_algebra(request)
            self.assertEqual(status, 200, configured)
            self.assertEqual(configured["result"], record["result"])
            self.assertNotEqual(configured["execution"].get("implementation"), "native-cli")

    def test_isolated_worker_deadline_and_server_close(self):
        request = {"operation": "ae-codimension", "ring": {"variables": ["t"]},
                   "operands": {"components": ["t^2"]}, "compute": "cpu"}
        for deadline in (True, False):
            with self.subTest(deadline=deadline):
                owner = WorkerGroup()
                self.addCleanup(owner.close)
                start = owner.start
                started = threading.Event()
                children = []

                def hold_native_worker(command, **options):
                    self.assertEqual(command[-1], "--ae-worker")
                    child = start([sys.executable, "-c", "import signal; signal.pause()"], **options)
                    children.append(child)
                    started.set()
                    return child

                with patch.object(owner, "start", side_effect=hold_native_worker), ThreadPoolExecutor(1) as pool:
                    future = pool.submit(calculate_algebra, request, timeout=0.2 if deadline else None, owner=owner)
                    self.assertTrue(started.wait(2))
                    if not deadline:
                        owner.close()
                    status, record = future.result(timeout=3)
                self.assertEqual(status, 422 if deadline else 503, record)
                self.assertIsNotNone(children[0].poll())

    def test_rejected_requests_preserve_input_errors(self):
        for ring, fields in [({"variables": []}, {"components": ["0"]}),
                             ({"variables": ["t"]}, {"components": []}),
                             ({"variables": ["t"]}, {"components": ["t+1"]}),
                             ({"variables": ["t"]}, {"components": [3]}),
                             ({"variables": ["t"]}, {"components": ["t"], "point": [1]}),
                             ({"variables": ["t"]}, {"ideal": ["t"]})]:
            with self.subTest(ring=ring, fields=fields):
                status, record = post(self.url, "ae-codimension", fields, ring)
                self.assertEqual(status, 400, record)
                self.assertEqual(record["error"]["code"], "invalid_request")
