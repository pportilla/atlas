"""Regression tests that exercise the native CPU executable through ./atlas."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from time import monotonic, sleep
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.access(ROOT / "native/build/local_algebra", os.X_OK), "make groebner first")
class LocalNativeCliTests(unittest.TestCase):
    def invoke(self, args, *, interpreter=False, stdin=None):
        argv = [str(ROOT / "atlas"), *args]
        if interpreter:
            argv.insert(0, sys.executable)
        env = {**os.environ, "ATLAS_GROEBNER": "native"}
        env.pop("ATLAS_GROEBNER_GPU", None)
        env.pop("PYTHONINTMAXSTRDIGITS", None)
        return subprocess.run(argv, cwd=ROOT, env=env, text=True, input=stdin, capture_output=True, timeout=15)

    def record(self, args):
        reply = self.invoke(["algebra", *args, "--json", "--compute", "cpu"])
        self.assertEqual(reply.returncode, 0, reply.stderr)
        data = json.loads(reply.stdout)
        self.assertEqual(data["execution"]["implementation"], "native-cli")
        return data

    def test_shifted_projection_and_both_replays(self):
        args = ["local-standard-basis", "--vars", "x,y,z", "--point", "-1/2,0,0",
                "--gen", "3*x^6000*y^2", "--gen", "x*y^2*z", "--gen", "-2*z^2"]
        record = self.record(args)
        self.assertEqual(record["result"]["generators"], ["z^2", "y^2"])
        reference = self.invoke(["algebra", *args, "--compute", "cpu", "--json"], interpreter=True)
        command = self.invoke(shlex.split(record["replay"]["command"])[1:])
        script = self.invoke(["run", "-"], stdin=record["replay"]["script"])
        for response in (reference, command, script):
            self.assertEqual(response.returncode, 0, response.stderr)
            self.assertEqual(json.loads(response.stdout)["result"], record["result"])

    def test_shifted_unit_and_wide_reductions(self):
        unit = self.record(["local-standard-basis", "--vars", "x", "--point", "1",
                            "--gen", "x^800000+1"])
        self.assertEqual(unit["result"]["generators"], ["1"])
        for names in ("x,y", "x,y,z"):
            result = self.record(["local-normal-form", "--vars", names,
                                  "--gen", "x-2*y", "--gen", "y", "--poly", "x^10000000000"])
            self.assertEqual((result["result"]["unit"], result["result"]["remainder"]), ("1", "0"))

    def test_fractional_point_restores_mixed_basis_and_normal_form(self):
        # In centered coordinates u=x-1/2 and v=y+2/3, u=-v^2 and v^4=0.
        u, v = "(x-1/2)", "(y+2/3)"
        expected_basis = {"y^2+x+4/3*y-1/18", "y^4+8/3*y^3+8/3*y^2+32/27*y+16/81"}
        for order in ("lex", "grevlex"):
            with self.subTest(order=order):
                args = ["--vars", "x,y", "--point", "1/2,-2/3", "--order", order,
                        "--gen", f"{u}+{v}^2", "--gen", f"{v}^4"]
                basis = self.record(["local-standard-basis", *args])["result"]
                self.assertEqual(set(basis["generators"]), expected_basis)
                nf = self.record(["local-normal-form", *args, "--poly", f"{u}+{v}^3"])["result"]
                self.assertEqual((nf["unit"], nf["remainder"], nf["reduction"]),
                                 ("1", "y^3+y^2-4/27", "standard"))

    def test_finite_jet_requires_a_certificate(self):
        for order in ("lex", "grevlex"):
            for finite, second in ((True, "x^4+y^5+x*y^4"), (False, "(x^4+y^5)*(1+x)")):
                args = ["local-normal-form", "--vars", "x,y", "--order", order,
                        "--gen", "x^4+y^5", "--gen", second, "--poly", "x^4+y^7"]
                with self.subTest(order=order, finite=finite):
                    result = self.record(args)
                    reference = self.invoke(["algebra", *args, "--compute", "cpu", "--json"], interpreter=True)
                    self.assertEqual(reference.returncode, 0, reference.stderr)
                    self.assertEqual(json.loads(reference.stdout)["result"], result["result"])
                    stages = [s["operation"] for s in result["execution"]["stages"]]
                    self.assertEqual("finite_jet_standard_basis" in stages, finite)

    def test_homogeneous_division_respects_the_local_tie_order(self):
        # x*z leads y^2 under lex, while y^2 leads x*z under grevlex.
        # Adding z^3 makes the local quotient finite in both orders.
        for finite in (False, True):
            for order, quadratic in (("lex", "y^2"), ("grevlex", "x*z")):
                with self.subTest(finite=finite, order=order):
                    args = ["local-normal-form", "--vars", "x,y,z", "--order", order,
                            "--gen", "x*z-y^2", "--gen", "x^3", "--poly", "x*z/2+x+1"]
                    if finite:
                        args.extend(["--gen", "z^3"])
                    record = self.record(args)
                    stages = {s["operation"]: s.get("label") for s in record["execution"]["stages"]}
                    self.assertEqual(stages.get("homogeneous_native_standard_basis"), "local_standard_basis")
                    self.assertEqual(stages.get("homogeneous_native_normal_form"), "local_normal_form")
                    result = record["result"]
                    self.assertEqual((result["unit"], result["remainder"], result["reduction"]),
                                     ("1", f"1/2*{quadratic}+x+1", "standard"))

    def test_homogeneous_division_at_a_shifted_point_uses_a_finite_jet(self):
        # u=x-1, v=y+2/3 satisfy u^2=-v^2 and v^3=0.
        # Hence (1+u)^6000 = 1+6000*u-C(6000,2)*v^2-C(6000,3)*u*v^2.
        u, v = "(x-1)", "(y+2/3)"
        args = ["local-normal-form", "--vars", "x,y", "--point", "1,-2/3",
                "--gen", f"{u}^2+{v}^2", "--gen", f"{v}^3", "--poly", "x^6000"]
        expected = ("-35982002000*x*y^2-143928008000/3*x*y+35964005000*y^2"
                    "-143927954000/9*x+143856020000/3*y+143855966009/9")
        for order in ("lex", "grevlex"):
            with self.subTest(order=order):
                result = self.record([*args, "--order", order])["result"]
                self.assertEqual((result["unit"], result["remainder"], result["reduction"]),
                                 ("1", expected, "standard"))

    def test_homogeneous_division_beyond_packed_exponents(self):
        exponent = 2**64 + 1
        cases = [
            (["x-y", "z^2"], f"x^{exponent}+z+1", f"y^{exponent}+z+1"),
            ([f"x^{exponent}-y^{exponent}", "z^2"], "x+y+z^2+1", "x+y+1"),
        ]
        for order in ("lex", "grevlex"):
            for generators, query, expected in cases:
                with self.subTest(order=order, query=query):
                    args = ["local-normal-form", "--vars", "x,y,z", "--order", order, "--poly", query]
                    for g in generators:
                        args.extend(["--gen", g])
                    record = self.record(args)
                    result = record["result"]
                    self.assertEqual((result["unit"], result["remainder"], result["reduction"]),
                                     ("1", expected, "standard"))
                    self.assertNotIn("homogeneous_native_normal_form",
                                     [s["operation"] for s in record["execution"]["stages"]])

    def test_invalid_point_keeps_cli_diagnostics(self):
        args = ["algebra", "local-normal-form", "--vars", "x", "--point", "1/0",
                "--poly", "x", "--gen", "x", "--json"]
        direct, reference = self.invoke(args), self.invoke(args, interpreter=True)
        self.assertNotEqual(direct.returncode, 0)
        self.assertEqual((direct.returncode, direct.stdout, direct.stderr),
                         (reference.returncode, reference.stdout, reference.stderr))

    def test_primary_decomposition_both_replays_and_cli_output(self):
        args = ["primary-decomposition", "--vars", "y,x", "--order", "lex",
                "--gen", "-x^2", "--gen", "x*y"]
        record = self.record(args)
        self.assertCountEqual([c["prime"] for c in record["result"]["components"]], [["x"], ["y", "x"]])
        self.assertNotIn("point", record["result"])
        stage, = [s for s in record["execution"]["stages"] if s["operation"] == "primary_decomposition"]
        self.assertEqual(stage["generators"], 2)
        replays = (self.invoke(["algebra", *args, "--compute", "cpu", "--json"], interpreter=True),
                   self.invoke(shlex.split(record["replay"]["command"])[1:]),
                   self.invoke(["run", "-"], stdin=record["replay"]["script"]))
        for response in replays:
            self.assertEqual(response.returncode, 0, response.stderr)
            self.assertEqual(json.loads(response.stdout)["result"], record["result"])
        text_args = ["algebra", *args, "--compute", "cpu"]
        self.assertEqual(self.invoke(text_args).stdout, self.invoke(text_args, interpreter=True).stdout)

    def test_primary_decomposition_rejects_local_options(self):
        args = ["algebra", "primary-decomposition", "--vars", "x", "--gen", "x", "--point", "0"]
        direct, reference = self.invoke(args), self.invoke(args, interpreter=True)
        self.assertNotEqual(direct.returncode, 0)
        self.assertEqual((direct.returncode, direct.stdout, direct.stderr),
                         (reference.returncode, reference.stdout, reference.stderr))

    @unittest.skipUnless(Path("/proc/self/task").exists(), "observes the worker using Linux procfs")
    def test_primary_worker_interruption_and_unexpected_exit(self):
        args = [str(ROOT / "atlas"), "algebra", "primary-decomposition", "--compute", "cpu",
                "--vars", "x,y", "--gen", "(x+y+1)^260", "--json"]
        env = dict(os.environ, ATLAS_GROEBNER="native", ATLAS_GROEBNER_GPU="", PYTHONINTMAXSTRDIGITS="")
        for cancel in (True, False):
            with self.subTest(cancel=cancel), subprocess.Popen(
                args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            ) as cli:
                try:
                    children = Path(f"/proc/{cli.pid}/task/{cli.pid}/children")
                    deadline = monotonic() + 5
                    while not (workers := children.read_text().split()):
                        self.assertLess(monotonic(), deadline, "Native CLI did not start a worker")
                        sleep(0.001)
                    worker, = map(int, workers)
                    os.kill(worker, signal.SIGSTOP)
                    if cancel:
                        cli.send_signal(signal.SIGINT)
                    else:
                        os.kill(worker, signal.SIGKILL)
                    stdout, stderr = cli.communicate(timeout=5)
                    if cancel:
                        self.assertEqual((cli.returncode, stdout, stderr),
                                         (130, "", "Algebra calculation interrupted.\n"))
                    else:
                        self.assertEqual(cli.returncode, 2)
                        self.assertEqual(json.loads(stdout)["error"]["code"], "worker_unavailable")
                    self.assertFalse(Path(f"/proc/{worker}").exists())
                finally:
                    if cli.poll() is None:
                        os.killpg(cli.pid, signal.SIGKILL)
