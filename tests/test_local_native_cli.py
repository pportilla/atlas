"""Regression tests that exercise the point-local executable through ./atlas."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
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

    def test_invalid_point_keeps_cli_diagnostics(self):
        args = ["algebra", "local-normal-form", "--vars", "x", "--point", "1/0",
                "--poly", "x", "--gen", "x", "--json"]
        direct, reference = self.invoke(args), self.invoke(args, interpreter=True)
        self.assertNotEqual(direct.returncode, 0)
        self.assertEqual((direct.returncode, direct.stdout, direct.stderr),
                         (reference.returncode, reference.stdout, reference.stderr))
