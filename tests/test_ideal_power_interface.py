"""Shared ideal-power results across the shell, prompt, scripts and HTTP."""
import json
import shlex
import unittest
from textwrap import dedent

from tests.cli_support import run_prompt, run_script, shell
from tests.http_support import endpoint, post


class IdealPowerInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def test_ideal_outputs_and_portable_replays(self):
        wide = 2**64 + 17
        for op, variables, gens, n, expected in (
            ("ideal-power", ["x", "y"], ["x", "y"], "0002", ["x^2", "x*y", "y^2"]),
            ("symbolic-power", ["x", "y", "z"], ["x*y", "x*z", "y*z"], 2,
             ["x^2*y^2", "x^2*z^2", "x*y*z", "y^2*z^2"]),
            ("ideal-power", ["ideal_power"], [f"{2**150+3}*ideal_power^3"], str(wide),
             [f"ideal_power^{3*wide}"]),
            ("symbolic-power", ["x"], [f"{2**150+3}*x"], wide, [f"x^{wide}"]),
            ("symbolic-power", [], [], 0, ["1"]),
            ("ideal-power", [], [], 3, []),
        ):
            ring = {"variables": variables, "field": "QQ", "order": "lex"}
            with self.subTest(op=op, n=n):
                status, record = post(self.url, op, {"ideal": gens, "exponent": n}, ring)
                self.assertEqual(status, 200, record)
                result = record["result"]
                self.assertEqual(result, {"type": "ideal", "ring": ring, "generators": expected})
                args = shlex.split(record["replay"]["command"])
                native = shell(args)
                self.assertEqual(native.returncode, 0, native.stderr)
                native_record = json.loads(native.stdout)
                self.assertEqual(native_record["result"], result)
                for replay in (shell(["python3", *args]), run_script(record["replay"]["script"]),
                               run_script(native_record["replay"]["script"])):
                    self.assertEqual(replay.returncode, 0, replay.stderr)
                    self.assertEqual(json.loads(replay.stdout)["result"], result)
                args.remove("--json")
                self.assertEqual(shell(args).stdout, shell(["python3", *args]).stdout)

    def test_reuse_at_prompt_and_in_scripts(self):
        source = dedent("""\
            x, y, z = coordinates("x", "y", "z", coeffs=QQ)
            I = ideal(x*y, x*z, y*z)
            A = ideal_power(I, 2)
            B = symbolic_power(I, 2)
            print(A.coordinates == I.coordinates, B.coordinates == I.coordinates)
            print(membership(x*y*z, A).value, membership(x*y*z, B).value)
            print(ideal_power(B, 1).generators == B.generators)
            print(ideal_power(ideal(1, variables=[x,y,z]), 10^4400).generators)
            """)
        expected = ["True True", "False True", "True", "(1,)"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        code, output, errors = run_prompt(source + "quit\n")
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_invalid_input(self):
        for op in ("ideal-power", "symbolic-power"):
            for bad in (-1, True, "1/2", 2.5, None, "2e3"):
                with self.subTest(op=op, bad=bad):
                    status, record = post(self.url, op, {"ideal": ["x"], "exponent": bad})
                    self.assertEqual(status, 400, record)
                    self.assertIn("exponent", record["error"]["message"].lower())
        for n in (0, 1, 2):
            status, record = post(self.url, "symbolic-power", {"ideal": ["x^2"], "exponent": n})
            self.assertEqual(status, 400, record)
            self.assertIn("radical", record["error"]["message"].lower())
        for op, exponent in (("ideal-power", "-1"), ("symbolic-power", "1")):
            failed = shell(["./atlas", "algebra", op, "--vars", "x,y", "--gen", "x^2",
                            "--exponent", exponent, "--json"])
            self.assertEqual(failed.returncode, 2, failed.stderr)
            self.assertEqual(json.loads(failed.stdout)["error"]["code"], "invalid_request")
