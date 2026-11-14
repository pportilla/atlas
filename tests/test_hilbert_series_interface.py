"""Global Hilbert series through the public algebra API, CLI and replay scripts."""

import json
import shlex
import unittest

from tests.cli_support import run_script, shell
from tests.http_support import RING, endpoint, post


OPERATION = "hilbert-series"


class HilbertSeriesInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def assert_replays(self, record):
        args = shlex.split(record["replay"]["command"])
        self.assertEqual(args[:3], ["./atlas", "algebra", OPERATION])
        self.assertIn("--json", args)
        for entry, process in (("shell", shell(args)), ("script", run_script(record["replay"]["script"]))):
            with self.subTest(entry=entry):
                self.assertEqual(process.returncode, 0, process.stderr)
                replay = json.loads(process.stdout)
                self.assertEqual(replay["operation"], OPERATION)
                self.assertEqual(replay["result"], record["result"])
                self.assertEqual(replay["execution"]["requested"], record["execution"]["requested"])

    def test_signed_series_in_http_shell_and_replays(self):
        status, record = post(self.url, OPERATION, {"ideal": ["x^2", "x*y", "y^2"]})
        self.assertEqual(status, 200, record)
        self.assertEqual(record["result"], {
            "type": OPERATION, "scope": "global", "ring": RING,
            "numerator": [
                {"degree": 0, "coefficient": 1},
                {"degree": 2, "coefficient": -3},
                {"degree": 3, "coefficient": 2},
            ],
            "denominator_power": 2, "dimension": 0, "multiplicity": 3,
        })
        self.assert_replays(record)

        args = shlex.split(record["replay"]["command"])
        args.remove("--json")
        process = shell(args)
        self.assertEqual(process.returncode, 0, process.stderr)
        lines = process.stdout.splitlines()
        for line in ("Scope: global", "H(t) = (1 - 3*t^2 + 2*t^3)/(1 - t)^2",
                     "Dimension: 0", "Multiplicity: 3"):
            self.assertIn(line, lines)

    def test_sparse_degree_and_multiplicity_above_machine_word_size(self):
        degree = 2**64 + 17
        ring = {**RING, "variables": ["x"]}
        status, record = post(self.url, OPERATION, {"ideal": [f"x^{degree}"]}, ring)
        self.assertEqual(status, 200, record)
        result = record["result"]
        self.assertEqual(result["numerator"], [
            {"degree": 0, "coefficient": 1}, {"degree": degree, "coefficient": -1},
        ])
        self.assertEqual((result["denominator_power"], result["dimension"], result["multiplicity"]),
                         (1, 0, degree))
        self.assert_replays(record)

    def test_reject_nonhomogeneous_ideal_over_http_and_cli(self):
        status, record = post(self.url, OPERATION, {"ideal": ["x^2+x"]})
        self.assertEqual(status, 400, record)
        self.assertEqual(record["error"]["code"], "invalid_request")
        self.assertIn("homogeneous", record["error"]["message"])
        self.assertNotIn("result", record)

        process = shell(["./atlas", "algebra", OPERATION, "--vars", "x,y", "--gen", "x^2+x", "--json"])
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["error"], record["error"])

    def test_reject_point_and_degree_inputs(self):
        for operand, value, cli_value in (("point", [0, 0], "0,0"), ("degrees", [0, 1], "0,1")):
            with self.subTest(operand=operand):
                status, record = post(self.url, OPERATION, {"ideal": [], operand: value})
                self.assertEqual(status, 400, record)
                self.assertEqual(record["error"]["code"], "invalid_request")
                self.assertIn(operand, record["error"]["message"])
                self.assertNotIn("result", record)
                process = shell(["./atlas", "algebra", OPERATION, "--vars", "x,y", f"--{operand}", cli_value])
                self.assertEqual(process.returncode, 2)
                self.assertIn(f"--{operand}", process.stderr)
