"""Exercise standalone native ideal sums and products and their two replay formats."""
import json
import os
import shlex
import sys
import unittest
from unittest.mock import patch

from tests.cli_support import ROOT, run_script, shell


@unittest.skipUnless((ROOT / "native/build/ideal_sum").is_file(), "Requires the standalone native ideal commands")
class BinaryIdealCommandTests(unittest.TestCase):
    def test_standalone_results_and_replays(self):
        cases = [
            ("ideal-sum", "", [], ["--other-gen=-3/7"], ["1"]),
            ("ideal-product", "ideal_product,x", ["--gen", "ideal_product^2+x", "--gen=-x"],
             ["--other-gen=-2/3*x"], ["-2/3*ideal_product^2*x-2/3*x^2", "2/3*x^2"]),
            ("ideal-product", "x", [], ["--other-gen", "x"], []),
        ]
        with patch.dict(os.environ, {"ATLAS_GROEBNER": "native", "ATLAS_GROEBNER_GPU": "", "PYTHONINTMAXSTRDIGITS": ""}):
            for op, names, generators, other, expected in cases:
                with self.subTest(operation=op, names=names):
                    cmd = ["./atlas", "algebra", op, "--vars", names, "--compute", "cpu", *generators, *other]
                    cli = shell([*cmd, "--json"])
                    self.assertEqual(cli.returncode, 0, cli.stderr)
                    record = json.loads(cli.stdout)
                    self.assertEqual(record["execution"]["implementation"], "native-cli")
                    self.assertEqual(record["result"]["generators"], expected)
                    self.assertIn(f"result = {op.replace('-', '_')}(I, J", record["replay"]["script"])
                    for replay in (run_script(record["replay"]["script"]), shell(shlex.split(record["replay"]["command"]))):
                        self.assertEqual(replay.returncode, 0, replay.stderr)
                        self.assertEqual(json.loads(replay.stdout)["result"], record["result"])
                    self.assertEqual(shell(cmd).stdout, shell([sys.executable, *cmd]).stdout)

            bad = ["./atlas", "algebra", "ideal-product", "--vars", "x", "--gen", "x", "--other-gen", "y", "--json"]
            native, shared = shell(bad), shell([sys.executable, *bad])
            self.assertEqual((native.returncode, shared.returncode), (2, 2))
            self.assertEqual(json.loads(native.stdout)["error"], json.loads(shared.stdout)["error"])
