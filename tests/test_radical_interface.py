"""Radical requests through the public algebra endpoint and replay commands."""
import json
import shlex
import unittest

from tests.cli_support import run_script, shell
from tests.http_support import RING, endpoint, post


class RadicalInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def test_mixed_dimensional_ideal_and_replays(self):
        # A doubled line and a separate isolated point.
        operands = {"ideal": ["x^2*(x-1)", "x^2*y"]}
        for compute in ("cpu", "gpu"):
            with self.subTest(compute=compute):
                status, record = post(self.url, "radical", operands, compute=compute)
                self.assertEqual(status, 200, record)
                self.assertEqual(record["result"], {
                    "ring": RING, "type": "ideal", "generators": ["x^2-x", "x*y"],
                })
                self.assertEqual(record["execution"]["requested"], compute)
                args = shlex.split(record["replay"]["command"])
                for replay in (shell(args), run_script(record["replay"]["script"])):
                    self.assertEqual(replay.returncode, 0, replay.stderr)
                    self.assertEqual(json.loads(replay.stdout)["result"], record["result"])
                args.remove("--json")
                displayed = shell(args)
                self.assertEqual(displayed.returncode, 0, displayed.stderr)
                self.assertIn("Generators: x^2-x, x*y", displayed.stdout.splitlines())

    def test_reserved_coordinate_scalar_and_empty_ideals(self):
        # Naming a coordinate after the operation must allow the independent script replay.
        for names, generators, expected in ((["radical"], ["(radical^2+1)^2"], ["radical^2+1"]),
                                            ([], [], []), ([], ["-2/5"], ["1"])):
            with self.subTest(names=names, generators=generators):
                status, record = post(self.url, "radical", {"ideal": generators}, {"variables": names})
                self.assertEqual(status, 200, record)
                self.assertEqual(record["result"]["generators"], expected)
                self.assertEqual(record["result"]["ring"], {"field": "QQ", "order": "grevlex", "variables": names})
                replay = run_script(record["replay"]["script"])
                self.assertEqual(replay.returncode, 0, replay.stderr)
                self.assertEqual(json.loads(replay.stdout)["result"], record["result"])

    def test_invalid_ring_or_operands(self):
        for ring, operands in (({**RING, "field": "RR"}, {"ideal": ["x"]}),
                               (RING, {"ideal": ["x"], "point": [0, 0]}),
                               (RING, {"ideal": ["x-z"]})):
            with self.subTest(ring=ring, operands=operands):
                status, error = post(self.url, "radical", operands, ring)
                self.assertEqual((status, error["error"]["code"]), (400, "invalid_request"))
