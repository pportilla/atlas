"""Finite-colength multiplicities through HTTP, CLI and exported calculations."""
import json
import shlex
import unittest

from tests.test_cli_session import run_script
from tests.test_http_basis_refs import RING, endpoint, post
from tests.test_point_standard_interface import shell


class MultiplicityInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def test_global_length_and_infinite_colength_at_the_cli(self):
        for generators, expected in ((["x^2*(x-1)", "y"], 3), (["x*y"], None), (["1"], 0)):
            with self.subTest(ideal=generators):
                status, record = post(self.url, "multiplicity", {"ideal": generators})
                self.assertEqual(status, 200, record)
                self.assertEqual(record["result"], {
                    "type": "integer", "ring": RING, "scope": "global",
                    "finite": expected is not None, "value": expected,
                })
                args = shlex.split(record["replay"]["command"])
                args.remove("--json")
                shown = shell(args)
                self.assertEqual(shown.returncode, 0, shown.stderr)
                display = f"Multiplicity: {expected}" if expected is not None else "The ideal has infinite colength."
                self.assertIn(display, shown.stdout.splitlines())

    def test_saved_global_basis_and_portable_replays(self):
        # These input leaders omit the relation y^2 = 1 when used directly.
        generators = ["x*y-1", "x-y"]
        status, basis = post(self.url, "groebner", {"ideal": generators})
        self.assertEqual(status, 200, basis)
        operands = {"basis_ref": basis["basis_ref"]}
        status, saved = post(self.url, "multiplicity", operands)
        self.assertEqual(status, 200, saved)
        stages = [s["operation"] for s in saved["execution"]["stages"]]
        self.assertIn("restore_global_basis", stages)
        self.assertNotIn("groebner_basis", stages)
        self.assertEqual(saved["result"]["value"], 2)
        status, fresh = post(self.url, "multiplicity", {"ideal": generators}, compute="gpu")
        self.assertEqual(status, 200, fresh)
        self.assertEqual(saved["result"], fresh["result"])
        self.assertEqual(fresh["execution"]["requested"], "gpu")
        self.assertNotIn(basis["basis_ref"], str(saved["replay"]))
        for replay in (shell(shlex.split(saved["replay"]["command"])), run_script(saved["replay"]["script"])):
            self.assertEqual(replay.returncode, 0, replay.stderr)
            self.assertEqual(json.loads(replay.stdout)["result"], fresh["result"])
        status, mismatch = post(self.url, "multiplicity", operands, ring={**RING, "order": "grevlex"})
        self.assertEqual((status, mismatch["error"]["code"]), (400, "basis_ring_mismatch"))

    def test_rejects_point_and_keeps_exact_large_integer(self):
        status, error = post(self.url, "multiplicity", {"ideal": ["x", "y"], "point": [0, 0]})
        self.assertEqual((status, error["error"]["code"]), (400, "invalid_request"))
        # A coordinate sharing the operation's name must also survive script export.
        degree = 2**96 + 7
        args = ["./atlas", "algebra", "multiplicity", "--vars", "multiplicity", "--gen", f"multiplicity^{degree}"]
        human = shell(args)
        self.assertEqual(human.returncode, 0, human.stderr)
        self.assertIn(f"Multiplicity: {degree}", human.stdout.splitlines())
        raw = shell([*args, "--json"])
        self.assertEqual(raw.returncode, 0, raw.stderr)
        record = json.loads(raw.stdout)
        self.assertEqual(record["result"]["value"], degree)
        exported = run_script(record["replay"]["script"])
        self.assertEqual(exported.returncode, 0, exported.stderr)
        self.assertEqual(json.loads(exported.stdout)["result"], record["result"])
