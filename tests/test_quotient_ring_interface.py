"""Public quotient-ring requests, saved declarations and portable replays."""
import json
import shlex
import unittest

from tests.cli_support import run_script, shell
from tests.http_support import RING, endpoint, post


class QuotientRingInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def test_cli_http_and_record_replay(self):
        n = 2**80 + 5
        coeff = 2**140 + 1
        for action, left, right, exponent, expected in (
            ("declare", None, None, None, None),
            ("reduce", "x^7", None, None, "x"),
            ("add", "x+1", "x-1", None, "2*x"),
            ("subtract", "x+1", "x-1", None, "2"),
            ("multiply", "x+y", "x-y", None, "-y^2+1"),
            ("negate", "x^3", None, None, "-x"),
            ("power", "x", None, "-" + str(n), "x"),
            ("equal", "x^4", "1", None, True),
            ("unit", "x-1", None, None, False),
            ("inverse", "x", None, None, "x"),
            ("divide", f"{coeff}*y", "x", None, f"{coeff}*x*y"),
        ):
            with self.subTest(action=action):
                fields = {"ideal": ["x^2-1"], "action": action}
                fields.update({k: v for k, v in (("polynomial", left), ("other_polynomial", right),
                                                  ("exponent", exponent)) if v is not None})
                status, record = post(self.url, "quotient-ring", fields)
                self.assertEqual(status, 200, record)
                result = record["result"]
                self.assertEqual(result["relations"], ["x^2-1"])
                self.assertEqual(result["ring"], RING)
                self.assertEqual(result.get("value", result.get("representative")), expected)
                args = shlex.split(record["replay"]["command"])
                native = shell(args)
                self.assertEqual(native.returncode, 0, native.stderr)
                native_record = json.loads(native.stdout)
                self.assertEqual(native_record["result"], result)
                for proc in (shell(["python3", *args]), run_script(record["replay"]["script"]),
                             run_script(native_record["replay"]["script"])):
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertEqual(json.loads(proc.stdout)["result"], result)
                self.assertEqual(shell(args[:-1]).stdout, shell(["python3", *args[:-1]]).stdout)

    def test_saved_quotient_ring_and_arity_validation(self):
        status, decl = post(self.url, "quotient-ring", {"ideal": ["x*y-1"]})
        self.assertEqual(status, 200, decl)
        request = {"basis_ref": decl["basis_ref"], "polynomial": "x", "action": "inverse"}
        status, inverse = post(self.url, "quotient-ring", request)
        self.assertEqual(status, 200, inverse)
        self.assertEqual(inverse["result"]["representative"], "y")
        self.assertNotIn(decl["basis_ref"], str(inverse["replay"]))
        for fields in ({"action": "add", "polynomial": "1"}, {"action": "declare", "polynomial": "0"},
                       {"action": "power", "polynomial": "x", "exponent": "1.5"},
                       {"action": "reduce", "other_polynomial": "x"}, {"action": "inverse", "polynomial": "0"}):
            with self.subTest(fields=fields):
                status, failure = post(self.url, "quotient-ring", dict(fields, ideal=["x*y-1"]))
                self.assertEqual((status, failure["error"]["code"]), (400, "invalid_request"))
        status, failure = post(self.url, "quotient-ring", {"ideal": []}, known_basis=True)
        self.assertEqual((status, failure["error"]["code"]), (400, "invalid_request"))

    def test_finite_inverse_handles_nonreduced_binomials_and_singular_matrices(self):
        relations = ["x^2-9", "y^3-2*y^2"]
        status, record = post(self.url, "quotient-ring", {
            "ideal": relations, "action": "inverse", "polynomial": "1+x*y^2"})
        self.assertEqual(status, 200, record)
        inverse = record["result"]["representative"]
        status, product = post(self.url, "quotient-ring", {
            "ideal": relations, "action": "multiply", "polynomial": inverse, "other_polynomial": "1+x*y^2"})
        self.assertEqual(status, 200, product)
        self.assertEqual(product["result"]["representative"], "1")
        # This element vanishes at (3, 2) but has nonzero constant coefficient.
        fields = {"ideal": relations, "action": "unit", "polynomial": "x+y-5"}
        status, nonunit = post(self.url, "quotient-ring", fields)
        self.assertEqual(status, 200, nonunit)
        self.assertFalse(nonunit["result"]["value"])
        status, failure = post(self.url, "quotient-ring", dict(fields, action="inverse"))
        self.assertEqual((status, failure["error"]["code"]), (400, "invalid_request"))
