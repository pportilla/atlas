"""A replay must preserve native errors and the byte framing of each transport."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine.kernel_json import kernel_json
from engine.native_library import affine_request
from engine.polynomial import InputError
from tools.replay_requests import replay


class RequestRecording(unittest.TestCase):
    def test_record_replay_and_detect_single_byte_difference(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.dict(os.environ, ATLAS_RECORD_REQUESTS=temp):
                self.assertEqual(["1 2"], affine_request(["parse_input", "x", "x ^ 2"]))
                with self.assertRaises(ArithmeticError):
                    affine_request(["unknown_request"])
                with self.assertRaises(InputError):
                    kernel_json("atlas_resolution_candidate", "unknown_request", width=1)
            with patch.dict(os.environ, ATLAS_LINEAR_ALGEBRA="0"):
                counts = replay(directory)
                self.assertEqual(sum(counts.values()), 3)
                corpus, = directory.iterdir()
                rows = [json.loads(line) for line in corpus.read_text().splitlines()]
                self.assertEqual(rows[0]["response"], "1 2\n")
                rows[0]["response"] = "1 2"
                corpus.write_text("\n".join(map(json.dumps, rows)) + "\n")
                with self.assertRaisesRegex(ValueError, "reply differs"):
                    replay(directory)

    def test_implicit_divisor_orders_and_residual_are_replayable(self):
        from engine.kernel_json import kernel_json
        # In the graph z=x*y, test a nonconstant open, a positive order
        # followed by a division, and a divisor entirely outside the open.
        frame = dict(names=["x", "y", "z"], identities=[0]*3,
                     coordinates=[0, 1], packed_present=True, scalar_unit=False,
                     equations=["1 0 0 1;-1 1 1 0"], kind="remove_coordinate_powers")
        for poly, units, orders in (
            ("1 0 0 0;1 0 0 1", ["1 0 1 0;1 0 0 0"], ["0", "0"]),
            ("1 0 0 1", [], ["1", "1"]),
            ("1 0 0 1", ["1 1 0 0"], ["0", "1"]),
        ):
            args = dict(frame, extra=units, poly=poly, divisors=[0, 1])
            with self.subTest(poly=poly, units=units):
                with patch.dict(os.environ, ATLAS_LINEAR_ALGEBRA="0"):
                    old = kernel_json("atlas_centre_maximum", "replay_inner", **args)
                self.assertEqual(old["orders"], orders)
                self.assertEqual(kernel_json("atlas_centre_maximum", "replay_inner", **args), old)

    def test_order_locus_and_working_set_dimensions(self):
        from engine import centres, native_polynomial as np
        x, y, z = variables = np.symbols("x y z")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ATLAS_RECORD_REQUESTS=directory):
            centres.maximum(variables, (), 3, (x*y,), [("E", (z,))],
                            [variables], {"E": (2, 0, 1, 0, centres.INFINITE)}, ["E"])
            filename, = Path(directory).glob("centre-profile-*.jsonl")
            profile = json.loads(filename.read_text())
            self.assertEqual(profile["dimensions"], [
                dict(order=2, order_locus=1, order_locus_on_working=0, working=0)])
            self.assertGreater(profile["operations"]["affine_maximum"]["us"], 0)
