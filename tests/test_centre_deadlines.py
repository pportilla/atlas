"""Native centre calls stop at their deadline, and the centre search saturates in grevlex.

The fixture is a surface step from the Newton repair of
x^9+x^4*y*z+9/8*x^3*y^3+5/2*x^3*z^2-25/32*x*y^4-23/2*x*y*z^2-2*y^3*z+y^2*z^2-9*z^3.
Without a deadline it takes about 50 seconds, most of them in modular Groebner
computations whose lifting runs on worker threads.
"""
import json
import subprocess
import sys
import time
import unittest
from pathlib import Path
from time import monotonic
from unittest.mock import patch

from engine.kernel_json import kernel_json
from engine.native_library import ideal_request

ROOT = Path(__file__).resolve().parents[1]
SLOW_STEP = Path(__file__).resolve().parent / "fixtures" / "slow-surface-step.json"
GIVE_UP = {"raise": "GiveUp", "message": "The time allowed for the blowups is over."}
# At T = 0 of the same mu-constant family the complete rule's first centre
# choice used to convert a saturation to an enormous lexicographic basis.
T0_ROW = "1 9 0 0;1 4 1 1;1/8 3 3 0;-3/2 3 0 2;-1/32 1 4 0;1/2 1 1 2;1 0 2 2;-1 0 0 3"


class NativeDeadline(unittest.TestCase):
    def test_a_surface_step_stops_at_its_deadline(self):
        operands = json.loads(SLOW_STEP.read_text())
        for seconds in (1, 5):
            with self.subTest(seconds=seconds):
                begin = time.monotonic()
                answer = kernel_json("atlas_centre_maximum", "surface_step", **dict(operands, seconds=seconds))
                elapsed = time.monotonic() - begin
                self.assertEqual(answer, GIVE_UP)
                self.assertGreaterEqual(elapsed, seconds)
                self.assertLess(elapsed, seconds + 2)

    def test_the_surface_step_passes_its_deadline_to_the_kernel(self):
        from engine import native_polynomial, surface
        symbols = tuple(map(native_polynomial.Symbol, "xyz"))
        x = native_polynomial.from_terms(symbols, {(1, 0, 0): 1})
        point = tuple(native_polynomial.from_terms(symbols, {e: 1}) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        sent = []

        def kernel(symbol, operation, **operands):
            sent.append(operands["seconds"])
            return {"step": None}

        with patch("engine.kernel_json.kernel_json", side_effect=kernel):
            for deadline in (None, monotonic() + 30, monotonic() - 1):
                self.assertIsNone(surface._native_step(symbols, (), (x,), [], [point], (), deadline))
        self.assertIsNone(sent[0])
        self.assertTrue(25 < sent[1] <= 30)
        self.assertEqual(sent[2], 0)


class GrevlexSaturation(unittest.TestCase):
    CASES = (
        # (x*y, x*z) : x^infinity = (y, z)
        (("1 1 1 0", "1 1 0 1", "1 1 0 0"), ["1 0 1 0", "1 0 0 1"]),
        # An embedded component: (x^2, x*y) : y^infinity = (x)
        (("1 2 0 0", "1 1 1 0", "1 0 1 0"), ["1 1 0 0"]),
        # ((x+y)*(y-z^2), (x+y)*(x-z)) : (x+y)^infinity = (y-z^2, x-z)
        (("1 1 1 0;-1 1 0 2;1 0 2 0;-1 0 1 2", "1 2 0 0;-1 1 0 1;1 1 1 0;-1 0 1 1", "1 1 0 0;1 0 1 0"),
         ["1 1 0 0;-1 0 0 1", "1 0 1 0;-1 0 0 2"]),
    )

    def basis(self, rows):
        """The reduced grevlex basis of the ideal: saturation by a constant."""
        return ideal_request(f"saturate_grevlex 3 {len(rows)}", [*rows, "1 0 0 0"])

    def test_the_grevlex_saturation_is_the_lexicographic_one(self):
        for (*ideal, by), expected in self.CASES:
            with self.subTest(ideal=ideal, by=by):
                grevlex = ideal_request(f"saturate_grevlex 3 {len(ideal)}", [*ideal, by])
                lex = ideal_request(f"saturate 3 {len(ideal)}", [*ideal, by])
                self.assertNotEqual(grevlex, self.basis(ideal))
                self.assertEqual(grevlex, self.basis(expected))
                self.assertEqual(grevlex, self.basis(lex))

    def test_the_first_centre_choice_of_a_stalling_germ_finishes(self):
        operands = dict(names=["x", "y", "z"], identities=[0, 0, 0], ambient=[], dimension=3, strict=[T0_ROW],
                        divisors=[], working=[["1 1 0 0", "1 0 1 0", "1 0 0 1"]], away=[], births=[],
                        order_only=False, records={}, seconds=None, packed_present=True, gpu=False)
        program = ("import json, sys\nfrom engine.kernel_json import kernel_json\n"
                   "print(json.dumps(kernel_json('atlas_centre_maximum', 'maximum', **json.load(sys.stdin))))\n")
        # It used to run for many minutes; a regression fails here instead of hanging.
        result = subprocess.run([sys.executable, "-c", program], input=json.dumps(operands), cwd=ROOT,
                                capture_output=True, text=True, timeout=240)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        found = json.loads(result.stdout)["maximum"]
        self.assertIsNotNone(found)
        self.assertTrue(found["centre"])


if __name__ == "__main__":
    unittest.main()
