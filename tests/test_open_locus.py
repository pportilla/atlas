import gzip
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from engine import kernel
from engine.chart_expressions import runtime_expressions
from engine.local_blowups import _present


class OpenLocus(unittest.TestCase):
    def test_reducible_nonreduced_and_deleted_charts(self):
        x, y = runtime_expressions().symbols("x y")
        cases = (
            ((x*y,), ((x,),), True),
            ((x*y,), ((x,), (y,)), False),
            ((x*y,), ((x, y),), True),
            ((x**2, y), ((x,),), False),
            ((x**2, y), ((x, y-1),), True),
            ((), ((),), False),
            ((), ((x-x+1,),), True),
        )
        for ideal, removed, expected in cases:
            with self.subTest(ideal=ideal, removed=removed):
                self.assertIs(kernel.present_outside((x,y), ideal, removed), expected)
                with patch("engine.kernel.present_outside", return_value=None):
                    self.assertIs(_present((x,y), ideal, removed), expected)

    def test_large_open_cover_and_unpacked_exponents_defer(self):
        x, y = runtime_expressions().symbols("x y")
        for ideal, removed in (((), ((x,y),)*8), ((), ((x,),)*17), ((x**40000,), ((y,),))):
            with self.subTest(ideal=ideal, removed=removed):
                self.assertIsNone(kernel.present_outside((x,y), ideal, removed))
                self.assertTrue(_present((x,y), ideal, removed))

    def test_over_budget_headers_decline_before_reading_equations(self):
        from engine.native_library import ideal_request
        for sizes in ((1,)*17, (3,)*5):
            header = " ".join(map(str, (2, 1, len(sizes), *sizes)))
            self.assertEqual(ideal_request("open_locus "+header, ()), ["unknown"])

    def test_axis_roots_and_deleted_centres(self):
        x, y = runtime_expressions().symbols("x y")
        f = (x**2 + 1)**4 * (x-2)**3
        cases = (
            # The only remaining geometric points have nonrational coordinates.
            ((y, f), ((x-2,),), True),
            # Each removed centre covers only some roots, but their union covers all.
            ((y, f), ((x-2,), (x**2+1,)), False),
            ((y, f), ((x-2, x**2+1),), True),
            # The whole axis remains after removal of two finite sets.
            ((y,), ((x,), (x-1,)), True),
            ((y,), ((y,),), False),
            # Both axes miss this hyperbola. The general test must still find it.
            ((x*y-1,), ((x-1,),), True),
            ((x*y-1, x-1), ((y-1,),), False),
            # A restriction beyond the dense univariate limit has an off-axis solution.
            ((x**257-1, x*y-1), (), True),
            ((x**257-1, y), ((x,),), True),
        )
        for ideal, removed, expected in cases:
            with self.subTest(ideal=ideal, removed=removed):
                self.assertIs(kernel.present_outside((x,y), ideal, removed), expected)

    def test_wide_axis_coefficients_and_unpacked_fallback(self):
        x, y = runtime_expressions().symbols("x y")
        linear = (2**131 + 7)*x-1
        self.assertIs(kernel.present_outside((x,y), (linear, y), ((x,),)), True)
        self.assertIs(kernel.present_outside((x,y), (linear, y), ((linear,),)), False)
        # This exponent cannot enter the packed multivariate kernel. Its zero
        # set still meets the open y!=0 through the normal caller.
        large = x**(2**32 + 1)
        self.assertIsNone(kernel.present_outside((x,y), (large,), ((y,),)))
        self.assertTrue(_present((x,y), (large,), ((y,),)))

    def test_dense_b8_strict_divisor_survives_in_its_first_blowup_chart(self):
        root = Path(__file__).resolve().parents[1]
        with gzip.open(root/"tests/fixtures/b8-strict-open-locus.json.gz", "rt") as stream:
            data = json.load(stream)
        # In the original chart coordinates (0,u,0,0,0), the three equations
        # restrict to (3211264*u+112,0,0). This chart must retain the divisor.
        # Isolate the real call so a regression to the full basis stays bounded.
        program = (
            "import json,sys\n"
            "from engine.native_library import ideal_request\n"
            "rows=json.load(sys.stdin)\n"
            "print(ideal_request('open_locus 5 3 0', rows))\n"
        )
        result = subprocess.run([sys.executable, "-c", program], cwd=root,
                                input=json.dumps(data["rows"]), capture_output=True,
                                text=True, check=True, timeout=5)
        self.assertEqual(result.stdout.strip(), "['yes']")

    def test_unrecognized_native_answer_defers(self):
        x, y = runtime_expressions().symbols("x y")
        for response in ([], ["unexpected"], ["no", "extra"]):
            with patch("engine.kernel._ask", return_value=response):
                self.assertIsNone(kernel.present_outside((x,y), (x,), ((y,),)))
