import unittest
from unittest.mock import patch

from engine import kernel
from engine.chart_expressions import runtime_expressions
from engine.local_blowups import _present
from engine.native_library import _MODE


@unittest.skipUnless(_MODE == "native", "These checks exercise the native open-locus calculation.")
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

    def test_unrecognized_native_answer_defers(self):
        x, y = runtime_expressions().symbols("x y")
        for response in ([], ["unexpected"], ["no", "extra"]):
            with patch("engine.kernel._ask", return_value=response):
                self.assertIsNone(kernel.present_outside((x,y), (x,), ((y,),)))
