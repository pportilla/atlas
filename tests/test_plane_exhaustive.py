"""Constructive point blowups beyond the quick plane solver's budgets."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import unittest

from engine.plane_backend import resolve_plane
from engine.polynomial import LimitError, NonReducedError, parse
from engine.resolve import resolve


VARIABLES = ["x","y"]


def solve(polynomial, mode="embedded", search="exhaustive"):
    return resolve({"variables":VARIABLES,"polynomial":polynomial,"mode":mode,"search":search})


class ExhaustivePlane(unittest.TestCase):
    def check_pullbacks(self, result, expression):
        for chart in result["charts"]:
            source = expression.format(**{v:"("+word+")" for v,word in chart["map"].items()})
            lhs = parse(source,chart["coordinates"],bounded=False)
            rhs = parse(chart["ambient_total_transform"],chart["coordinates"],bounded=False)
            self.assertEqual(lhs,rhs,chart["id"])

    def test_long_cusp_enters_constructive_algorithm(self):
        # For y²=x^(2m+1), m successive x-chart blowups produce v²=x.
        # Embedded resolution needs two more blowups for tangency and the
        # resulting triple point. Intrinsic resolution stops at the smooth curve.
        expression = "{y}^2-{x}^391"
        public_input = expression.format(x="x",y="y")
        for mode, steps in [("embedded",197),("intrinsic",195)]:
            result = solve(public_input,mode)
            self.assertEqual(result["engine"], "C++ / FLINT plane point blowups")
            self.assertEqual(result["status"],"resolved")
            self.assertEqual(len(result["steps"]),steps)
            self.assertEqual(len(result["charts"]),2*steps)
            self.assertEqual(len(result["branches"]),1)
            self.check_pullbacks(result,expression)
            dual = result["dual_complex"]
            if mode=="embedded":
                self.assertEqual(len(dual["vertices"]),steps)
                self.assertEqual(len(dual["cells"]),steps-1)
                self.assertEqual(dual["vertices"][-1]["total_multiplicity"],782)
                reached = {"E1"}
                for _ in dual["vertices"]:
                    for edge in dual["cells"]:
                        if reached.intersection(edge["vertices"]):
                            reached.update(edge["vertices"])
                self.assertEqual(reached,{v["id"] for v in dual["vertices"]})
            else:
                self.assertEqual([v["id"] for v in dual["vertices"]],["P1"])
                self.assertEqual(dual["cells"],[])
        self.assertEqual(solve(public_input,search="quick")["status"],"inconclusive")
        with self.assertRaisesRegex(LimitError,"192"):
            resolve_plane(parse(public_input,VARIABLES,bounded=False),VARIABLES,"embedded")

    def test_strict_support_above_the_old_chart_degree_limit(self):
        expression = "(1+{x}^2051)*({y}^2-{x}^3)"
        polynomial = expression.format(x="x",y="y")
        result = solve(polynomial)
        self.assertEqual(len(result["steps"]),3)
        self.check_pullbacks(result,expression)
        with self.assertRaisesRegex(LimitError,"2048"):
            resolve_plane(parse(polynomial,VARIABLES,bounded=False),VARIABLES,"embedded")

    def test_nonreduced_high_degree_divisor(self):
        polynomial = "(x^2+y^3)^45"
        with self.assertRaises(NonReducedError):
            solve(polynomial,"intrinsic")
        result = solve(polynomial)
        self.assertEqual(result["status"],"resolved")
        self.assertEqual([v["total_multiplicity"] for v in result["dual_complex"]["vertices"]],
                         [90,135,270])

    def test_representation_overflow_is_reported_before_arithmetic(self):
        with self.assertRaisesRegex(LimitError,"signed native word"):
            resolve_plane({(sys.maxsize+1,0):Fraction(1)},VARIABLES,"embedded",exhaustive=True)
        worker = Path(__file__).resolve().parents[1]/"native"/"build"/"plane"
        payload = {"exhaustive":True, "variables":VARIABLES, "mode":"embedded",
                   "terms":[[sys.maxsize,1,"1"]]}
        process = subprocess.run([str(worker)],input=json.dumps(payload),capture_output=True,
                                 text=True,check=True,timeout=5)
        output = json.loads(process.stdout)
        self.assertEqual(output["kind"],"limit")
        self.assertIn("signed FLINT word",output["error"])


if __name__=="__main__":
    unittest.main()
