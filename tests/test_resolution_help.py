"""Resolution help uses saved object identities and reads only recorded data."""
import unittest
from unittest.mock import patch

from engine.resolution_help import resolution_help, resolution_topics
from engine.session_values import DualComplex, Resolution
from tests.cli_support import run_prompt


class ResolutionHelpTests(unittest.TestCase):
    def test_saved_geometry_uses_names_from_the_prompt(self):
        status, output, error = run_prompt("""
            x,y=coordinates("x","y")
            f=y^2-x^3
            K=resolve(f)
            patch=K.chart("B3.u")
            branch=K.branches[0]
            E=branch.meets[0]
            D=K.dual_complex
            help K
            help patch
            help branch
            help E
            help D
            E.ideal_in(patch)
            quit
        """)
        self.assertEqual((status, error), (0, ""))
        for label in ("K is the embedded resolution record for f in QQ[x, y].",
                      "patch is chart B3.u of K.", "branch is branch P1 of K.",
                      "E is exceptional divisor E3 of K.", "D is the complete exceptional dual complex of K",
                      "E.ideal_in(patch)", "docs/resolution-records.md#chart-and-complex-conventions"):
            self.assertIn(label, output)

    def test_inspection_of_partial_or_rebuilt_records_performs_no_algebra(self):
        record = {
            "status": "resolved", "mode": "intrinsic", "variables": ["x", "y"], "polynomial": "y^2-x^3",
            "charts": [{"id": "U", "coordinates": ["u", "v"], "strict_transform": "v",
                        "map": {"x": "u"}, "exceptional": [{"id": "P", "ideal": ["u", "v"]}]}],
            "dual_complex": {"status": "complete", "vertices": [{"id": "P"}], "cells": []},
        }
        result = Resolution(record)
        chart, divisor = result.charts[0], result.divisors[0]
        rebuilt = DualComplex(result, {"status": "partial", "vertices": [], "cells": []})
        names = dict(K=result, C=chart, E=divisor, D=rebuilt, resolution=4)
        with (patch("engine.session_algebra.polynomial", side_effect=AssertionError("parsed archived equation")),
              patch("engine.session_values.chart_space", side_effect=AssertionError("computed chart ring")),
              patch("engine.session_values.chart_polynomial", side_effect=AssertionError("computed equation"))):
            topics = resolution_topics(names)
            self.assertNotIn("resolution.charts", topics)
            self.assertIsNone(resolution_help("resolution.charts", names))
            for name in ("K", "K.polynomial", "K.dual_complex", "C", "C.parent", "C.pullback_map",
                         "E", "E.equation_in", "E.ideal_in", "D", "D.vertices"):
                with self.subTest(name=name):
                    self.assertIn(name, topics)
                    self.assertIn("github.com/pportilla/atlas/blob/main/docs/", resolution_help(name, names))
            self.assertIn("partial chart map", resolution_help("C.pullback_map", names))
            self.assertIn("No chart of E records a single local equation.", resolution_help("E.equation_in", names))
            self.assertIn("E.ideal_in(C)", resolution_help("E.ideal_in", names))
            self.assertIn("partial recomputed exceptional dual complex of K", resolution_help("D", names))
            self.assertNotIn("[0]", resolution_help("D.vertices", names))
