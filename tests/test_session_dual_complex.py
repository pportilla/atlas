from contextlib import redirect_stdout
import io
import json
import unittest

from engine.session import new_session
from engine.session_values import DualComplex, Resolution


class SessionDualComplexTests(unittest.TestCase):
    def setUp(self):
        # Two distinct strata can have the same incident vertices.
        self.graph = {
            "status": "complete", "convention": "Geometric exceptional components.",
            "vertices": [{"id": "A", "total_multiplicity": 4}, {"id": "B"}],
            "cells": [{"id": label, "dimension": 1, "vertices": ["A", "B"], "faces": ["B", "A"]}
                      for label in ("AB.1", "AB.2")],
        }
        self.resolution = Resolution({"variables": ["x", "y"], "mode": "embedded",
                                      "dual_complex": self.graph, "charts": []})
        self.session = new_session(None)
        self.session.variables["R"] = self.resolution

    def test_recorded_view_shares_divisors_and_preserves_the_full_record(self):
        view = self.resolution.dual_complex
        self.assertIs(view.vertices[0], self.resolution.divisor("A"))
        self.assertEqual(view.vertices, self.resolution.divisors)
        self.assertEqual(view.cells, self.graph["cells"])
        self.assertEqual(view.record, self.graph)
        view.record["vertices"][0]["total_multiplicity"] = 0
        view.cells.pop()
        self.assertEqual(repr(view), "Dual complex (embedded, complete): 2 vertices, 2 edges\n  A — B, A — B")

        out = io.StringIO()
        with redirect_stdout(out):
            list(self.session.run('G = R.dual_complex\n'
                                  'n = G["vertices"][0]["total_multiplicity"]\n'
                                  'm = G.vertices[0]["total_multiplicity"]\n'
                                  'c = G.cells[1].id\n'
                                  'print_json(complex=G, vertices=G.vertices)'))
        self.assertEqual([self.session.variables[name] for name in ("n", "m", "c")], [4, 4, "AB.2"])
        self.assertEqual(json.loads(out.getvalue()), {"complex": self.graph, "vertices": self.graph["vertices"]})

    def test_rebuilt_vertices_only_offer_the_annotations_replay_produced(self):
        replay = DualComplex(self.resolution, {"status": "complete", "vertices": [{"id": "A"}, {"id": "C"}],
                                               "cells": []})
        self.assertEqual([v.id for v in replay.vertices], ["A", "C"])
        with self.assertRaisesRegex(ValueError, "total_multiplicity"):
            replay.vertices[0].session_attribute("total_multiplicity")
        self.assertEqual(self.resolution.divisor("A").session_attribute("total_multiplicity"), 4)

    def test_incomplete_display_reports_status_and_diagnostic(self):
        for status in ("partial", "unavailable"):
            with self.subTest(status=status):
                graph = {"status": status, "reason": "No certified attachments.", "cells": [], "vertices": []}
                view = DualComplex(self.resolution, graph)
                self.assertIn(status, repr(view))
                self.assertIn(graph["reason"], repr(view))
                self.assertEqual(view.session_attribute("reason"), graph["reason"])
