"""Geometric sections of two nondegenerate graph face equations."""
from copy import deepcopy
import unittest

from engine.polynomial import InputError
from engine.session_values import Resolution
from engine.toric_graph import graph_complex, resolve_graph


class IntrinsicGraphSections(unittest.TestCase):
    def test_connected_projective_quadric(self):
        graph = dict(auxiliary="v", equation="v-x-y-z",
                     divisor="x^2+y^2+z^2+v*(v-x-y-z)")
        record = resolve_graph(["x", "y", "z"], "x^2+y^2+z^2", graph, mode="intrinsic")
        dual = record["dual_complex"]
        self.assertEqual([vertex["dimension"] for vertex in dual["vertices"]], [1])
        self.assertEqual(dual["cells"], [])
        self.assertEqual(graph_complex(record), dual)

    def test_node_characters_restrict_from_the_edge_to_the_correct_curve(self):
        graph = dict(auxiliary="v", equation="v-x-y-z",
                     divisor="x^2+y^2+z^3+v^2*(v-x-y-z)")
        record = resolve_graph(["x", "y", "z"], "x^2+y^2+z^3", graph, mode="intrinsic")
        dual = record["dual_complex"]
        self.assertEqual(graph_complex(record), dual)
        vertices = {item["id"]: item for item in dual["vertices"]}
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(dual["cells"]), 2)
        resolution = Resolution(record)
        used = set()
        for edge in dual["cells"]:
            self.assertEqual(set(edge["vertices"]), set(edge["faces"]))
            split, = [vertices[v] for v in edge["vertices"] if "characters" in vertices[v]]
            condition, = split["characters"]
            self.assertIn(condition, edge["characters"])
            value = condition["value"]
            used.add(value)
            self.assertEqual(dual["root_families"][value]["polynomial"], "T^2+1")
            rows = [d for chart in record["charts"] for d in chart["exceptional"] if d["id"] == split["id"]]
            self.assertTrue(rows)
            self.assertTrue(all(value in row["ideal"][1] for row in rows))
            component = resolution.divisor(split["id"])
            self.assertEqual(len(component.ideal_in(component.charts[0])), 2)
        self.assertEqual(len(used), 2)
        damaged = deepcopy(record)
        sheet = next(d for chart in damaged["charts"] for d in chart["exceptional"] if "ideal" in d)
        sheet["ideal"][1] = "1"
        with self.assertRaisesRegex(InputError, "saved charts"):
            graph_complex(damaged)

    def test_two_characters_mark_each_point_of_the_resolved_plane_node(self):
        graph = dict(auxiliary="v", equation="v-x-y", divisor="x^2+y^2+v^2*(v-x-y)")
        record = resolve_graph(["x", "y"], "x^2+y^2", graph, mode="intrinsic")
        dual = record["dual_complex"]
        self.assertEqual(len(dual["vertices"]), 2)
        self.assertEqual(dual["cells"], [])
        for point in dual["vertices"]:
            self.assertEqual(point["dimension"], 0)
            self.assertEqual(len(point["characters"]), 2)
            rows = [d for chart in record["charts"] for d in chart["exceptional"] if d["id"] == point["id"]]
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual(len(row["ideal"]), 3)
                for condition, equation in zip(point["characters"], row["ideal"][1:]):
                    self.assertIn(condition["value"], equation)
        self.assertEqual(graph_complex(record), dual)
