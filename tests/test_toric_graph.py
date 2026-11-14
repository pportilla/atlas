import unittest
from copy import deepcopy

from engine.monodromy import monodromy_zeta
from engine.polynomial import InputError
from engine.topological import topological_zeta
from engine.toric_graph import _request, graph_complex, resolve_graph


class SmoothPolynomialGraph(unittest.TestCase):
    def test_incidence_rejects_holes_and_duplicate_maximal_cones(self):
        names = ["x", "y"]
        graph = dict(auxiliary="v", equation="v-x-y", divisor="v+x+2*y")
        record = resolve_graph(names, "2*x+3*y", graph)
        cones = record["fan"]["cones"]
        for invalid in (cones[:-1], [*cones, list(reversed(cones[0]))]):
            with self.subTest(cones=len(invalid)), self.assertRaisesRegex(InputError, "subdivide"):
                _request("incidence", names, "2*x+3*y", graph, fan={**record["fan"], "cones": invalid})

    def test_graph_replay_and_tamper_detection(self):
        names = ["x", "y"]
        graph = dict(auxiliary="v", equation="v-x-y", divisor="v+x+2*y")
        record = resolve_graph(names, "2*x+3*y", graph)
        self.assertEqual(record["dual_complex"], graph_complex(record))
        self.assertEqual(monodromy_zeta(record)["expression"], "1/(1-T)")
        self.assertEqual(topological_zeta(record)["expression"], "1/(s + 1)")
        damaged = deepcopy(record)
        damaged["charts"][0]["strict_transform"] = "1"
        with self.assertRaisesRegex(InputError, "saved charts"):
            graph_complex(damaged)
        with self.assertRaisesRegex(InputError, "saved charts"):
            monodromy_zeta(damaged)
        broken = deepcopy(record)
        broken["fan"]["cones"][0].pop()
        with self.assertRaisesRegex(InputError, "generators"):
            monodromy_zeta(broken)
        with self.assertRaisesRegex(InputError, "coordinate origin"):
            graph_complex({**record, "origin": [1, 0]})
        record["polynomial"] = "x+y"
        with self.assertRaisesRegex(InputError, "different polynomial"):
            monodromy_zeta(record)

    def test_graph_dependency_and_wide_coordinates(self):
        width = 2**78+1
        coeff = 2**170+33
        graph = dict(auxiliary="v", equation=f"v-{coeff}*x^{width}", divisor="v+y")
        support = _request("support", ["x","y"], f"{coeff}*x^{width}+y", graph)
        self.assertIn([str(width), "1", "0"], support)
        with self.assertRaisesRegex(InputError, "different polynomial"):
            _request("support", ["x","y"], f"x^{width}+y", graph)
        # A high-order term strictly above the Newton boundary changes the
        # chart equations but induces exactly the same exceptional incidences.
        graph = dict(auxiliary="v", equation="v-x-y", divisor="v+x+2*y")
        old = resolve_graph(["x", "y"], "2*x+3*y", graph)
        graph["equation"] += f"+{coeff}*x^{width}"
        verified = _request("charts", ["x", "y"], f"2*x+3*y-{coeff}*x^{width}",
                            graph, fan=old["fan"])
        self.assertEqual(verified["dual_complex"], old["dual_complex"])
        self.assertNotEqual(verified["charts"], old["charts"])

    def test_geometric_components_match_the_same_algebraic_root_across_cones(self):
        coefficient = 2**150 + 19
        graph = dict(auxiliary="v", equation=f"v+x^2+{coefficient}*y^2", divisor="x^4+y^4+v")
        record = resolve_graph(["x", "y"], f"x^4+y^4-x^2-{coefficient}*y^2", graph)
        dual = record["dual_complex"]
        self.assertEqual(dual["status"], "complete")
        self.assertEqual(len(dual["vertices"]), 5)
        self.assertEqual(len(dual["cells"]), 4)
        vertices = {v["id"]: v for v in dual["vertices"]}
        paired = [c for c in dual["cells"] if all("root" in vertices[v] for v in c["vertices"])]
        self.assertEqual(len(paired), 2)
        self.assertEqual(len({c["root"] for c in paired}), 2)
        for edge in paired:
            self.assertEqual({vertices[v]["root"] for v in edge["vertices"]}, {edge["root"]})
            self.assertEqual(set(edge["faces"]), set(edge["vertices"]))
        for chart in record["charts"]:
            for divisor in chart["exceptional"]:
                vertex = vertices[divisor["id"]]
                if "root" in vertex:
                    self.assertIn(vertex["root"], divisor["ideal"][1])
                    self.assertEqual(vertex["ray_divisor"], divisor["ray_divisor"])
        from engine.session_values import Resolution
        resolution = Resolution(record)
        sheet = resolution.divisor(paired[0]["vertices"][0])
        self.assertEqual(len(sheet.ideal_in(sheet.charts[0])), 2)
        self.assertEqual(graph_complex(record), dual)
        # Existing graph records stored the hyperplane equation shared by all
        # its geometric components. The charts still certify this refinement.
        legacy = deepcopy(record)
        legacy["dual_complex"] = dict(status="unavailable", vertices=[], cells=[])
        for chart in legacy["charts"]:
            parents = {}
            for divisor in chart["exceptional"]:
                if "ideal" in divisor:
                    divisor["id"] = divisor.pop("ray_divisor")
                    divisor["equation"] = divisor.pop("ideal")[0]
                parents[divisor["id"]] = divisor
            chart["exceptional"] = list(parents.values())
        self.assertEqual(graph_complex(legacy), dual)

    def test_singular_orbit_rejects_root_count(self):
        graph = dict(auxiliary="v", equation="v+(x+y)^2", divisor="x^4+y^4+v")
        with self.assertRaisesRegex(InputError, "Singular graph face"):
            resolve_graph(["x", "y"], "x^4+y^4-(x+y)^2", graph)
