import json
from pathlib import Path
import subprocess
import unittest

import sympy as sp

from engine import atlas
from engine.newton_backend import resolve_newton
from engine.newton_repair import _point_count, _rational, records, strata
from engine.weighted_record import nonreduced_resolution
from tests.nonreduced_oracle import _strict_parts
from engine.origin_backend import _resolve
from engine.polynomial import parse
from engine.resolve import resolve


class RepairedNewtonAtlas(unittest.TestCase):
    def test_intrinsic_nonisolated_quartic_has_two_chains_of_seven(self):
        cli = Path(__file__).resolve().parents[1] / "atlas"
        record = json.loads(subprocess.check_output(
            [cli, "resolve", "x^4-x^2+y^2*z^2", "--vars", "x,y,z", "--mode", "intrinsic", "--json"],
            text=True, timeout=120))
        self.assertEqual(record["status"], "resolved")
        graph = record.pop("dual_complex")
        self.assertEqual(graph["status"], "complete", graph.get("reason"))
        # The two smooth local sheets yz = ±x sqrt(1-x²) have separate chains.
        adjacency = {v["id"]: set() for v in graph["vertices"]}
        self.assertEqual(len(adjacency), 14)
        self.assertEqual(len(graph["vertices"]), 14)
        self.assertEqual(len(graph["cells"]), 12)
        for edge in graph["cells"]:
            self.assertEqual(edge["dimension"], 1)
            a, b = edge["vertices"]
            self.assertCountEqual(edge["faces"], (a, b))
            self.assertNotEqual(a, b)
            self.assertNotIn(b, adjacency[a])
            adjacency[a].add(b)
            adjacency[b].add(a)

        # The terminal chart divisors contain both punctured base axes.
        # Each of the four axis-sheets therefore carries both local names.
        tips = {v["id"] for v in graph["vertices"] if set(v.get("ambient_components", ())) == {"P1.B3","P2.B3"}}
        self.assertEqual(tips, {v for v, neighbours in adjacency.items() if len(neighbours) == 1})
        remaining = set(adjacency)
        while remaining:
            component = {remaining.pop()}
            pending = list(component)
            while pending:
                neighbours = adjacency[pending.pop()] & remaining
                remaining.difference_update(neighbours)
                component.update(neighbours)
                pending.extend(neighbours)
            self.assertEqual(sorted(len(adjacency[v]) for v in component), [1, 1, 2, 2, 2, 2, 2])

        replay = subprocess.check_output([cli, "dual-complex", "-"], input=json.dumps(record),
                                         text=True, timeout=120)
        self.assertEqual(json.loads(replay), graph)

    def test_repaired_cycle_over_conjugate_centres_in_a_product(self):
        from collections import Counter
        from engine.repair_gluing import repair_charts
        names = list("xyz")
        text = "x^3+y^3+z^3-3*x*y*z+x^4+y^4+z^4"
        result = resolve(dict(polynomial=text, variables=names, mode="intrinsic", search="quick"))
        expected = result["dual_complex"]
        self.assertEqual(expected["status"], "complete")
        # The tangent cubic is a triangle. The rational point and the two
        # conjugate points are repaired by chains subdividing its three edges.
        vertices = {v["id"] for v in expected["vertices"]}
        edges = [c["vertices"] for c in expected["cells"]]
        self.assertEqual((len(vertices),len(edges)), (10,10))
        self.assertTrue(all(c["dimension"] == 1 and c["faces"] == c["vertices"] for c in expected["cells"]))
        self.assertEqual(Counter(v for edge in edges for v in edge), {v:2 for v in vertices})
        reached = {next(iter(vertices))}
        for _ in vertices:
            for edge in edges:
                if reached.intersection(edge):
                    reached.update(edge)
        self.assertEqual(reached, vertices)
        stored = json.loads(json.dumps(result["atlas"]))
        for chart in stored:
            free = ["free_a","free_b"]
            chart["coordinates"].extend(free)
            chart["map"].update(zip(("a","b"),free))
            chart["fibre"].extend(free)
            if chart.get("repair_projection"):
                chart["repair_projection"]["map"].extend(free)
            if chart.get("torus"):
                for matrix in chart["torus"].values():
                    for row in matrix:
                        row.extend((0,0))
                    matrix.extend(([0,0,0,1,0],[0,0,0,0,1]))
        found = atlas.boundary_complex(atlas.from_json(stored), 5, intrinsic=True)
        self.assertEqual(found["status"], "complete")
        self.assertEqual(found["cells"], expected["cells"])
        self.assertEqual({(v["id"],v["dimension"]) for v in found["vertices"]}, {(v,3) for v in vertices})
        damaged = json.loads(json.dumps(result["atlas"]))
        link = next(row["repair_projection"] for row in damaged if row.get("repair_projection"))
        link["map"][0] += "+1"
        self.assertIsNone(repair_charts(atlas.from_json(damaged)))

    def test_separating_cover_when_one_coordinate_is_constant(self):
        u, x, y, theta = sp.symbols("u x y theta")
        coords, free = (u, x, y), (x, y)
        for points, expected in (((x-1, y+1), 1), ((x-1, y**2+1), 2)):
            count = _point_count(free, points)
            self.assertEqual(count, expected)
            space, cover = _rational(coords, free, points, count)
            if expected == 1:
                self.assertEqual((space, cover), (coords, ()))
            else:
                self.assertEqual(space, (*coords, theta))
                # The first coordinate does not separate. The next form x+y
                # gives the two sheets theta=1±i and one point on each.
                self.assertEqual(cover, (theta**2-2*theta+2, x-1, y-theta+1))

    def test_intrinsic_graph_and_zeta_after_a_newton_repair(self):
        text = "z^3+y^4+(1/432)*x^6+x*y^2*z+x^3*y*z+x*y*z^2"
        intrinsic, embedded = [resolve({"polynomial": text, "variables": list("xyz"), "mode": mode,
                                        "monodromy": True}) for mode in ("intrinsic", "embedded")]
        graph = intrinsic["dual_complex"]
        self.assertEqual(graph["status"], "complete")
        self.assertEqual({v["id"] for v in graph["vertices"]}, {"E1", "E4.1", "E4.2", "E4.3", "P1.B2"})
        endpoints = [frozenset(edge["vertices"]) for edge in graph["cells"]]
        self.assertCountEqual(endpoints, [frozenset(("E1", v)) for v in ("E4.1", "E4.2", "E4.3", "P1.B2", "P1.B2")])
        ambient = embedded["dual_complex"]
        self.assertEqual(len(ambient["vertices"]) + sum((-1)**c["dimension"] for c in ambient["cells"]), 1)
        for name in ("monodromy_zeta", "topological_zeta"):
            self.assertEqual(intrinsic[name]["expression"], embedded[name]["expression"])

    def test_repaired_fibre_and_strata(self):
        examples = (
            ("z^3+y^4+(1/432)*x^6+x*y^2*z+x^3*y*z+x*y*z^2", list("xyz"), [1]),
            ("(y-x)^2*(x^2+y^2)^2+x^7", list("xy"), [1, 2]),
            ("x^3+y^3+z^3-3*x*y*z+x^4+y^4+z^4", list("xyz"), [1, 2]),
        )
        for text, names, point_counts in examples:
            with self.subTest(text=text):
                polynomial = parse(text, names)
                record = resolve_newton(polynomial, names, "embedded", defer_outputs=True)
                record.update(variables=names, mode="embedded", polynomial=text)
                repairs = [step for step in record["steps"] if "points" in step]
                self.assertEqual(sorted(step["points"] for step in repairs), point_counts)
                charts = atlas.from_record(json.loads(json.dumps(record)), polynomial)
                self.assertIsNotNone(charts)
                for chart in charts:
                    self.assertEqual(dict(chart.coordinates).keys(), set(names))
                    self.assertTrue(all(g.free_symbols <= set(chart.variables) for ideal in chart.removed for g in ideal))
                found = atlas.outputs(atlas.from_json(atlas.to_json(charts)), len(names), complex=len(repairs) == 1)
                certificate = record["certificate"]
                if "strata" in certificate:  # the independent orbit integral already ran in check mode
                    expected = certificate["numerical_data"], certificate["strata"]
                else:
                    adapted = parse(certificate["adapted_polynomial"], names)
                    expected = records(strata(adapted, names, record["fan"]))
                self.assertEqual((found["numerical_data"], found["strata"]), expected)
                if len(repairs) == 1:
                    cells = found["cells"]
                    self.assertIsNotNone(cells)
                    self.assertEqual(7 + sum((-1)**c["dimension"] for c in cells), 1)
                else:
                    self.assertTrue(any(v.name == "theta" for chart in charts for v in chart.variables))

    def test_weights_survive_serialization_on_the_origin_and_repaired_runs(self):
        names = list("xyz")
        examples = (
            ("(x+y)^2+z^3+x^3", lambda polynomial: _resolve(polynomial, names, "embedded", 60)),
            ("z^3+y^4+(1/432)*x^6+x*y^2*z+x^3*y*z+x*y*z^2", lambda p: resolve_newton(p, names, "embedded")),
        )
        for text, producer in examples:
            with self.subTest(text=text):
                reduced = producer(parse(text, names))
                reduced.update(variables=names, mode="embedded")
                support = atlas.from_record(reduced, parse(text, names))
                oracle = {label: 2 * order for c in support for label, order, _, _ in c.components}
                polynomial = parse(f"({text})^2", names)
                weighted = nonreduced_resolution(polynomial, names, "embedded", lambda _: reduced)
                charts = atlas.from_record(json.loads(json.dumps(weighted)), polynomial)
                actual = atlas.outputs(charts, 3, complex=False)["numerical_data"]
                self.assertEqual({row["component"]: row["multiplicity"] for row in actual}, oracle)

    def test_separate_strict_factors_on_an_implicit_chart(self):
        x, y, z = sp.symbols("x y z")
        factors = (x**3 * y**2 * (x+y), x*y*(y+1))
        self.assertEqual(_strict_parts((x, y, z), (z-x-y,), ((x, z-y), (y,)), factors),
                         [((3, 2), (x+y, z)), ((1, 1), (x-z-1, y+1))])


if __name__ == "__main__":
    unittest.main()
