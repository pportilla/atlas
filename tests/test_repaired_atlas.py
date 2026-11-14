import json
import unittest

import sympy as sp

from engine import atlas
from engine.newton_backend import resolve_newton
from engine.newton_repair import _point_count, _rational, records, strata
from engine.nonreduced import _strict_parts, nonreduced_resolution
from engine.origin_backend import _resolve
from engine.polynomial import parse
from engine.resolve import resolve


class RepairedNewtonAtlas(unittest.TestCase):
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
