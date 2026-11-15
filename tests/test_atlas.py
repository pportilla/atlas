import unittest

import sympy as sp

from engine import atlas
from engine.atlas_gluing import glued_complex
from engine.binomial_backend import resolve_binomial
from engine.newton_backend import resolve_newton
from engine.monodromy import monodromy_zeta
from engine.polynomial import InputError, parse
from engine.resolve import resolve
from tests.strata_oracle import resolution_strata as reference_strata
from engine.topological import topological_zeta

NAMES = ["x", "y", "z"]


def general(text, record):
    """Both zeta functions from the atlas of the record alone."""
    p = parse(text, NAMES)
    charts = atlas.from_record(record, p)
    if charts is None:
        return None
    found = atlas.outputs(atlas.from_json(atlas.to_json(charts)), len(NAMES), complex=False)
    strata = found["numerical_data"], found["strata"], "Unified atlas"
    zeta = monodromy_zeta(record, polynomial=p, strata=strata)
    return zeta["expression"], topological_zeta(record, polynomial=p, monodromy=zeta, strata=strata)["expression"]


class OneMachineForEveryEngine(unittest.TestCase):
    """The outputs computed from the atlas, by the same code for every
    engine, agree with what each engine computes in its own way."""

    def test_zeta_functions_from_the_atlas(self):
        cases = {"x^2+y^3+z^5": "native-toric", "x^2*y-z^3": "binomial-coordinate-blowups",
                 "x^4+y^4+z^4": "native-ordinary", "x*y*z+x^4+y^4+z^4": "newton-polyhedron",
                 "x^3+y^3+z^4": "native-toric"}
        for text, engine in cases.items():
            with self.subTest(germ=text):
                record = resolve({"polynomial": text, "variables": NAMES, "mode": "embedded", "monodromy": True})
                self.assertEqual(record["engine"], engine)
                p = parse(text, NAMES)
                expected = reference_strata(record, p)
                zeta = monodromy_zeta(record, polynomial=p, strata=expected)
                top = topological_zeta(record, polynomial=p, monodromy=zeta, strata=expected)
                self.assertEqual(general(text, record), (zeta["expression"], top["expression"]))

    def test_polynomial_grammar_on_saved_charts(self):
        names = ["for", "u", "u10"]
        variables = tuple(map(sp.Symbol, names))
        f, u, u10 = variables
        item = {"coordinates": names, "map": {"x": "(for-u10)**2/(4/3)-for^2+u"},
                "region": [], "ambient_ideal": [], "components": [], "removed": [], "fibre": []}
        chart, = atlas.from_json([item], native=False)
        self.assertEqual(dict(chart.coordinates)["x"], sp.expand((f-u10)**2*sp.Rational(3, 4)-f**2+u))
        from engine import native_polynomial
        default_chart, = atlas.from_json([item])
        self.assertTrue(native_polynomial.native_ring(default_chart.variables))
        self.assertEqual(atlas.to_json([default_chart]), atlas.to_json([chart]))
        for value in ("u/u", "u(2)", "u10junk", "for.__class__", "u^2^2", "u^-1", "1/0", "u\n1"):
            with self.subTest(value=value), self.assertRaises(InputError):
                atlas.from_json([{**item, "map": {"x": value}}])
        with self.assertRaises(InputError):
            atlas.from_json([{**item, "coordinates": ["u", "u"]}])

    def test_translated_and_algebraic_plane_centres(self):
        names = ["x", "y"]
        for text, multiplicities in (("(y-x)^2-x^5", [2, 4, 5, 10]),
                                     ("(y^2-2*x^4)^2-x^11", [4, 8, 10, 11, 22, 10, 11, 22])):
            record = resolve({"polynomial": text, "variables": names, "mode": "embedded"})
            charts = atlas.from_record(record, parse(text, names))
            self.assertEqual(atlas.from_json(atlas.to_json(charts)), charts)
            result = atlas.outputs(charts, 2)
            self.assertEqual([r["multiplicity"] for r in result["numerical_data"] if r["component"] != "strict"],
                             multiplicities)
            self.assertEqual(len(multiplicities) - len(result["cells"]), 1)
            self.assertEqual(sum(r["euler_characteristic"] for r in result["strata"]), 1 + len(multiplicities))
            if "2*x^4" in text:
                self.assertTrue(any(c.coefficient_field and c.coefficient_field[2] == 2 for c in charts))

    def test_quadratic_chart_field(self):
        text = "x^2+y^2+z^4"
        record = resolve({"polynomial": text, "variables": NAMES, "mode": "embedded"})
        charts = atlas.from_record(record, parse(text, NAMES))
        self.assertTrue(all(c.coefficient_field[2] == 2 for c in charts))
        self.assertEqual(atlas.from_json(atlas.to_json(charts)), charts)
        result = atlas.outputs(charts, 3)
        self.assertIsNone(result["reason"])
        self.assertEqual(sum(r["euler_characteristic"] for r in result["strata"]), 5)
        archived = {**record, "atlas": [], "charts": [{k: v for k, v in item.items() if k != "split_cone"}
                                                     for item in record["charts"]]}
        self.assertEqual(atlas.outputs(atlas.from_record(archived, parse(text, NAMES)), 3), result)
        saved = atlas.to_json(charts)[0]
        for degree in (0, 1, True):
            with self.subTest(degree=degree), self.assertRaises(InputError):
                atlas.from_json([{**saved, "coefficient_field": {**saved["coefficient_field"], "degree": degree}}])

    def test_sheared_fan_agrees_with_the_ordinary_blowup(self):
        text = "x^3+x*y^2+z^3"
        poly = parse(text, NAMES)
        # Exercise the sheared fan even when a different certified monomial
        # modification can resolve the same equation in the original axes.
        from unittest.mock import patch
        with patch("engine.newton_trials.resolve_trials", return_value=None):
            sheared = resolve_newton(poly, NAMES, "embedded")
        self.assertNotEqual(sp.Matrix(sheared["certificate"]["linear_map_to_input"]), sp.eye(3))
        sheared.update(variables=NAMES)
        ordinary = resolve({"variables": NAMES, "polynomial": text})
        self.assertEqual(ordinary["engine"], "native-ordinary")
        self.assertEqual(atlas.outputs(atlas.from_record(sheared, poly), 3),
                         atlas.outputs(atlas.from_record(ordinary, poly), 3))

    def test_a_finite_cover_is_not_a_birational_chart_overlap(self):
        from engine.local_blowups import Chart
        x, y, t = sp.symbols("x y t")
        cover = Chart((x, y, t), (t**2 + 1,), (("E1", 1, 1, (x,)),),
                      (x, y), (), (("x", x), ("y", y)), ())
        self.assertIsNone(glued_complex([cover], 2))

    def test_meeting_strict_branches_are_transverse_but_intrinsically_singular(self):
        from engine.local_blowups import Chart, GiveUp
        x, y = sp.symbols("x y")
        item = Chart((x, y), (), (("S1", 1, 1, (x,)), ("S2", 1, 1, (y,))),
                     (x, y), (), (("x", x), ("y", y)), ())
        atlas.outputs([item], 2, complex=False, integrals=False)
        with self.assertRaisesRegex(GiveUp, "Distinct strict components meet"):
            atlas.outputs([item], 2, complex=False, intrinsic=True, integrals=False)

    def test_gluing_assigns_split_faces_to_the_correct_component(self):
        poly = parse("x^4*y^2-z^6", NAMES)
        record = resolve_binomial(poly, NAMES, "intrinsic")
        record.update(variables=NAMES)
        charts = atlas.from_record(record, poly)
        dual = glued_complex(charts, 3, intrinsic=True)
        # Two disjoint copies of one length-two chain, one per strict branch.
        adjacency = {v["id"]: set() for v in dual["vertices"]}
        for edge in dual["cells"]:
            a, b = edge["vertices"]
            adjacency[a].add(b)
            adjacency[b].add(a)
        self.assertEqual(sorted(map(len, adjacency.values())), [1, 1, 1, 1, 2, 2])
        for a in adjacency:
            connected, pending = {a}, [a]
            while pending:
                node = pending.pop()
                new = adjacency[node] - connected
                pending.extend(new)
                connected.update(new)
            self.assertEqual(len(connected), 3)

    def test_the_charts_partition_the_fibre(self):
        # The Euler characteristics of all the strata add up to that of the
        # fibre over the origin, counted once.
        p = parse("x^4+y^4+z^4", NAMES)
        record = resolve({"polynomial": "x^4+y^4+z^4", "variables": NAMES, "mode": "embedded"})
        found = atlas.outputs(atlas.from_record(record, p), 3, complex=False)
        self.assertEqual(sum(row["euler_characteristic"] for row in found["strata"]), 3)  # the projective plane


if __name__ == "__main__":
    unittest.main()
