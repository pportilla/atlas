"""Overlaps of marked number fields and punctured blowup charts."""
import unittest
from dataclasses import replace

import sympy as sp

from engine import atlas
from engine.atlas_gluing import _character_field, _geometric_primes, glued_complex
from engine.chain_backend import resolve_chain
from engine.local_blowups import Chart
from engine.plane_backend import resolve_plane
from engine.polynomial import parse
from engine.resolve import resolve


class FieldAndOpenOverlaps(unittest.TestCase):
    def test_conjugate_character_curves_attach_on_the_same_branch(self):
        result = resolve({"polynomial": "2*x^2+3*y^4-z^7+x*y*z^4", "variables": list("xyz"),
                          "mode": "intrinsic"})
        dual = result["dual_complex"]
        self.assertEqual(dual["status"], "complete")
        self.assertNotEqual(dual.get("source"), "resolution engine certificate")
        self.assertEqual(len(dual["vertices"]), 8)
        self.assertEqual(len(dual["cells"]), 7)
        roots = {v["id"]: v.get("root") for v in dual["vertices"]}
        arms = [tuple(roots[v] for v in e["vertices"]) for e in dual["cells"]
                if all(roots[v] and roots[v].startswith("alpha12") for v in e["vertices"])]
        self.assertCountEqual(arms, [(root, root) for root in ("alpha12_1", "alpha12_2") for _ in range(2)])

    def test_primitive_character_with_two_nonsquare_classes(self):
        x, y, z = sp.symbols("x y z")
        c = Chart((x, y, z), (), (("D", 1, 1, (z,)), ("strict", 1, 1, (x**2*y**4-2,))),
                  (x, y, z), (), (("x", x), ("y", y), ("z", z)), ())
        (marked,), d = _character_field([c])
        theta, relation, _, _ = marked.coefficient_field
        eq = (relation, z, x**2*y**4-8)
        primes = _geometric_primes(marked, eq, 1, d)
        self.assertCountEqual(primes, [(z, x*y**2-2*theta), (z, x*y**2+2*theta)])
        self.assertIsNone(_geometric_primes(marked, (relation, z, x**2*y**4-3), 1, d))

    def test_split_incidences_over_a_fixed_quadratic_field(self):
        names = list("xyz")
        p = parse("x^2+y^2+z^6", names)
        record = resolve_chain(p, names, "intrinsic")
        record["variables"] = names
        charts = atlas.from_json(atlas.to_json(atlas.from_record(record, p)))
        dual = glued_complex(charts, 3, intrinsic=True)
        self.assertEqual(dual["status"], "complete")
        adjacency = {v["id"]: set() for v in dual["vertices"]}
        for edge in dual["cells"]:
            a, b = edge["vertices"]
            adjacency[a].add(b)
            adjacency[b].add(a)
        self.assertEqual(len(adjacency), 5)
        self.assertEqual(adjacency["E3"], {"E2.1", "E2.2"})
        self.assertEqual(len(adjacency["E1.1"]), 1)
        self.assertEqual(len(adjacency["E1.2"]), 1)
        self.assertNotEqual(adjacency["E1.1"], adjacency["E1.2"])

    def test_meeting_curves_of_one_ambient_divisor_in_the_minimal_resolution(self):
        names = list("xyz")
        for text in ("x*y-z^5", "x^2+y^2+z^5"):
            with self.subTest(text=text):
                p = parse(text, names)
                record = resolve_chain(p, names, "intrinsic")
                record.update(variables=names, mode="intrinsic")
                charts = atlas.from_record(record, p)
                expected = {frozenset(c["vertices"]) for c in record["dual_complex"]["cells"]}
                record["dual_complex"] = atlas.record_complex(record, charts)
                self.assertEqual(record["dual_complex"]["status"], "complete")
                self.assertNotEqual(record["dual_complex"].get("source"), "resolution engine certificate")
                edges = {frozenset(c["vertices"]) for c in record["dual_complex"]["cells"]}
                self.assertEqual(edges, expected)
                self.assertIn(frozenset(("E2u", "E2v")), edges)

    def test_public_odd_a2_has_one_internal_edge(self):
        result = resolve({"polynomial": "x*y-z^3", "variables": list("xyz"), "mode": "intrinsic"})
        dual = result["dual_complex"]
        self.assertEqual(dual["status"], "complete")
        self.assertNotEqual(dual.get("source"), "resolution engine certificate")
        self.assertCountEqual([v["id"] for v in dual["vertices"]], ["E1u", "E1v"])
        self.assertEqual([set(cell["vertices"]) for cell in dual["cells"]], [{"E1u", "E1v"}])

    def test_two_branches_can_meet_on_a_chart_overlap_without_gluing(self):
        x, y, z = sp.symbols("x y z")
        # On the graph surface z=x*y, the plane z=0 cuts two transverse
        # lines. Two copies of the chart describe the same neighbourhood;
        # the first owns its fibre and the second owns the empty region.
        c = Chart((x, y, z), (), (("D", 1, 1, (z,)), ("strict", 1, 1, (z-x*y,))),
                  (x, y, z), (), (("x", x), ("y", y), ("z", z)), ())
        overlap = replace(c, region=(sp.S.One,))
        found = glued_complex([c, overlap], 3, intrinsic=True)
        self.assertCountEqual([v["id"] for v in found["vertices"]], ["D.1", "D.2"])
        self.assertEqual([e["vertices"] for e in found["cells"]], [["D.1", "D.2"]])
        # The same two distinct curves can be tangent. Their graph would
        # require an additional blowup before the crossing is transverse.
        tangent = replace(c, components=(c.components[0], ("strict", 1, 1, (z-y*(y-x**2),))))
        self.assertIsNone(glued_complex([tangent], 3, intrinsic=True))

    def test_only_the_retained_complement_counts_and_separate_embeddings_glue(self):
        names = list("xy")
        for text in ("y^2-x^3", "(y^2-2*x^4)^2-x^11"):
            with self.subTest(text=text):
                p = parse(text, names)
                record = resolve_plane(p, names, "embedded")
                record["variables"] = names
                charts = atlas.from_json(atlas.to_json(atlas.from_record(record, p)))
                self.assertTrue(any(c.removed for c in charts))
                dual = glued_complex(charts, 2)
                expected = {frozenset(edge["vertices"]) for edge in record["dual_complex"]["cells"]}
                edges = [frozenset(edge["vertices"]) for edge in dual["cells"]]
                self.assertEqual(set(edges), expected)
                self.assertEqual(len(edges), len(expected))
                self.assertEqual(len(dual["vertices"]) - len(edges), 1)
                # Blowing up the intersection removes its old edge completely.
                if text == "y^2-x^3":
                    self.assertNotIn(frozenset(("E1", "E2")), edges)


if __name__ == "__main__":
    unittest.main()
