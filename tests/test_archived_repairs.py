"""Replay Newton repair records written by 27abada and cfa1e57."""
import json
import re
import unittest
from collections import defaultdict
from pathlib import Path

from engine import atlas
from engine.native_library import _MODE
from engine.polynomial import parse


FIXTURE = Path(__file__).with_name("fixtures") / "archived-newton-repair.json"
BEFORE_OWNERSHIP = FIXTURE.with_name("newton-before-ownership.json")


@unittest.skipUnless(_MODE == "native", "The archived conjugate repair runs in the production kernel.")
class ArchivedNewtonRepair(unittest.TestCase):
    def assert_triangle_with_subdivided_edges(self, boundary, dimension):
        # The tangent cubic has a rational line and two conjugate lines.
        # Resolving their double points inserts one vertex on each edge of
        # the triangle, with the rational repair and the conjugate pair
        # belonging to different Newton charts.
        self.assertEqual(boundary["status"], "complete", boundary.get("reason"))
        vertices = {v["id"] for v in boundary["vertices"]}
        lines = {"E1.1", "E1.2", "E1.3"}
        repairs = {"P1.B2.1", "P1.B2.2", "P2.B2"}
        self.assertEqual(vertices, lines | repairs)
        self.assertEqual({v["dimension"] for v in boundary["vertices"]}, {dimension})
        self.assertEqual(len(boundary["cells"]), 6)
        neighbors = defaultdict(set)
        for cell in boundary["cells"]:
            self.assertEqual(cell["dimension"], 1)
            self.assertCountEqual(cell["faces"], cell["vertices"])
            a, b = cell["vertices"]
            neighbors[a].add(b)
            neighbors[b].add(a)
        self.assertEqual({frozenset(neighbors[r]) for r in repairs},
                         {frozenset(lines - {line}) for line in lines})

    def test_saved_atlas_with_two_parents(self):
        record = json.loads(FIXTURE.read_text())
        self.assertEqual(record["dual_complex"]["status"], "partial")
        self.assertTrue(all("repair_projection" not in c and "parent" not in c for c in record["atlas"]))
        charts = atlas.from_record(record, parse(record["polynomial"], record["variables"]))
        self.assert_triangle_with_subdivided_edges(atlas.record_complex(record, charts), 1)
        restored = atlas.to_json(charts)
        self.assertEqual(sum("repair_projection" in c for c in restored), 13)
        # Already populated maps in a partially upgraded copy must agree
        # with the ones recovered from the remaining equations.
        for c in restored[::2]:
            c.pop("repair_projection", None)
        mixed = atlas.from_record({**record, "atlas": restored}, None)
        self.assertEqual(atlas.to_json(mixed), atlas.to_json(charts))

    def test_raw_newton_charts_with_only_composite_maps(self):
        from engine.newton_backend import resolve_newton
        archived = json.loads(FIXTURE.read_text())
        variables = archived["variables"]
        polynomial = parse(archived["polynomial"], variables)
        raw = resolve_newton(polynomial, variables, "intrinsic", defer_outputs=True)
        raw.update(variables=variables, mode="intrinsic", dual_complex=archived["dual_complex"])
        raw.pop("atlas", None)
        for chart in raw["charts"]:
            chart.pop("repair_projection", None)
        self.assert_triangle_with_subdivided_edges(atlas.record_complex(raw, atlas.from_record(raw, polynomial)), 1)

    def test_reordered_product_from_equations(self):
        stored = json.loads(FIXTURE.read_text())["atlas"]
        for chart in stored:
            chart["coordinates"].append("t")
            chart["map"]["w"] = "t"
            chart["fibre"].append("t")
            if "torus" in chart:
                for matrix in chart["torus"].values():
                    for row in matrix:
                        row.append(0)
                    matrix.append([0, 0, 0, 1])
            else:
                chart["map"] = dict(reversed(chart["map"].items()))
        charts = atlas.from_json(list(reversed(stored)))
        self.assert_triangle_with_subdivided_edges(atlas.boundary_complex(charts, 4, intrinsic=True), 2)

    def test_missing_parent_centre_cannot_certify_attachments(self):
        stored = json.loads(FIXTURE.read_text())["atlas"]
        # Replace the conjugate parent centre by a different pair of points
        # on the same exceptional divisor. The lift is still polynomial,
        # but its owned fibre cannot replace the points removed here.
        centre, = stored[0]["removed"]
        centre[1] = "u2^2+2*u2+3"
        boundary = atlas.boundary_complex(atlas.from_json(stored), 3, intrinsic=True)
        self.assertEqual(boundary["status"], "partial")

    def test_incomplete_or_duplicate_cover_without_parent_hints(self):
        record = json.loads(FIXTURE.read_text())
        stored = [{k:v for k,v in c.items() if k != "torus"} for c in record["atlas"]]
        self.assert_triangle_with_subdivided_edges(atlas.boundary_complex(atlas.from_json(stored),3,True), 1)
        for damaged in (stored[:3], stored[:3] + stored[4:], [*stored,stored[3]]):
            with self.subTest(size=len(damaged)):
                boundary = atlas.boundary_complex(atlas.from_json(damaged), 3, intrinsic=True)
                self.assertEqual(boundary["status"], "partial")
                self.assertEqual(boundary["cells"], [])

    def test_archive_before_ownership_used_a_different_field_generator(self):
        record = json.loads(BEFORE_OWNERSHIP.read_text())
        polynomial = parse(record["polynomial"], record["variables"])
        parents = [c for c in record["charts"] if "ray_indices" in c]
        patches = [c for c in record["charts"] if "parent" in c]
        self.assertTrue(all("region" not in c and "fibre" not in c for c in patches))
        self.assertIn("theta^2 + 2*theta + 2", patches[0]["ambient_ideal"])
        # Coordinate names and chart IDs are display choices. Keep the
        # ordered coordinates and equations, while changing their spelling.
        first = patches[0]
        rename = {name: f"local_{i}" for i, name in enumerate(first["coordinates"])}
        def renamed(text):
            return re.sub(r"[A-Za-z_][A-Za-z0-9_]*", lambda m: rename[m[0]], text)
        first["id"] = "renamed_patch"
        first["coordinates"] = list(rename.values())
        first["map"] = {k: renamed(v) for k,v in first["map"].items()}
        for key in ("ambient_ideal", "strict_transform"):
            first[key] = list(map(renamed, first[key]))
        for divisor in first["exceptional"]:
            divisor["ideal"] = list(map(renamed, divisor["ideal"]))
            divisor["equation"] = ", ".join(divisor["ideal"])
        record["charts"] = parents + list(reversed(patches))
        charts = atlas.from_record(record, polynomial)
        self.assertIsNotNone(charts)
        self.assert_triangle_with_subdivided_edges(atlas.record_complex(record, charts), 1)
        # All the other pivots still map to the same centre. Ownership
        # reconstruction must account for the missing pivot itself.
        del record["charts"][-1]
        self.assertIsNone(atlas.from_record(record, polynomial))

    def test_truncated_strict_equation_in_early_archive(self):
        record = json.loads(BEFORE_OWNERSHIP.read_text())
        patch = next(c for c in record["charts"] if c.get("parent"))
        patch["strict_transform"] = ["1"]
        self.assertIsNone(atlas.from_record(record, parse(record["polynomial"], record["variables"])))


class ArchivedLocalCover(unittest.TestCase):
    def test_projective_pivots_recover_the_archived_partition(self):
        record = json.loads(BEFORE_OWNERSHIP.read_text())
        charts = atlas.from_record(record, parse(record["polynomial"], record["variables"]))
        self.assertIsNotNone(charts)
        self.assertEqual(len(charts), len(record["charts"]))
        self.assertCountEqual((c.repair_projection[0] for c in charts if c.repair_projection), [0]*7 + [1]*6)
