"""Global characters and faces on a certified common torus atlas."""
from collections import Counter
from copy import deepcopy
import unittest

import sympy as sp

from engine import atlas
from engine.atlas_torus import _orbit, torus_complex
from engine.local_blowups import Chart
from engine.newton_backend import resolve_newton
from engine.polynomial import parse
from engine.resolve import resolve
from engine.toric_backend import resolve_toric


class TorusChartCertification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names = list("xyz")
        polynomial = parse("x^2+y^3+z^5", cls.names)
        record = resolve_toric(polynomial, cls.names, "intrinsic")
        record.update(variables=cls.names, mode="intrinsic")
        cls.charts = atlas.from_record(record, polynomial)

    def test_complete_cover_and_ownership_of_permuted_regular_cones(self):
        from engine.fan_geometry import fan_cover
        x, y, z = (1,0,0), (0,1,0), (0,0,1)
        a = (1,1,1)
        cones = ((x,y,a), (a,z,x), (y,z,a))
        self.assertEqual(fan_cover(cones, 3), ((), (frozenset({1}),), (frozenset({0}), frozenset({1}))))
        self.assertIsNone(fan_cover(cones[:2], 3))
        self.assertIsNone(fan_cover((*cones, (x,a,y)), 3))
        self.assertIsNone(fan_cover(((x,y,(0,0,2)),), 3))

    def test_altered_serialized_coordinates_and_torus_metadata_are_refused(self):
        original = atlas.to_json(self.charts)
        altered = []
        wrong_map = deepcopy(original)
        wrong_map[0]["map"]["x"] += "+1"
        altered.append(wrong_map)
        wrong_ray = deepcopy(original)
        wrong_ray[0]["torus"]["rays"] = [list(r) for r in wrong_ray[0]["torus"]["rays"]]
        wrong_ray[0]["torus"]["rays"][0][0] += 1
        altered.append(wrong_ray)
        wrong_linear = deepcopy(original)
        wrong_linear[0]["torus"]["linear_map"] = [list(r) for r in wrong_linear[0]["torus"]["linear_map"]]
        wrong_linear[0]["torus"]["linear_map"][0][0] += 1
        altered.append(wrong_linear)
        wrong_fibre = deepcopy(original)
        wrong_fibre[0]["fibre"] = ["1"]
        altered.append(wrong_fibre)
        omitted_label = deepcopy(original)
        omitted = next(d for d in omitted_label[0]["components"] if d["id"] != "strict")
        omitted_label[0]["components"].remove(omitted)
        altered.append(omitted_label)
        changed_label = deepcopy(original)
        next(d for d in changed_label[0]["components"] if d["id"] == omitted["id"])["id"] = "changed"
        altered.append(changed_label)
        for record in altered:
            with self.subTest(record=record[0]["torus"]):
                self.assertIsNone(torus_complex(atlas.from_json(record), 3, True))

    def test_common_character_handles_reciprocals_and_coefficient_scales(self):
        identity = ((1, 0), (0, 1))
        inversion = ((-1, 0), (0, 1))
        first = _orbit(identity, (((0, 0), 3), ((2, 0), 2)))
        reciprocal = _orbit(inversion, (((0, 0), 2), ((2, 0), 3)))
        scaled = _orbit(identity, (((0, 0), 21), ((2, 0), 14)))
        different = _orbit(identity, (((0, 0), 5), ((2, 0), 2)))
        self.assertEqual(first, reciprocal)
        self.assertEqual(first, scaled)
        self.assertNotEqual(first, different)

    def test_root_indices_match_on_all_split_incident_curves(self):
        result = resolve({"variables": self.names, "polynomial": "2*x^2+3*y^4-z^7+x*y*z^4",
                          "mode": "intrinsic"})
        dual = result["dual_complex"]
        self.assertEqual((len(dual["vertices"]), len(dual["cells"])), (8, 7))
        self.assertNotEqual(dual.get("source"), "resolution engine certificate")
        roots = {v["id"]: v.get("root") for v in dual["vertices"]}
        pairs = [tuple(roots[v] for v in edge["vertices"]) for edge in dual["cells"]
                 if all(roots[v] and roots[v].startswith("alpha12") for v in edge["vertices"])]
        self.assertCountEqual(pairs, [(root, root) for root in ("alpha12_1", "alpha12_2") for _ in range(2)])

    def test_local_smoothness_of_one_affine_chart_does_not_certify_connectedness(self):
        u, v, w = sp.symbols("u v w")
        rays = ((1, 1, 1), (0, 1, 0), (0, 0, 1))
        identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        chart = Chart((u, v, w), (), (("E", 1, 1, (u,)),
                      ("strict", 1, 1, ((v+w+1)*(v+w+2),))),
                      (u, u*v, u*w), (), (("x", u), ("y", u*v), ("z", u*w)), (),
                      torus=(rays, identity))
        # The two parallel affine lines only meet on missing charts at infinity.
        self.assertIsNone(torus_complex([chart], 3, True))


class FourVariableTorus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = list("xyzt")
        polynomial = parse("x^5+z*y^3+z^15+t^4*x+t^8", names)
        cls.record = resolve_newton(polynomial, names, "intrinsic")
        cls.record.update(variables=names, mode="intrinsic")
        cls.charts = atlas.from_record(cls.record, polynomial)

    def test_sheared_598_chart_complex_and_its_integral_boundaries(self):
        self.assertEqual(len(self.charts), 598)
        identity = tuple(tuple(int(i == j) for j in range(4)) for i in range(4))
        self.assertNotEqual(self.charts[0].torus[1], identity)
        charts = atlas.from_json(atlas.to_json(self.charts))
        dual = torus_complex(charts, 4, True)
        self.assertIsNotNone(dual)
        vertices = {v["id"] for v in dual["vertices"]}
        edges = {c["id"]: c for c in dual["cells"] if c["dimension"] == 1}
        triangles = [c for c in dual["cells"] if c["dimension"] == 2]
        self.assertEqual((len(vertices), len(edges), len(triangles)), (99, 251, 153))
        for edge in edges.values():
            self.assertEqual(edge["vertices"], edge["faces"])
            self.assertTrue(set(edge["vertices"]) <= vertices)
        for cell in triangles:
            boundary = Counter()
            for face_id in cell["faces"]:
                face = edges[face_id]
                omitted, = (i for i, v in enumerate(cell["vertices"]) if v not in face["vertices"])
                self.assertEqual(cell["vertices"][:omitted]+cell["vertices"][omitted+1:], face["vertices"])
                sign = (-1)**omitted
                boundary[face["vertices"][0]] -= sign
                boundary[face["vertices"][1]] += sign
            self.assertTrue(all(value == 0 for value in boundary.values()))
        labelled = atlas.record_complex(self.record, charts, dual)
        by_id = {c["id"]: c for c in (*labelled["vertices"], *labelled["cells"])}
        comparisons = 0
        for cell in labelled["cells"]:
            if "root" in cell:
                for face in cell["faces"]:
                    if "root" in by_id[face]:
                        self.assertEqual(cell["root"], by_id[face]["root"])
                        comparisons += 1
        self.assertGreater(comparisons, 0)


if __name__ == "__main__":
    unittest.main()
