import unittest

from engine.monodromy import monodromy_zeta
from engine.origin_backend import resolve_origin
from engine.polynomial import format_poly, parse
from engine.resolve import resolve
from engine.topological import topological_zeta

NAMES = ["x", "y", "z"]


def from_origin(text):
    """Both zeta functions from the blowups started at the origin."""
    p = parse(text, NAMES)
    record = resolve_origin(p, NAMES, "embedded")
    record.update(variables=NAMES, polynomial=format_poly(p, NAMES), mode="embedded", origin=[0, 0, 0])
    zeta = monodromy_zeta(record, polynomial=p)
    return record, zeta["expression"], topological_zeta(record, polynomial=p, monodromy=zeta)["expression"]


def default(text):
    result = resolve({"polynomial": text, "variables": NAMES, "mode": "embedded", "monodromy": True})
    return result, result["monodromy_zeta"]["expression"], result["topological_zeta"]["expression"]


class BlowupsFromTheOrigin(unittest.TestCase):
    def test_shared_curve_ends_the_repeated_surface_point_choice(self):
        record = resolve(dict(polynomial="x^5+(y^2-z^3)^2", variables=NAMES, mode="embedded",
                              search="quick", compute="cpu", monodromy=True))
        self.assertEqual(record["dual_complex"]["status"], "complete")
        shared = [s for s in record["steps"] if s["rule"] == "shared boundary curve"]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]["center"], "curve")
        self.assertEqual(set(shared[0]["components"]), {"strict", "B1", "B11"})
        self.assertEqual(len(record["steps"]), 34)
        # The squared cusp has two disjoint cusp Milnor fibres (chi=-1 each).
        # Joining with the five-point x^5 fibre gives reduced Euler number 12.
        self.assertEqual(record["monodromy_zeta"]["milnor_fiber_euler_characteristic"], 13)

    def test_same_zeta_functions_as_the_newton_resolutions(self):
        # Two resolutions of the same germ, by different centres.
        for text in ("x^2+y^3+z^5", "x^2+y^3+z^3", "x^3+y^3+z^4", "x^2+y^2+z^4"):
            with self.subTest(germ=text):
                other, zeta, topological = default(text)
                self.assertNotEqual(other["engine"], "canonical-blowups")
                record, own_zeta, own_topological = from_origin(text)
                self.assertEqual(record["engine"], "canonical-blowups")
                self.assertEqual((own_zeta, own_topological), (zeta, topological))

    def test_the_central_divisor_of_e8(self):
        record, _, _ = from_origin("x^2+y^3+z^5")
        data = {(row["multiplicity"], row["nu"]) for row in record["certificate"]["numerical_data"]}
        self.assertIn((30, 31), data)

    def test_a_germ_that_needs_other_coordinates(self):
        # (x+y)^2 + z^3 + x^3 is u^2 + z^3 + x^3 in the coordinates u = x + y, z, x.
        result, zeta, topological = default("(x+y)^2+z^3+x^3")
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["engine"], "canonical-blowups")
        _, expected_zeta, expected_topological = default("x^2+y^3+z^3")
        self.assertEqual((zeta, topological), (expected_zeta, expected_topological))

    def test_a_tangent_cone_with_a_double_component(self):
        result, zeta, topological = default("(x*y-z^2)^2+x^5+y^5")
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["engine"], "canonical-blowups")
        self.assertEqual(result["topological_zeta"]["status"], "computed")
        # The Milnor number is 43 (4 + 80 - 40 - 1).
        self.assertEqual(zeta, "(1-T^5)^8/((1-T^4)*(1-T^10)^8)")
        self.assertEqual(topological, "-(22*s^2 - 11*s - 21)/((s + 1)*(4*s + 3)*(10*s + 7))")


if __name__ == "__main__":
    unittest.main()


class DualComplexAfterBlowups(unittest.TestCase):
    """The exceptional divisor of blowups of a smooth point has a contractible
    dual complex: its Euler characteristic is 1."""

    def euler(self, text):
        result = resolve({"polynomial": text, "variables": NAMES, "mode": "embedded"})
        dual = result["dual_complex"]
        self.assertEqual(dual["status"], "complete", dual.get("reason"))
        ids = {vertex["id"] for vertex in dual["vertices"]} | {cell["id"] for cell in dual["cells"]}
        for cell in dual["cells"]:
            self.assertEqual(len(cell["faces"]), len(cell["vertices"]))
            self.assertLessEqual(set(cell["faces"]), ids)
        return result["engine"], len(dual["vertices"]), sum((-1)**cell["dimension"] for cell in dual["cells"])

    def test_newton_modification_completed_by_blowups(self):
        engine, vertices, rest = self.euler("z^3+y^4+(1/432)*x^6+x*y^2*z+x^3*y*z+x*y*z^2")
        self.assertEqual((engine, vertices, vertices + rest), ("newton-polyhedron", 7, 1))

    def test_blowups_from_the_origin(self):
        engine, vertices, rest = self.euler("x^2+(y+z)^3+y^4")
        self.assertEqual((engine, vertices, vertices + rest), ("canonical-blowups", 6, 1))

    def test_incident_edges_and_faces_follow_each_conjugate_component(self):
        from itertools import combinations
        for text, degree in (("(x+y)^2+z^3+x^3", 3), ("(x*y-z^2)^2+x^5+y^5", 10)):
            with self.subTest(text=text):
                result = resolve({"polynomial": text, "variables": NAMES, "mode": "embedded"})
                dual = result["dual_complex"]
                self.assertEqual(dual["status"], "complete", dual.get("reason"))
                vertices = {v["id"] for v in dual["vertices"]}
                edges = {c["id"]: frozenset(c["vertices"]) for c in dual["cells"] if c["dimension"] == 1}
                triangles = [c for c in dual["cells"] if c["dimension"] == 2]
                self.assertEqual((len(vertices), len(edges), len(triangles)),
                                 (degree+3, 3*degree+2, 2*degree))
                for i in range(1, degree+1):
                    sheet = f"B2.{i}"
                    self.assertEqual({v for edge in edges.values() if sheet in edge for v in edge if v != sheet},
                                     {"B1", "B3", "B4"})
                    self.assertCountEqual([set(c["vertices"])-{sheet} for c in triangles if sheet in c["vertices"]],
                                          [{"B1", "B4"}, {"B3", "B4"}])
                for c in triangles:
                    self.assertCountEqual([edges[e] for e in c["faces"]],
                                          [frozenset(pair) for pair in combinations(c["vertices"], 2)])


class NormalCrossingsAtTheOrigin(unittest.TestCase):
    def test_the_identity_is_the_resolution(self):
        for text, expected in (("x*y*z", "(s + 1)^(-3)"), ("x^2*y^3*z", "1/((s + 1)*(2*s + 1)*(3*s + 1))"),
                               ("(x+y^2)*(y+z^2)*(z+x^2)", "(s + 1)^(-3)")):
            with self.subTest(divisor=text):
                result = resolve({"polynomial": text, "variables": NAMES, "mode": "embedded", "monodromy": True})
                self.assertEqual((result["status"], result["engine"]), ("resolved", "normal-crossings-at-origin"))
                self.assertEqual(result["monodromy_zeta"]["expression"], "1")
                self.assertEqual(result["topological_zeta"]["expression"], expected)

    def test_dependent_differentials_are_not_normal_crossings(self):
        from engine.monodromy_special import _factor_snc
        self.assertIsNone(_factor_snc(parse("x*y*(x+y+z^2)", NAMES), NAMES))
        self.assertEqual(len(_factor_snc(parse("x*y*(x+y+z)", NAMES), NAMES)), 3)
