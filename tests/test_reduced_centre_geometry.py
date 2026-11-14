"""Regression cases for discovering and proving a single blowup along the reduced singular locus."""
import unittest

from engine import native_polynomial as np
from engine.blowup_ideal import ideal_basis
from engine.monodromy import monodromy_zeta
from engine.general_resolution import smooth_reduced_graph
from engine.native_library import _MODE
from engine.polynomial import parse
from engine.resolve import resolve
from engine.resolution_complex import dual_complex


@unittest.skipUnless(_MODE == "native", "Reduced singular centre discovery uses the native solver.")
class ReducedCentreGeometry(unittest.TestCase):
    names = ["x", "y", "z"]
    parabola = "(y-x^2)^2-x*z^2"
    insufficient = "(y-x^2)^2-z^5"
    remote_crossing = "(y-x^2)^2-x*(x-1)^3*z^2"

    def proposal(self, polynomial, mode="embedded"):
        return smooth_reduced_graph(parse(polynomial, self.names), self.names, mode, seconds=2)

    def resolve_public(self, polynomial, mode="embedded"):
        return resolve({"polynomial": polynomial, "variables": self.names, "mode": mode,
                        "search": "exhaustive", "compute": "cpu", "monodromy": True})

    def test_curved_centre_on_the_public_path(self):
        record = self.resolve_public(self.parabola)
        self.assertEqual(record["engine"], "verified-ideal-blowup")
        self.assertEqual(record["certificate"]["center_source"], "smooth reduced singular locus")
        self.assertEqual(len(record["charts"]), 2)
        x, y, z = np.symbols("x y z")
        generators = [np.from_terms((x, y, z), parse(f, self.names)) for f in record["steps"][0]["generators"]]
        self.assertEqual(ideal_basis((x, y, z), generators), ideal_basis((x, y, z), (y-x**2, z)))
        self.assertEqual(record["dual_complex"]["vertices"], [{"id": "E1", "dimension": 2}])
        self.assertEqual(record["dual_complex"]["cells"], [])
        self.assertEqual(dual_complex(record)["vertices"], record["dual_complex"]["vertices"])
        self.assertEqual(record["monodromy_zeta"]["expression"], "1/(1-T^2)")
        self.assertEqual(record["topological_zeta"]["expression"], "(s + 2)/(2*(s + 1)^2)")
        self.assertEqual(monodromy_zeta(record)["expression"], record["monodromy_zeta"]["expression"])

    def test_nonlinear_centre_with_a_free_parameter(self):
        names = ["x", "y", "z", "t"]
        cases = (("(y-x^2)^2-(x+t)*z^2", 2), ("(y-x^2)^2+z^2+t^2", 3))
        for equation, count in cases:
            polynomial = parse(equation, names)
            for mode, dimension in (("embedded", 3), ("intrinsic", 2)):
                with self.subTest(equation=equation, mode=mode):
                    record = smooth_reduced_graph(polynomial, names, mode, seconds=2)
                    self.assertEqual(len(record["charts"]), count)
                    self.assertEqual(record["dual_complex"]["vertices"], [{"id": "E1", "dimension": dimension}])

    def test_smooth_radical_does_not_prove_one_blowup_suffices(self):
        for polynomial in (self.insufficient, self.remote_crossing, "x^2-y^2*z^2", "x^2+y^2+z^2",
                           "((y-x^2)^2-x*z^2)^2"):
            with self.subTest(polynomial=polynomial):
                self.assertIsNone(self.proposal(polynomial))
        record = self.resolve_public(self.insufficient)
        self.assertEqual(record["status"], "resolved")
        self.assertNotEqual(record["certificate"].get("center_source"), "smooth reduced singular locus")
        self.assertEqual(record["monodromy_zeta"]["expression"], "(1-T^10)/((1-T^2)*(1-T^5))")

    def test_embedded_and_intrinsic_exceptional_components(self):
        # The two sections of the exceptional P¹ are disjoint on the strict transform.
        polynomial = "(y-x^2)^2-z^2"
        ambient = self.proposal(polynomial)
        intrinsic = self.proposal(polynomial, "intrinsic")
        self.assertEqual(len(ambient["dual_complex"]["vertices"]), 1)
        self.assertEqual(len(intrinsic["dual_complex"]["vertices"]), 2)
        self.assertEqual(intrinsic["dual_complex"]["cells"], [])
        for chart in intrinsic["charts"]:
            self.assertEqual(len(chart["exceptional"]), 2)
            self.assertEqual(len(chart["ambient_components"]), 1)

    def test_quadratic_conjugate_sheets_do_not_merge_in_intrinsic_mode(self):
        polynomial = "(y-x^2)^2+2*z^2"
        ambient = self.proposal(polynomial)
        self.assertEqual(len(ambient["dual_complex"]["vertices"]), 1)
        intrinsic = self.proposal(polynomial, "intrinsic")
        if intrinsic is not None:
            self.assertEqual(len(intrinsic["dual_complex"]["vertices"]), 2)
            self.assertEqual(intrinsic["dual_complex"]["cells"], [])

    def test_units_and_input_bounds_transfer_to_the_existing_search(self):
        record = self.resolve_public("(1+x)*("+self.parabola+")")
        self.assertEqual(record["monodromy_zeta"]["expression"], "1/(1-T^2)")
        self.assertEqual(monodromy_zeta(record)["expression"], "1/(1-T^2)")
        for polynomial in ("(y-x^40)^2-x*z^2", str(2**300 + 1)+"*("+self.parabola+")"):
            self.assertIsNone(self.proposal(polynomial))
        self.assertIsNone(smooth_reduced_graph(parse(self.parabola, self.names), self.names, "embedded", seconds=0))

    def test_only_owned_origin_fibre_is_reported(self):
        # The smooth disconnected global centre consists of the lines y=0 and y=1.
        record = self.proposal("x^2-z*(y*(y-1))^2")
        self.assertEqual(record["dual_complex"]["vertices"], [{"id": "E1", "dimension": 2}])
        x, y, z = np.symbols("x y z")
        for chart in record["charts"]:
            # Factor ideals are equations in the base for these rational lines.
            self.assertEqual(len(chart["exceptional"]), 1)
            piece = chart["exceptional"][0]["ideal"]
            rows = [np.from_terms((x, y, z), parse(f, self.names)) for f in piece]
            self.assertEqual(ideal_basis((x, y, z), rows), ideal_basis((x, y, z), (x, y)))


if __name__ == "__main__":
    unittest.main()
