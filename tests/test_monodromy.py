import json
from copy import deepcopy
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import sympy as sp

from engine.general_resolution import verify_candidate
from engine.monodromy import monodromy_zeta
from engine.polynomial import InputError
from engine.resolution_search import ResolutionCandidate
from engine.resolve import resolve


ROOT = Path(__file__).resolve().parents[1]


class MonodromyTests(unittest.TestCase):
    def test_acampo_published_plane_curve(self):
        # A'Campo (1975), Example 1. Multiple exceptional curves are punctured
        # by different geometric branches. The fiber has Milnor number 34.
        r = resolve({"polynomial":"(x^2+y^3)*(x^2*y^2+x^6+y^6)", "variables":["x","y"],
                     "monodromy":True})
        zeta = r["monodromy_zeta"]
        self.assertEqual(zeta["expression"], "(1-T^8)*(1-T^9)*(1-T^16)")
        self.assertEqual(zeta["milnor_fiber_euler_characteristic"], 1-34)

    def test_saved_binomial_cli_and_factor_sign(self):
        saved = ROOT / "examples/xy2z3-t6-embedded.json"
        run = subprocess.run([ROOT/"atlas", "monodromy", saved, "--json"], capture_output=True,
                             text=True, check=True)
        zeta = json.loads(run.stdout)
        self.assertEqual(zeta["expression"], "1/(1-T^6)")
        self.assertEqual(zeta["factors"], [{"multiplicity":6, "exponent":-1}])
        self.assertEqual(zeta["milnor_fiber_euler_characteristic"], 6)

    def test_intrinsic_cusp_needs_embedded_contact_data(self):
        req = {"polynomial":"y^2-x^3", "variables":["x","y"], "mode":"intrinsic"}
        intrinsic = resolve(req)
        self.assertEqual(monodromy_zeta(intrinsic)["status"], "unavailable")
        r = resolve(dict(req, monodromy=True))
        self.assertEqual(len(r["steps"]), len(intrinsic["steps"]))
        self.assertEqual(r["monodromy_zeta"]["source"], "companion embedded resolution")
        self.assertEqual(r["monodromy_zeta"]["expression"], "(1-T^6)/((1-T^2)*(1-T^3))")
        self.assertEqual(r["monodromy_zeta"]["milnor_fiber_euler_characteristic"], -1)

    def test_general_projective_graph_matches_plane_blowups(self):
        x,y = sp.symbols("x y")
        u,v,w = sp.symbols("U V W")
        f = y**2-x**4
        center = (x**3,x*y,y**2)
        candidate = ResolutionCandidate(center, (u,v,w), ((x,y,u),(x,y,w)), sp.QQ, None, "test")
        record = verify_candidate((x,y), f, candidate, "embedded")
        req = {"variables":["x","y"], "polynomial":str(f), "mode":"embedded"}
        graph = monodromy_zeta(record | req)
        plane = resolve(req | {"monodromy":True})["monodromy_zeta"]
        self.assertEqual(graph["factors"], plane["factors"])
        self.assertEqual(graph["expression"], "(1-T^4)/(1-T^2)")

    def test_graph_over_curve_and_coefficient_field(self):
        x,y,z = sp.symbols("x y z")
        u,v = sp.symbols("U V")
        a = sp.sqrt(2)
        center = (x,y)
        candidate = ResolutionCandidate(center, (u,v), ((x+a*y,y),), sp.QQ.algebraic_field(a), a, "test")
        record = verify_candidate((x,y,z), x**2-y**2*z, candidate, "embedded", effort=3)
        req = {"variables":list("xyz"), "polynomial":"x^2-y^2*z", "mode":"embedded"}
        graph = monodromy_zeta(record | req)
        self.assertEqual(graph["expression"], "1/(1-T^2)")
        self.assertEqual(graph["contributions"][0]["euler_characteristic"], 1)
        # Old records have the same marked root but omit its polynomial.
        del record["coefficient_field"]["minimal_polynomial_coefficients"]
        archived = monodromy_zeta(record | req)
        self.assertEqual(archived["expression"], graph["expression"])
        self.assertEqual(archived["contributions"], graph["contributions"])

    def test_coordinate_graph_replay_and_unavailable_sparse_kernel(self):
        from engine import native_library
        from engine.graph_record import graph_outputs
        from engine.incidence import incidence
        from engine.polynomial import parse
        from engine.strata import InvalidChart
        if native_library._MODE != "native":
            self.skipTest("Sparse graph replay runs in native mode.")
        x, y, z = sp.symbols("x y z")
        u, v = sp.symbols("U V")
        centre = (x, y)
        candidate = ResolutionCandidate(centre, (u, v), (centre,), sp.QQ, None, "test")
        equation = (x+y)**2 - z*(x-y)**2
        record = verify_candidate((x,y,z), equation, candidate, "embedded")
        names = ["x", "y", "z"]
        record.update(variables=names, polynomial=str(equation), mode="embedded")
        poly = parse(str(equation), names)
        expected = graph_outputs(record, poly)
        atlas, outputs = expected
        self.assertEqual(outputs["dual_complex"]["vertices"], [{"id":"E1", "dimension":2}])
        self.assertEqual([row["euler_characteristic"] for row in outputs["strata"]], [1, 1])
        saved = {**record, "atlas":atlas}

        for failure in ({"return_value":None}, {"side_effect":native_library.Unavailable("Exponent out of range.")}):
            with (self.subTest(failure=failure), patch("engine.chart_kernel.crossings_rows", **failure),
                  patch("engine.general_atlas.native_general_outputs", return_value=outputs) as general):
                self.assertEqual(graph_outputs(saved, poly), expected)
                general.assert_called_once()
        self.assertIsNone(incidence("coordinate_exceptional", variables=names, centre=["x", "x"],
                                    charts=[{"exceptional":[{"id":"E1", "ideal":["x"]}]}]*2))
        for change in ("region", "order", "pivot"):
            broken = deepcopy(saved)
            if change == "region":
                broken["atlas"][1]["region"] = []
            elif change == "order":
                broken["atlas"][0]["components"][0]["nu"] = 17
            else:
                broken["charts"][1] = deepcopy(broken["charts"][0])
            with self.subTest(change=change), self.assertRaises(InvalidChart):
                graph_outputs(broken, poly)


    def test_archived_graph_reconstructs_its_marked_field(self):
        from tests.graph_monodromy_oracle import _archived_field, _field_coefficients
        from engine.polynomial import InputError
        x = sp.Symbol("x")
        for text, root, coefficients in (
                ("sqrt(2)", sp.sqrt(2), (1, 0, -2)),
                ("-sqrt(1/2)", -sp.sqrt(2)/2, (2, 0, -1)),
                ("sqrt(9/4)", sp.Rational(3,2), (2, -3)),
                ("-I", -sp.I, (1, 0, 1)),
                ("CRootOf(T**4 - T - 1, 3)", sp.CRootOf(x**4-x-1,3), (1,0,0,-1,-1)),
                ("CRootOf((x**2-2)^2, 2)", sp.CRootOf(x**2-2,1), (1,0,-2)),
                ("CRootOf((x-1)*(x**2-2), 2)", sp.CRootOf(x**2-2,1), (1,0,-2))):
            with self.subTest(text=text):
                self.assertEqual(_field_coefficients({"exact_root": text}), coefficients)
                field, marked = _archived_field(text)
                # The bound variable can differ between serialized polynomials.
                if root.is_Rational:
                    self.assertEqual(marked, root)
                else:
                    self.assertTrue(field.ext.minpoly.same_root(marked, root))
        for text in ("CRootOf(x^2-2, 2)", "object.attribute(1)"):
            with self.assertRaises(InputError):
                _archived_field(text)
        self.assertEqual(_field_coefficients({"exact_root": "sqrt(2) + sqrt(3)"}), (1, 0, -10, 0, 1))

    def test_inconclusive_input_and_option_validation(self):
        self.assertEqual(monodromy_zeta({"status":"inconclusive"})["status"], "unavailable")
        with self.assertRaises(InputError):
            resolve({"polynomial":"x", "variables":["x"], "monodromy":"false"})


if __name__ == "__main__":
    unittest.main()
