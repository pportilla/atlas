import json
from copy import deepcopy
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import sympy as sp

from engine.general_resolution import verify_candidate
from engine.monodromy import _transverse_compositions, monodromy_zeta
from engine.polynomial import InputError, parse
from engine.resolution_search import ResolutionCandidate
from engine.resolve import resolve


ROOT = Path(__file__).resolve().parents[1]


class MonodromyTests(unittest.TestCase):
    def test_isolated_curve_has_only_the_origin_zeta(self):
        record = resolve({"polynomial": "y^2-x^3", "variables": ["x", "y"], "search": "quick",
                          "zetas": ["monodromy"]})
        self.assertEqual(record["monodromy_zeta"]["nearby_strata"], [])
        self.assertNotIn("topological_zeta", record)

    def test_transverse_node_has_zeta_one_and_supplies_no_zeta_matches(self):
        for polynomial, model in (("x^2-z*y^2", "X^a + Y^b"), ("x^2+z*y^2+y^3", "sum of squares")):
            with self.subTest(polynomial=polynomial):
                record = resolve({"polynomial": polynomial, "variables": list("xyz"), "search": "quick",
                                  "zetas": ["monodromy", "topological"]})
                nearby = record["monodromy_zeta"]["nearby_strata"]
                index, row = next((i, row) for i, row in enumerate(nearby) if row["model"] == model)
                self.assertEqual((set(row["coordinates"]), row["parameters"]), ({"x", "y"}, ["z"]))
                self.assertEqual((row["expression"], row["factors"]), ("1", []))
                if model == "sum of squares":
                    self.assertEqual((row["transverse_hessian_determinant"], row["parameter_values"]), ("4", [1]))
                for pole in record["monodromy_check"]["poles"]:
                    self.assertNotIn(index, [match["stratum_index"] for match in pole["nearby_matches"]])

    def test_cusp_cylinder_cli_replays_nearby_zetas_from_original_geometry(self):
        record = resolve({"polynomial": "x^2+y^3+y^4", "variables": list("xyz"), "search": "quick",
                          "zetas": ["monodromy"]})
        zeta = record["monodromy_zeta"]
        cylinder, = [row for row in zeta["nearby_strata"] if row["model"] == "constant transverse type"]
        self.assertEqual((cylinder["coordinates"], cylinder["parameters"]), (["x", "y"], ["z"]))
        self.assertEqual(cylinder["expression"], "(1-T^6)/((1-T^2)*(1-T^3))")
        self.assertEqual(cylinder["factors"], zeta["factors"])
        record["monodromy_zeta"] = {**zeta, "nearby_strata": [{"model": "stale", "expression": "1"}]}
        output = subprocess.run([ROOT / "atlas", "monodromy", "-", "--json"], input=json.dumps(record),
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(output.stdout), zeta)

    def test_repeated_branch_equations_follow_the_checked_chart_projection(self):
        names = list("xyzw")
        record = resolve({"polynomial": "(x+y^2)^6*(z+w^2)^4", "variables": names,
                          "search": "quick", "zetas": ["monodromy"]})
        nearby = record["monodromy_zeta"]["nearby_strata"]
        branches = {s["component"]: (s["equation"], s["expression"]) for s in nearby if s["model"] == "smooth component"}
        self.assertEqual(branches, {"S1": ("w^2+z", "1/(1-T^4)"), "S2": ("y^2+x", "1/(1-T^6)")})

        # Reverse the chart coordinates, preserving the actual projection and divisor.
        chart, = record["atlas"]
        chart["coordinates"] = chart["fibre"] = names[::-1]
        chart["map"] = dict(zip(names, names[::-1]))
        for row in chart["components"]:
            row["ideal"] = ["y+x^2" if row["id"] == "S1" else "w+z^2"]
        record["certificate"]["components"][0]["equation"] = "stale"
        self.assertEqual(monodromy_zeta(record)["nearby_strata"], nearby)

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
        self.assertEqual(monodromy_zeta(intrinsic)["expression"], "(1-T^6)/((1-T^2)*(1-T^3))")
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

    def test_equal_multiplicity_coordinate_branches_each_appear_once(self):
        record = resolve({"polynomial": "x^2*y^2", "variables": ["x", "y"], "search": "quick",
                          "mode": "embedded", "zetas": ["monodromy"]})
        nearby = record["monodromy_zeta"]["nearby_strata"]
        self.assertCountEqual([(row["model"], row["component"], row["expression"]) for row in nearby],
                              [("smooth component", label, "1/(1-T^2)") for label in ("S1", "S2")])

    def test_quadric_composition_witnesses_the_missing_order_ten_eigenvalue(self):
        # The singular lines x = y^2+z^2 = 0 have no rational point except the origin.
        record = resolve({"polynomial": "(x^2+y^2+z^2)^2+x^5", "variables": list("xyz"), "search": "quick",
                          "zetas": ["monodromy", "topological"]})
        zeta = record["monodromy_zeta"]
        self.assertEqual(zeta["expression"], "1/(1-T^4)")
        (index, row), = [(i, r) for i, r in enumerate(zeta["nearby_strata"]) if r["model"] == "transverse composition"]
        self.assertEqual((row["coordinates"], row["transverse_coordinates"], row["transverse_germ"], row["dimension"]),
                         (["x^2+y^2+z^2", "x"], ["u1", "u2"], "u2^5+u1^2", 1))
        self.assertEqual(row["expression"], "(1-T^10)/((1-T^2)*(1-T^5))")
        check = record["monodromy_check"]
        self.assertEqual(check["status"], "verified")
        pole, = [p for p in check["poles"] if p["value"] == "-7/10"]
        self.assertEqual((pole["eigenvalue_order"], pole["cyclotomic_exponent"], pole["witness"]),
                         (10, 0, "nearby transverse composition"))
        self.assertEqual(pole["nearby_matches"], [{"stratum_index": index, "cyclotomic_exponent": 1}])
        self.assertEqual(pole["transverse"]["transverse_germ"], "u2^5+u1^2")
        # Replaying the saved record recomputes the strata from the original polynomial.
        output = subprocess.run([ROOT / "atlas", "monodromy", "-", "--json"], input=json.dumps(record),
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(output.stdout), zeta)

    def test_transverse_compositions_certify_factorizations_through_submersions(self):
        names = list("xyz")
        cases = {
            "(x^2+y^2+z^2)^3+(x+y)^4": (["x+y", "x^2+y^2+z^2"], "u1^4+u2^3", "(1-T^12)/((1-T^3)*(1-T^4))"),
            "(x+y*z)^2+x^3": (["x", "y*z+x"], "u1^3+u2^2", "(1-T^6)/((1-T^2)*(1-T^3))"),
            "(x^2+y^2+z^2)^2+123456789012345678901234567890*x^3":
                (["x^2+y^2+z^2", "x"], "123456789012345678901234567890*u2^3+u1^2", "(1-T^6)/((1-T^2)*(1-T^3))"),
            # This G is not of Brieskorn-Pham type, so Atlas resolves it at the origin.
            "(x^2-y^3)^2+z^7*(x^2-y^3)+z^20":
                (["z^7-2*y^3+2*x^2", "z"], "u2^20-1/4*u2^14+1/4*u1^2", "(1-T^14)/(1-T^2)"),
        }
        for polynomial, expected in cases.items():
            with self.subTest(polynomial=polynomial):
                row, = _transverse_compositions(parse(polynomial, names), names)
                self.assertEqual((row["coordinates"], row["transverse_germ"], row["expression"]), expected)
        # Names for G avoid the input coordinates.
        renamed = ["u1", "u2", "v"]
        row, = _transverse_compositions(parse("(u1^2+u2^2+v^2)^2+u1^5", renamed), renamed)
        self.assertEqual((row["coordinates"], row["transverse_coordinates"], row["transverse_germ"]),
                         (["u1^2+u2^2+v^2", "u1"], ["v1", "v2"], "v2^5+v1^2"))
        # Isolated singularities, coordinate models, inputs without a factorization
        # through a submersion, and inputs beyond the search bounds give no row.
        for polynomial in ("x^5+y^7+z^9+x^2*y^2*z^2", "x^2+y^3+y^4", "(x*y-z^2)^2+x^5",
                           "(x^2+y^2+z^2)^2+x^33", "x^9007199254740993*y^4+z^2*y^4"):
            with self.subTest(polynomial=polynomial):
                terms = parse(polynomial, names, bounded=False)
                self.assertEqual(_transverse_compositions(terms, names), [])

    def test_inconclusive_input_and_option_validation(self):
        with self.assertRaises(InputError):
            monodromy_zeta({"status": "inconclusive"})
        with self.assertRaises(InputError):
            resolve({"polynomial":"x", "variables":["x"], "monodromy":"false"})


if __name__ == "__main__":
    unittest.main()
