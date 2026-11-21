"""The strata of the total transform and the topological zeta function, for every kind of resolution."""
import unittest

import sympy as sp

from engine.general_resolution import verify_candidate
from engine.monodromy import monodromy_zeta
from engine.newton_backend import resolve_newton
from engine.polynomial import InputError, format_poly, integer_text, parse
from engine.resolution_search import ResolutionCandidate
from engine.resolve import resolve
from engine.strata import InvalidChart, InvalidZetaCertificate, resolution_strata
from tests.strata_oracle import _power_factors
from engine.topological import _reduce, topological_zeta


def answer(polynomial, names, mode="embedded"):
    return resolve({"polynomial": polynomial, "variables": list(names), "mode": mode, "search": "quick",
                    "monodromy": True})


def newton(polynomial, names):
    """Both zeta functions from the Newton modification, whatever engine resolve would pick."""
    names = list(names)
    p = parse(polynomial, names)
    record = resolve_newton(p, names, "embedded")
    record.update(variables=names, polynomial=format_poly(p, names), mode="embedded", origin=[0]*len(names))
    zeta = monodromy_zeta(record)
    return topological_zeta(record, monodromy=zeta)["expression"], zeta["expression"]


class TopologicalZeta(unittest.TestCase):
    def test_arbitrary_precision_denominator_survives_python_decimal_limit(self):
        exponent = 10**5000 + 7
        digits = integer_text(exponent)
        value, numerator, _, poles = _reduce({"S": (exponent, 1)},
                                             [{"components": ["S"], "euler_characteristic": 1}])
        self.assertEqual(value, f"1/({digits}*s + 1)")
        self.assertEqual(numerator, "1")
        self.assertEqual(poles, [{"value": "-1/" + digits, "order": 1, "components": ["S"]}])

    def test_stored_factors_are_recomputed_and_conflicting_explicit_factors_are_rejected(self):
        record = answer("y^2-x^3", "xy")
        expected = record["topological_zeta"]
        record["monodromy_zeta"] = dict(record["monodromy_zeta"], expression="1", factors=[])
        self.assertEqual(topological_zeta(record), expected)
        with self.assertRaisesRegex(InvalidZetaCertificate, "monodromy factors"):
            topological_zeta(record, monodromy=record["monodromy_zeta"])
        with self.assertRaisesRegex(InvalidChart, "polynomial"):
            monodromy_zeta(record, polynomial=parse("x*y", list("xy")))

    def test_legacy_numerical_table_needs_geometric_evidence(self):
        # This table even satisfies Z(0)=1. Without charts it does not
        # establish the claimed order 17 for the cusp.
        record = {"status": "resolved", "mode": "embedded", "variables": ["x", "y"],
                  "polynomial": "y^2-x^3", "certificate": {
                      "numerical_data": [{"component": "S", "multiplicity": 17, "nu": 1}],
                      "strata": [{"components": ["S"], "euler_characteristic": 1}]}}
        self.assertEqual(monodromy_zeta(record)["expression"], "(1-T^6)/((1-T^2)*(1-T^3))")
        self.assertEqual(topological_zeta(record)["expression"], "(4*s + 5)/((s + 1)*(6*s + 5))")

    def test_repeated_and_cancelled_poles(self):
        numbers = {"unused": (6, 3), "C": (3, 1), "B": (4, 2),
                   "D": (6, 2), "A": (2, 1), "S": (1, 1)}
        # C and D cancel; A and B contribute a double pole. An unused
        # component with the same candidate pole retains its input position.
        terms = [(2, ["A", "B"]), (1, ["S"]), (1, ["C"]),
                 (-1, ["A"]), (-2, ["D"]), (0, ["C", "S"])]
        rows = [dict(components=names, euler_characteristic=chi) for chi, names in terms]
        expected = ("(2*s^2 + 2*s + 1)/((s + 1)*(2*s + 1)^2)", "2*s^2 + 2*s + 1",
                    "(s + 1)*(2*s + 1)^2",
                    [{"value": "-1", "order": 1, "components": ["S"]},
                     {"value": "-1/2", "order": 2, "components": ["unused", "B", "A"]}])
        self.assertEqual(_reduce(numbers, rows), expected)
        self.assertEqual(_reduce(numbers, list(reversed(rows))), expected)

    def test_constant_terms_and_origin_validation(self):
        for weights, expected in (([], None), ([0], None), ([1, -1], None),
                                  ([2], None), ([3, -2], ("1", "1", "1", []))):
            with self.subTest(weights=weights):
                rows = [dict(components=[], euler_characteristic=chi) for chi in weights]
                self.assertEqual(_reduce({}, rows), expected)

    def test_zero_terms_validate_component_indices(self):
        rows = [dict(components=[], euler_characteristic=1),
                dict(components=["A", "A"], euler_characteristic=0)]
        with self.assertRaisesRegex(InputError, "Invalid component in stratum"):
            _reduce({"A": (1, 1), "B": (2, 1)}, rows)

    def test_every_engine(self):
        expected = {
            ("y^2-x^3", "xy"): ("C++ / FLINT plane point blowups", "(4*s + 5)/((s + 1)*(6*s + 5))"),
            ("x*y", "xy"): ("normal-crossings-at-origin", "(s + 1)^(-2)"),
            ("x^2+y^2+z^2", "xyz"): ("native-a-series", "(s + 3)/((s + 1)*(2*s + 3))"),
            ("x*y-z^3", "xyz"): ("native-a-series", "(s + 4)/((s + 1)*(3*s + 4))"),
            ("x*y-z^4", "xyz"): ("native-a-series", "(s + 5)/((s + 1)*(4*s + 5))"),
            ("z^3+y^4+x^7", "xyz"): ("native-toric", "(48*s + 61)/((s + 1)*(84*s + 61))"),
            ("x^2*y-z^3", "xyz"): ("binomial-coordinate-blowups", "(2*s + 5)/((s + 1)*(6*s + 5))"),
            ("x^3+y^3+z^3+w^3", "xyzw"): ("native-ordinary", "-(5*s - 4)/((s + 1)*(3*s + 4))"),
            ("x^2+y^3+z^7+w^2", "xyzw"): ("newton-polyhedron", "(9*s + 31)/((s + 1)*(21*s + 31))"),
            ("x+y^2+z^5", "xyz"): ("native-ordinary", "1/(s + 1)"),
        }
        for (polynomial, names), (engine, value) in expected.items():
            with self.subTest(polynomial=polynomial):
                result = answer(polynomial, names)
                self.assertEqual(result["engine"], engine)
                self.assertEqual(result["topological_zeta"]["expression"], value)

    def test_poles_are_those_of_the_candidates(self):
        zeta = answer("z^3+y^4+x^7", "xyz")["topological_zeta"]
        self.assertEqual([(pole["value"], pole["order"]) for pole in zeta["poles"]], [("-1", 1), ("-61/84", 1)])
        self.assertTrue(all(pole["in_monodromy_zeta_at_origin"] for pole in zeta["poles"]))

    def test_intrinsic_mode_uses_an_embedded_resolution(self):
        zeta = answer("y^2-x^3", "xy", "intrinsic")["topological_zeta"]
        self.assertEqual(zeta["expression"], "(4*s + 5)/((s + 1)*(6*s + 5))")
        self.assertEqual(zeta["source"], "companion embedded resolution")

    def test_the_blowup_and_the_identity_on_a_plane_crossing(self):
        for mode, steps in (("intrinsic", 1), ("embedded", 0)):
            with self.subTest(mode=mode):
                result = answer("x*y", "xy", mode)
                self.assertEqual(len(result["steps"]), steps)
                self.assertEqual(result["topological_zeta"]["expression"], "(s + 1)^(-2)")
                self.assertEqual(result["monodromy_zeta"]["expression"], "1")

    def test_identity_replays_transverse_plane_factors_with_their_weights(self):
        result = answer("(1+x)*(y+x^2)^2*(x+y)^3", "xy")
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["dual_complex"]["vertices"], [])
        chart, = result["atlas"]
        self.assertEqual(sorted(part["multiplicity"] for part in chart["components"]), [2, 3])
        self.assertEqual(topological_zeta(result)["expression"], "1/((2*s + 1)*(3*s + 1))")
        self.assertEqual(monodromy_zeta(result)["expression"], "1")

    def test_conjugate_plane_branches_have_a_verified_marked_identity_chart(self):
        from engine.resolution_complex import dual_complex
        from engine.session_values import Resolution
        f = "(1+a)*((a-y^2)^2+2*(a-y^2)*y+2*y^2)^3"
        result = answer(f, "ay")
        self.assertEqual(result["steps"], [])
        chart, = Resolution(result).charts
        a, y = chart.coordinates
        reduced = (a-y**2)**2+2*(a-y**2)*y+2*y**2
        self.assertEqual(chart.strict_transform, (1+a)*reduced**3)
        left, right = chart.session_attribute("strict_transform_components")
        self.assertEqual((left["multiplicity"], right["multiplicity"]), (3, 3))
        self.assertEqual(left["ideal"][0]*right["ideal"][0], reduced)
        self.assertEqual(dual_complex(result)["vertices"], [])
        self.assertEqual(topological_zeta(result)["expression"], "(3*s + 1)^(-2)")
        self.assertEqual(monodromy_zeta(result)["expression"], "1")
        owned, = result["atlas"]
        self.assertEqual(owned["coefficient_field"]["degree"], 2)
        owned["components"][0]["ideal"] = owned["components"][1]["ideal"]
        with self.assertRaises(InvalidChart):
            topological_zeta(result)

    def test_conjugate_cylinders_preserve_free_axes_and_mark(self):
        from engine.session_values import Resolution
        for mode, steps in (("embedded", 0), ("intrinsic", 1)):
            with self.subTest(mode=mode):
                result = answer("x^2+y^2", "xzy", mode)
                self.assertEqual(len(result["steps"]), steps)
                self.assertEqual(topological_zeta(result)["expression"], "(s + 1)^(-2)")
        for names in (["x", "a", "z", "b"], ["c", "b", "e", "q", "xx", "f", "gg", "a"]):
            with self.subTest(names=names):
                result = answer("(1+a)*((a-b^2)^2+b^2)^2", names)
                self.assertEqual(result["steps"], [])
                chart, = Resolution(result).charts
                coordinates = dict(zip(names, chart.coordinates))
                a, b = coordinates["a"], coordinates["b"]
                reduced = (a-b**2)**2+b**2
                self.assertEqual(chart.strict_transform, (1+a)*reduced**2)
                left, right = chart.session_attribute("strict_transform_components")
                self.assertEqual((left["multiplicity"], right["multiplicity"]), (2, 2))
                self.assertEqual(left["ideal"][0]*right["ideal"][0], reduced)
                owned, = result["atlas"]
                generator = owned["coefficient_field"]["generator"]
                self.assertNotIn(generator, names)
                self.assertEqual(owned["coordinates"], [*names, generator])
                self.assertEqual(owned["fibre"], names)
                self.assertEqual(owned["map"], dict(zip(names, names)))
                self.assertEqual(owned["split_cone"], [[int(i == j) for j in names] for i in names])
                self.assertEqual(topological_zeta(result)["expression"], "(2*s + 1)^(-2)")
                self.assertEqual(monodromy_zeta(result)["expression"], "1")
                owned["fibre"].remove(names[0])  # even a free coordinate belongs to the origin fibre
                with self.assertRaises(InvalidChart):
                    topological_zeta(result)

    def test_independent_of_the_resolution(self):
        # The same germ through its own engine and through the Newton modification.
        for polynomial, names in (("y^2-x^4", "xy"), ("x^5+y^5+x^2*y^2", "xy"), ("x^3*y+y^4+x^6", "xy"),
                                  ("x*y+z^5", "xyz"), ("x*y+z^6", "xyz"), ("x^3+y^3+z^3", "xyz"),
                                  ("x^2+y^2+z^2+w^2", "xyzw"), ("x^2+y^3+z^5", "xyz")):
            with self.subTest(polynomial=polynomial):
                result = answer(polynomial, names)
                self.assertNotEqual(result["engine"], "newton-polyhedron")
                self.assertEqual((result["topological_zeta"]["expression"], result["monodromy_zeta"]["expression"]),
                                 newton(polynomial, names))

    def test_divisors_with_multiplicities(self):
        plane = answer("(y^2-x^3)^2*x*(x-y)^3", "xy")["topological_zeta"]
        self.assertEqual([(row["component"], row["multiplicity"], row["nu"]) for row in plane["numerical_data"]],
                         [("E1", 8, 2), ("E2", 10, 3), ("E3", 20, 5), ("S1", 1, 1), ("S2", 3, 1), ("S3", 2, 1)])
        self.assertEqual(plane["expression"],
                         "-(36*s^4 - 20*s^3 - 72*s^2 - 36*s - 5)/(5*(s + 1)*(2*s + 1)*(3*s + 1)*(4*s + 1)^2)")
        # Conjugate points of the strict transform, with factors of different multiplicities.
        conjugate = answer("(x^3-2*y^3)^2*(x-y)*(y^2-x^3+x^4)^3", "xy")["topological_zeta"]
        self.assertEqual(conjugate["status"], "computed")
        self.assertEqual(answer("(x^2+y^2+z^2)^3", "xyz")["topological_zeta"]["expression"],
                         "(s + 1)/((2*s + 1)*(3*s + 1))")
        self.assertEqual(answer("(z^3+y^4+x^7)^2", "xyz")["topological_zeta"]["expression"],
                         "(96*s + 61)/((2*s + 1)*(168*s + 61))")

    def test_blowup_of_an_ideal(self):
        x, y, z = sp.symbols("x y z")
        u, v, w = sp.symbols("U V W")

        def certified(variables, f, centre, coordinates, boundary):
            candidate = ResolutionCandidate(centre, coordinates, boundary, sp.QQ, None, "test certificate")
            record = verify_candidate(variables, f, candidate, "embedded")
            record.update(variables=[str(s) for s in variables], polynomial=str(sp.expand(f)).replace("**", "^"),
                          mode="embedded", origin=[0]*len(variables))
            return topological_zeta(record, monodromy=monodromy_zeta(record))

        # The Whitney umbrella with its double curve moved to a parabola: nu = 2 for a curve.
        h = y - x**2
        umbrella = certified((x, y, z), h*h - x*z*z, (h, z), (u, v), ((h, z),))
        self.assertEqual([(row["component"], row["multiplicity"], row["nu"]) for row in umbrella["numerical_data"]],
                         [("E1", 2, 2), ("strict", 1, 1)])
        self.assertEqual(umbrella["expression"], answer("x^2-y*z^2", "xyz")["topological_zeta"]["expression"])
        # A nonradical ideal that encodes two point blowups: nu comes from the Jacobian of the projection.
        tacnode = certified((x, y), y**2 - x**4, (x**3, x*y, y**2), (u, v, w), ((x, y, u), (x, y, w)))
        self.assertEqual([(row["component"], row["multiplicity"], row["nu"]) for row in tacnode["numerical_data"]],
                         [("E1", 2, 2), ("E2", 4, 3), ("strict", 1, 1)])
        self.assertEqual(tacnode["expression"], answer("y^2-x^4", "xy")["topological_zeta"]["expression"])

    def test_degenerate_face_completed_by_blowups(self):
        result = answer("z^3+y^4+(1/432)*x^6+x*y^2*z+x^3*y*z+x*y*z^2", "xyz")
        self.assertEqual(result["monodromy_zeta"]["expression"], "(1-T^6)^2/((1-T^3)*(1-T^4)*(1-T^12)^2*(1-T^13))")
        self.assertEqual(result["monodromy_zeta"]["milnor_fiber_euler_characteristic"], 32)   # mu = 31
        self.assertEqual(result["topological_zeta"]["expression"],
                         "(77*s^2 + 175*s + 90)/(3*(s + 1)*(4*s + 3)*(13*s + 10))")
        created = [row for row in result["topological_zeta"]["numerical_data"] if row["component"].startswith("P")]
        self.assertEqual([(row["multiplicity"], row["nu"]) for row in created], [(13, 11), (26, 20)])

    def test_zeta_uses_the_atlas_independently_of_display_metadata(self):
        record = answer("y^2-x^3", "xy")
        data, strata, _ = resolution_strata(record)
        self.assertEqual(sum(sp.Rational(row["euler_characteristic"])
                             / sp.prod([next(d["nu"] for d in data if d["component"] == c) for c in row["components"]])
                             for row in strata), 1)
        record["dual_complex"]["vertices"][0]["self_intersection"] = -4
        self.assertEqual(topological_zeta(record)["expression"], record["topological_zeta"]["expression"])
        # Damaging the actual strict transform invalidates the normal-crossings certificate.
        for chart in record["atlas"]:
            for component in chart["components"]:
                if component["id"] == "strict":
                    component["ideal"] = [f"({component['ideal'][0]})^2"]
        with self.assertRaises(InvalidChart):
            topological_zeta(record)

    def test_power_products(self):
        self.assertEqual(_power_factors("(1)*(1 - v)^3*(u - v^2)^2"), [("1", 1), ("1 - v", 3), ("u - v^2", 2)])
        self.assertIsNone(_power_factors("u - v"))


if __name__ == "__main__":
    unittest.main()
