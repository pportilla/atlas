"""Global characters and faces on a certified common torus atlas."""
from collections import Counter
from copy import deepcopy
from itertools import combinations
import json
import unittest

import sympy as sp

from engine import atlas
from engine.atlas_torus import _orbit, torus_complex
from engine.local_blowups import Chart
from engine.newton_backend import resolve_newton
from engine.polynomial import parse
from engine.resolve import resolve
from engine.toric_backend import resolve_toric


class SurfaceSectionFallback(unittest.TestCase):
    def test_relative_fan_prunes_only_curves_missing_the_origin(self):
        from engine.resolution_complex import dual_complex
        from engine.strata import InvalidChart

        # Independent affine factorization over Q(i) and fibre Groebner
        # bases in benchmarks/relative_fan_sections.py give these graphs.
        cases = (("x^4+y^5+z*x^2*y^2", 4, 3, 7),
                 ("z^4+x^2*y^2*(x^3+y^4)", 18, 17, 13),
                 ("y*(x^2+y^3+y*z^3)", 15, 13, 4))
        for f, vertices, edges, invisible in cases:
            with self.subTest(f=f):
                record = resolve(dict(variables=list("xyz"), polynomial=f, mode="intrinsic",
                                      compute="cpu", search="quick"))
                self.assertEqual(record["status"], "resolved")
                answer = record["dual_complex"]
                self.assertEqual((len(answer["vertices"]), len(answer["cells"])), (vertices, edges))
                public = record["charts"]
                visible = {v["ambient_divisor"] for c in public for v in c["exceptional"]}
                all_labels = {v["id"] for c in public for v in c["ambient_components"]}
                self.assertEqual(len(all_labels - visible), invisible)
                for owned in (True, False):
                    saved = deepcopy(record)
                    if not owned:
                        del saved["atlas"]
                    self.assertEqual(dual_complex(saved), answer)
                    chart = next(c for c in saved["charts"] if any(
                        a["id"] not in visible for a in c["ambient_components"]))
                    phantom = next(a for a in chart["ambient_components"] if a["id"] not in visible)
                    chart["exceptional"].append(dict(id=phantom["id"], ambient_divisor=phantom["id"],
                                                     ideal=[phantom["equation"], "1"], equation="1"))
                    with self.assertRaises(InvalidChart):
                        dual_complex(saved)

    def test_owned_curves_and_their_attachments_survive_chart_reordering(self):
        from engine.fan_surface import surface_complex
        from engine.polynomial import parse_expression

        rays = ((1, 1, 1), (0, 1, 0), (0, 0, 1))
        # The normalized Laurent equation on the middle ray splits into the
        # two rational lines y=±1. Each meets the sole curve on the z divisor.
        terms = tuple(sorted(parse_expression("x+y^2-1", list("xyz")).items()))
        owned = ("1 1 0 0",)
        chart = (rays, {0: "A", 2: "B"}, terms, owned)
        other = (tuple(reversed(rays)), {2: "A", 0: "B"},
                 tuple((tuple(reversed(e)), c) for e, c in terms), ("1 0 0 0",))
        output = surface_complex([chart, other])
        self.assertEqual([v["id"] for v in output["vertices"]], ["A.1", "A.2", "B"])
        self.assertEqual([(c["id"], c["faces"]) for c in output["cells"]],
                         [("CA-B.1", ["A.1", "B"]), ("CA-B.2", ["A.2", "B"])])
        self.assertEqual(output, surface_complex([chart]))
        presented = surface_complex([chart], local_coordinates=[["u", "v", "w"]])
        sections, = presented.pop("chart_sections")
        self.assertEqual(presented, output)
        self.assertEqual({(s["id"], s["ambient_divisor"]) for s in sections},
                         {("A.1", "A"), ("A.2", "A"), ("B", "B")})
        self.assertEqual({tuple(s["ideal"]) for s in sections if s["ambient_divisor"] == "A"},
                         {("u", "v-1"), ("u", "v+1")})
        # An orbit curve lying in both exceptional divisors cannot carry
        # two independent labels. Leave this case to the general gluer.
        shared = tuple(sorted(parse_expression("x+y*z", list("xyz")).items()))
        self.assertIsNone(surface_complex([(rays, {0: "A", 1: "B"}, shared, owned)]))


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

    def test_coordinate_and_open_torus_components_have_distinct_attachments(self):
        from engine.atlas_torus import section_complex
        names = list("abuv")
        rays = ((1,1,1,1), (1,2,1,1), (0,0,1,0), (0,0,0,1))
        # On A the section splits into u=0 and u*v=1, which are disjoint.
        # B carries a single smooth graph. Each component on A meets it.
        terms = tuple(parse("a+u*(u*v-1)", names).items())
        charts = [(rays, {0:"A", 1:"B"}, terms)]
        self.assertIsNone(section_complex(charts, 4, True, common_source=True))
        result = section_complex(charts, 4, True, common_source=True, embedded_snc=True,
                                 local_coordinates=[names])
        self.assertEqual({c["id"]:c["ideal"] for c in result["chart_sections"][0]},
                         {"A.1":["a", "u*v-1"], "A.2":["a", "u"], "B":["b", "u^2*v+a-u"]})
        self.assertEqual([c["faces"] for c in result["cells"]], [["A.1","B"], ["A.2","B"]])

    def test_threefold_split_factors_multiply_to_every_affine_restriction(self):
        from engine.monomial_atlas import fan_outputs
        names = list("xyzt")
        f = "x^2+y^3+y*z^3+t^2"
        record = resolve(dict(variables=names, polynomial=f, mode="intrinsic", search="quick", compute="cpu"))
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["root_families"]["alpha1"]["polynomial"], "T^2+1")
        coordinates = sp.symbols("u1:5")
        constants = {str(x):x for x in coordinates}
        constants.update(alpha1_1=-sp.I, alpha1_2=sp.I)
        def poly(text):
            return sp.sympify(text.replace("^","**"), locals=constants)
        visible = set()
        for chart in record["charts"]:
            for ambient in chart["ambient_components"]:
                axis = poly(ambient["equation"])
                restriction = sp.Poly(poly(chart["strict_transform"]).subs(axis,0), *coordinates)
                factors = []
                for component in chart["exceptional"]:
                    if component["ambient_divisor"] == ambient["id"]:
                        visible.add(component["id"])
                        first, second = map(poly, component["ideal"])
                        self.assertEqual(first, axis)
                        factors.append(second)
                # Over Q(i), the product of the local geometric factors must
                # recover the entire affine section, including coordinate factors.
                product = sp.Poly(sp.prod(factors), *coordinates)
                self.assertEqual(product.monic(), restriction.monic())
        self.assertEqual(visible, {v["id"] for v in record["dual_complex"]["vertices"]})
        _, replay = fan_outputs(record, parse(f, names), intrinsic=True, integrals=False)
        self.assertEqual(replay["dual_complex"], record["dual_complex"])

    def test_nonprimitive_simplex_on_a_relative_threefold_fan(self):
        from engine.resolution_complex import dual_complex
        names = list("xyzt")
        equation = "x^2+y^2+z^3+t^2*z^2"
        record = resolve(dict(variables=names, polynomial=equation, mode="intrinsic", search="quick"))
        complex_ = record["dual_complex"]
        self.assertEqual(complex_["status"], "complete")
        # E1 has face x²+y²+z²t², irreducible over the whole torus.
        # E3 and E5 have face x²+y², with the same two geometric sheets.
        self.assertEqual({v["id"] for v in complex_["vertices"]},
                         {"E1", "E2", "E3.1", "E3.2", "E5.1", "E5.2"})
        self.assertEqual(Counter(c["dimension"] for c in complex_["cells"]), {1: 7, 2: 2})
        self.assertEqual({tuple(c["vertices"]) for c in complex_["cells"] if c["dimension"] == 2},
                         {("E2", "E3.1", "E5.1"), ("E2", "E3.2", "E5.2")})
        self.assertEqual(dual_complex(record), complex_)
        del record["atlas"]
        self.assertEqual(dual_complex(record), complex_)

    def test_fan_uses_original_equation_across_differently_named_charts(self):
        from engine.monomial_atlas import fan_outputs
        from engine.strata import InvalidChart
        variables = ["x", "y"]
        record = {
            "variables": variables, "fan": {"rays": [[1, 0], [0, 1], [1, 1]]},
            "certificate": {"linear_map_to_input": [[1, -1], [1, 1]], "adapted_polynomial": "1"},
            "charts": [
                {"coordinates": ["u", "v"], "map": {"x": "(u-1)*v", "y": "u*v+v"}, "ray_indices": [0, 2]},
                {"coordinates": ["p", "q"], "map": {"x": "p-p*q", "y": "p+p*q"}, "ray_indices": [2, 1]},
            ],
        }
        polynomial = parse("x^2-y^2", variables)
        charts, proved = fan_outputs(record, polynomial)
        self.assertEqual([c["components"][-1]["ideal"] for c in charts], [["-4*u"], ["-4*q"]])
        self.assertEqual(proved["numerical_data"][0], {"component": "E1", "multiplicity": 2, "nu": 2})
        saved = {"variables": variables, "atlas": json.loads(json.dumps(charts))}
        self.assertEqual(fan_outputs(saved, polynomial)[1], proved)
        saved["atlas"][1]["map"]["x"] += "+1"
        with self.assertRaises(InvalidChart):
            fan_outputs(saved, polynomial)

    def test_same_orbit_sections_do_not_certify_a_changed_strict_equation(self):
        from engine.monomial_atlas import fan_outputs
        from engine.strata import InvalidChart
        original = atlas.to_json(self.charts)
        self.assertIsNotNone(torus_complex(self.charts, 3, True))
        chart = original[-1]
        strict = next(c for c in chart["components"] if c["id"] == "strict")
        # The added term and its first derivatives vanish on every boundary
        # hyperplane. All orbit restrictions agree with the original chart.
        strict["ideal"][0] += "+(" + "*".join(chart["coordinates"]) + ")^2"
        self.assertIsNone(torus_complex(atlas.from_json(original), 3, True))
        with self.assertRaises(InvalidChart):
            fan_outputs(dict(variables=self.names, atlas=original),
                        parse("x^2+y^3+z^5", self.names), intrinsic=True)

    def test_projection_terms_follow_chart_order_and_certify_the_whole_map(self):
        from engine.monomial_atlas import fan_outputs
        from engine.strata import InvalidChart

        # The three charts of the blowup of the origin. The two degree-two
        # terms tie on total degree in every chart. Coordinate order differs
        # from alphabetical order, and the supplied expressions are factored.
        names = ["zeta", "alpha", "mu"]
        variables = ["x", "y", "z"]
        linear = [[-1, 2, -3], [0, -1, 1], [0, 0, 1]]
        prescribed = [
            ([3, 1, 2], ["zeta*(2*alpha-3*mu-1)", "zeta*(mu-alpha)", "zeta*mu"]),
            ([0, 3, 2], ["alpha*(2-zeta-3*mu)", "alpha*(mu-1)", "alpha*mu"]),
            ([0, 1, 3], ["(4*alpha-2*zeta-6)*mu/2", "(1-alpha)*mu", "mu"]),
        ]
        record = {
            "variables": variables,
            "fan": {"rays": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]},
            "certificate": {"linear_map_to_input": linear},
            "charts": [{"coordinates": names, "map": dict(zip(variables, images)), "ray_indices": cone}
                       for cone, images in prescribed],
        }
        expected = [
            ["2*zeta*alpha-3*zeta*mu-zeta", "-zeta*alpha+zeta*mu", "zeta*mu"],
            ["-zeta*alpha-3*alpha*mu+2*alpha", "alpha*mu-alpha", "alpha*mu"],
            ["-zeta*mu+2*alpha*mu-3*mu", "-alpha*mu+mu", "mu"],
        ]
        # The centre must also be admissible for a resolution. A smooth
        # projectivized quadric supplies a singular source for this map test.
        polynomial = parse("x^2+y^2+z^2", variables)
        with self.assertRaisesRegex(InvalidChart, "smooth locus"):
            fan_outputs(record, parse("z", variables))
        charts, proved = fan_outputs(record, polynomial)
        for chart, images in zip(charts, expected):
            self.assertEqual(chart["map"], dict(zip(variables, images)))
            self.assertEqual(chart["fibre"], images)

        saved = {"variables": variables, "atlas": deepcopy(charts)}
        for chart, (_, images) in zip(saved["atlas"], prescribed):
            chart["map"] = dict(zip(variables, images))
        replay, found = fan_outputs(saved, polynomial)
        self.assertEqual(found, proved)
        self.assertEqual(replay, saved["atlas"])

        # This perturbation and its first derivatives vanish on the whole
        # origin fibre of this chart, but the projection on the chart differs.
        saved["atlas"][1]["map"]["x"] += "+alpha^2"
        with self.assertRaises(InvalidChart):
            fan_outputs(saved, polynomial)

    def test_shared_face_is_checked_once_and_coordinate_faces_stay_distinct(self):
        from unittest.mock import patch
        from engine.atlas_torus import section_complex

        # Blow up the origin of the plane curve x+y=0. Its strict transform
        # meets the open exceptional orbit but neither torus fixed point.
        x, y, ray = (1, 0), (0, 1), (1, 1)
        charts = (((ray, y), {0: "E"}, (((0, 0), 1), ((0, 1), 1))),
                  ((x, ray), {1: "E"}, (((0, 0), 1), ((1, 0), 1))))
        with patch("engine.atlas_torus._orbit", wraps=_orbit) as orbit:
            expected = section_complex(charts, 2, True)
            self.assertEqual(orbit.call_count, 4)
            orbit.reset_mock()
            self.assertEqual(section_complex(charts, 2, True, common_source=True), expected)
            self.assertEqual(orbit.call_count, 3)
        self.assertEqual(expected["vertices"], [{"id": "E", "dimension": 0}])

    def test_split_sections_require_the_same_character_on_incident_faces(self):
        from unittest.mock import patch
        from engine.atlas_torus import section_complex
        # A regular 3-cone whose two exceptional rays are internal. Supply
        # independent section certificates to exercise their restriction map.
        charts = ((((1, 1, 1), (2, 2, 1), (0, 1, 0)), {0: "E", 1: "F"},
                   (((0, 0, 0), 1), ((0, 1, 0), 1), ((1, 0, 0), 1))),)
        def certificate(rays, polynomial, exhaustive):
            if len(polynomial) == 1:
                return 2, ("different primitive character",)
            return 2, ("common primitive character",)
        with patch("engine.atlas_torus._orbit", side_effect=certificate):
            self.assertIsNone(section_complex(charts, 3, True, common_source=True))

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

    def test_sheared_427_chart_complex_and_its_integral_boundaries(self):
        # One chart per cone of the regular fan the solver chose: 598 with
        # sampled subdivision points, 427 with the exact search.
        self.assertEqual(len(self.charts), 427)
        identity = tuple(tuple(int(i == j) for j in range(4)) for i in range(4))
        self.assertNotEqual(self.charts[0].torus[1], identity)
        charts = atlas.from_json(atlas.to_json(self.charts))
        dual = torus_complex(charts, 4, True)
        self.assertIsNotNone(dual)
        vertices = {v["id"] for v in dual["vertices"]}
        edges = {c["id"]: c for c in dual["cells"] if c["dimension"] == 1}
        triangles = [c for c in dual["cells"] if c["dimension"] == 2]
        self.assertEqual((len(vertices), len(edges), len(triangles)), (99, 252, 154))
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


class BoundaryRaysOverPositiveDimensionalCentres(unittest.TestCase):
    def test_coordinate_component_keeps_only_the_sections_over_zero(self):
        from engine.atlas_torus import section_complex
        from engine.polynomial import parse_expression
        names = list("abuv")
        rays = ((1,1,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1))
        # The fibre ideal is (a,u,v). On A=(a) there are three components.
        # The orbit closure u=0 meets the fibre; the torus roots v=±1 miss it.
        strict = tuple(parse("a+u*(v^2-1)", names).items())
        result = section_complex([(rays, {0:"A"}, strict)], 4, True, common_source=True,
                                 embedded_snc=True, local_coordinates=[names])
        self.assertEqual(result["vertices"], [{"id":"A", "dimension":2}])
        self.assertEqual(result["chart_sections"], [[{"id":"A", "ambient_divisor":"A", "ideal":["a","u"]}]])
        strict = tuple(parse_expression("a+v^2-1", names).items())
        result = section_complex([(rays, {0:"A"}, strict)], 4, True, common_source=True,
                                 embedded_snc=True, local_coordinates=[names])
        self.assertEqual((result["vertices"], result["cells"], result["chart_sections"]), ([], [], [[]]))
        # Both torus components through u=v=0 would meet in the fibre.
        # A whole contained orbit cannot assign an individual square root.
        strict = tuple(parse("a+u^2-v^2", names).items())
        self.assertIsNone(section_complex([(rays, {0:"A"}, strict)], 4, True, common_source=True,
                                          embedded_snc=True, local_coordinates=[names]))

    def test_relative_coordinate_primes_and_all_faces_replay(self):
        from engine.resolution_complex import dual_complex
        for extra, size in (("", (15,13,0)), ("+y*t^3", (21,37,18)), ("+y*z*t", (40,82,44))):
            with self.subTest(extra=extra):
                record = resolve(dict(polynomial=f"y*(x^2+y^3+y*z^3{extra})", variables=list("xyzt"),
                                      mode="intrinsic", search="quick", compute="cpu"))
                answer = record["dual_complex"]
                dimensions = Counter(c["dimension"] for c in answer["cells"])
                self.assertEqual((len(answer["vertices"]), dimensions[1], dimensions[2]), size)
                for omit in ("atlas", "charts"):
                    saved = {k:v for k,v in record.items() if k not in ("dual_complex", omit)}
                    self.assertEqual(dual_complex(saved), answer)

    def test_higher_rank_relative_fan_from_saved_equations(self):
        from engine.native_library import affine_request
        from engine.polynomial import format_poly, sparse_row
        from engine.strata import rational_chart_outputs
        names = list("xyzt")
        section = "x^2+y^3+z^3+t^6+y*z*t^2"
        family = section+"+w*y*z*t^2"
        record = resolve(dict(variables=names, polynomial=section, mode="intrinsic", search="quick"))
        full = [*names, "w"]
        charts = json.loads(json.dumps(record["atlas"]))
        for chart in charts:
            coords = chart["coordinates"]
            coords.append("extra")
            chart["map"]["w"] = "extra"
            chart["fibre"].append("extra")
            for key in ("rays", "linear_map"):
                matrix = chart["torus"][key]
                for row in matrix:
                    row.append(0)
                matrix.append([0, 0, 0, 0, 1])
            # This is a family on the relative fan. Its incidence must be
            # computed from the new strict equation, which depends on w.
            exceptional = [c for c in chart["components"] if c["id"] != "strict"]
            indices = [coords.index(c["ideal"][0]) for c in exceptional]
            rows = affine_request(["chart", " ".join(coords), " ".join(full),
                                   *(chart["map"][v] for v in full), "0", family, " ".join(map(str, indices))])
            for c, row in zip(exceptional, rows):
                self.assertEqual((c["multiplicity"], c["nu"]), tuple(map(int, row.split())))
            strict = next(c for c in chart["components"] if c["id"] == "strict")
            strict["ideal"] = [format_poly(sparse_row(rows[-1]), coords)]
        dual = torus_complex(atlas.from_json(charts), 5, True)
        self.assertEqual((dual["status"], len(dual["vertices"]), len(dual["cells"])), ("complete", 4, 3))
        def adjacencies(complex):
            return Counter(frozenset(c["vertices"]) for c in complex["cells"])
        self.assertEqual(adjacencies(dual), adjacencies(record["dual_complex"]))
        _, found = rational_chart_outputs(dict(variables=full, atlas=charts), parse(family, full), intrinsic=True)
        self.assertEqual(found["dual_complex"], dual)

    def test_two_sheets_and_every_face_from_the_saved_fan(self):
        from engine.strata import InvalidChart, rational_chart_outputs
        names = list("xyztu")
        source = "(x*y)^2-2*(z*t*u)^2"
        polynomial = parse(source, names)
        record = resolve(dict(polynomial=source, variables=names, mode="intrinsic", search="quick"))
        def check(boundary):
            self.assertEqual(boundary["status"], "complete")
            # On each character sheet the maximal faces of the regular fan
            # meeting the origin are these three coordinate tetrahedra.
            simplices = [(1, 4, 5, 6), (1, 2, 5, 6), (1, 2, 3, 6)]
            expected = {frozenset(f"E{i}.{sheet}" for i in face)
                        for sheet in (1, 2) for simplex in simplices
                        for size in range(1, 5) for face in combinations(simplex, size)}
            by_id = {v["id"]: frozenset([v["id"]]) for v in boundary["vertices"]}
            by_id.update((c["id"], frozenset(c["vertices"])) for c in boundary["cells"])
            self.assertCountEqual(by_id.values(), expected)
            for cell in boundary["cells"]:
                vertices = by_id[cell["id"]]
                self.assertCountEqual((by_id[f] for f in cell["faces"]), (vertices-{v} for v in vertices))

        check(record["dual_complex"])
        saved = json.loads(json.dumps(dict(variables=names, atlas=record["atlas"])))
        charts = atlas.from_json(saved["atlas"])
        check(torus_complex(charts, len(names), True))
        check(rational_chart_outputs(saved, polynomial, intrinsic=True)[1]["dual_complex"])
        strict, = (c for c in saved["atlas"][0]["components"] if c["id"] == "strict")
        strict["ideal"][0] += "+1"
        self.assertIsNone(torus_complex(atlas.from_json(saved["atlas"]), len(names), True))
        with self.assertRaises(InvalidChart):
            rational_chart_outputs(saved, polynomial, intrinsic=True)

    def test_three_term_section_on_the_same_toric_modification(self):
        from engine.affine_kernel import chart_data
        from engine.chart_expressions import runtime_expressions
        from engine.strata import rational_chart_outputs
        names = list("xyzt")
        source = "(x*y)^2+3*x*y*z*t+(z*t)^2"
        api = runtime_expressions()
        x, y, z, t = base = tuple(map(api.Symbol, names))
        f = (x*y)**2+3*x*y*z*t+(z*t)**2
        # The projected toric charts come from the subdivision for the same
        # segment. Recompute their pullbacks for this different hypersurface.
        support = resolve(dict(variables=names, polynomial="(x*y)^2-2*(z*t)^2",
                               mode="intrinsic", search="quick"))
        charts = atlas.from_json(support["atlas"])
        for chart, item in zip(charts, support["atlas"]):
            axes = [ideal[0] for label, _, _, ideal in chart.components if label != "strict"]
            _, strict = chart_data(chart.variables, [g for _, g in chart.coordinates], axes, f, base)
            equation = atlas._record_text(strict, chart.variables)
            next(c for c in item["components"] if c["id"] == "strict")["ideal"] = [equation]
        saved = dict(variables=names, atlas=support["atlas"])
        _, found = rational_chart_outputs(saved, parse(source, names), intrinsic=True)
        dual = found["dual_complex"]
        self.assertEqual(dual["status"], "complete")
        self.assertEqual(len(dual["vertices"]), 8)
        self.assertEqual(Counter(c["dimension"] for c in dual["cells"]), {1:10, 2:4})


if __name__ == "__main__":
    unittest.main()
