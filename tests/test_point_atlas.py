"""The standard projective cover over Q and over a marked quadratic field."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

from engine import native_library
from engine.point_atlas import point_outputs
from engine.polynomial import InputError, parse
from engine.resolve import resolve
from engine.strata import InvalidChart


@unittest.skipIf(native_library._MODE == "sympy", "The sparse chart verifier is disabled in pure reference mode.")
class PointAtlas(unittest.TestCase):
    def test_owned_crossing_keeps_the_linear_normal_term(self):
        from engine.monomial_atlas import fan_outputs
        names = ["x", "y"]
        record = dict(variables=names, fan={"rays": [[1,0], [0,1], [1,1]]}, charts=[
            dict(ray_indices=[2,1], coordinates=["u","v"], map={"x": "u", "y": "u*v"}),
            dict(ray_indices=[0,2], coordinates=["u","v"], map={"x": "u*v", "y": "v"}),
        ])
        def verify(f, intrinsic):
            return fan_outputs(record, parse(f, names, bounded=False),
                               complex=False, intrinsic=intrinsic, integrals=False)

        # On u=0 both strict equations restrict to v². Their derivatives
        # distinguish a smooth branch tangent to the exceptional divisor
        # from a singular branch. The discarded term exceeds packed degree.
        smooth = "y^2+x^3+x^65540"
        with self.assertRaisesRegex(InvalidChart, "Normal crossings fail"):
            verify(smooth, False)
        with self.assertRaisesRegex(InvalidChart, "strict transform is singular"):
            verify("y^2+x^4+x^65540", True)
        charts, _ = verify(smooth, True)
        strict = charts[0]["components"][-1]["ideal"]
        self.assertEqual(strict, ["u^65538+v^2+u"])

    def test_mixed_monomial_jet_over_a_field(self):
        if not hasattr(native_library._library(), "atlas_crossing_jet_abi"):
            self.skipTest("Requires the native chart crossing jet.")
        from engine.polynomial import sparse_row
        base = ["chart", "u v w t", "x y z a", "u", "v", "w", "t", "t^2+1"]
        # J=(uv,vw). Neither generator squared divides uv²w, but their
        # product does. The coefficient parameter is free in the jet, as
        # is every power of u alone, including a beyond-word-size power.
        power = 2**65
        f = f"x^{power}/3+a*y*z/2+x*y+x*y^2*z+x^2*y^2+y^2*z^2"
        rows = native_library.affine_request([*base, f, "", "3 6"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(sparse_row(rows[0]), parse(f, ["x","y","z","a"], bounded=False))
        self.assertEqual(sparse_row(rows[1]), parse(f"x^{power}/3+a*y*z/2+x*y",
                                                   ["x","y","z","a"], bounded=False))

    def test_surface_sections_on_toric_orbits_and_inside_the_torus(self):
        from engine.monomial_atlas import fan_outputs
        names = list("xyz")
        charts = resolve(dict(polynomial="x*y+z^3", variables=names, mode="intrinsic"))["atlas"]
        cases = [
            ("x*(x+y)+z^3", ("1+b+a*c^3", "a^2+a+b*c^3", "a^2+a*b+c")),
            ("(x+2*y)*(x+3*y)+z^3", ("(1+2*b)*(1+3*b)+a*c^3", "(a+2)*(a+3)+b*c^3",
                                      "(a+2*b)*(a+3*b)+c")),
        ]
        for f, equations in cases:
            for c, equation in zip(charts, equations):
                c["components"][0]["id"] = "outer"
                c["components"][-1]["ideal"] = [equation]
            _, found = fan_outputs(dict(variables=names, atlas=charts), parse(f, names), intrinsic=True)
            self.assertEqual([edge["vertices"] for edge in found["cells"]], [["outer.1", "outer.2"]])
            self.assertNotIn("strata", found)
        # Three concurrent lines on an otherwise smooth strict surface have
        # a triple point. Pair Euler counts alone would produce a false graph.
        for c, equation in zip(charts, ("b+b^2+a*c^4", "a^2+a+b*c^4", "a^2*b+a*b^2+c")):
            c["components"][0]["multiplicity"] = 3
            c["components"][-1]["ideal"] = [equation]
        self.assertIsNone(fan_outputs(dict(variables=names, atlas=charts), parse("x^2*y+x*y^2+z^4", names),
                                      intrinsic=True))

    def test_intrinsic_and_companion_reread_without_symbolic_imports(self):
        code = textwrap.dedent("""
            import json, sys
            class Block:
                def find_spec(self, name, *args):
                    if name.partition('.')[0] == 'sympy':
                        raise AssertionError(name)
            sys.meta_path.insert(0, Block())
            from engine.resolve import resolve
            from engine.polynomial import parse
            from engine.strata import rational_chart_outputs, resolution_strata
            from engine.topological import topological_zeta
            from engine.monodromy import monodromy_zeta
            names = list('xyz')
            for f, k in [('x*y+z^3',3),('x^2+y^2+z^5',5),('x*y+z^6',6),('x^7+y^2-z^2',7)]:
                p = parse(f,names)
                r = resolve(dict(polynomial=f, variables=names, mode='intrinsic', search='quick', monodromy=True))
                saved = {key: r[key] for key in ('variables','polynomial','mode','status','atlas')}
                if 'embedded_companion' in r:
                    saved['embedded_companion'] = r['embedded_companion']
                saved = json.loads(json.dumps(saved))
                _, found = rational_chart_outputs(saved,p,intrinsic=True)
                assert len(found['dual_complex']['vertices']) == k-1 and len(found['cells']) == k-2
                assert ('strata' not in found) == (k % 2 == 1) == ('embedded_companion' in saved)
                assert monodromy_zeta(saved)['expression'] == r['monodromy_zeta']['expression']
                assert topological_zeta(saved)['expression'] == r['topological_zeta']['expression']
                if 'embedded_companion' in saved:
                    companion = saved['embedded_companion']['atlas']
                    _, embedded = rational_chart_outputs(dict(variables=names, atlas=companion), p)
                    assert embedded['strata'] == r['topological_zeta']['strata']
                    companion[-1]['components'][0]['multiplicity'] += 1
                    assert resolution_strata(saved,p) is None
        """)
        subprocess.run([sys.executable, "-B", "-c", code], check=True,
                       cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, ATLAS_GROEBNER="native"))

    def test_split_cone_round_trip_through_general_charts(self):
        from engine import atlas
        from engine.strata import rational_chart_outputs
        names = list("xyz")
        for f in ("x^2+y^2+z^2", "x^2+y^2+z^5"):
            p = parse(f, names)
            r = resolve(dict(polynomial=f, variables=names, mode="embedded"))
            raw = dict(r)
            del raw["atlas"]
            generated = atlas.to_json(atlas.from_record(raw, p))
            saved = json.loads(json.dumps(r["atlas"]))
            charts = atlas.from_json(saved)
            encoded = atlas.to_json(charts)
            self.assertEqual([c.split_cone for c in charts], [c["split_cone"] for c in generated])
            self.assertEqual([c["split_cone"] for c in encoded], [c["split_cone"] for c in generated])
            self.assertIsNotNone(rational_chart_outputs(dict(variables=names, atlas=encoded), p))
            encoded[-1]["split_cone"] = encoded[0]["split_cone"]
            with self.assertRaises(InvalidChart):
                rational_chart_outputs(dict(variables=names, atlas=atlas.to_json(atlas.from_json(encoded))), p)
            # Records predating this optional certificate still parse and round trip.
            for c in saved:
                del c["split_cone"]
            self.assertTrue(all("split_cone" not in c for c in atlas.to_json(atlas.from_json(saved))))

    def test_split_fan_maps_labels_and_owned_fibres_survive_serialization(self):
        from engine.monomial_atlas import fan_outputs
        names = list("xyz")
        for f in ("x*y+z^5", "x^2+3*x*y+2*y^2+z^4", "x^2+y^2+z^5", "x^2-2*y^2+3*z^6"):
            with self.subTest(f=f):
                p = parse(f, names)
                output = resolve(dict(polynomial=f, variables=names, mode="embedded", monodromy=True))
                charts = output["atlas"]
                self.assertGreater(len(charts), 3)
                self.assertTrue(all("split_cone" in c for c in charts))
                _, found = fan_outputs(dict(variables=names, atlas=charts), p)
                self.assertEqual(found["strata"], output["topological_zeta"]["strata"])
                self.assertEqual([v["id"] for v in found["dual_complex"]["vertices"]],
                                 [v["id"] for v in output["dual_complex"]["vertices"]])
        # Reuse only the saved equations and cones, with no producer record.
        for error in ("map", "cone", "label", "region", "field"):
            with self.subTest(error=error):
                changed = copy.deepcopy(charts)
                item = changed[-1]
                if error == "map":
                    item["map"]["x"] = "2*(" + item["map"]["x"] + ")"
                elif error == "cone":
                    item["split_cone"] = changed[0]["split_cone"]
                elif error == "label":
                    item["components"][0]["id"] = "altered"
                elif error == "region":
                    item["region"] = ["1"]
                else:
                    item["coefficient_field"] = dict(item["coefficient_field"], embedding="other root")
                with self.assertRaises(InvalidChart):
                    fan_outputs(dict(variables=names, atlas=changed), p)

    def test_fabricated_gluing_on_standalone_split_charts_is_rejected(self):
        from engine.strata import resolution_strata
        f = "x*y+z^5"
        names = list("xyz")
        p = parse(f, names)
        charts = resolve(dict(variables=names, polynomial=f, mode="embedded"))["atlas"]
        record = dict(status="resolved", mode="embedded", variables=names)
        for bad in ("removed", "strict unit", "changed sheet"):
            changed = copy.deepcopy(charts)
            for c in changed:
                c["components"][-1]["id"] = "S1"
            if bad == "removed":
                changed[-1]["removed"] = [["0"]]
            elif bad == "strict unit":
                for c in changed:
                    c["components"][-1]["ideal"] = ["1"]
            else:
                changed[-1]["components"][-1]["id"] = "S2"
            self.assertIsNone(resolution_strata(dict(record, atlas=changed), p))

        # The same smooth pair on every affine chart. Swapping two labels
        # in only one chart preserves EVERY local product, order and locus.
        # Its common Laurent branches, however, no longer glue.
        charts = resolve(dict(variables=names, polynomial="x*y+z^3", mode="intrinsic"))["atlas"]
        p = parse("(x+y+z)*(x+2*y+3*z)", names)
        for c in charts:
            i = c["coordinates"].index(c["components"][0]["ideal"][0])
            directions = list("abc")
            directions[i] = "1"
            a,b,d = directions
            c["components"][1:] = [dict(id="S1", multiplicity=1, nu=1, ideal=[f"{a}+{b}+{d}"]),
                                   dict(id="S2", multiplicity=1, nu=1, ideal=[f"{a}+2*{b}+3*{d}"])]
        self.assertIsNotNone(resolution_strata(dict(record, atlas=charts), p))
        charts[-1]["components"][1]["id"], charts[-1]["components"][2]["id"] = "S2", "S1"
        self.assertIsNone(resolution_strata(dict(record, atlas=charts), p))

        # Uniformly altering the field mark used to bypass the check that
        # the coefficients name an actual root of their defining equation.
        f = "x^2+y^2+z^5"
        charts = resolve(dict(variables=names, polynomial=f, mode="embedded"))["atlas"]
        for c in charts:
            c["coefficient_field"]["embedding"] = "root 3 (FLINT order)"
        self.assertIsNone(resolution_strata(dict(record, atlas=charts), parse(f, names)))

    def test_weighted_split_fan_without_producer_metadata(self):
        from engine.strata import resolution_strata
        from engine.monomial_atlas import fan_outputs
        for f in ("(1+x)*(x*y+z^5)^2", "(x^2+y^2+z^5)^3"):
            names = list("xyz")
            result = resolve(dict(polynomial=f, variables=names, mode="embedded", monodromy=True))
            saved = dict(status="resolved", variables=names, mode="embedded", atlas=result["atlas"])
            p = parse(f, names)
            self.assertEqual(resolution_strata(saved, p)[1], result["topological_zeta"]["strata"])
            with self.assertRaises(InvalidChart):
                fan_outputs(saved, p, intrinsic=True)
            saved["atlas"][-1]["components"][-1]["multiplicity"] += 1
            self.assertIsNone(resolution_strata(saved, p))

    def test_regions_and_recovery_without_producer_metadata(self):
        names = list("xyz")
        for f in ("x^2+y^2+z^2+x*y/2", "x^3+y^3+z^3", "x^3+x*y^2+z^3"):
            with self.subTest(f=f):
                p = parse(f, names)
                record = resolve(dict(polynomial=f, variables=names, mode="intrinsic", monodromy=True))
                charts = record["atlas"]
                self.assertEqual([c["region"] for c in charts], [charts[i]["coordinates"][:i] for i in range(3)])
                self.assertFalse(any("torus" in chart for chart in charts))
                archived = dict(variables=names, atlas=charts)
                _, result = point_outputs(archived, p, intrinsic=True)
                self.assertEqual(result["strata"], record["topological_zeta"]["strata"])
                self.assertEqual(result["dual_complex"]["vertices"], [{"id":"E1", "dimension":1}])

    def test_actual_pullback_and_projective_partition_are_verified(self):
        names = list("xyz")
        f = "x^2+y^2+z^2"
        p = parse(f, names)
        record = resolve(dict(polynomial=f, variables=names, mode="embedded"))
        for field in ("matrix", "multiplicity", "region", "fibre", "strict", "modulus", "extra generator"):
            with self.subTest(field=field):
                charts = copy.deepcopy(record["atlas"])
                c = charts[-1]
                if field == "matrix":
                    c["map"]["x"] += "+a*c"
                elif field == "multiplicity":
                    c["components"][0]["multiplicity"] = 4
                elif field == "extra generator":
                    c["components"][0]["ideal"].append("a")
                    c["components"][0]["multiplicity"] += 1
                elif field == "region":
                    c["region"] = ["a"]
                elif field == "fibre":
                    c["fibre"] = []
                elif field == "strict":
                    c["components"][-1]["ideal"] = ["a+b"]
                else:
                    for item in charts:
                        item["coefficient_field"]["minimal_polynomial"] = "atlas_field^2-1"
                with self.assertRaises((InputError,InvalidChart)):
                    point_outputs(dict(variables=names, atlas=charts), p)

    def test_redundant_generators_over_the_marked_field(self):
        names = list("xyz")
        f = "x^2+y^2+z^2"
        record = resolve(dict(polynomial=f, variables=names))
        c = record["atlas"][0]
        c["components"][0]["ideal"].append("a*b")
        c["components"][1]["ideal"] = ["atlas_field*(" + c["components"][1]["ideal"][0] + ")"]
        self.assertIsNotNone(point_outputs(record, parse(f, names)))

    def test_tangent_projective_section_must_be_smooth(self):
        names = list("xyz")
        record = resolve(dict(polynomial="x^3+y^3+z^3", variables=names))
        del record["atlas"]
        # On the exceptional divisor xy=0 is reducible and its intersection
        # is singular. Changing the input cannot preserve the old certificate.
        p = parse("x*y+z^4", names)
        self.assertIsNone(point_outputs(record, p, intrinsic=True))
        with self.assertRaises(InvalidChart):
            point_outputs(record, p)
