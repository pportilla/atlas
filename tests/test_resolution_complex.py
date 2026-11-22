import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from engine.atlas_gluing import glued_complex
from engine.chart_expressions import runtime_expressions
from engine.local_blowups import Chart
from engine.polynomial import InputError
from engine.resolution_complex import dual_complex
from engine.resolve import resolve
from engine.session import new_session
from engine.session_values import Resolution
from tests.cli_support import run_prompt


class ResolutionComplexTests(unittest.TestCase):
    def test_relative_component_requires_a_connected_cover_of_every_owned_fibre_point(self):
        from engine import native_polynomial as np
        from engine.incidence import counted_boundary
        x, y, z = np.symbols("x y z")
        components = (("E",1,1,(x,)), ("F",1,1,(y,)), ("G",1,1,(z,)))
        chart = Chart((x,y,z), (), components, (y*z,), (), (), ())
        counts = {frozenset(names): (None if names == "E" else 1)
                  for names in ("E", "F", "G", "EF", "EG", "FG", "EFG")}
        def reason(charts, incidences=counts):
            return counted_boundary(incidences, {}, 3, False, False, charts=charts)["reason"]
        # The fibre on E is the union of two lines meeting at the origin.
        self.assertIsNone(reason([chart]))
        extra = replace(chart, fibre=(y*z*(z-1),))
        self.assertIsNotNone(reason([extra]))
        self.assertIsNone(reason([replace(extra, removed=((z-1,),))]))
        # Two disjoint visible sections do not identify their parent.
        parallel = replace(chart, fibre=(y*(y-1),),
                           components=(*components[:2], ("G",1,1,(y-1,))))
        separate = {k:v for k,v in counts.items() if not frozenset("FG") <= k}
        self.assertIsNotNone(reason([parallel], separate))

    def test_graph_cannot_remove_an_entire_smooth_component_as_its_centre(self):
        # Blowing up the Cartier ideal (x) is the identity. Saturating the
        # pullback of x by that pivot would incorrectly erase the whole curve.
        names = list("xy")
        chart = dict(coordinates=names, ambient_ideal=[], map=dict(x="x", y="y"),
                     exceptional=[dict(id="E1", ideal=["x"])], strict_transform=["1"],
                     pivot_generator="x", projective_coordinates=["1"])
        for mode, equation in (("intrinsic", "x"), ("embedded", "x"), ("embedded", "x^3")):
            record = dict(status="resolved", engine="verified-ideal-blowup", mode=mode, variables=names,
                          polynomial=equation, steps=[dict(generators=["x"])], charts=[chart])
            with self.subTest(mode=mode, equation=equation), self.assertRaisesRegex(InputError, "graph centre"):
                dual_complex(record)

    def test_large_computed_coefficient_survives_resolution_and_incidence_replay(self):
        import sys
        limit = sys.get_int_max_str_digits()
        if not limit:
            self.skipTest("This interpreter permits unrestricted decimal literals.")
        source = f"(10^{limit // 2 + 1})^2*x+y"
        for mode in ("embedded", "intrinsic"):
            with self.subTest(mode=mode):
                record = resolve(dict(polynomial=source, variables=["x","y"], mode=mode))
                self.assertEqual(record["steps"], [])
                self.assertEqual(dual_complex(record)["status"], "complete")
                with self.assertRaisesRegex(InputError, "active limit"):
                    resolve(dict(polynomial=record["polynomial"], variables=["x","y"], mode=mode))
        self.assertEqual(sys.get_int_max_str_digits(), limit)

    def test_smooth_chart_for_a_different_hypersurface_cannot_certify_incidence(self):
        from engine.resolution_zetas import replay_zetas
        chart = dict(coordinates=list("uvw"), ambient_ideal=[], fibre=list("uvw"), region=[], removed=[],
                     map=dict(x="u+v", y="v", z="w"),
                     components=[dict(id="strict", ideal=["w"], multiplicity=1, nu=1)])
        for mode in ("embedded", "intrinsic"):
            record = dict(status="resolved", mode=mode, variables=list("xyz"), atlas=[chart], steps=[])
            with self.subTest(mode=mode):
                self.assertEqual(dual_complex(dict(record, polynomial="z"))["status"], "complete")
                with self.assertRaisesRegex(InputError, "strict transform differs"):
                    dual_complex(dict(record, polynomial="x^2+y^3+z^7"))
                bad = dict(record, polynomial="x^2+y^3+z^7")
                with self.assertRaisesRegex(InputError, "strict transform differs"):
                    replay_zetas(dict(record=bad, zetas=["topological"]))
                with patch("engine.resolve._resolve_polynomial", return_value=bad):
                    result = resolve(dict(mode=mode, polynomial=bad["polynomial"], variables=list("xyz"), search="quick"))
                self.assertEqual(result["status"], "inconclusive")
                self.assertIn("strict transform differs", result["diagnostic"]["description"])

    def test_saved_polynomial_graph_certifies_its_exceptional_bundle(self):
        from engine.graph_record import graph_outputs
        from engine.polynomial import parse
        variables = ["x", "y", "z"]
        for equation in ("(x+y+z)^2-z*(x-y)^2", "(y-x^2)^2-x*z^2"):
            with self.subTest(equation=equation):
                record = resolve(dict(polynomial=equation, variables=variables, mode="embedded", monodromy=True))
                expected = record.pop("dual_complex")
                del record["certificate"]
                with patch("engine.general_atlas.native_general_outputs", side_effect=AssertionError("generic replay")):
                    charts, result = graph_outputs(record, parse(equation, variables))
                self.assertEqual(charts, record["atlas"])
                self.assertEqual(expected, result["dual_complex"])
                self.assertEqual({v["component"]: (v["multiplicity"], v["nu"]) for v in result["numerical_data"]},
                                 {"E1": (2, 2), "strict": (1, 1)})

    def test_bundle_certificate_requires_the_same_exceptional_ideal_on_every_open(self):
        from engine.incidence import incidence
        from engine.monomial_atlas import _row
        names = ("x", "y", "z")

        def verify(generators, chart_ideals, labels=("E1", "E1")):
            centre = [_row(f, names) for f in generators]
            charts = [{"exceptional": [{"id": label, "ideal": [_row(f, names) for f in polynomials]}]}
                      for label, polynomials in zip(labels, chart_ideals)]
            return incidence("polynomial_graph_exceptional", variables=names, centre=centre, charts=charts)

        parabola = ("y-x^2", "z")
        self.assertEqual(verify(parabola, [("3*z", "2*x^2-2*y"), parabola]), "E1")
        self.assertIsNone(verify(parabola, [parabola, ("y", "z")]))
        self.assertIsNone(verify(parabola, [parabola]))
        self.assertIsNone(verify(parabola, [parabola, parabola], labels=("E1", "E2")))
        for centre in (("x-1", "y"), ("x*(x-1)", "y"), ("x-y*z", "y-x*z")):
            with self.subTest(centre=centre):
                self.assertIsNone(verify(centre, [centre, centre]))

    def test_weighted_graph_replays_exceptional_multiplicity_from_the_charts(self):
        equation = "((x+y+z)^2-z*(x-y)^2)^2"
        record = resolve(dict(polynomial=equation, variables=["x", "y", "z"], mode="embedded"))
        self.assertEqual(record["engine"], "verified-ideal-blowup")
        expected = record.pop("dual_complex")
        self.assertEqual(expected["vertices"], [{"id": "E1", "dimension": 2, "total_multiplicity": 4}])
        self.assertEqual(dual_complex(record), expected)

    def test_archived_record_replays_without_cached_incidence(self):
        r = resolve(dict(variables=["x", "y"], polynomial="y^2-x^3", mode="embedded"))
        r["dual_complex"] = dict(status="partial", vertices=[], cells=[])
        expected = dual_complex(r)
        self.assertEqual(expected["status"], "complete")
        self.assertEqual({tuple(c["vertices"]) for c in expected["cells"]}, {("E1","E3"), ("E2","E3")})
        session = new_session(None)
        session.variables["R"] = Resolution(r)
        self.assertEqual(list(session.run("dual_complex(R)"))[0][1], expected)
        list(session.run("D = recompute_dual_complex(R)"))
        self.assertEqual(session.variables["D"].record, expected)
        self.assertEqual([v.id for v in session.variables["D"].vertices], ["E1", "E2", "E3"])
        self.assertEqual(session.variables["R"].dual_complex.record, r["dual_complex"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "archived.json.gz"
            with gzip.open(path, "wt") as stream:
                json.dump(r, stream)
            cli = Path(__file__).resolve().parents[1] / "atlas"
            for command in ("recompute-dual-complex", "dual-complex"):
                with self.subTest(command=command):
                    output = subprocess.check_output([cli, command, path], text=True)
                    self.assertEqual(json.loads(output), expected)
            status, output, error = run_prompt(f'recompute-dual-complex ./"{os.path.relpath(path)}"\nquit\n')
            self.assertEqual((status, error), (0, ""))
            self.assertEqual(json.loads(output.split("atlas> ")[1]), expected)
        with self.assertRaisesRegex(InputError, "chart cover"):
            dual_complex(dict(r, charts=[], atlas=[]))

    def test_conjugate_sections_glued_on_overlapping_charts(self):
        # Model the open radial blowup chart (u, u*x, u*y). Over u=0
        # the disjoint sheets x=±sqrt(2) intersect the unsplit u divisor.
        api = runtime_expressions()
        x, y, u = api.symbols("x y u")
        images = (u, u*x, u*y)
        chart = Chart((x,y,u), (), (("D",1,1,(x*x-2,)), ("E",1,1,(u,))),
                      images, (), tuple(zip(("a","b","c"),images)), ())
        empty_owner = replace(chart, region=(api.S.One,))
        # Direct diagonal gluing exercises the general section solver,
        # bypassing the special finite-parameter certificate.
        for cover in ([chart], [chart, empty_owner]):
            dual = glued_complex(cover, 3)
            self.assertEqual(dual["status"], "complete")
            self.assertEqual({v["id"] for v in dual["vertices"]}, {"D.1", "D.2", "E"})
            self.assertEqual({frozenset(c["faces"]) for c in dual["cells"]},
                             {frozenset(("D.1","E")), frozenset(("D.2","E"))})
