import gzip
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from dataclasses import replace

from engine.atlas_gluing import glued_complex
from engine.chart_expressions import runtime_expressions
from engine.local_blowups import Chart
from engine.polynomial import InputError
from engine.resolution_complex import dual_complex
from engine.resolve import resolve
from engine.session import new_session
from engine.session_values import Resolution


class ResolutionComplexTests(unittest.TestCase):
    def test_archived_record_replays_without_cached_incidence(self):
        r = resolve(dict(variables=["x", "y"], polynomial="y^2-x^3", mode="embedded"))
        r["dual_complex"] = dict(status="partial", vertices=[], cells=[])
        expected = dual_complex(r)
        self.assertEqual(expected["status"], "complete")
        self.assertEqual({tuple(c["vertices"]) for c in expected["cells"]}, {("E1","E3"), ("E2","E3")})
        session = new_session(None)
        session.variables["R"] = Resolution(r)
        self.assertEqual(list(session.run("dual_complex(R)"))[0][1], expected)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "archived.json.gz"
            with gzip.open(path, "wt") as stream:
                json.dump(r, stream)
            cli = Path(__file__).resolve().parents[1] / "atlas"
            output = subprocess.check_output([cli, "dual-complex", path], text=True)
            self.assertEqual(json.loads(output), expected)
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
