"""Cross-check arrangement incidence, fibre integration and saved projections."""

from copy import deepcopy
import unittest

from engine import atlas
from engine.plane_arrangement import _construction
from engine.polynomial import parse
from engine.resolution_complex import dual_complex
from engine.resolution_zetas import replay_zetas
from engine.resolve import resolve
from engine.strata import InvalidChart


class PlaneArrangementTests(unittest.TestCase):
    def test_chart_verifier_recovers_arrangement_strata(self):
        for equation, expected_steps, expected_chi in (
            ("x*y*z*(x+y+z)", 1, 4),
            ("x*y*z*(x+y)", 2, 0),
            ("x*y*z*(x-y)*(y-z)*(x-z)", 5, 12),
        ):
            with self.subTest(equation=equation):
                record = resolve(
                    dict(
                        polynomial=equation,
                        variables=list("xyz"),
                        search="quick",
                        compute="cpu",
                        mode="embedded",
                        monodromy=True,
                    )
                )
                self.assertEqual(record["engine"], "central-plane-arrangement")
                self.assertEqual(len(record["steps"]), expected_steps)
                self.assertEqual(record["monodromy_zeta"]["milnor_fiber_euler_characteristic"], expected_chi)
                # Independently check every Jacobian, and integrate on the disjoint affine pieces.
                charts = atlas.from_json(record["atlas"], native=True)
                self.assertEqual(atlas.to_json(charts), record["atlas"])
                checked = atlas.outputs(charts, 3, complex=False, integrals=True)

                def nonzero(rows):
                    return {
                        frozenset(row["components"]): row["euler_characteristic"]
                        for row in rows
                        if row["euler_characteristic"]
                    }

                self.assertEqual(nonzero(checked["strata"]), nonzero(record["topological_zeta"]["strata"]))
                replay = replay_zetas(dict(record=record, zetas=["topological"], search="quick"))
                self.assertEqual(replay["topological_zeta"], record["topological_zeta"])
                archived = {key: record[key] for key in ("status", "mode", "variables", "polynomial", "atlas")}
                self.assertEqual(dual_complex(archived), record["dual_complex"])
                partial = dict(archived, atlas=archived["atlas"][1:])
                with self.assertRaises(InvalidChart):
                    dual_complex(partial)
                broken = deepcopy(record)
                broken["atlas"][0]["map"]["x"] = "1"
                with self.assertRaises(InvalidChart):
                    replay_zetas(dict(record=broken, zetas=["topological"]))

    def test_intrinsic_boundary_contains_one_star_per_plane(self):
        record = resolve(
            dict(
                polynomial=f"x*y*z*(x+{2**140}*y+z)",
                variables=list("xyz"),
                search="quick",
                mode="intrinsic",
                compute="cpu",
            )
        )
        boundary = dual_complex(record)
        self.assertEqual(len(boundary["vertices"]), 16)
        self.assertEqual(len(boundary["cells"]), 12)
        self.assertTrue(
            all(c["vertices"][0].split(".")[1] == c["vertices"][1].split(".")[1] for c in boundary["cells"])
        )
        local = {div["id"] for chart in record["charts"] for div in chart["exceptional"]}
        self.assertEqual(local, {vertex["id"] for vertex in boundary["vertices"]})
        for chart in record["charts"]:
            for divisor in chart["exceptional"]:
                self.assertEqual(len(divisor["ideal"]), 2)
        with self.assertRaises(InvalidChart):
            dual_complex(dict(record, atlas=record["atlas"][:-1]))

    def test_pullbacks_retain_a_rational_unit_and_noncoordinate_planes(self):
        import sympy as sp

        names = list("xyz")
        x, y, z = sp.symbols(names)
        # Four rational planes with a non-coordinate triple line and a wide rational unit.
        equation = f"(-{2**137 + 1}/7)*(2*x+y)*(y+3*z)*(2*x+2*y+3*z)*(x-z)"
        original = sp.sympify(equation, locals=dict(zip(names, (x, y, z))))
        for mode in ("embedded", "intrinsic"):
            record = resolve(dict(polynomial=equation, variables=names, mode=mode, compute="cpu", search="quick"))
            self.assertEqual(record["engine"], "central-plane-arrangement")
            for chart in record["charts"]:
                with self.subTest(mode=mode, chart=chart["id"]):
                    coordinates = sp.symbols(chart["coordinates"])
                    chart_ring = dict(zip(chart["coordinates"], coordinates))
                    def expression(source):
                        return sp.sympify(source, locals=chart_ring)
                    projection = {symbol: expression(chart["map"][name]) for name, symbol in zip(names, (x, y, z))}
                    divisors = chart.get("ambient_components", chart["exceptional"])
                    factor = sp.prod(expression(d["equation"])**d["total_multiplicity"] for d in divisors)
                    difference = original.xreplace(projection) - factor * expression(chart["strict_transform"])
                    self.assertTrue(sp.Poly(difference, *coordinates).is_zero)

    def test_declines_nonarrangements_pencils_and_repetitions(self):
        names = list("xyz")
        for equation in (
            "x*y*(x-y)*(x+y)",
            "x^2*y*z",
            "x*y*z*(x+y+z)+x^5",
            "x*y*(z^2+x*y)",
            "x^12345678901234567890123456789+y^2",
        ):
            self.assertIsNone(_construction(parse(equation, names, bounded=False), names, "embedded"))
