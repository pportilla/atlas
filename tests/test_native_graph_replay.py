from copy import deepcopy
import unittest

from engine.graph_record import _graph_request, graph_record_charts
from engine.polynomial import parse
from engine.resolve import resolve
from engine.strata import InvalidChart


class NativeGraphReplayTests(unittest.TestCase):
    def test_displayed_strict_is_checked_in_its_chart_ring(self):
        names = list("xyz")
        equation = "(x+y+z)^2-z*(x-y)^2"
        record = resolve(dict(polynomial=equation, variables=names, compute="cpu"))
        polynomial = parse(equation, names)
        expected = graph_record_charts(record, polynomial)
        alternate = deepcopy(record)
        chart = alternate["charts"][0]
        chart["strict_transform"][0] += "+(" + chart["ambient_ideal"][0] + ")"
        self.assertEqual(graph_record_charts(alternate, polynomial), expected)
        for bad in (["1"], [], ["x"]):
            with self.subTest(bad=bad), self.assertRaisesRegex(InvalidChart, "strict transform"):
                chart["strict_transform"] = bad
                graph_record_charts(alternate, polynomial)

    def test_weighted_graph_checks_the_reduced_and_weighted_equations(self):
        names = list("xyz")
        equation = "((y-x^2)^2-x*z^2)^2"
        record = resolve(dict(polynomial=equation, variables=names, compute="cpu"))
        polynomial = parse(equation, names)
        self.assertEqual(graph_record_charts(record, polynomial), record["atlas"])
        for key in ("strict_transform", "reduced_strict_transform"):
            altered = deepcopy(record)
            altered["charts"][0][key] = ["1"]
            with self.subTest(key=key), self.assertRaisesRegex(InvalidChart, "strict transform"):
                graph_record_charts(altered, polynomial)

    def test_exceptional_cover_is_relative_to_the_supplied_fibre(self):
        chart = ["x", "y", "t"]
        # On x=y=0 the union is all of the fibre. Away from the fibre it
        # misses part of V(x); an empty family misses the fibre too.
        pieces = [["x", "y"], ["x", "t"]]
        self.assertTrue(_graph_request("cover", chart, ambient=["x", "y"], pivot="x", pieces=pieces))
        self.assertFalse(_graph_request("cover", chart, ambient=[], pivot="x", pieces=pieces))
        self.assertFalse(_graph_request("cover", chart, ambient=["x", "y"], pivot="x", pieces=[]))

    def test_native_cells_require_a_unique_parent_and_visible_fibre(self):
        stratum = lambda subset, groups: {"subset": subset, "groups": groups, "dimension": 2-len(subset)}
        piece = lambda ideal: [{"chart": 0, "ideal": ideal, "display": ideal}]
        x, y = piece(["x"]), piece(["y"])
        invisible = piece(["x-1", "y"])
        strata = [stratum([0], [x, invisible]), stratum([1], [y]), stratum([0, 1], [piece(["x", "y"])])]
        args = dict(charts=1, relation=[], fibre=["x", "y"], strata=strata)
        data = _graph_request("stratum_faces", ["x", "y"], **args)
        self.assertEqual([v["id"] for v in data["vertices"]], ["E1", "E2"])
        self.assertEqual(data["cells"], [{"id": "C1", "dimension": 1, "vertices": ["E1", "E2"],
                                         "faces": ["E1", "E2"]}])
        strata[0]["groups"].append(piece(["x+y"]))
        with self.assertRaisesRegex(InvalidChart, "ambiguous"):
            _graph_request("stratum_faces", ["x", "y"], **args)

    def test_generic_component_unknown_declines_the_graph(self):
        from unittest.mock import patch
        from engine import native_polynomial as np
        from engine.general_resolution import verify_candidate
        from engine.resolution_search import _candidate

        coordinates = np.symbols("x y z")
        x, y, z = coordinates
        candidate = _candidate(coordinates, (x, y), "fixture")
        with patch("engine.geometric_components.geometric_components", return_value=None) as proof:
            result = verify_candidate(coordinates, x*x-z*y*y, candidate, "intrinsic")
        self.assertIsNone(result)
        proof.assert_called()

    def test_wide_smooth_locus_and_snc_return_exact_bad_ideals(self):
        from engine import native_polynomial as np
        from engine.blowup_ideal import radical_subset, smooth_locus, verify_snc
        from engine.candidate_field import indexed_field

        axes = np.symbols("wide0:35")
        x, y = axes[:2]
        self.assertTrue(radical_subset(axes, (1,), (x, x-1)))
        self.assertFalse(radical_subset(axes, (1,), (x,)))
        exponent = 2**65 + 17
        coefficient = 2**301 + 3
        self.assertTrue(radical_subset(axes, (x,), (coefficient*x**exponent,)))
        thick = smooth_locus(axes, (coefficient*x**exponent,), 34)
        self.assertFalse(thick["empty"])
        self.assertFalse(thick["smooth"])
        self.assertEqual(thick["bad_ideal"], (x**(exponent-1),))

        field, a = indexed_field((1, 0, -2), 1)
        simple = x+a*y
        proof = verify_snc(axes, (), (simple,), ((y,), (x,)), 35, domain=field)
        self.assertFalse(proof["ok"])
        self.assertEqual([z["components"] for z in proof["bad_strata"]], [["strict","E1","E2"]])
        self.assertEqual(proof["bad_strata"][0]["bad_ideal"], (x, y))
        self.assertTrue(smooth_locus(axes, (x*(x-1), x*y, (x-1)*axes[2], y*axes[2]), 33)["smooth"])

    def test_projection_eliminant_over_marked_field_and_wide_graph_pivot(self):
        from engine import native_polynomial as np
        from engine.candidate_field import indexed_field
        from engine.component_ring import ComponentRing

        u, t = np.symbols("u t")
        field, a = indexed_field((1, 0, -2), 1)
        ring = ComponentRing((u, t), field)
        # The second component lies over a parameter value. Recover the
        # generic eliminant by taking a gcd in K[u,t], with theta a coefficient.
        generators = ((t-a)*(t+u), (u-1)*(t-a))
        shape = ring.request("projection_shape", dimension=1, generators=[ring.row(f) for f in generators])
        self.assertEqual(ring.expression(shape["eliminant"]), t-a)

        from engine.geometric_components import geometric_components
        self.assertIsNone(geometric_components((u, t), (), 2**130, domain=field))
        self.assertEqual(geometric_components((u, t), (1,), 2**130, domain=field), [])

        n = 2**65 + 11
        k = 2**301 + 3
        pivot = a*t + k*u**n
        step = ring.request("decomposition_step", generators=[ring.row(pivot), ring.row(u*u-1)], dimension=0)
        self.assertEqual((step["action"], step["axis"]), ("graph", 1))
        self.assertEqual(ring.expression(step["value"]), -k*a*u**n/2)

    def test_orders_with_wide_exponents_and_coefficients(self):
        exponent = 2**65 + 11
        coefficient = str(2**301 + 3)
        request = dict(base=1, ambient=[], prime=["x"], equation=f"{coefficient}*x^{exponent}")
        self.assertEqual(_graph_request("orders", ["x"], **request), [str(exponent), "1"])
