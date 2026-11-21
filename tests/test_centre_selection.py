"""Independent comparisons for the geometric centre chosen across chart boundaries."""
from fractions import Fraction
from itertools import combinations
from random import Random
import unittest

from engine.centre_weights import level_weights, maximum_indices, surface_maxima
from engine.centres import _intersection
from engine import native_polynomial as np


class CentreSelection(unittest.TestCase):
    def test_global_maximum_agrees_with_exact_padded_order(self):
        rng = Random(28)
        huge = 2**130 + 1
        values = [Fraction(huge, huge+1), Fraction(huge+1, huge+2), Fraction(3, 2), float("inf")]
        for _ in range(80):
            width = rng.randrange(6)
            found = [None]
            for _ in range(7):
                value = tuple(rng.choices(values, k=rng.randrange(1, 4)))
                key = tuple(rng.choices([0, 1], k=rng.randrange(width+1)))
                found.extend([(value, key), (value, key+(0,)*(width-len(key)))])
            ranks = [None if item is None else (item[0], item[1]+(0,)*(width-len(item[1]))) for item in found]
            best = max(rank for rank in ranks if rank is not None)
            self.assertEqual(maximum_indices(found, width), [i for i,rank in enumerate(ranks) if rank==best])
        self.assertEqual(maximum_indices([None, None]), [])
        self.assertEqual(maximum_indices([((1,), (1,)), ((1,), (1,0))]), [1])

    def test_surface_rank_uses_exact_numbers_and_ordered_component_names(self):
        rows = [(2, 2**90, 1, ("E10", "E2")), (2, 2**90, 1, ("E2",)), (2, 2**90, 1, ("E2",))]
        self.assertEqual(surface_maxima([None, *((key, "points", ()) for key in rows)]), [2, 3])
        self.assertEqual(level_weights("birth_keys", (), births=["E2","E10","E1"], groups=[["E1","E2"], []]),
                         [[1,0,1], [0,0,0]])
        for descending in (True, False):
            labels = ["E2", "E2", "E1", "E10"]
            ordered = sorted(set(labels), reverse=descending)
            self.assertEqual(level_weights("label_order", (), labels=labels, descending=descending), ordered)
        self.assertEqual(level_weights("label_order", (), labels=labels, pairs=True),
                         list(map(list, reversed(list(combinations(sorted(set(labels)), 2))))))

    def test_points_and_curves_split_by_owned_incidence(self):
        self.assertEqual(level_weights("direct_stratum", (), labels=["E2","E1","E2","E3"],
                                       subsets=[["E1"], ["E1","E3"], ["E2","E3"]]), ["E2","E3"])
        mixed = [[["E1"], ["E1","E2"], ["E2","E1"]], [], [["E2"]]]
        self.assertEqual(level_weights("point_signatures", (), charts=mixed, single=False), [[1,2],[],[]])
        self.assertEqual(level_weights("point_signatures", (), charts=mixed, single=True), [[1],[],[]])
        self.assertIsNone(level_weights("point_signatures", (), charts=[[["E1"]], [], [["E1"]]], single=False))
        for single in (True, False):
            self.assertIsNone(level_weights("point_signatures", (), charts=[[["E1"],["E1"]], [["E1"]]], single=single))
            # Slopes 1 and 2 can appear in reverse order in the reciprocal chart.
            self.assertIsNone(level_weights("point_signatures", (), charts=[[["E1"]]*2, [["E1"]]*2], single=single))
        for parts in ([],[False],[False,True]):
            self.assertEqual(level_weights("direct_locus", (), parts=parts),any(parts))
            for outside in (True,False):
                self.assertEqual(level_weights("split_curve", (), outside=outside, parts=parts),outside and any(parts))

    def test_native_closure_intersection_does_not_narrow_coefficients_or_exponents(self):
        x,y = variables = np.symbols("x y")
        q = Fraction(2**110+3, 2**75+1)
        self.assertEqual(_intersection(variables, (x**(2**65),), (y-q,)), (x**(2**65)*(y-q),))
        self.assertEqual(_intersection(variables, (x*(x-y), 2*y,), (1,)), (x*x,y))
        self.assertEqual(_intersection(variables, (), (x,)), ())

    def test_wide_crossing_loci_and_cartier_order_match_the_packed_chart(self):
        from unittest.mock import patch
        from engine import kernel
        from engine.local_blowups import _failures, _order, Chart, GiveUp
        x,y = variables = np.symbols("x y")
        samples = [((x*y,), (x+y,), (y,)), ((x**2-y**3,), (x,), (y-1,))]
        for divisors in samples:
            for region in ((), (x,), (x-1,)):
                self.assertEqual(kernel.wide_crossings(variables, 2, (), divisors, (), region),
                                 kernel.crossings(variables, 2, (), divisors, (), region))
        ample = 2**65
        p = x**ample
        chart = Chart(variables, (), (("strict", 1, 1, (p,)),), (), (x-1,), (), ())
        with patch.object(kernel, "crossings", return_value=None):
            over, owned = _failures(chart, 2)
        self.assertEqual(over, {("strict",): (x**(ample-1),)})
        self.assertEqual(owned, set())
        self.assertEqual(_order(variables, (x-2*y,), y, (x**3,)), 3)
        self.assertEqual(_order(variables, (), y, (x*y**2, y**6)), 2)
        self.assertEqual(_order(variables, (), y, (x**ample*y**3,)), 3)
        with self.assertRaises(GiveUp):
            _order(variables, (x,), y, (x,), limit=3)

    def test_retained_complements_use_radical_containment_and_preserve_hole_order(self):
        from engine.centre_weights import centre_rows
        x,y = variables = np.symbols("x y")
        holes = [(x*x,), (y,), (x-1,), (x, y)]
        result = level_weights("retained_centres", (), width=2, ambient=[], centre=centre_rows(variables,(x,)),
                               holes=[centre_rows(variables,h) for h in holes])
        self.assertEqual(result, [1,2])

    def test_linear_graph_recognition_in_permuted_ambient_and_wide_coefficients(self):
        from engine.local_blowups import Chart, _linear
        x,y,p,q = np.symbols("x y p q")
        variables = (q, y, p, x)
        chart = Chart(variables, (p*q,), (), (), (), (), ())
        large = Fraction(2**108+1, 3**43)
        centre = (2*x-2*large*p**(2**65), y-q, p*q*x)
        self.assertEqual(_linear(chart, centre, 2), [(y, q), (x, large*p**(2**65))])
        self.assertIsNone(_linear(chart, centre, 1))
        self.assertIsNone(_linear(chart, (x*x-p, y-q), 2))
        self.assertIsNone(_linear(chart, (x, y, p), 2))

    def test_generators_and_principal_preserve_the_quotient_and_presentation_order(self):
        from engine.local_blowups import _generators
        from engine.centres import _principal, GiveUp
        x,y,p,q = variables = np.symbols("x y p q")
        ambient = (p*y-q*x,)
        self.assertEqual(_generators(variables, ambient, (x, y, p*y)), (x,y))
        self.assertEqual(_generators(variables, ambient, ambient), ())
        self.assertEqual(_principal(variables, (x-y,), (x-y,y,x)), y)
        self.assertEqual(_principal(variables, (), (p*p,p,p*y)), p)
        with self.assertRaises(GiveUp):
            _principal(variables, ambient, (x,y))

    def test_finite_piece_display_order_and_unique_strict_surface(self):
        from engine.centre_weights import ordered_parts
        from engine.local_blowups import Chart, _Canonical
        x,y = np.symbols("x y")
        p = np.Dummy("x")
        variables = (p,y,x)
        parts = [(p+y,), (x+y,), (), (p,), (x,), (x*y+Fraction(1,3),), (y,p)]
        degrees = [1,2,3,1,1,1,1]
        self.assertEqual(ordered_parts(variables, parts, degrees),
                         sorted(range(len(parts)), key=lambda i: (degrees[i],np.default_sort_key(parts[i]))))
        chart = Chart((x,y), (), (("S1",1,1,(x,)),("S2",1,1,(y-1,)),("B1",1,2,(y,))), (), (), (), ())
        rule = _Canonical(chart.components)
        failures = [{("S1","B1"): (x,y)}, {("S1","S2"): (x,y-1,)},
                    {("S1","B1"): (x,y), ("S2","B1"): (y-1,)}, {}]
        phase = rule._phase([chart]*4, [(f, {}) for f in failures], 3)
        self.assertEqual(phase, {"phase": "order", "charts": [1, 2], "surfaces": [0, None, None, None]})
        surface = rule._phase([chart, chart], [(failures[0], {}), ({}, {})], 3)
        self.assertEqual(surface, {"phase": "surface", "charts": [0], "surfaces": [0, None]})

    def test_unpacked_surface_priority_and_callback_interrupt(self):
        from engine import surface
        from engine.centre_frame import Frame
        from engine.centre_weights import centre_search, divisor_subsets, jet_order
        from engine.centres import GiveUp
        from unittest.mock import patch
        x,y,z = np.symbols("x y z")
        # QQ[sqrt(2)] is represented on both sheets by the same parameter
        # equation. Neither chart rank nor a unit scaling changes priority.
        q = Fraction(2**93+3, 11)
        work = [(x,y)]
        with patch("engine.kernel.surface_step", return_value=None):
            marked = list(surface._on_model((x,y,z), (), {"E1":q*y*y,"E2":x}, work,
                                            (z*z-2,), 0, None))
        self.assertEqual({entry[0] for entry in marked}, {(2,2,0,("E1",))})
        self.assertEqual(surface._plane_step((x,y), (), {"E1":y*y-x*x*x}, work)[:2], ((3,0,0,()),"points"))
        self.assertEqual(surface._implicit_step(Frame((x,y,z),(x,y),(z-x*y,)),
                                                {"E1":x*y,"E2":y}, work)[0], (2,1,0,("E1","E2")))
        self.assertEqual(divisor_subsets("abc",lambda group: set(group)<=set("ab"),largest=True), [("a","b")])
        with self.assertRaises(GiveUp):
            jet_order(1,lambda order: True)
        with self.assertRaises(TimeoutError):
            centre_search("maximum_threshold", lambda task: (_ for _ in ()).throw(TimeoutError()), count=3)

    def test_unpacked_linear_chart_residuals_holes_maps_and_sheet(self):
        from unittest.mock import patch
        from engine.local_blowups import Chart, _blow_up_linear
        from engine.blowup_ideal import ideal_basis
        x,y,t = variables = np.symbols("x y t")
        field=(t,t*t-2,2,"sqrt(2)")
        example=Chart(variables,(t*t-2,), (("strict",1,1,(x-t, (y+t)*(x-t))),("E1",2,3,(y+t,))),
                      (x-t,y+t), (x-t,), (("a",x),("b",y),("c",t**4)), (), ((x-t,y+t),),field)
        pivots=[(x,t),(y,-t)]
        packed=_blow_up_linear(example,pivots,"B1",1,2)
        with patch("engine.kernel.blowup_linear",return_value=None):
            fallback=_blow_up_linear(example,pivots,"B1",1,2)
        for a,b in zip(packed,fallback):
            self.assertEqual((a.variables,a.coordinates,a.history,a.coefficient_field,a.removed),
                             (b.variables,b.coordinates,b.history,b.coefficient_field,b.removed))
            self.assertEqual([c[:3] for c in a.components],[c[:3] for c in b.components])
            for g,h in zip(a.components,b.components):
                self.assertEqual(ideal_basis(a.variables,(*a.ambient,*g[3])),ideal_basis(b.variables,(*b.ambient,*h[3])))
        large=2**67
        chart=Chart((x,y),(),(("strict",1,1,((2**97+1)*x**large*y,)),), (x,y), (), (("x",x),("y",y)), ())
        children=_blow_up_linear(chart,[(x,0),(y,0)],"B1",1,2)
        self.assertEqual([r[1] for c in children for r in c.components if r[0]=="B1"],[large+1]*2)

    def test_boundary_cover_products_and_native_grouped_decision(self):
        from engine.centre_weights import centre_search, cover_products
        x,y = axes = np.symbols("x y")
        a = 2**70
        self.assertEqual(cover_products(axes,[(x**a,y),(x**a,y)]),[x**(2*a),x**a*y,y*y])
        self.assertEqual(cover_products(axes,[]),[1])
        self.assertEqual(cover_products(axes,[()]),[])
        def split(q):
            if q["query"] in {"regular","vanishes"}:
                return False
            return dict(on=[1],off=[2],overlap=True)
        self.assertIsNone(centre_search("grouped_loci", split, count=1,work=[0]))
        attempts = []
        def first(q):
            attempts.append((q["dependent"],q["rows"]))
            return len(attempts)==4
        self.assertTrue(centre_search("frame_cover",first, width=3,dimension=2,equations=2,
                                      divisors=[True,False,False]))
        self.assertEqual(attempts,[([1],[0]),([1],[1]),([2],[0]),([2],[1])])
