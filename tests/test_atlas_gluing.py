"""Overlaps of marked number fields and punctured blowup charts."""
import unittest
from dataclasses import replace

import sympy as sp

from engine import atlas
from engine.atlas_gluing import _character_field, _geometric_primes, glued_complex
from engine.chain_backend import resolve_chain
from engine.local_blowups import Chart
from engine.plane_backend import resolve_plane
from engine.polynomial import parse
from engine.resolve import resolve


class FieldAndOpenOverlaps(unittest.TestCase):
    def test_ordinary_diagonal_uses_named_rational_projections_and_ordered_holes(self):
        from fractions import Fraction
        from engine import native_polynomial as np
        from engine.atlas_gluing import _ordinary_overlaps
        from engine.blowup_ideal import saturate

        x, y = np.symbols("x y")
        z = np.Symbol("z")
        a = Chart((x, y), (), (), (), (), (("u", (x+1)*Fraction(-3,7)), ("v", y), ("w", np.S.Zero)), (),
                  removed=((x-1, y), ()))
        b = Chart((z,), (), (), (), (), (("w", np.S.Zero), ("v", z*z), ("u", (z+2)*Fraction(5,13))), (),
                  removed=((z+1,),))
        boundaries = [{"D": (np.S.Zero, x, y)}, {"E": (z,)}]
        factory = _ordinary_overlaps([a,b], boundaries)
        if factory is None:
            self.skipTest("the packed native overlap operation is unavailable")
        coords, left, right, graph, holes = factory(0, 1)
        xl, yl, zr = coords
        self.assertEqual(left, {x: xl, y: yl})
        self.assertEqual(right, {z: zr})
        self.assertEqual(holes, ((xl-1, yl), (), (zr+1,)))
        self.assertEqual(graph, saturate(coords, ((xl+1)*Fraction(-3,7)-(zr+2)*Fraction(5,13), yl-zr*zr), xl*zr))
        for ineligible in (replace(a, ambient=(x,)), replace(a, coefficient_field=(x,x*x-2,2,"root")),
                           replace(a, repair_projection=(0,(x,y)))):
            self.assertIsNone(_ordinary_overlaps([ineligible,b], boundaries)(0,1))

        # Empty exceptional ideals must still reach the general validator;
        # the operation must likewise decline exponents outside its packed range.
        self.assertIsNone(_ordinary_overlaps([a,b], [{"D": ()}, {}])(0,1))
        wide = replace(a, coordinates=(("u",x**65536), ("v",y), ("w",np.S.Zero)))
        self.assertIsNone(_ordinary_overlaps([wide,b], boundaries)(0,1))
        with self.assertRaisesRegex(ValueError, "different base coordinates"):
            _ordinary_overlaps([replace(a, coordinates=(("bad",x),)),b], boundaries)(0,1)

    def test_generic_product_diagonal_and_character_on_wide_sparse_charts(self):
        from engine import native_polynomial as np
        from engine.atlas_gluing import _ordinary_overlaps
        from engine.chart_gluing_kernel import chart_input, values
        from engine.kernel_json import kernel_json

        x, y, a, b = np.symbols("x y a b")
        degree = 2**64 + 17
        scale = 2**180 + 31
        shear = scale*x**degree
        chart = Chart((y,x), (), (("D",1,1,(x,)),), (), (), (("Y",y+shear),("X",x)), ())
        other = Chart((b,a), (), (("D",1,1,(a,)),), (), (), (("Y",b),("X",a)), (), removed=((b,),))
        packed = _ordinary_overlaps((chart,other),({"D":(x,)},{"D":(a,)}))
        self.assertIsNone(packed(0,1))
        diagonal = kernel_json("atlas_chart_geometry","diagonal",a=chart_input(chart),b=chart_input(other),
                               transition=None,descended=False)
        product = (*chart.variables,*other.variables)
        self.assertEqual(set(values(product,diagonal["graph"])), {y-b+scale*a**degree,x-a})
        self.assertEqual([values(product,h) for h in diagonal["removed"]],[(b,)])

        z = np.Symbol("z")
        monomial = x*y**degree
        carrier = Chart((x,y,z), (), (("D",1,1,(z,)),("strict",1,1,(monomial**2-2,))),
                        (), (), (), ())
        (marked,), ratio = _character_field([carrier])
        theta, relation, _, _ = marked.coefficient_field
        section = (relation,z,monomial**2-2*scale**2)
        self.assertCountEqual(_geometric_primes(marked,section,1,ratio),
                              [(z,monomial-scale*theta),(z,monomial+scale*theta)])

    def test_branch_aliases_preserve_parallel_edges_and_loops(self):
        from engine.strata import _vertex_aliases

        def graph(labels, edges):
            return {"vertices": [{"id": label} for label in labels],
                    "cells": [{"vertices": vertices} for vertices in edges]}

        old = graph(("Eu", "Ev", "F"), [("Eu", "Eu"), ("Eu", "F"), ("Eu", "F"), ("Ev", "F")])
        new = graph(("E.1", "E.2", "F"), [("E.2", "E.2"), ("E.1", "F"), ("E.2", "F"), ("E.2", "F")])
        self.assertEqual(_vertex_aliases(new, old), {"E.1": "Ev", "E.2": "Eu", "F": "F"})
        new["cells"][0]["vertices"] = ("E.1", "E.1")
        self.assertIsNone(_vertex_aliases(new, old))

    def test_conjugate_character_curves_attach_on_the_same_branch(self):
        result = resolve({"polynomial": "2*x^2+3*y^4-z^7+x*y*z^4", "variables": list("xyz"),
                          "mode": "intrinsic"})
        dual = result["dual_complex"]
        self.assertEqual(dual["status"], "complete")
        self.assertNotEqual(dual.get("source"), "resolution engine certificate")
        self.assertEqual(len(dual["vertices"]), 8)
        self.assertEqual(len(dual["cells"]), 7)
        roots = {v["id"]: v.get("root") for v in dual["vertices"]}
        arms = [tuple(roots[v] for v in e["vertices"]) for e in dual["cells"]
                if all(roots[v] and roots[v].startswith("alpha12") for v in e["vertices"])]
        self.assertCountEqual(arms, [(root, root) for root in ("alpha12_1", "alpha12_2") for _ in range(2)])

    def test_primitive_character_with_two_nonsquare_classes(self):
        x, y, z = sp.symbols("x y z")
        c = Chart((x, y, z), (), (("D", 1, 1, (z,)), ("strict", 1, 1, (x**2*y**4-2,))),
                  (x, y, z), (), (("x", x), ("y", y), ("z", z)), ())
        (marked,), d = _character_field([c])
        theta, relation, _, _ = marked.coefficient_field
        eq = (relation, z, x**2*y**4-8)
        primes = _geometric_primes(marked, eq, 1, d)
        self.assertCountEqual(primes, [(z, x*y**2-2*theta), (z, x*y**2+2*theta)])
        self.assertIsNone(_geometric_primes(marked, (relation, z, x**2*y**4-3), 1, d))

    def test_split_incidences_over_a_fixed_quadratic_field(self):
        names = list("xyz")
        p = parse("x^2+y^2+z^6", names)
        record = resolve_chain(p, names, "intrinsic")
        record["variables"] = names
        charts = atlas.from_json(atlas.to_json(atlas.from_record(record, p)))
        dual = glued_complex(charts, 3, intrinsic=True)
        self.assertEqual(dual["status"], "complete")
        adjacency = {v["id"]: set() for v in dual["vertices"]}
        for edge in dual["cells"]:
            a, b = edge["vertices"]
            adjacency[a].add(b)
            adjacency[b].add(a)
        self.assertEqual(len(adjacency), 5)
        self.assertEqual(adjacency["E3"], {"E2.1", "E2.2"})
        self.assertEqual(len(adjacency["E1.1"]), 1)
        self.assertEqual(len(adjacency["E1.2"]), 1)
        self.assertNotEqual(adjacency["E1.1"], adjacency["E1.2"])

    def test_meeting_curves_of_one_ambient_divisor_in_the_minimal_resolution(self):
        names = list("xyz")
        for text in ("x*y-z^5", "x^2+y^2+z^5"):
            with self.subTest(text=text):
                p = parse(text, names)
                record = resolve_chain(p, names, "intrinsic")
                record.update(variables=names, mode="intrinsic")
                charts = atlas.from_record(record, p)
                expected = {frozenset(c["vertices"]) for c in record["dual_complex"]["cells"]}
                record["dual_complex"] = atlas.record_complex(record, charts)
                self.assertEqual(record["dual_complex"]["status"], "complete")
                self.assertNotEqual(record["dual_complex"].get("source"), "resolution engine certificate")
                edges = {frozenset(c["vertices"]) for c in record["dual_complex"]["cells"]}
                self.assertEqual(edges, expected)
                self.assertIn(frozenset(("E2u", "E2v")), edges)

    def test_public_odd_a2_has_one_internal_edge(self):
        result = resolve({"polynomial": "x*y-z^3", "variables": list("xyz"), "mode": "intrinsic"})
        dual = result["dual_complex"]
        self.assertEqual(dual["status"], "complete")
        self.assertNotEqual(dual.get("source"), "resolution engine certificate")
        self.assertCountEqual([v["id"] for v in dual["vertices"]], ["E1u", "E1v"])
        self.assertEqual([set(cell["vertices"]) for cell in dual["cells"]], [{"E1u", "E1v"}])

    def test_two_branches_can_meet_on_a_chart_overlap_without_gluing(self):
        x, y, z = sp.symbols("x y z")
        # On the graph surface z=x*y, the plane z=0 cuts two transverse
        # lines. Two copies of the chart describe the same neighbourhood;
        # the first owns its fibre and the second owns the empty region.
        c = Chart((x, y, z), (), (("D", 1, 1, (z,)), ("strict", 1, 1, (z-x*y,))),
                  (x, y, z), (), (("x", x), ("y", y), ("z", z)), ())
        overlap = replace(c, region=(sp.S.One,))
        found = glued_complex([c, overlap], 3, intrinsic=True)
        self.assertCountEqual([v["id"] for v in found["vertices"]], ["D.1", "D.2"])
        self.assertEqual([e["vertices"] for e in found["cells"]], [["D.1", "D.2"]])
        # The same two distinct curves can be tangent. Their graph would
        # require an additional blowup before the crossing is transverse.
        tangent = replace(c, components=(c.components[0], ("strict", 1, 1, (z-y*(y-x**2),))))
        self.assertIsNone(glued_complex([tangent], 3, intrinsic=True))

    def test_higher_intersections_of_one_restricted_exceptional_divisor(self):
        from itertools import combinations
        from math import prod
        from engine.chart_expressions import runtime_expressions
        api = runtime_expressions()
        for dimension in (3, 4):
            with self.subTest(dimension=dimension):
                coordinates = api.symbols(f"x0:{dimension}")
                w = api.Symbol("w")
                variables = (*coordinates, w)
                images = tuple(w*v for v in coordinates) + (w,)
                strict = w-prod(coordinates)
                chart = Chart(variables, (), (("D", 1, 1, (w,)), ("strict", 1, 1, (strict,))),
                              images, (), tuple(zip(map(str, variables), images)), ())
                # Blowup of the origin with w as pivot. The second presentation
                # shears coordinates; over w=0, the branches still have distinct
                # generic points, though they meet on this nonempty overlap.
                move = {coordinates[0]: coordinates[0]+coordinates[1]}
                other = replace(chart, region=(api.S.One,),
                                components=(chart.components[0], ("strict", 1, 1, (strict.xreplace(move),))),
                                coordinates=tuple((k,g.xreplace(move)) for k,g in chart.coordinates),
                                fibre=tuple(g.xreplace(move) for g in chart.fibre))
                dual = glued_complex([chart, other], dimension+1, intrinsic=True)
                self.assertEqual(dual["status"], "complete")
                vertices = [v["id"] for v in dual["vertices"]]
                self.assertEqual(len(vertices), dimension)
                faces = {frozenset([v]): v for v in vertices}
                for cell in dual["cells"]:
                    ends = cell["vertices"]
                    self.assertCountEqual(cell["faces"], [faces[frozenset(k)] for k in combinations(ends,len(ends)-1)])
                    faces[frozenset(ends)] = cell["id"]
                self.assertEqual(len(faces), 2**dimension-1)
                tangent = replace(chart, components=(chart.components[0],
                                  ("strict", 1, 1, (w-coordinates[0]*(coordinates[0]-coordinates[1]**2),))))
                self.assertIsNone(glued_complex([tangent], dimension+1, intrinsic=True))

    def test_one_intrinsic_vertex_lies_in_two_ambient_divisors(self):
        from engine.chart_expressions import runtime_expressions
        api = runtime_expressions()
        x,y,z,w = variables = api.symbols("x y z w")
        rows = (("D",1,1,(x,)), ("E",1,1,(y,)), ("F",1,1,(z,)), ("strict",1,1,(y-x,)))
        images = (x*y*z,x*x*y*z,x*y*z*z,x*y*z*w)
        chart = Chart(variables, (), rows, images, (), tuple(zip(("a","b","c","d"),images)), ())
        dual = glued_complex([chart,replace(chart,region=(api.S.One,))],4,intrinsic=True)
        self.assertEqual(dual["status"], "complete")
        self.assertEqual(len(dual["vertices"]), 2)
        shared, = [v for v in dual["vertices"] if v.get("ambient_components") == ["D","E"]]
        edge, = dual["cells"]
        self.assertCountEqual(edge["vertices"], [shared["id"],"F"])
        from engine.strata import annotate_complex
        annotated = {**dual, "vertices": [{"id": v["id"], "dimension": 99} for v in dual["vertices"]]}
        self.assertEqual(annotate_complex(annotated, dual)["vertices"], dual["vertices"])

    def test_two_triangles_have_different_parallel_edges(self):
        from engine.chart_expressions import runtime_expressions
        api = runtime_expressions()
        x,y,z,w = variables = api.symbols("x y z w")
        # The sections of D and E on the threefold are disjoint lines at
        # x=0 and x=1. F cuts each in one point. The second chart exchanges
        # these roots, so counting faces or matching them by local order fails.
        base = w*y*z
        images = (base, x*base, w*base, z*base)
        rows = (("D",1,1,(y,)), ("E",1,1,(w,)), ("F",1,1,(z,)), ("strict",1,1,(w-y-x*(x-1),)))
        chart = Chart(variables, (), rows, images, (), tuple(zip(("X","Y","Z","W"), images)), ())
        move = {x: 1-x, y: y+z}
        other = replace(chart, components=tuple((k,n,j,tuple(g.xreplace(move) for g in ideal))
                                                for k,n,j,ideal in rows),
                        coordinates=tuple((k,g.xreplace(move)) for k,g in chart.coordinates),
                        fibre=tuple(g.xreplace(move) for g in images), region=(api.S.One,))
        for charts in ([chart,other], [other,chart]):
            dual = glued_complex(charts,4,intrinsic=True)
            self.assertEqual(dual["status"], "complete")
            edges = {c["id"]: c["vertices"] for c in dual["cells"] if c["dimension"] == 1}
            parallel = {k for k,v in edges.items() if set(v) == {"D","E"}}
            triangles = [c for c in dual["cells"] if c["dimension"] == 2]
            self.assertEqual((len(edges),len(parallel),len(triangles)), (4,2,2))
            self.assertCountEqual([f for c in triangles for f in c["faces"] if f in parallel], parallel)
            for c in triangles:
                self.assertCountEqual([edges[f] for f in c["faces"]], [["D","E"],["D","F"],["E","F"]])

    def test_birational_inverse_is_verified_on_the_marked_sheet(self):
        from engine.chart_expressions import runtime_expressions
        from engine.repair_gluing import _birational_parameters
        api = runtime_expressions()
        x,y,t = variables = api.symbols("x y t")
        m = t**2-2
        field = (t,m,2,"sqrt(2)")
        c = Chart(variables,(m,),(),(),(),(),(),coefficient_field=field)
        self.assertIsNotNone(_birational_parameters(c,(t*x,x*y)))
        # Over Q(t), x -> (t²-2)x has an inverse. Both its coordinate and
        # its inverse denominator vanish identically on the stated sheet.
        self.assertIsNone(_birational_parameters(c,(m*x,y)))
        self.assertIsNone(_birational_parameters(c,(x**2,y)))
        from engine.repair_gluing import _reduced_fraction, transition_on_field_sheet
        with self.assertRaisesRegex(ArithmeticError, "denominator"):
            _reduced_fraction(c,x,api.S.Zero)
        # In K=Q(sqrt(2)), the target inverse y=B/A specializes to
        # (x²-2)/(x-t). Cancellation makes the transition regular on x=t,
        # including this entire exceptional branch.
        source = (x-t,y*(x*x-2))
        target = (x,x*y)
        transfer, holes = transition_on_field_sheet(c,c,source,target)
        self.assertEqual(holes, ())
        from engine.groebner import groebner
        answer = groebner((m,),variables).reduce(transfer(y)-y*(x+t))[1]
        self.assertEqual(answer, 0)


    def test_a_regular_transition_can_contract_a_prime_outside_the_overlap(self):
        from engine.chart_expressions import runtime_expressions
        from engine.atlas_gluing import _prime_groups
        from engine.repair_gluing import transition_on_field_sheet
        api = runtime_expressions()
        x,y = axes = api.symbols("x y")
        charts = [Chart(axes,(),(),(),(),(),()),Chart(axes,(),(),(),(),(),())]
        maps = [(x,x*y),(x,y)]
        def transition(i,j):
            return transition_on_field_sheet(charts[i],charts[j],maps[i],maps[j])
        def overlap(i,j):
            self.fail("Both rational transitions are available.")
        # x=0 is contracted to a point on y=0. These two curves have no
        # common generic point where the forward and inverse maps agree.
        pieces = [((0,"strict"),(x,)),((1,"strict"),(y,))]
        self.assertEqual(len(_prime_groups(charts,pieces,overlap,transition)),2)

    def test_closed_diagonal_needs_a_common_regular_open(self):
        from engine.atlas_gluing import _prime_groups
        from engine.chart_expressions import runtime_expressions
        from tests.gluing_examples import crossed_blowup_charts

        variables, maps, _, graph, pairs = crossed_blowup_charts(runtime_expressions())
        # Over x=a=0 the diagonal equations give y=c=0. The inverse
        # images of the two exceptional surfaces coincide in this graph,
        # but both project to lines and the surfaces lie over different axes.
        charts = [Chart(v, (), (), (), (), tuple(zip("XYZ", m)), ()) for v, m in zip(variables, maps)]
        renames = [{v: v for v in chart.variables} for chart in charts]

        def overlap(i, j):
            return (*charts[i].variables, *charts[j].variables), renames[i], renames[j], graph, ()

        # The two lines (x,y)=(a,c)=0 have both projections dominant,
        # but the relation on them is the entire (z,b)-plane. The charts
        # have no transition there. The strict surfaces z=0 and c=0,
        # in contrast, agree via the inverse coordinates where x=a≠0.
        for name, left, right, decision in pairs:
            for order in ((0, 1), (1, 0)):
                with self.subTest(name=name, order=order):
                    pieces = [((i,"strict"), (left, right)[i]) for i in order]
                    groups = _prime_groups(charts, pieces, overlap, lambda i,j: None)
                    self.assertEqual(len(groups), 1 if decision == 1 else 2)

    def test_implicit_chart_uses_alternative_regular_denominators(self):
        from engine.atlas_gluing import _prime_groups
        from engine.chart_expressions import runtime_expressions
        from tests.gluing_examples import implicit_cone_charts

        variables, maps, ambient, graph, pairs = implicit_cone_charts(runtime_expressions())
        charts = [Chart(v, eqs, (), (), (), tuple(zip("XYZW", m)), ())
                  for v, m, eqs in zip(variables, maps, ambient)]
        names = [{v: v for v in chart.variables} for chart in charts]

        def overlap(i, j):
            return (*charts[i].variables, *charts[j].variables), names[i], names[j], graph, ()

        # On the cone x*w=y*z the regularity ideal for b is (x,z).
        # Along (x,y) only w/z works; along (z,w) only y/x works.
        for name, left, right, _ in pairs:
            for order in ((0, 1), (1, 0)):
                with self.subTest(name=name, order=order):
                    pieces = [((i,"strict"), (left, right)[i]) for i in order]
                    groups = _prime_groups(charts, pieces, overlap, lambda i,j: None)
                    self.assertEqual(len(groups), 1)

    def test_closed_diagonal_transports_wide_sparse_prime_equations(self):
        from engine import native_polynomial as np
        from engine.kernel_json import kernel_json

        x, y, a, b = np.symbols("x y a b")
        blocks = ((y,x), (b,a))
        power = 2**64 + 17
        coefficient = 2**180 + 31
        # The identity chart identifies y=C*x^N with b=C*a^N. The
        # opposite sign defines a distinct irreducible curve in char 0.
        for sign, expected in ((1,1), (-1,-1)):
            primes = (y-coefficient*x**power, b-sign*coefficient*a**power)
            for i, j in ((0,1), (1,0)):
                with self.subTest(sign=sign, source=i):
                    variables = (*blocks[i], *blocks[j])
                    got = kernel_json("atlas_component_geometry", "generic_overlap",
                                      left_width=2, right_width=2, removed=[],
                                      graph=[np.row(variables, f) for f in (x-a,y-b)],
                                      left_prime=[np.row(blocks[i], primes[i])],
                                      right_prime=[np.row(blocks[j], primes[j])])
                    self.assertEqual(got, expected)

    def test_glued_prime_representatives_on_full_and_partial_overlaps(self):
        from engine.atlas_gluing import _prime_groups
        from engine.chart_expressions import runtime_expressions
        x, left, right = runtime_expressions().symbols("x left right")
        chart = Chart((x,), (), (), (), (), (("x", x),), ())
        for removed in ((), ((x,),)):
            with self.subTest(removed=removed):
                charts = [chart, chart, replace(chart, removed=removed), chart]
                pieces = [((i, "strict"), (prime,)) for i, prime in enumerate((x, x, x-1, x))]
                comparisons = []

                def overlap(i,j):
                    comparisons.append((i,j))
                    a, b = {x: left}, {x: right}
                    holes = tuple(tuple(g.xreplace(swap) for g in hole)
                                  for k, swap in ((i, a), (j, b)) for hole in charts[k].removed)
                    return (left, right), a, b, (left-right,), holes

                def transition(i,j):
                    if (i,j) == (0,3):
                        # Only the restriction of the identity to D(x) is
                        # available; it supplies no comparison at x=0.
                        return (lambda f, basis: basis.reduce(f)[1]), ((x,),)

                groups = _prime_groups(charts, pieces, overlap, transition)
                self.assertEqual([{i for i,_ in group} for group in groups], [{0,1,3}, {2}])
                self.assertEqual(comparisons, [(0,1), (0,2), (1,3), (2,3)])

    def test_only_the_retained_complement_counts_and_separate_embeddings_glue(self):
        names = list("xy")
        for text in ("y^2-x^3", "(y^2-2*x^4)^2-x^11"):
            with self.subTest(text=text):
                p = parse(text, names)
                record = resolve_plane(p, names, "embedded")
                record["variables"] = names
                charts = atlas.from_json(atlas.to_json(atlas.from_record(record, p)))
                self.assertTrue(any(c.removed for c in charts))
                dual = glued_complex(charts, 2)
                expected = {frozenset(edge["vertices"]) for edge in record["dual_complex"]["cells"]}
                edges = [frozenset(edge["vertices"]) for edge in dual["cells"]]
                self.assertEqual(set(edges), expected)
                self.assertEqual(len(edges), len(expected))
                self.assertEqual(len(dual["vertices"]) - len(edges), 1)
                # Blowing up the intersection removes its old edge completely.
                if text == "y^2-x^3":
                    self.assertNotIn(frozenset(("E1", "E2")), edges)


if __name__ == "__main__":
    unittest.main()
