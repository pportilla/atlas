"""Checks for the full Newton resolver, including non-simplicial dual cells."""
import ast
from collections import Counter, defaultdict
from fractions import Fraction
from functools import cache
from itertools import combinations
from math import gcd
import unittest

from engine.newton_backend import resolve_newton
from engine.newton_fan import build_fan
from engine.polynomial import parse


VARS = list("xyzt")
EXAMPLE = "x^5 + z*y^3 + z^15 + t^4*x + t^8"


def add(p, q, sign=1):
    out = p.copy()
    for k,v in q.items():
        out[k] = out.get(k,0)+sign*v
        if not out[k]:
            del out[k]
    return out


def mul(p, q):
    out = {}
    for e,a in p.items():
        for f,b in q.items():
            k = tuple(i+j for i,j in zip(e,f))
            out[k] = out.get(k,0)+a*b
    return {k:v for k,v in out.items() if v}


@cache
def expression(source):
    return ast.parse(source.replace("^","**"), mode="eval").body


def eval_poly(source, symbols, dimension):
    # Output exponents can be far larger than the input degree limit. Use
    # independent exact sparse arithmetic to compare polynomial identities.
    unit = {(0,)*dimension: 1}

    def visit(node):
        if isinstance(node, ast.Name):
            return symbols[node.id]
        if isinstance(node, ast.Constant):
            return {(0,)*dimension: node.value}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return {k:-v for k,v in visit(node.operand).items()}
        p = visit(node.left)
        if isinstance(node.op, ast.Pow):
            power = node.right.value
            if len(p)==1:
                (e,c),=p.items()
                return {tuple(i*power for i in e):c**power}
            out=unit
            while power:
                if power%2: out=mul(out,p)
                power//=2
                if power: p=mul(p,p)
            return out
        q = visit(node.right)
        if isinstance(node.op, ast.Mult): return mul(p,q)
        if isinstance(node.op, ast.Add): return add(p,q)
        if isinstance(node.op, ast.Sub): return add(p,q,-1)
        if isinstance(node.op, ast.Div):
            return {k:Fraction(v,q[(0,)*dimension]) for k,v in p.items()}
        raise AssertionError(type(node))

    return visit(expression(source))


class GeneralNewton(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.poly = parse(EXAMPLE, VARS)
        cls.result = resolve_newton(cls.poly,VARS,"intrinsic")

    def test_change_of_coordinates_and_every_pullback(self):
        result = self.result
        self.assertEqual(result["engine"], "newton-polyhedron")
        self.assertEqual(result["dual_complex"]["status"], "complete")
        # The original coordinates have boundary modifications. The rational
        # change recorded by the solver yields an isomorphism off the origin.
        self.assertFalse(build_fan(self.poly)["isomorphism_off_origin"])
        self.assertTrue(result["fan"]["isomorphism_off_origin"])
        n=len(VARS)
        identity = tuple(tuple(int(i==j) for j in range(n)) for i in range(n))
        base = {s:{e:1} for s,e in zip(VARS,identity)}
        linear = result["certificate"]["linear_map_to_input"]
        transformed = {s:{e:c for e,c in zip(identity,row) if c} for s,row in zip(VARS,linear)}
        adapted = eval_poly(result["certificate"]["adapted_polynomial"],base,n)
        self.assertEqual(eval_poly(EXAMPLE,transformed,n),adapted)
        self.assertEqual({tuple(ch["ray_indices"]) for ch in result["charts"]},
                         {tuple(c) for c in result["fan"]["cones"]})
        for chart in result["charts"]:
            symbols={s:{e:1} for s,e in zip(chart["coordinates"],identity)}
            mapped={x:eval_poly(expr,symbols,n) for x,expr in chart["map"].items()}
            strict=eval_poly(chart["strict_transform"],symbols,n)
            factor=eval_poly(chart["pullback_factor"],symbols,n)
            self.assertEqual(eval_poly(EXAMPLE,mapped,n),mul(strict,factor),chart["id"])

    def test_all_compact_faces_independent_of_the_emitted_fan(self):
        # Enumerate supporting hyperplanes directly from the input after the
        # recorded change of coordinates. Every compact facet contains four
        # affinely independent support points and has a strictly positive
        # inward normal. Coordinatewise dominated points cannot minimize it.
        adapted = parse(self.result["certificate"]["adapted_polynomial"], VARS)
        points = [e for e in adapted if not any(
            a != e and all(i <= j for i, j in zip(a, e)) for a in adapted)]

        def minor(rows, omit):
            (a,b,c), (d,e,f), (g,h,i) = [
                [v for k,v in enumerate(row) if k != omit] for row in rows]
            return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)

        facets = set()
        for origin, *other in combinations(points, 4):
            rows = [[a-b for a,b in zip(e,origin)] for e in other]
            w = [(-1)**i * minor(rows,i) for i in range(4)]
            if max(w) < 0:
                w = [-v for v in w]
            if min(w) <= 0:
                continue
            heights = {e:sum(i*j for i,j in zip(e,w)) for e in adapted}
            if heights[origin] == min(heights.values()):
                facets.add(frozenset(e for e,h in heights.items() if h==heights[origin]))
        self.assertEqual(len(facets),4)
        self.assertTrue(all(len(facet)==4 for facet in facets))  # tetrahedra
        faces = {frozenset(face) for facet in facets for size in range(1,5)
                 for face in combinations(facet,size)}
        self.assertEqual(Counter(map(len, faces)), {1:6,2:13,3:12,4:4})
        reported = {frozenset(map(tuple, p["exponents"])) for p in
                    self.result["certificate"]["compact_faces"]}
        self.assertEqual(reported,faces)

    def test_components_and_boundaries_on_every_orbit(self):
        result=self.result
        p=parse(result["certificate"]["adapted_polynomial"],VARS)
        dual=result["dual_complex"]
        vertices={v["id"]:v for v in dual["vertices"]}
        cells={c["id"]:c for c in dual["cells"]}
        face_ids=set(vertices)|set(cells)
        grouped=defaultdict(list)
        for v in vertices.values():
            grouped[(v["ray_index"],)].append(v)
        for c in dual["cells"]:
            subset=tuple(sorted(vertices[i]["ray_index"] for i in c["vertices"]))
            grouped[subset].append(c)
            self.assertEqual(c["dimension"],len(subset)-1)
            self.assertEqual(len(c["faces"]),len(subset))
            self.assertTrue(set(c["faces"])<=face_ids)
            if c["dimension"]==1:
                self.assertEqual(set(c["vertices"]),set(c["faces"]))
            else:
                self.assertEqual(
                    {frozenset(cells[i]["vertices"]) for i in c["faces"]},
                    {frozenset(q) for q in combinations(c["vertices"],len(subset)-1)})
            if "root" in c:
                for f in c["faces"]:
                    small=(cells.get(f) or vertices[f])
                    if "root" in small:
                        self.assertEqual(small["root"],c["root"])
        rays=result["fan"]["rays"]
        minima=[]
        for r in rays:
            values={e:sum(a*b for a,b in zip(r,e)) for e in p}
            minima.append({e for e,v in values.items() if v==min(values.values())})
        seen=set()
        for cone in result["fan"]["cones"]:
            interior=sorted(i for i in cone if i>=len(VARS))
            for size in range(1,len(interior)+1):
                for subset in combinations(interior,size):
                    if subset in seen: continue
                    seen.add(subset)
                    support=sorted(set.intersection(*(minima[i] for i in subset)))
                    self.assertTrue(support)
                    if len(support)==1:
                        count=0
                    else:
                        # A segment has as many components as its lattice
                        # length; a nondegenerate face of dimension >=2
                        # gives a connected smooth torus section.
                        a,b=support[0],support[-1]
                        diff=[j-i for i,j in zip(a,b)]
                        k=next(i for i,x in enumerate(diff) if x)
                        edge=all(all((e[j]-a[j])*diff[k]==(e[k]-a[k])*diff[j]
                                     for j in range(len(a))) for e in support)
                        count=gcd(*diff) if edge else 1
                    self.assertEqual(len(grouped[subset]),count,subset)
        self.assertTrue(set(grouped)<=seen)
        # A degree-four edge face of this polyhedron yields four components
        # of a single pairwise intersection. Preserve the four corresponding
        # two-cells and attach each one to the edge with the same root.
        repeated=[cs for sub,cs in grouped.items() if len(sub)==2 and len(cs)==4
                  and len({tuple(c["vertices"]) for c in cs})==1]
        self.assertTrue(repeated)
        for group in repeated:
            self.assertEqual(len({c["root"] for c in group}),4)
            ids={c["id"] for c in group}
            adjacent=[c for c in cells.values() if ids.intersection(c["faces"])]
            self.assertTrue(adjacent)
            for c in adjacent:
                edge,=ids.intersection(c["faces"])
                self.assertEqual(c["root"],cells[edge]["root"])

    def test_edge_character_gluing_across_charts(self):
        result=self.result
        dual=result["dual_complex"]
        vertices={v["id"]:v for v in dual["vertices"]}
        families=result["root_families"]
        seen=Counter()
        n=len(VARS)
        identity=[tuple(int(i==j) for i in range(n)) for j in range(n)]
        for c in result["charts"]:
            rr=[result["fan"]["rays"][i] for i in c["ray_indices"]]
            symbols={s:{e:1} for s,e in zip(c["coordinates"],identity)}
            for v in c["exceptional"]:
                self.assertIn(v["id"],vertices)
                vertex=vertices[v["id"]]
                if "root" not in vertex: continue
                axis,eq=v["ideal"]
                self.assertEqual(rr[c["coordinates"].index(axis)],vertex["ray"])
                root=vertex["root"]
                pos,neg=eq.split("-"+root+"*")
                (e,), (f,) = eval_poly(pos,symbols,n), eval_poly(neg,symbols,n)
                d=families[root.rsplit("_",1)[0]]["direction"]
                # Each chart's ratio represents the same primitive character
                # of the original torus. Thus the indexed roots agree on
                # every overlap, including charts where it is a reciprocal.
                self.assertEqual([p-q for p,q in zip(e,f)],
                                 [sum(i*j for i,j in zip(ray,d)) for ray in rr])
                seen[root]+=1
        self.assertGreater(max(seen.values()),1)

    def test_integral_cellular_boundaries_of_four_variable_germ(self):
        from sympy import ZZ
        from sympy.polys.matrices import DomainMatrix
        from sympy.polys.matrices.normalforms import invariant_factors

        complex_ = self.result['dual_complex']
        layers = [complex_['vertices'], *(
            [cell for cell in complex_['cells'] if cell['dimension']==k] for k in (1,2))]
        boundaries = []
        for lower, upper in zip(layers,layers[1:]):
            positions = {face['id']:i for i,face in enumerate(lower)}
            entries = {}
            for j,cell in enumerate(upper):
                order = cell['vertices']
                for face_id in cell['faces']:
                    i = positions[face_id]
                    facet = lower[i].get('vertices',[face_id])
                    omitted, = (k for k,v in enumerate(order) if v not in facet)
                    self.assertEqual(order[:omitted]+order[omitted+1:], facet)
                    entries.setdefault(i,{})[j] = ZZ((-1)**omitted)
            boundaries.append(DomainMatrix(entries,(len(lower),len(upper)),ZZ))
        d1,d2 = boundaries
        self.assertTrue((d1*d2).is_zero_matrix)
        self.assertEqual([len(layer) for layer in layers],[99,251,153])
        self.assertEqual((d1.rank(),d2.rank()),(98,153))
        # The image of d2 is saturated, so H1 has no torsion either.
        self.assertEqual({abs(s) for s in invariant_factors(d2)},{1})

    def test_degenerate_subface_refuses_connectedness_claim(self):
        # On the dense torus the faces x+y=0 and x+y+z=0 are disjoint and
        # smooth. Their common edge on z=0 is repeated, which disqualifies
        # the polygon from the nondegenerate connectedness theorem. The
        # modification is completed by blowups. Its atlas determines some
        # exceptional curves even when a geometric component count is open.
        p=parse("(x+y)*(x+y+z)+z^4",list("xyz"))
        result=resolve_newton(p,list("xyz"),"intrinsic")
        self.assertEqual(result["status"],"resolved")
        dual = result["dual_complex"]
        self.assertIn(dual["status"], {"partial", "complete"})
        self.assertTrue(any(v["id"].startswith("P1.") for v in dual["vertices"]))
        if dual["status"] == "partial":
            self.assertIn("components", dual["reason"])
        self.assertTrue(any(step.get("centers") for step in result["steps"]))
        # The germ is uv + (v-u)^4 with u = x+y, v = x+y+z: two smooth surfaces
        # crossing normally, so the strata must give 1/(s+1)^2.
        from engine.topological import topological_zeta
        result.update(variables=list("xyz"),polynomial="(x+y)*(x+y+z)+z^4",mode="intrinsic",origin=[0,0,0])
        self.assertEqual(topological_zeta(result)["expression"],"(s + 1)^(-2)")

    def test_failed_repair_skips_chart_expansion(self):
        from unittest.mock import patch
        names = list("xyz")
        polynomial = parse("(x+y)^2+z^3+x^3", names)
        with patch("engine.newton_backend.transform_atlas") as transform:
            self.assertIsNone(resolve_newton(polynomial, names, "embedded", defer_outputs=True))
            transform.assert_not_called()

    def test_nonprimitive_support_lattice(self):
        # Support differences span a proper finite-index sublattice of the
        # rank-two saturated plane. The Fermat curve is still irreducible.
        v=list("xyz")
        r=resolve_newton(parse("x^4+y^4+z^4",v),v,"intrinsic")
        self.assertEqual(len(r["dual_complex"]["vertices"]),1)
        self.assertEqual(r["dual_complex"]["cells"],[])

    def test_smoothness_certificate_localizes_to_the_origin(self):
        v = list("xyz")
        # The repeated plane x = 1 is a unit factor at the origin. It still
        # gives singular points on the full affine charts outside the germ.
        result = resolve_newton(parse("(1-x)^2*(x^3+y^3+z^3)",v),v,"intrinsic")
        self.assertEqual(result["status"], "resolved")
        self.assertIn("neighborhood of the fiber over the origin",
                      result["certificate"]["description"])
        chart = result["charts"][0]
        units = [tuple(int(i==j) for i in range(3)) for j in range(3)]
        symbols = {s:{e:1} for s,e in zip(chart["coordinates"],units)}
        self.assertTrue(all(sum(eval_poly(s,symbols,3).values())==1
                            for s in chart["map"].values()))
        strict = eval_poly(chart["strict_transform"],symbols,3)
        self.assertEqual(sum(strict.values()),0)
        for axis in range(3):
            self.assertEqual(sum(e[axis]*c for e,c in strict.items()),0)


if __name__=="__main__":
    unittest.main()
