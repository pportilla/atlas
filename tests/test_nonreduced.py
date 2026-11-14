import unittest

import sympy as sp
from sympy import QQ

from engine.blowup_ideal import ideal_subset, saturate
from engine.nonreduced import nonreduced_resolution
from engine.polynomial import InputError, format_poly, parse
from engine.resolve import resolve


def weighted(source, variables, mode="embedded"):
    def support(q):
        return resolve({"polynomial":format_poly(q,variables),"variables":variables,"mode":mode})
    return nonreduced_resolution(parse(source,variables), variables, mode, support)


def expr(text, names):
    return sp.sympify(text.replace("^","**"),locals=names)


class WeightedEmbedded(unittest.TestCase):
    def test_quadratic_display_preserves_archived_signs_and_rational_cancellation(self):
        from engine.field_coefficient import rational_sqrt
        from engine.nonreduced import _weight_embedding
        for c,b,a in ((2,-3,1), (1,2,1), (-2,3,4), (7,2,3), (-7,-2,3), (0,-4,2)):
            for position, sign in ((1,1),(2,-1)):
                marked = f"PolynomialRootIndexed(List({c}, {b}, {a}), {position})"
                reference = (-b + sign*sp.sqrt(b*b-4*a*c))/(2*a)
                self.assertEqual(_weight_embedding(marked), str(reference))
        for q in ("-72/125", "0", "-1", "81/49", "2/3", str(2*1000000007**2)):
            self.assertEqual(str(rational_sqrt(q)), str(sp.sqrt(sp.Rational(q))))

    def test_quadratic_display_extracts_square_beyond_legacy_factoring_limit(self):
        from engine import groebner as gb
        from engine.field_coefficient import rational_sqrt
        if gb._MODE == "sympy" or not hasattr(gb._kernel(), "atlas_field_radical"):
            self.skipTest("requires the native radical display")
        p,q = 1000000007, 1000000009
        self.assertEqual(str(rational_sqrt(p*p*q)), f"{p}*sqrt({q})")
        self.assertEqual(str(rational_sqrt(-p*p*q)), f"{p}*sqrt({q})*I")

    def test_graph_lift_over_a_marked_field(self):
        from engine.nonreduced import _lift_graph
        x, y, r, t = sp.symbols("x y r weight_generator")
        atlas = {"coefficient_field": {"generator": "a", "minimal_polynomial_coefficients": ["1", "0", "1"],
                                       "exact_root": "I"}}
        chart = {"coordinates": ["x", "y", "r"], "map": {"x": "x", "y": "y"},
                 "ambient_ideal": ["y-x*(r+a)"], "pivot_generator": "x",
                 "exceptional": [{"id": "E", "equation": "x", "ideal": ["x"]}]}
        from copy import deepcopy
        archived = deepcopy(chart)
        orders = _lift_graph(atlas, chart, [(x**2+y**2, 2), (x, 3)], x+1, (x, y), {})
        self.assertEqual(orders, {"E": 7})
        self.assertEqual(chart["weight_field"], {"generator": t.name, "exact": "I"})
        del atlas["coefficient_field"]["minimal_polynomial_coefficients"]
        self.assertEqual(_lift_graph(atlas, archived, [(x**2+y**2, 2), (x, 3)], x+1, (x, y), {}), orders)
        self.assertEqual(archived, chart)
        frame = (x, y, r, t)
        ambient = (t**2 + 1, y-x*(r+t))
        proper = tuple(expr(f, {v.name: v for v in frame}) for f in chart["strict_transform"])
        expected = ((x+1)*(r**2+2*r*t)**2,)
        self.assertTrue(ideal_subset(frame, proper, (*ambient, *expected)))
        self.assertTrue(ideal_subset(frame, expected, (*ambient, *proper)))

    def test_weighted_pullback_in_the_marked_quadratic_field(self):
        from engine.nonreduced import _lift_free
        x, y, u, v, t = sp.symbols("x y u v weight_generator")
        atlas = {"coefficient_field": {"generator": "theta", "minimal_polynomial": "theta^2-(-1)"}}
        chart = {"coordinates": ["u", "v"], "map": {"x": "u", "y": "u*(v+theta)"},
                 "exceptional": [{"id": "E", "equation": "u"}]}
        orders = _lift_free(atlas, chart, [(x**2+y**2, 2), (x, 3)], x+1, (x, y))
        self.assertEqual(orders, {"E": 7})
        self.assertEqual(chart["weight_field"], {"generator": t.name, "exact": "I"})
        names = {w.name: w for w in (u, v, t)}
        self.assertEqual(expr(chart["strict_transform"], names), (u+1)*(v**2+2*t*v)**2)
        self.assertEqual(expr(chart["pullback_factor"], names), u**7)

    def test_uniform_weight_preserves_the_unit_on_marked_charts(self):
        r = weighted("(1+y)*((x^2+y^2)^2+x^7)^2", ["x", "y"])
        names = {name: sp.Symbol(name) for name in ("u", "v", *r["algebraic_numbers"])}
        marked = [ch for ch in r["charts"] if expr(ch["map"]["y"], names).has(
            *(names[a] for a in r["algebraic_numbers"]))]
        self.assertTrue(marked)
        for ch in marked:
            unit = 1 + expr(ch["map"]["y"], names)
            strict = expr(ch["reduced_strict_transform"], names)
            self.assertEqual(sp.expand(unit*strict**2 - expr(ch["strict_transform"], names)), 0)

    def test_mixed_orders_and_every_pullback(self):
        vs=["x","y"]
        source="(1+x)^2*(x-y)^2*(y^2-x^3)^3"
        result=weighted(source,vs)
        orders={v["id"]:v["total_multiplicity"] for v in result["dual_complex"]["vertices"]}
        self.assertEqual(orders["E1"],8)
        self.assertEqual(sorted(v["multiplicity"] for v in result["input_divisor"]["factors"]),[2,3])
        for ch in result["charts"]:
            symbols=sp.symbols(" ".join(ch["coordinates"]))
            coordinates=dict(zip(ch["coordinates"],symbols))
            if ch.get("weight_field"):
                f=ch["weight_field"];coordinates[f["generator"]]=sp.sympify(f["exact"])
            mapping={s:expr(a,coordinates) for s,a in ch["map"].items()}
            expected=expr(source,mapping)
            strict=expr(ch["strict_transform"],coordinates)
            exceptional=expr(ch["pullback_factor"],coordinates)
            self.assertEqual(sp.expand(expected-exceptional*strict),0)
            for v in ch["exceptional"]:
                self.assertEqual(v["total_multiplicity"],orders[v["id"]])

    def test_saturated_graph_ideals(self):
        from engine.general_resolution import search

        vars=list("xyz")
        result=nonreduced_resolution(parse("(x^2-y^2*z)^2",vars), vars, "embedded",
            lambda q: search(q,vars,"embedded",seconds=15))
        self.assertEqual(result["engine"],"verified-ideal-blowup")
        self.assertEqual(result["dual_complex"]["vertices"][0]["total_multiplicity"],4)
        for ch in result["charts"]:
            v=tuple(sp.Symbol(name) for name in ch["coordinates"])
            names=dict(zip(ch["coordinates"],v))
            ambient=tuple(expr(p,names) for p in ch["ambient_ideal"])
            strict=tuple(expr(p,names) for p in ch["strict_transform"])
            total=expr(ch["ambient_total_transform"],names)
            pivot=expr(ch["pivot_generator"],names)
            expected=saturate(v,(*ambient,total),pivot,domain=QQ)
            self.assertTrue(ideal_subset(v,expected,(*ambient,*strict)))
            self.assertTrue(ideal_subset(v,(*ambient,*strict),expected))

    def test_algebraic_chart_weighted_factor(self):
        r=weighted("((x^2+y^2)^2+x^7)^2*x^3",["x","y"])
        self.assertTrue(any(ch.get("weight_field") for ch in r["charts"]))
        ch=next(ch for ch in r["charts"] if ch.get("weight_field") and "weight_generator" in ch["strict_transform"])
        f=ch["weight_field"];x,y=sp.symbols("u v")
        names={"u":x,"v":y,f["generator"]:sp.sympify(f["exact"])}
        for label, definition in r["algebraic_numbers"].items():
            # For this fixture all recorded coefficients are Gaussian integers.
            from engine.nonreduced import _algebraics
            names.update(_algebraics({label:definition},{label}))
        mapping={s:expr(a,names) for s,a in ch["map"].items()}
        expected=((mapping["x"]**2+mapping["y"]**2)**2+mapping["x"]**7)**2 * mapping["x"]**3
        strict=expr(ch["strict_transform"],names)
        self.assertEqual(sp.expand(expected-strict*expr(ch["pullback_factor"],names)),0)

    def test_multiple_conjugate_pairs_and_native_labels(self):
        from engine.nonreduced import _weight_embedding
        x = sp.Symbol("x")
        # Non-real roots in this quintic have different real parts and
        # imaginary signs. The two numbering conventions run in different orders.
        for flint_index, sympy_index in enumerate((0, 4, 3, 2, 1), 1):
            label = f"PolynomialRootIndexed(List(1, -1, 0, 0, 0, 1), {flint_index})"
            self.assertTrue(sp.sympify(_weight_embedding(label)).equals(sp.CRootOf(x**5-x+1, sympy_index)))
        # Four purely imaginary roots in two conjugate pairs with the same
        # real part. FLINT orders positive imaginary first within each pair.
        from engine.nonreduced import _flint_roots
        roots=_flint_roots((1,0,3,0,1))
        small=(sp.sqrt(5)-1)/2
        large=(sp.sqrt(5)+1)/2
        field=QQ.algebraic_field(sp.I,sp.sqrt(5))
        for actual,want in zip(roots,[small*sp.I,-small*sp.I,large*sp.I,-large*sp.I]):
            self.assertEqual(field.from_sympy(actual),field.from_sympy(want))
        r=weighted("((x^4+3*x^2*y^2+y^4)^2+x^9)^2",["x","y"])
        self.assertGreaterEqual(len(r["algebraic_numbers"]),4)
        self.assertTrue(any("a" in ch["strict_transform"] for ch in r["charts"]))
        self.assertEqual(r["dual_complex"]["vertices"][0]["total_multiplicity"],16)

    def test_display_order_uses_exact_rectangles_not_sorted_real_parts(self):
        from unittest.mock import patch
        from engine import groebner
        from engine.field_coefficient import marked_root_display
        from engine.nonreduced import _weight_embedding

        if groebner._MODE == "sympy" or not hasattr(groebner._kernel(), "atlas_field_display_order"):
            self.skipTest("Requires the native marked-root ordering kernel.")
        # Two coincident roots cannot be separated by any rational rectangle.
        with self.assertRaisesRegex(ArithmeticError, "squarefree"):
            marked_root_display((1, 0, 2, 0, 1))
        # This irreducible has three real roots and three conjugate pairs.
        # Sorting the real parts of the pairs interchanges the last two.
        coeffs = (3, -3, -4, -3, -1, 2, -5, 7, 4, 3)
        expected = (2, 1, 0, 8, 7, 4, 3, 6, 5)
        self.assertEqual(marked_root_display(coeffs)[2], expected)
        # The bounds for public CRootOf numbering refer to its normalized
        # polynomial. Multiplying all roots by two changes its root bound.
        scaled = (16, 0, 12, 0, 1)
        self.assertEqual(marked_root_display(scaled), (2, (1, 0, 3, 0, 1), (1, 0, 3, 2)))
        if not groebner.CHECK:
            with patch.object(sp.CRootOf, "_get_interval", side_effect=AssertionError("Unexpected root isolation")):
                for i, position in enumerate(expected, 1):
                    label = f"PolynomialRootIndexed(List({', '.join(map(str, coeffs))}), {i})"
                    self.assertTrue(_weight_embedding(label).endswith(f", {position})"))

    def test_intrinsic_rejects_repetition_and_units_are_local(self):
        with self.assertRaises(InputError):
            weighted("x^2*(x-y)",["x","y"],"intrinsic")
        self.assertIsNone(weighted("(x+1)^3*(x-y)",["x","y"],"intrinsic"))
        r=weighted("-2*(1+x)^3*x^2",["x","y"])
        self.assertEqual(r["dual_complex"]["vertices"],[])
        self.assertEqual(r["input_divisor"]["factors"],[{"equation":"x","multiplicity":2}])
