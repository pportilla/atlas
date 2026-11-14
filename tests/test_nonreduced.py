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

    def test_intrinsic_rejects_repetition_and_units_are_local(self):
        with self.assertRaises(InputError):
            weighted("x^2*(x-y)",["x","y"],"intrinsic")
        self.assertIsNone(weighted("(x+1)^3*(x-y)",["x","y"],"intrinsic"))
        r=weighted("-2*(1+x)^3*x^2",["x","y"])
        self.assertEqual(r["dual_complex"]["vertices"],[])
        self.assertEqual(r["input_divisor"]["factors"],[{"equation":"x","multiplicity":2}])
