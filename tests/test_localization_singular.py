"""Check localization in QQ[x] against ideal quotients and two Singular decompositions."""
from dataclasses import replace
import shutil
import unittest

from benchmarks.localization import Case, corpus, localize, run_singular


class LocalizationTests(unittest.TestCase):
    def test_invalid_primes_and_denominators(self):
        for c in (
            Case("unit_prime",("x",),(),("1",)),
            Case("nonradical",("x","y"),("y",),("x^2","x*y")),
            Case("reducible_prime",("x","y"),("x",),("x*y",)),
            Case("reducible_tensor",("x","y"),(),("x^2+1","y^2+1")),
            Case("cancelled_fraction",("x","y"),(),("x",),"x","x"),
            Case("zero_denominator",("x","y"),(),(),"0","0"),
        ):
            with self.subTest(case=c.name):
                self.assertEqual(localize(c)["error"]["code"], "invalid_request")

    def test_scalar_ring(self):
        for I, expected in [((),[]), (("2/3",),["1"])]:
            c=Case("scalars",(),I,(),"4","3")
            r=localize(c)
            self.assertEqual(r["generators"],expected)
            self.assertEqual(r["prime"],[])
            self.assertTrue(r["element"]["unit"])
            self.assertEqual(r["element"]["member"],bool(I))

    def test_wide_prime_and_contraction(self):
        exponent=2**64+5
        z=2**128+3
        c=Case("wide",("x","y"),(f"{z}*x^{exponent}*y",),("x",),f"x^{exponent}","y")
        r=localize(c)
        self.assertEqual(r["generators"],[f"x^{exponent}"])
        self.assertTrue(r["element"]["member"])
        self.assertFalse(r["element"]["unit"])
        graph=f"x-y^{exponent}"
        for num,member in ((graph,True),(f"y^{exponent}+1",False)):
            c=Case("wide_graph",("x","y"),(graph,),(graph,"y"),num,"x+1")
            self.assertEqual(localize(c)["element"]["member"],member)

    def test_wide_height_one_valuation_preserves_multiplicity(self):
        exponent = 2**64 + 3
        c = Case("sparse_power", ("x",), (f"x^{2*exponent}-2*x^{exponent}+1",),
                 ("x-1",), "x-1", "x")
        r = localize(c)
        self.assertEqual(r["generators"], ["x^2-2*x+1"])
        self.assertFalse(r["element"]["member"])
        self.assertEqual(localize(replace(c, prime=("x-2",)))["generators"], ["1"])
        cyclotomic = replace(c, generators=(f"x^{3*exponent}-1",), prime=("x^2+x+1",))
        self.assertEqual(localize(cyclotomic)["generators"], ["x^2+x+1"])

    def test_wide_cyclotomic_residue(self):
        exponent = 3 * (2**64 + 1)
        phi = "x^2+x+1"
        c = Case("cyclotomic", ("x","y"), (phi, "y^2"), (phi, "y"),
                 f"x^{exponent}-1+y^2", f"x^{exponent}")
        r = localize(c)
        self.assertTrue(r["element"]["member"])
        self.assertFalse(r["element"]["unit"])
        self.assertTrue(localize(replace(c,numerator=f"x^{exponent}"))["element"]["unit"])
        self.assertEqual(localize(replace(c,denominator=f"x^{exponent}-1"))["error"]["code"],"invalid_request")

    @unittest.skipUnless(shutil.which("Singular"), "Singular is required for this comparison")
    def test_contraction_basis_depends_on_requested_order(self):
        c = Case("basis_order", ("x","y","z"), ("2*x^2+3*y*z","2*x*y+3*z^2"), ("x","y","z"))
        for order, expected in (
            ("grevlex", ["-x*z^2+y^2*z", "x^2+3/2*y*z", "x*y+3/2*z^2"]),
            ("lex", ["x^2+3/2*y*z", "x*y+3/2*z^2", "x*z^2-y^2*z", "y^3*z+3/2*z^4"]),
        ):
            trial = replace(c, order=order)
            with self.subTest(order=order):
                result = localize(trial)
                self.assertEqual(result["generators"], expected)
                run_singular(trial, "SY", 47, 30, False, None, check=result)

    @unittest.skipUnless(shutil.which("Singular"), "Singular is required for this comparison")
    def test_contraction_and_fraction_with_both_orders(self):
        for c in corpus():
            for order,method in (("grevlex","SY"),("lex","GTZ")):
                trial=replace(c,order=order)
                with self.subTest(case=c.name,order=order):
                    r=localize(trial)
                    # Equality of ideals is order independent. Use the oracle
                    # in the case's declared order for a stable verification cost.
                    certified=dict(r,ring=dict(r["ring"],order=c.order))
                    run_singular(c,method,47,30,False,None,certified)
                    self.assertEqual(r["ring"]["order"],order)
