"""Independent reduced ideal checks, symbolic/ordinary distinctions and wide inputs."""
from dataclasses import replace
import shutil
import unittest

from benchmarks.ideal_powers import Case, call_wire, corpus, methods, run_singular


class IdealPowersTests(unittest.TestCase):
    def test_disjoint_monomials_of_different_degrees(self):
        # Removing the repeated generator and the divisible generator exposes
        # three disjoint supports, one of which spans two coordinates.
        c = Case("disjoint", ("x", "y", "z", "w"), ("x^2*y", "z^3", "w", "x^2*y*z^3", "w"), 3, False)
        expected = {"x^6*y^3", "x^4*y^2*z^3", "x^4*y^2*w", "x^2*y*z^6", "x^2*y*z^3*w",
                    "x^2*y*w^2", "z^9", "z^6*w", "z^3*w^2", "w^3"}
        for order in ("grevlex", "lex"):
            self.assertCountEqual(call_wire(replace(c, order=order))["generators"], expected)

    @unittest.skipUnless(shutil.which("Singular"), "Singular is required")
    def test_corpus_in_both_orders(self):
        for c in corpus():
            for order in ("grevlex","lex"):
                t=replace(c,order=order)
                with self.subTest(name=c.name,order=order):
                    result=call_wire(t)
                    self.assertEqual(result["ring"],{"variables":list(c.variables),"field":"QQ","order":order})
                    self.assertEqual(result["type"],"ideal")
                    for m in methods(t):
                        run_singular(t,m,0,19,40,check=result)

    def test_triangle_symbolic_square(self):
        c=Case("triangle",("x","y","z"),("x*y","x*z","y*z"),2,False)
        ordinary=set(call_wire(c)["generators"])
        symbolic=set(call_wire(replace(c,symbolic=True))["generators"])
        self.assertEqual(ordinary,{"x^2*y^2","x^2*z^2","y^2*z^2","x^2*y*z","x*y^2*z","x*y*z^2"})
        self.assertEqual(symbolic,{"x*y*z","x^2*y^2","x^2*z^2","y^2*z^2"})

    def test_reject_nonradical_even_for_exponents_zero_and_one(self):
        for gens in (("x^2","x*y"),("(x+y)^2",),("x^2+2*x*y+y^2","x-y"),("y-x^2","y^2")):
            for n in (0,1,3):
                with self.subTest(gens=gens,n=n):
                    c=Case("nonradical",("x","y"),gens,n,True)
                    self.assertEqual(call_wire(c)["error"]["code"],"invalid_request")

    def test_symbolic_product_checks_all_radicality_conditions(self):
        # A repeated hypersurface, a nonradical residual, or a hypersurface
        # through a residual component each makes hK nonradical.
        for generators in (
            ("(x-1)^2*x", "(x-1)^2*y"),
            ("(x-1)*x^2", "(x-1)*y"),
            ("(x+y)*x", "(x+y)*y"),
        ):
            for exponent in (0, 1, 3):
                with self.subTest(generators=generators, exponent=exponent):
                    case = Case("common_factor", ("x", "y"), generators, exponent, True)
                    self.assertEqual(call_wire(case)["error"]["code"], "invalid_request")

    def test_redundant_nonsquarefree_generators_of_radical_ideals(self):
        for gens in (("x*y","x^3*y^2"),("x+y","x*(x+y)^2"),("x^2+y","x^2+y+x","y^3")):
            for n in (0,1,2):
                with self.subTest(gens=gens,n=n):
                    c=Case("redundant",("x","y"),gens,n,True)
                    self.assertEqual(call_wire(c),call_wire(replace(c,symbolic=False)))

    def test_prime_with_nonradical_initial_ideal(self):
        c=Case("curve",("x","y","z"),("y^2-x*z","y*z-x^3","z^2-x^2*y"),2,True)
        out=call_wire(c)
        ordinary=call_wire(replace(c,symbolic=False))
        self.assertNotEqual(out["generators"],ordinary["generators"])
        if shutil.which("Singular"):
            run_singular(c,"Jacobian",0,41,40,check=out)

    def test_scalar_rings_and_wide_exponents(self):
        n=2**80+91
        for symbolic in (False,True):
            for gens, exp, expected in (((),0,["1"]),((),n,[]),(("2/5",),n,["1"])):
                c=Case("scalar",(),gens,exp,symbolic)
                self.assertEqual(call_wire(c)["generators"],expected)
            c=Case("principal",("x","y"),(f"{2**180+33}*x*y",),n,symbolic)
            self.assertEqual(call_wire(c)["generators"],[f"x^{n}*y^{n}"])
        c=Case("wide_generator",("x","y"),(f"x^{n}*y",),3,False)
        self.assertEqual(call_wire(c)["generators"],[f"x^{3*n}*y^3"])
        self.assertEqual(call_wire(replace(c,symbolic=True))["error"]["code"],"invalid_request")
