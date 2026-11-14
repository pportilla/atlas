"""Radicals of rational polynomial ideals in the original ordered coordinate ring."""
import shutil
import unittest

from tests.test_local_algebra_singular import calculate, singular


class IdealRadicalTests(unittest.TestCase):
    def check(self, names, generators, expected, order="grevlex"):
        radical = calculate("radical", variables=names, order=order, ideal=generators)
        basis = calculate("groebner", variables=names, order=order, ideal=expected)
        self.assertEqual(radical, {"type": "ideal", "ring": basis["ring"], "generators": basis["generators"]})

    def test_zero_unit_scalar_and_arbitrary_precision_monomials(self):
        for names in ((), ("x", "y")):
            self.check(names, [], [])
            self.check(names, ["0", "0"], [])
            self.check(names, ["0", "-2/7"], ["1"])
        self.check(("x",), ["x-1", "x-2"], ["1"])
        n = 2**64 + 17
        self.check(("x", "y", "z"), [f"x^{n}*y^2", f"x^3*z^{n}", f"y^{n}*z^4", "0"],
                   ["x*y", "x*z", "y*z"])

    def test_finite_loci_and_principal_ideals_over_qq(self):
        # x²+1 contributes its two conjugate geometric points over QQ.
        for order in ("lex", "grevlex"):
            self.check(("y", "x"), ["(x^2+1)^2*(x-1/2)^3", "y-x"],
                       ["(x^2+1)*(x-1/2)", "y-x"], order)
            self.check(("y", "x"), ["-3/7*(x^2+y^2)^3*(x-y)^2"], ["(x^2+y^2)*(x-y)"], order)

    def test_hidden_nilpotents_and_exceptional_components(self):
        for order in ("lex", "grevlex"):
            # Both submitted generators are squarefree. Their combination gives y³.
            self.check(("x", "y", "z"), ["x*y", "x*z+y^2"], ["y", "x*z"], order)
            # The nilpotent at the origin is embedded in a double line.
            self.check(("x", "y"), ["x^3+x^2*y", "x^2*y"], ["x"], order)
            # The origin remains as an isolated point beside the affine line x=1.
            self.check(("y", "x"), ["(x-1)*x^2", "(x-1)*y^2"], ["x*(x-1)", "(x-1)*y"], order)

    @unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
    def test_mutual_containment_with_singular_radical(self):
        cases = [
            (("x", "y"), ["0"]),
            (("x", "y"), ["x*y-1", "x"]),
            (("y", "x"), ["x^3-x", "(y^2+x^2+1)^2"]),
            (("x", "y", "z"), ["x*y", "x*z+y^2"]),
            # The reduced curve (x²-t, y²-t) has a nonreduced fiber over t=0.
            (("t", "y", "x"), ["x^2-t", "(y^2-t)^2+(x^2-t)"]),
            # An isolated degree-two closed point sits over t=0, outside the
            # generic graph (t*x=1, y=t²). The two branches have different dimensions.
            (("x", "y", "t"), [f"({h})*({f})^2" for h in ("t", "y", "x^2+1")
                                 for f in ("t*x-1", "y-t^2")]),
            # Two independent parameters. Over QQ(a,b) the common component
            # satisfies y=b/a and x=y. Both generators mix the hidden squares.
            (("x", "y", "a", "b"), ["(a*y-b)^2+2*(x-y)^2", "(a*y-b)^2-3*(x-y)^2"]),
        ]
        for names, generators in cases:
            for order, singular_order in (("lex", "lp"), ("grevlex", "dp")):
                with self.subTest(generators=generators, order=order):
                    received = calculate("radical", variables=names, order=order, ideal=generators)["generators"]
                    commands = [f"ideal I=ideal({','.join(generators)});",
                                "ideal G=std(radical(I));",
                                f"ideal A=std(ideal({','.join(received) or '0'}));",
                                'print("ANSWER "+string(size(reduce(G,A))==0));',
                                'print("ANSWER "+string(size(reduce(A,G))==0));']
                    self.assertEqual(singular(singular_order, commands, names), [1, 1])
