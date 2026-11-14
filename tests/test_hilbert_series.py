"""Standard-graded Hilbert series from a global homogeneous ideal over QQ."""
import shutil
import unittest

from engine.algebra import dispatch
from engine.polynomial import InputError
from tests.test_local_algebra_singular import singular


def series(ideal, variables=("x", "y", "z"), order="grevlex"):
    return dispatch({"operation": "hilbert-series", "ring": {"variables": list(variables), "order": order},
                     "operands": {"ideal": ideal}, "compute": "cpu"})["result"]


def terms(numerator):
    return [{"degree": d, "coefficient": c} for d, c in sorted(numerator.items())]


class GradedHilbertSeriesTests(unittest.TestCase):
    def check(self, ideal, names, numerator, dim, mult, order="grevlex"):
        self.assertEqual(series(ideal, names, order), {
            "type": "hilbert-series", "scope": "global", "ring": {"field": "QQ", "variables": list(names), "order": order},
            "denominator_power": len(names), "numerator": terms(numerator), "dimension": dim, "multiplicity": mult,
        })

    def test_polynomial_ring_scalar_ring_and_zero_quotient(self):
        for names in ((), ("x", "y")):
            for ideal in ([], ["0", "0"]):
                self.check(ideal, names, {0: 1}, len(names), 1)
            self.check(["-2/7"], names, {}, -1, 0)
        self.check(["x-1", "x-2"], ("x",), {}, -1, 0)
        self.check(["x^2+3*x", "1"], ("x",), {}, -1, 0)

    def test_homogeneity_of_the_ideal_and_nonmonomial_relations(self):
        # (x+y²,y²)=(x,y²); the presentation does not itself give homogeneous rows.
        for order in ("lex", "grevlex"):
            self.check(["x+y^2", "y^2"], ("x", "y"), {0: 1, 1: -1, 2: -1, 3: 1}, 0, 2, order)
            # The generators have a common linear factor, adding a cubic syzygy.
            self.check(["x*(x-y)/7", "2*y*(x-y)"], ("y", "x", "z"), {0: 1, 2: -2, 3: 1}, 2, 1, order)
            for ideal in (["x^2+y"], ["x+y^2", "y^3"]):
                with self.subTest(order=order, ideal=ideal), self.assertRaisesRegex(InputError, "homogeneous"):
                    series(ideal, order=order)

    def test_sparse_complete_intersection_without_enumerating_degrees(self):
        n = 2**64+17
        self.check([f"x^{n}", f"y^{n}"], ("x", "y", "z"), {0: 1, n: -2, 2*n: 1}, 1, n*n)
        self.check([f"x^{n}+y^{n}"], ("x", "y"), {0: 1, n: -1}, 1, n)

    @unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
    def test_full_numerators_dimension_and_multiplicity_against_singular(self):
        names = ("z", "y", "x")
        for ideal in ([], ["1"], ["x*y", "x*z"], ["x+y^2", "y^2"],
                      ["x^2+y*z", "x*y+z^2", "x*z+y^2"]):
            commands = [f"ideal G=std(ideal({','.join(ideal) or '0'}));",
                        'print("ANSWER "+string(dim(G)));', 'print("ANSWER "+string(mult(G)));',
                        "bigintvec H=hilb(G,1);",
                        'for(int i=1;i<=size(H);i++){print("ANSWER "+string(H[i]));}']
            # Singular's final zero is its module degree shift.
            dim, mult, *coefficients, shift = singular("dp", commands, names)
            self.assertEqual(shift, 0)
            numerator = {degree: value for degree, value in enumerate(coefficients) if value}
            for order in ("lex", "grevlex"):
                with self.subTest(ideal=ideal, order=order):
                    self.check(ideal, names, numerator, dim, mult, order)
