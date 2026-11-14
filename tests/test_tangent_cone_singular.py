"""Compare rational tangent-cone ideals and their full graded series with Singular."""
import shutil
import unittest
from fractions import Fraction

from tests.test_local_algebra_singular import calculate, singular
from tests.test_local_standard_singular import answer, translated_ring


@unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
class TangentConeSingularTests(unittest.TestCase):
    def check_cone(self, variables, ideal, point, order, dimension, multiplicity):
        operands = {"ideal": ideal}
        coords = ["0"] * len(variables)
        if point is not None:
            operands["point"] = point
            coords = [str(Fraction(a)) for a in point]
        cone = calculate("tangent-cone", variables=variables, order=order, **operands)
        series = calculate("tangent-cone-hilbert-series", variables=variables, order=order, **operands)
        for result in (cone, series):
            self.assertEqual(result["scope"], "origin" if point is None else "point")
            if point is not None:
                self.assertEqual(result["point"], coords)
            self.assertEqual(result["ring"], {"field": "QQ", "variables": list(variables), "order": order})
            self.assertEqual((result["dimension"], result["multiplicity"]), (dimension, multiplicity))
        self.assertEqual(cone["coordinate_system"], "tangent")
        self.assertEqual(series["denominator_power"], len(variables))

        answers = singular("dp", [
            f"ideal original={','.join(ideal) or '0'};",
            f"ideal proposed={','.join(cone['generators']) or '0'};",
            *translated_ring(order, coords),
            "ideal expected=tangentcone(J);",
            answer("dim(G)"), answer("mult(G)"),
            # hilb(G,1) gives the entire numerator over (1-t)^n, followed
            # by Singular's shift entry (zero for these unshifted ideals).
            "def hs=hilb(G,1);", answer("hs[size(hs)]"),
            "int i;", "for(i=1;i<size(hs);i++){" + answer("hs[i]") + "}",
            f"ring Directions=0,({','.join(variables)}),dp;",
            # fetch maps centered variables to the direction axes by position.
            # The Atlas output already uses those axes and receives NO shift.
            "ideal E=fetch(T,expected);", "ideal A=imap(R,proposed);",
            "ideal AS=std(A);", "ideal ES=std(E);",
            answer("homog(A)"), answer("size(reduce(E,AS))==0"), answer("size(reduce(A,ES))==0"),
        ], variables)
        self.assertEqual(answers[:3], [dimension, multiplicity, 0])
        self.assertEqual(answers[-3:], [1, 1, 1])
        self.assertEqual(series["numerator"], [{"degree": d, "coefficient": c}
                                               for d, c in enumerate(answers[3:-3]) if c])

    def test_cancellation_and_nonmonomial_forms(self):
        x, y = "(x-1/2)", "(y+2/3)"
        point = ["2/4", "-4/6"]
        for ideal, dimension, multiplicity in (
            # Both input heads are x, but the full initial ideal also contains y^4.
            ([f"{x}+{y}^2", f"{x}+{y}^2+{x}^2"], 0, 4),
            ([f"({x}+2*{y})^3+{x}^5"], 1, 3),
            ([f"{x}*({x}+{y})", f"{y}*({x}+{y})"], 1, 1),  # A line and an embedded point.
        ):
            for order in ("lex", "grevlex"):
                with self.subTest(ideal=ideal, order=order):
                    self.check_cone(("x", "y"), ideal, point, order, dimension, multiplicity)
        for order in ("lex", "grevlex"):
            with self.subTest(order=order):
                # x*z-y² has different leading terms in the two tie orders.
                self.check_cone(("x", "y", "z"), ["x^2+y^3", "y^3", "z^2", "x*z-y^2+z^3"],
                                None, order, 0, 7)

    def test_point_dependence_and_empty_support(self):
        x, y = "(x-1/2)", "(y+2/3)"
        curve = [f"{y}^3-{x}^3*({x}-1)^2"]
        for point, dimension, multiplicity in (
            (["1/2", "-2/3"], 1, 3), (["3/2", "-2/3"], 1, 2),
            (["19/2", "106/3"], 1, 1), (["5/2", "-2/3"], -1, 0),
        ):
            for order in ("lex", "grevlex"):
                with self.subTest(point=point, order=order):
                    self.check_cone(("x", "y"), curve, point, order, dimension, multiplicity)
        self.check_cone(("x", "y"), [], None, "grevlex", 2, 1)
        self.check_cone(("x", "y"), ["1"], [1, 2], "lex", -1, 0)

    def test_sparse_numerator_with_high_degree_cancellation(self):
        # The full numerator is 1-t^37-t^41+t^78. Long gaps and the final
        # positive coefficient are invisible in a short prefix of the series.
        for order in ("lex", "grevlex"):
            with self.subTest(order=order):
                self.check_cone(("x", "y"), ["x^37+y^38", "y^41"], None, order, 0, 37*41)
