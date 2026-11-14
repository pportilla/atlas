"""Compare local Krull dimensions and regularity over QQ with live Singular."""
from fractions import Fraction
import shutil
import unittest

from engine.polynomial import InputError
from tests.test_local_algebra_singular import calculate, singular
from tests.test_local_standard_singular import answer, translated_ring


@unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
class LocalSmoothnessSingularTests(unittest.TestCase):
    def check_local_ring(self, variables, ideal, point, order, dimension, embedding):
        coords = [str(Fraction(q)) for q in point] if point is not None else ["0"] * len(variables)
        oracle = singular("dp", [
            f"ideal original={','.join(ideal) or '0'};",
            *translated_ring(order, coords),
            answer("reduce(1,G)!=0"), answer("dim(G)"),
            # Independently of a Jacobian, dim_Q(m/m²) = length(A/m²)-1 at a QQ point.
            answer("vdim(std(J+maxideal(2)))-1"),
        ], variables)
        self.assertEqual(oracle, [1, dimension, embedding])
        arguments = {"ideal": ideal}
        if point is not None:
            arguments["point"] = point
        local = calculate("local-dimension", variables=variables, order=order, **arguments)
        smooth = calculate("local-smoothness", variables=variables, order=order, **arguments)
        for result in (local, smooth):
            self.assertEqual(result["scope"], "origin" if point is None else "point")
            if point is not None:
                self.assertEqual(result["point"], coords)
            self.assertEqual(result["ring"], {"field": "QQ", "variables": list(variables), "order": order})
            self.assertEqual(result["local_dimension"], dimension)
        self.assertEqual((local["type"], local["value"]), ("integer", dimension))
        self.assertEqual(smooth["type"], "boolean")
        self.assertIs(smooth["value"], dimension == embedding)
        self.assertEqual((smooth["embedding_dimension"], smooth["jacobian_rank"]),
                         (embedding, len(variables) - embedding))

    def test_reduced_and_thickened_local_schemes(self):
        xy, xyz = ("x", "y"), ("x", "y", "z")
        for variables, ideal, point, dimension, embedding in (
            (xyz, [], ["-1/3", 0, 2], 3, 3),
            (xy, ["x", "y"], [0, 0], 0, 0),
            (xy, ["x^2", "y"], None, 0, 1),
            (xyz, ["x*y", "x*z"], None, 2, 3),       # Plane and line at their intersection.
            (xyz, ["x*y", "x*z"], [1, 0, 0], 1, 1),  # On the line alone.
            (xyz, ["x*y", "x*z"], [0, 1, 0], 2, 2),  # On the plane alone.
            (xy, ["x*y", "y^2"], None, 1, 2),         # Embedded point on the x-axis.
            (xy, ["x*y", "y^2"], ["2/3", 0], 1, 1),
        ):
            for order in ("grevlex", "lex"):
                with self.subTest(ideal=ideal, point=point, order=order):
                    self.check_local_ring(variables, ideal, point, order, dimension, embedding)

    def test_translated_nonlinear_equations_and_redundant_generators(self):
        x, y, z = "(x-1/2)", "(y+2/3)", "(z-1/3)"
        cusp = [f"{y}^2-{x}^3"]
        row = f"({x}*{y}+{z}^2)"
        line = [f"2*{x}+3*{y}+{z}+{x}^2", f"{x}-2*{y}+3*{z}", f"2*({x}-2*{y}+3*{z})"]
        for variables, ideal, point, dimension, embedding in (
            (("x", "y"), cusp, ["2/4", "-4/6"], 1, 2),
            (("x", "y"), cusp, ["3/2", "1/3"], 1, 1),
            (("x", "y", "z"), line, ["1/2", "-2/3", "1/3"], 1, 1),
            (("x", "y", "z"), [row, f"{row}+{x}^2"], ["1/2", "-2/3", "1/3"], 1, 3),
        ):
            for order in ("grevlex", "lex"):
                with self.subTest(ideal=ideal, point=point, order=order):
                    self.check_local_ring(variables, ideal, point, order, dimension, embedding)

    def test_outside_support_has_no_local_point(self):
        for ideal, point in ((["1"], [0, 0]), (["x*y", "y^2"], ["1/2", 1])):
            with self.subTest(ideal=ideal):
                self.assertEqual(singular("dp", [
                    f"ideal original={','.join(ideal)};", *translated_ring("grevlex", point),
                    answer("reduce(1,G)==0"), answer("dim(G)"),
                ]), [1, -1])
                for operation in ("local-dimension", "local-smoothness"):
                    with self.subTest(operation=operation), self.assertRaises(InputError):
                        calculate(operation, ideal=ideal, point=point)

    def test_six_coordinate_determinantal_vertex(self):
        variables = ("a", "b", "c", "d", "e", "f")
        point = ["1/2", "-2/3", "1/3", 0, 0, 0]
        # A triangular coordinate change sends the 2x3 rank-one matrix
        # variety to this germ. Its height is two despite its three minors.
        a, b, c, d, e, f = ("((a-1/2)+(b+2/3)^2)", "(b+2/3)", "(c-1/3)",
                            "(d+(b+2/3)*(c-1/3))", "e", "f")
        ideal = [f"{a}*{e}-{b}*{d}", f"{a}*{f}-{c}*{d}", f"{b}*{f}-{c}*{e}"]
        for order, local_order in (("lex", "Ds"), ("grevlex", "ds")):
            with self.subTest(order=order):
                answers = singular("dp", [
                    f"ideal original={','.join(ideal)};", f"ring T=0,(u,v,w,h,j,k),{local_order};",
                    "map shift=R,u+1/2,v-2/3,w+1/3,h,j,k;", "ideal J=shift(original);",
                    answer("dim(std(J))"), answer("vdim(std(J+maxideal(2)))-1"),
                ], variables)
                dim, embedding = answers
                self.assertEqual(answers, [4, 6])
                for operation in ("local-dimension", "local-smoothness"):
                    result = calculate(operation, variables=variables, order=order, ideal=ideal, point=point)
                    self.assertEqual(result["local_dimension"], dim)
                    if operation == "local-smoothness":
                        self.assertFalse(result["value"])
                        self.assertEqual(result["embedding_dimension"], embedding)
