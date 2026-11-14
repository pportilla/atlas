"""Verify Zariski tangent spaces and embedding dimensions against Singular first jets."""
import shutil
import unittest
from fractions import Fraction

from engine.polynomial import InputError
from tests.test_local_algebra_singular import calculate, singular
from tests.test_local_standard_singular import answer, translated_ring


@unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
class ZariskiTangentSingularTests(unittest.TestCase):
    def check_space(self, variables, ideal, point, order, dimension):
        width = len(variables)
        coords = [str(Fraction(q)) for q in point] if point is not None else ["0"] * width
        operands = {"ideal": ideal}
        if point is not None:
            operands["point"] = point
        tangent = calculate("zariski-tangent-space", variables=variables, order=order, **operands)
        embedding = calculate("embedding-dimension", variables=variables, order=order, **operands)
        for result in (tangent, embedding):
            self.assertEqual(result["scope"], "origin" if point is None else "point")
            if point is not None:
                self.assertEqual(result["point"], coords)
            self.assertEqual(result["ring"], {"field": "QQ", "variables": list(variables), "order": order})
            self.assertEqual(result["ambient_dimension"], width)
            self.assertEqual(result["jacobian_rank"], width - dimension)
        self.assertEqual((tangent["type"], tangent["coordinate_system"], tangent["dimension"]),
                         ("zariski-tangent-space", "tangent", dimension))
        self.assertEqual((embedding["type"], embedding["value"]), ("integer", dimension))
        basis = tangent["basis"]
        self.assertEqual(len(basis), dimension)
        for row in basis:
            self.assertIsInstance(row, list)
            self.assertEqual(len(row), width)
            for coordinate in row:
                self.assertIsInstance(coordinate, str)
                self.assertEqual(str(Fraction(coordinate)), coordinate)

        commands = [
            f"ideal original={','.join(ideal) or '0'};",
            f"ideal proposed={','.join(tangent['equations']) or '0'};",
            *translated_ring(order, coords),
            # For a point on the scheme, length(A/m²) = 1 + dim(m/m²).
            answer("vdim(std(J+maxideal(2)))-1"),
            "ideal linear=jet(J,1);",
            f"ring Directions=0,({','.join(variables)}),dp;",
            # Only original generators receive the affine translation. The map
            # from centered coordinates to directions is the identity on axes.
            "ideal L=fetch(T,linear);", "ideal E=imap(R,proposed);",
            answer("size(reduce(L,std(E)))==0"), answer("size(reduce(E,std(L)))==0"),
            "int linearOnly=1; int i;",
            "for(i=1;i<=size(E);i++){if(deg(E[i])!=1 || jet(E[i],0)!=0){linearOnly=0;}}",
            answer("linearOnly"),
        ]
        expected = [dimension, 1, 1, 1]
        if basis:
            entries = ",".join(q for row in basis for q in row)
            commands += [f"matrix B[{dimension}][{width}]={entries};", answer("rank(B)")]
            expected.append(dimension)
        for i, row in enumerate(basis):
            commands += [f"map vector{i}=Directions,{','.join(row)};", answer(f"size(vector{i}(L))==0")]
            expected.append(1)
        self.assertEqual(singular("dp", commands, variables), expected)

    def test_curve_and_embedded_components_at_different_points(self):
        variables = ("x", "y")
        cusp = ["(y+2/3)^2-(x-1/2)^3"]
        embedded = ["x*y", "y^2"]
        for ideal, point, dimension in (
            (cusp, ["2/4", "-4/6"], 2), (cusp, ["3/2", "1/3"], 1),
            (embedded, None, 2), (embedded, [2, 0], 1),
            ([], ["2/3", "-1/5"], 2),
        ):
            for order in ("lex", "grevlex"):
                with self.subTest(ideal=ideal, point=point, order=order):
                    self.check_space(variables, ideal, point, order, dimension)

    def test_rational_oblique_linear_parts_and_nonlinear_generators(self):
        x, y, z = "(x-1/2)", "(y+2/3)", "(z-1/5)"
        a, b = f"(2*{x}+3*{y}+{z})", f"({x}-2*{y}+3*{z})"
        for ideal, dimension in (
            ([f"{a}+{x}^2", f"{b}+{y}^2", f"2*{a}-{b}+{z}^3"], 1),
            ([f"{a}+{y}^3", f"2*{a}+{x}*{z}", "0"], 2),
            ([x, y, z], 0),                      # A reduced point.
            ([f"{x}^2", f"{y}^2", f"{z}^2"], 3),  # A thickening supported at that same point.
        ):
            for order in ("lex", "grevlex"):
                with self.subTest(ideal=ideal, order=order):
                    self.check_space(("x", "y", "z"), ideal, ["1/2", "-2/3", "1/5"], order, dimension)

    def test_outside_support_is_a_local_unit(self):
        for ideal, point in ((["(y+2/3)^2-(x-1/2)^3"], [0, 0]), (["1"], ["1/2", "-2/3"])):
            for order in ("lex", "grevlex"):
                with self.subTest(ideal=ideal, point=point, order=order):
                    checks = singular("dp", [
                        f"ideal original={','.join(ideal)};", *translated_ring(order, point),
                        answer("reduce(1,G)==0"), answer("vdim(std(J+maxideal(2)))"),
                    ])
                    self.assertEqual(checks, [1, 0])
                    for operation in ("zariski-tangent-space", "embedding-dimension"):
                        with self.subTest(operation=operation), self.assertRaises(InputError):
                            calculate(operation, ideal=ideal, point=point, order=order)
