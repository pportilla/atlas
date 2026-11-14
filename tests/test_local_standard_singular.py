"""Live Singular comparisons for standard bases and weak normal forms at QQ points."""
import shutil
import unittest
from fractions import Fraction

from engine.algebra import dispatch
from tests.test_local_algebra_singular import singular


def check_at_point(operation, variables, order, ideal, point, query=None):
    ring = {"variables": list(variables), "field": "QQ", "order": order}
    operands = {"ideal": ideal}
    if point is not None:
        operands["point"] = point
    if query is not None:
        operands["polynomial"] = query
    result = dispatch({"operation": operation, "ring": ring, "operands": operands,
                       "compute": "cpu"})["result"]
    return ring, result


def translated_ring(order, point):
    # First declare every input/output polynomial in the original coordinates.
    # Singular changes rings and substitutes x_i = p_i + u_i in all of them.
    offsets = ("u", "v", "w")[:len(point)]
    shifts = [f"{v}+({q})" for v, q in zip(offsets, point)]
    local_order = {"lex": "Ds", "grevlex": "ds"}[order]
    return [
        f"ring T=0,({','.join(offsets)}),{local_order};",
        f"map shift=R,{','.join(shifts)};",
        "ideal J=shift(original);",
        "ideal G=std(J);",
        "ideal heads=std(lead(G));",
    ]


def answer(expression):
    return f'print("ANSWER "+string({expression}));'


@unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
class LocalStandardSingularTests(unittest.TestCase):
    def check_contract(self, operation, variables, order, ideal, point, query=None):
        ring, result = check_at_point(operation, variables, order, ideal, point, query)
        coords = [str(Fraction(q)) for q in point] if point is not None else ["0"] * len(variables)
        self.assertEqual(result["type"], operation)
        self.assertEqual(result["scope"], "point")
        self.assertEqual(result["point"], coords)
        self.assertEqual(result["ring"], ring)
        return result, coords

    def test_standard_bases_in_original_coordinates(self):
        curve = ["(x-1/2)*(1-(y+2))-(y+2)"]
        for variables, ideal, point, dimension in (
            (("x", "y"), [], None, 2),
            (("x", "y"), ["x^2*(x-1)", "y*(x-1)"], None, 0),
            (("x", "y"), ["x^2*(x-1)", "y*(x-1)"], [1, 3], 1),
            (("x", "y"), ["x^2*(x-1)", "y*(x-1)"], [2, 3], -1),
            (("x", "y"), curve, ["2/4", -2], 1),
            (("x", "y"), ["x-1/2+(y+2)^2", "x-1/2+(y+2)^3"], ["1/2", -2], 0),
            (("x", "y", "z"), ["(x-1/2)*z+(y+2)^2", "(x-1/2)^3"], ["1/2", -2, 0], 1),
        ):
            for order in ("lex", "grevlex"):
                with self.subTest(ideal=ideal, point=point, order=order):
                    result, coords = self.check_contract("local-standard-basis", variables, order, ideal, point)
                    checks = singular("dp", [
                        f"ideal original={','.join(ideal) or '0'};",
                        f"ideal proposed={','.join(result['generators']) or '0'};",
                        *translated_ring(order, coords),
                        "ideal A=shift(proposed);",
                        "ideal AS=std(A);",
                        "ideal AH=std(lead(A));",
                        answer("dim(G)"), answer("dim(AS)"),
                        answer("size(reduce(A,G))==0"), answer("size(reduce(J,AS))==0"),
                        # Use heads of the returned polynomials themselves: completing
                        # them with std first could hide a missing standard basis row.
                        answer("size(reduce(lead(A),heads))==0"),
                        answer("size(reduce(lead(G),AH))==0"),
                    ], variables)
                    self.assertEqual(checks, [dimension, dimension, 1, 1, 1, 1])

    def test_weak_normal_forms_and_units(self):
        for variables, ideal, point, query, member in (
            (("x", "y"), [], None, "1+y", False),
            # At the origin 1-y is a unit. Dividing x by the single
            # relation can require a multiplier for a polynomial remainder.
            (("x", "y"), ["x*(1-y)-y^2"], None, "x", False),
            (("x", "y"), ["(x-1/2)*(1-(y+2))-(y+2)^2"], ["1/2", -2], "x-1/2", False),
            (("x", "y"), ["(x-1/2)*(1-y)"], ["2/4", 0], "x-1/2", True),
            (("x", "y"), ["(x-1/2)*(1-y)"], ["1/2", 1], "x-1/2", False),
            (("x", "y"), ["x-1/2+(y+2)^2", "x-1/2+(y+2)^3"], ["1/2", -2], "(y+2)^2", True),
            (("x", "y"), ["x-1/2+(y+2)^2", "x-1/2+(y+2)^3"], ["1/2", -2], "y+2", False),
            # Lowest degree ties select x*z in lex and y^2 in grevlex.
            (("x", "y", "z"), ["(x-1/2)*z+(y+2)^2"], ["1/2", -2, 0], "(x-1/2)*z", False),
        ):
            for order in ("lex", "grevlex"):
                with self.subTest(ideal=ideal, point=point, order=order, query=query):
                    result, coords = self.check_contract("local-normal-form", variables, order, ideal, point, query)
                    checks = singular("dp", [
                        f"ideal original={','.join(ideal) or '0'};",
                        f"poly query={query};",
                        f"poly r={result['remainder']};",
                        f"poly m={result['unit']};",
                        *translated_ring(order, coords),
                        "poly F=shift(query); poly N=shift(r); poly U=shift(m);",
                        answer("reduce(U,std(maxideal(1)))==1"),
                        answer("reduce(U*F-N,G)==0"),
                        answer("N==0"), answer("reduce(F,G)==0"),
                        answer("N==0 || reduce(lead(N),heads)!=0"),
                    ], variables)
                    self.assertEqual(checks, [1, 1, int(member), int(member), 1])
