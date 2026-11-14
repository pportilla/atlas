"""Independent local leading coefficients and wide shifted standard bases."""
from fractions import Fraction
from itertools import product
from math import comb
import unittest

from engine.algebra import dispatch
from engine.polynomial import parse_expression


def basis(ideal, names, point, order="grevlex"):
    result = dispatch({"operation": "local-standard-basis", "ring": {"variables": list(names), "order": order},
                       "compute": "cpu", "operands": {"ideal": ideal, "point": list(point)}})["result"]
    return [parse_expression(f, list(names)) for f in result["generators"]]


def initial(row, point, order, degree_limit):
    # All moving wide exponents in these fixtures are at ±1, so only the
    # Taylor coefficients actually inspected below are materialized.
    for d in range(degree_limit+1):
        layer = {}
        for h in product(range(d+1), repeat=len(point)):
            if sum(h) != d:
                continue
            coefficient = Fraction(0)
            for e, c in row.items():
                if any(a < b for a, b in zip(e, h)):
                    continue
                for a, b, p in zip(e, h, point):
                    c *= comb(a, b)*Fraction(p)**(a-b)
                coefficient += c
            if coefficient:
                layer[h] = coefficient
        if layer:
            h = max(layer) if order == "lex" else min(layer, key=lambda e: e[::-1])
            return h, layer[h]
    raise AssertionError("Expected leading term in the tested Taylor range.")


class LargePointBasisTests(unittest.TestCase):
    def test_negative_wide_principal_and_monic_centered_coefficient(self):
        n = 2**64+7
        for order in ("lex", "grevlex"):
            row, = basis([f"x^{n}+1+y"], "xy", [-1,0], order)
            self.assertEqual(row, {(n,0): Fraction(1,n), (0,1): Fraction(1,n), (0,0): Fraction(1,n)})
            self.assertEqual(initial(row, [-1,0], order, 1), ((1,0),1))

    def test_wide_power_coordinates_preserve_declared_local_order(self):
        n = 2**64+7
        u, v = f"(x^{n}-1)", "(y-1/2)"
        rows = [f"({u}+2*{v})^3", f"(3*{u}+{v})^3*(5*(3*{u}+{v})-4)"]
        for order in ("lex", "grevlex"):
            result = basis(rows, "xy", [1,"1/2"], order)
            self.assertEqual({initial(f, [1,"1/2"], order, 5) for f in result},
                             {((3,0),1), ((2,1),1), ((1,3),1), ((0,5),1)})

    def test_wide_coordinate_slice_requires_a_new_standard_head(self):
        n = 2**64+7
        rows = ["x-1", "y^2+z^3", f"y*z+(x^{n}-1)*z^2"]
        result = basis(rows, "xyz", [1,0,0])
        # In x=1 the last row becomes yz. Its S-pair with y²+z³ gives z⁴.
        self.assertEqual({initial(f, [1,0,0], "grevlex", 4) for f in result},
                         {((1,0,0),1), ((0,2,0),1), ((0,1,1),1), ((0,0,4),1)})

    def test_noncompressible_zero_dimensional_jets(self):
        n = 2**64+7
        u = f"(x^{n}+x-2)"
        rows = [f"({u}+y)^2", f"({u}-y)^2"]
        for order in ("grevlex","lex"):
            result = basis(rows, "xy", [1,0], order)
            # u is an étale parameter at x=1. The length-four complete
            # intersection has leading ideal (X², Xy, y³).
            self.assertEqual({initial(f, [1,0], order, 3) for f in result},
                             {((2,0),1), ((1,1),1), ((0,3),1)})

    def test_positive_dimensional_sparse_mora_requires_a_new_head(self):
        n = 2**64+7
        g = f"(x^{n}+x-1)"
        rows = [f"y^2+{g}*z^3", "y*z"]
        # z times the first equation minus y times the second gives
        # g*z^4. Since g(1)=1 the z^4 row is a local generator. Neither
        # power strides nor an isolated-point Taylor certificate applies.
        for order in ("lex", "grevlex"):
            result = basis(rows, "xyz", [1,0,0], order)
            self.assertEqual({initial(f, [1,0,0], order, 4) for f in result},
                             {((0,2,0),1), ((0,1,1),1), ((0,0,4),1)})

    def test_independent_parameters_extend_to_full_coordinates(self):
        n = 2**64+7
        u, v = f"(x^{n}*y^{n}+x+y-3+z)", "(x-y+z)"
        # u and v are independent parameters in three variables. Completing
        # their Jacobian by a centered coordinate leaves a smooth free axis;
        # (u², uv, v²) then has tangent heads X², XY, Y².
        equations = [f"{u}^2", f"{u}*{v}", f"{v}^2"]
        for order in ("lex", "grevlex"):
            result = basis(equations, "xyz", [1,1,0], order)
            self.assertEqual({initial(f, [1,1,0], order, 2) for f in result},
                             {((2,0,0),1), ((1,1,0),1), ((0,2,0),1)})
