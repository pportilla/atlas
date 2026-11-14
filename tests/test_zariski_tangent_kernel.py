"""Exact Zariski tangent spaces and embedding dimensions, including sparse wide first jets."""

from fractions import Fraction
from math import prod
from random import Random
import unittest
from unittest.mock import patch

from engine import native_polynomial as np
from engine import zariski_tangent as tangent
from engine.point_germs import _coprime_columns
from engine.polynomial import InputError, parse_expression


def solve(ideal, *, names=("x", "y"), point=None, order="grevlex", dimension_only=False):
    names = list(names)
    variables = list(map(np.Symbol, names))
    operands = {"ideal": [np.from_terms(variables, parse_expression(f, names)) for f in ideal]}
    if point is not None:
        operands["point"] = point
    return tangent.handle("embedding-dimension" if dimension_only else "zariski-tangent-space",
                          {"field": "QQ", "variables": names, "order": order}, operands)


class ZariskiTangentKernelTests(unittest.TestCase):
    def test_shifted_equations_rref_and_free_variable_order(self):
        u, v, w = "(x-1/2)", "(y+2/3)", "(z-3)"
        f, g = f"(6*{u}+2*{v}-3*{w}+{w}^3)", f"(2*{v}+{w}+{u}^2)"
        for order in ("lex", "grevlex"):
            for ideal in ([f, g], [f"{f}+{g}", f"{g}*(1+{u})", f"7/9*{f}"], [g, f, "0"]):
                with self.subTest(order=order, ideal=ideal):
                    opts = dict(names=("x", "y", "z"), point=["2/4", "-2/3", 3], order=order)
                    space = solve(ideal, **opts)
                    self.assertEqual((space["point"], space["scope"], space["coordinate_system"]),
                                     (["1/2", "-2/3", "3"], "point", "tangent"))
                    self.assertEqual((space["ambient_dimension"], space["dimension"], space["jacobian_rank"]),
                                     (3, 1, 2))
                    self.assertEqual(space["equations"], ["x-2/3*z", "y+1/2*z"])
                    self.assertEqual(space["basis"], [["2/3", "-1/2", "1"]])
                    dimension = solve(ideal, dimension_only=True, **opts)
                    self.assertEqual((dimension["type"], dimension["value"], dimension["jacobian_rank"]),
                                     ("integer", 1, 2))
                    self.assertNotIn("basis", dimension)
                    self.assertNotIn("equations", dimension)

    def test_declared_columns_and_several_free_directions(self):
        result = solve(["2*x-3*y+w+z^2"], names=("z", "y", "x", "w"))
        self.assertEqual(result["scope"], "origin")
        self.assertNotIn("point", result)
        self.assertEqual(result["equations"], ["y-2/3*x-1/3*w"])
        self.assertEqual(result["basis"], [["1", "0", "0", "0"], ["0", "2/3", "1", "0"], ["0", "1/3", "0", "1"]])

    def test_zero_coordinate_ring_and_full_tangent_space(self):
        result = solve(["0"], names=(), point=[])
        self.assertEqual((result["point"], result["dimension"], result["equations"], result["basis"]),
                         ([], 0, [], []))
        for ideal in ([], ["x^2-y^3", "x*y"], ["(x+y)^7"]):
            with self.subTest(ideal=ideal):
                space = solve(ideal)
                self.assertEqual((space["dimension"], space["equations"], space["basis"]),
                                 (2, [], [["1", "0"], ["0", "1"]]))
        point = solve(["x-y", "x+y+x^3"])
        self.assertEqual((point["dimension"], point["equations"], point["basis"]), (0, ["x", "y"], []))

    def test_point_outside_support_has_no_local_ring(self):
        for scalar in (False, True):
            for names, ideal, point in (((), ["7/11"], []), (("x",), ["x*(x-1)"], ["1/2"]),
                                       (("x","y"), ["x", "y+1"], None)):
                with self.subTest(names=names, scalar=scalar), self.assertRaisesRegex(InputError, "point must lie on"):
                    solve(ideal, names=names, point=point, dimension_only=scalar)

    def test_wide_dependent_point_powers_and_vanishing_axis(self):
        k = 2**64+17
        # Each scalar factor at (2,4,0) is astronomically large, but the
        # gradient is proportional to (k,-k/4,1). No power or Taylor
        # expansion of that magnitude is needed for this row.
        f = f"x^{2*k}-y^{k}+x^{2*k}*z+z^{k}"
        result = solve([f], names=("x","y","z"), point=["2", "4", "0"])
        self.assertEqual(result["equations"], [f"x-1/4*y+1/{k}*z"])
        self.assertEqual(result["basis"], [["1/4","1","0"], [f"-1/{k}","0","1"]])
        # Multiplying by a huge power of a local unit preserves its equation.
        scaled = solve([f"x^{k}*(y+1/3+3*(x-2))"], point=["2", "-1/3"])
        self.assertEqual(scaled["basis"], [["-1/3","1"]])
        unit = solve([f"(1+x^{k})*y"], point=["2", 0])
        self.assertEqual((unit["equations"], unit["basis"]), (["y"], [["1","0"]]))

    def test_dimension_only_does_not_construct_nullspace(self):
        with patch.object(tangent, "_equations_and_basis", side_effect=AssertionError("materialized a basis")):
            result = solve(["(x-1/2)^2+y^3"], point=["1/2", 0], dimension_only=True)
        self.assertEqual((result["ambient_dimension"], result["jacobian_rank"], result["value"]), (2, 0, 2))

    def test_scaled_sparse_rows_against_independent_evaluated_derivatives(self):
        rng = Random(6217)
        for q in ((0,0,0), (-2,0,3), (2,4,-2), (Fraction(1,2),Fraction(-2,3),0)):
            point = tuple(map(Fraction,q))
            moving = [i for i,a in enumerate(point) if a]
            fixed = [i for i,a in enumerate(point) if not a]
            factors, columns = _coprime_columns([abs(point[i]) for i in moving])
            for _ in range(25):
                terms = {tuple(rng.randrange(5) for _ in point): Fraction(rng.randrange(-7,8),5)
                         for _ in range(16)}
                # The constant is immaterial for the differential. Adjust it
                # to put p on the scheme, keeping a mixture of signs and powers.
                terms[(0,0,0)] = -sum(c*prod(a**k for a,k in zip(point,e))
                                       for e,c in terms.items() if any(e))
                expected = [sum(c*e[i]*prod(a**(k-int(i==j)) for j,(a,k) in enumerate(zip(point,e)))
                                for e,c in terms.items() if e[i]) for i in range(3)]
                row = tangent._first_jet(terms, point, moving, fixed, factors, columns)
                self.assertEqual([bool(c) for c in row], [bool(c) for c in expected])
                if any(row):
                    pivot = next(i for i,c in enumerate(row) if c)
                    self.assertEqual([c*expected[pivot] for c in row], [c*row[pivot] for c in expected])
