"""Rational-point quotient bases and hypersurface invariants on native inputs."""

from contextlib import contextmanager
from fractions import Fraction
import unittest
from unittest.mock import patch

from engine import local_algebra
from engine import native_polynomial as np
from engine.polynomial import InputError, parse_expression


LOCAL_NUMBERS = ("milnor-number", "tjurina-number")
HYPERSURFACE_OPERATIONS = ("local-jacobian-algebra", *LOCAL_NUMBERS)


def calculate(operation, operand, variables, *, point=None, order="lex"):
    ring = {"field": "QQ", "variables": [v.name for v in variables], "order": order}
    key = "ideal" if operation in ("quotient-basis", "local-quotient-basis") else "polynomial"
    operands = {key: operand}
    if point is not None:
        operands["point"] = point
    return local_algebra.handle(operation, ring, operands)


def as_polynomials(texts, variables):
    names = [v.name for v in variables]
    return [np.from_terms(variables, parse_expression(text, names)) for text in texts]


@contextmanager
def guard_wide_powers(variables):
    """Forbid dense substitution and explicit wide rational powers in stress cases."""
    xreplace = np.Expr.xreplace
    rational_power = Fraction.__pow__

    def sparse_substitute(poly, replacements):
        for powers in np.terms(variables, poly):
            if any(v in replacements and e >= 2**63 for v, e in zip(variables, powers)):
                raise AssertionError("expanded a high power of a shifted coordinate")
        return xreplace(poly, replacements)

    def sparse_evaluate(base, power):
        if abs(base) not in (0, 1) and power >= 2**63:
            raise AssertionError("materialized a high power at the rational point")
        return rational_power(base, power)

    with patch.object(np.Expr, "xreplace", sparse_substitute), patch.object(Fraction, "__pow__", sparse_evaluate):
        yield


class PointLocalAlgebraKernel(unittest.TestCase):
    def test_fractional_cusp_in_declared_coordinate_order(self):
        # In the order (y,x) the derivative list starts with d/dy.
        y, x = variables = np.symbols("y x")
        a, b = x - Fraction(2, 3), y + Fraction(1, 2)
        f = b**2/3 - a**3
        point = ["-2/4", "4/6"]
        jacobian = [2*b/3, -3*a**2]
        for order in ("lex", "grevlex"):
            for operation in ("local-quotient-basis", *HYPERSURFACE_OPERATIONS):
                operand = jacobian if operation == "local-quotient-basis" else f
                with self.subTest(operation=operation, order=order):
                    result = calculate(operation, operand, variables, point=point, order=order)
                    self.assertEqual(result["ring"], {"variables": ["y", "x"], "field": "QQ", "order": order})
                    self.assertEqual((result["scope"], result["point"]), ("point", ["-1/2", "2/3"]))
                    self.assertIs(result["finite"], True)
                    if operation in LOCAL_NUMBERS:
                        self.assertEqual(result["value"], 2)
                    else:
                        self.assertEqual(result["dimension"], 2)
                        self.assertEqual(as_polynomials(result["basis"], variables), [1, a])
                    if operation == "local-jacobian-algebra":
                        self.assertEqual(as_polynomials(result["generators"], variables), jacobian)

    def test_smooth_off_hypersurface_and_nonisolated_points(self):
        x, y = variables = np.symbols("x y")
        f = y**2 - x**3
        for operation in HYPERSURFACE_OPERATIONS:
            with self.subTest(operation=operation):
                smooth = calculate(operation, f, variables, point=[1, 1])
                self.assertIs(smooth["finite"], True)
                self.assertEqual(smooth["value"] if operation in LOCAL_NUMBERS else smooth["basis"],
                                 0 if operation in LOCAL_NUMBERS else [])
                with self.assertRaises(InputError):
                    calculate(operation, f, variables, point=[1, 0])
                for nonisolated in ((x - 1)**2, np.S.Zero):
                    result = calculate(operation, nonisolated, variables, point=[1, 1])
                    self.assertIs(result["finite"], False)
                    if operation in LOCAL_NUMBERS:
                        self.assertIsNone(result["value"])
                    else:
                        self.assertIsNone(result["dimension"])
                        self.assertIsNone(result["basis"])

    def test_numbers_detect_a_smooth_point_without_building_derivative_polynomials(self):
        x, y, z, t, w = variables = np.symbols("x y z t w")
        point = ["2/3", "3/2", 0, 0, 0]
        tail = (z + t + w)**12
        f = x**2*y - Fraction(2, 3) + tail
        # Only d/dz survives at this point; x and y have nonzero values.
        g = x*y*z + tail
        with patch.object(local_algebra, "_jacobian", side_effect=AssertionError("built the full Jacobian")):
            for operation in LOCAL_NUMBERS:
                with self.subTest(operation=operation):
                    for equation in (f, g):
                        result = calculate(operation, equation, variables, point=point)
                        self.assertEqual((result["finite"], result["value"]), (True, 0))
                    # g+1 has the same nonzero gradient but misses the point.
                    with self.assertRaisesRegex(InputError, "pass through"):
                        calculate(operation, g + 1, variables, point=point)

    def test_point_quotients_and_explicit_zero_point_metadata(self):
        x, y = variables = np.symbols("x y")
        for generators, dimension, basis in (([], None, None), ([x], 0, []), ([x - 1], None, None)):
            result = calculate("local-quotient-basis", generators, variables, point=[1, 0])
            self.assertEqual((result["dimension"], result["basis"]), (dimension, basis))
            self.assertIs(result["finite"], dimension is not None)

        for operation in ("local-quotient-basis", *HYPERSURFACE_OPERATIONS):
            operand = [x, y] if operation == "local-quotient-basis" else x**2 + y**2
            implicit = calculate(operation, operand, variables)
            explicit = calculate(operation, operand, variables, point=["0/2", 0])
            self.assertEqual(implicit["scope"], "origin")
            self.assertNotIn("point", implicit)
            self.assertEqual(explicit, {**implicit, "scope": "point", "point": ["0", "0"]})
        with self.assertRaises(InputError):
            calculate("quotient-basis", [x], variables, point=[0, 0])

    def test_coefficient_ring_has_one_point(self):
        for generators, basis in (([], ["1"]), ([np.Rational(2, 7)], [])):
            result = calculate("local-quotient-basis", generators, (), point=[])
            self.assertEqual((result["scope"], result["point"], result["finite"]), ("point", [], True))
            self.assertEqual((result["dimension"], result["basis"]), (len(basis), basis))
        for operation in HYPERSURFACE_OPERATIONS:
            with self.subTest(operation=operation), self.assertRaises(InputError):
                calculate(operation, np.S.Zero, (), point=[])

    def test_monomial_unit_axis_can_be_removed_before_translation(self):
        x, y = variables = np.symbols("x y")
        # x is a unit at (2,0). Substituting (2+u)^N before removing that
        # unit would create N+1 terms, while the local quotient stays small.
        exponent = 2**64 + 17
        unit = np.from_terms(variables, {(exponent, 0): 1})
        multiple = np.from_terms(variables, {(exponent, 3): 1})
        with guard_wide_powers(variables):
            result = calculate("local-quotient-basis", [unit], variables, point=[2, 0])
            self.assertEqual((result["basis"], result["dimension"]), ([], 0))
            result = calculate("local-quotient-basis", [multiple, x - 2], variables, point=[2, 0])
            self.assertEqual((result["basis"], result["dimension"]), (["1", "y", "y^2"], 3))
            # f = x^N*(x-2)^3 has an isolated cubic critical point at 2.
            # The derivative returned to the caller includes the unit power.
            f = np.from_terms((x,), {(exponent + e,): c for e, c in enumerate((-8, 12, -6, 1))})
            df = np.from_terms((x,), {(k - 1,): k*c for (k,), c in np.terms((x,), f).items()})
            for operation in HYPERSURFACE_OPERATIONS:
                result = calculate(operation, f, (x,), point=[2])
                if operation in LOCAL_NUMBERS:
                    self.assertEqual(result["value"], 2)
                else:
                    self.assertEqual(as_polynomials(result["basis"], (x,)), [1, x - 2])
                    self.assertEqual(as_polynomials(result["generators"], (x,)), [df])

    def test_sparse_wide_binomials_at_points_where_terms_cancel(self):
        x, y = variables = np.symbols("x y")
        n = 2**64 + 1
        # At (1,0), x^N-1 has a simple zero. The Jacobian therefore has
        # local ideal (x-1,y). At (2,2), x^N-y^N vanishes and is smooth.
        node = np.from_terms(variables, {(2*n, 0): 1, (n, 0): -2, (0, 0): 1, (0, 2): 1})
        difference = np.from_terms(variables, {(n, 0): 1, (0, n): -1})
        node_dx = np.from_terms(variables, {(2*n - 1, 0): 2*n, (n - 1, 0): -2*n})
        with guard_wide_powers(variables):
            for f, point, basis in ((node, [1, 0], ["1"]), (difference, [2, 2], [])):
                for operation in HYPERSURFACE_OPERATIONS:
                    with self.subTest(point=point, operation=operation):
                        result = calculate(operation, f, variables, point=point)
                        self.assertIs(result["finite"], True)
                        if operation in LOCAL_NUMBERS:
                            self.assertEqual(result["value"], len(basis))
                        else:
                            self.assertEqual((result["dimension"], result["basis"]), (len(basis), basis))
                            if f is node:
                                self.assertEqual(as_polynomials(result["generators"], variables), [node_dx, 2*y])

    def test_wide_ideal_with_independent_linear_parts_at_the_point(self):
        x, y = variables = np.symbols("x y")
        n = 2**64 + 17
        equation = np.from_terms(variables, {(n, 0): Fraction(7, 3), (0, n): Fraction(-7, 3)})
        # At (2,2) its differential is a nonzero multiple of dx-dy.
        # Together with x-2 the rows generate the local maximal ideal.
        with guard_wide_powers(variables):
            result = calculate("local-quotient-basis", [equation, x - 2], variables, point=[2, 2])
        self.assertEqual((result["finite"], result["dimension"], result["basis"]), (True, 1, ["1"]))

    def test_moderate_degree_tangent_rank_and_rank_deficient_fallback(self):
        from engine import point_germs

        x, y = variables = np.symbols("x y")
        point = [1, 1]
        # The linear parts (10,1) and (1,10) span the cotangent space.
        ideal = [x**10 + y - 2, y**10 + x - 2]
        # The two squared summands also have independent linear parts.
        morse = (x**5 + y - 2)**2 + (y**5 + x - 2)**2
        with patch.object(point_germs, "_translate", side_effect=AssertionError("translated full-rank generators")):
            regular = calculate("local-quotient-basis", ideal, variables, point=point)
            self.assertEqual((regular["dimension"], regular["basis"]), (1, ["1"]))
            invariant = calculate("milnor-number", morse, variables, point=point)
            self.assertEqual((invariant["finite"], invariant["value"]), (True, 1))

        # At x=1 the first generator has a double root. Only y-1 has a
        # nonzero linear part, so the quotient still contains x-1.
        with patch.object(point_germs, "_translate", wraps=point_germs._translate) as translate:
            deficient = calculate("local-quotient-basis", [x**10 - 10*x + 9, y - 1], variables, point=point)
            self.assertTrue(translate.called)
        self.assertEqual(deficient["dimension"], 2)
        self.assertEqual(as_polynomials(deficient["basis"], variables), [1, x - 1])

    def test_mixed_wide_power_can_leave_a_finite_or_infinite_jacobian_algebra(self):
        x, y, z = variables = np.symbols("x y z")
        n = 2**64 + 1
        point = ["1/2", 0, 0]
        f = np.from_terms(variables, {(n, 2, 0): 1, (0, 0, 3): 1})
        # The unit x^N gives (y,z²) locally. Adding (x-1/2)² imposes
        # the remaining local parameter and leaves precisely 1,z.
        with guard_wide_powers(variables):
            for equation, expected in ((f, None), (f + (x - Fraction(1, 2))**2, 2)):
                for operation in HYPERSURFACE_OPERATIONS:
                    with self.subTest(finite=expected is not None, operation=operation):
                        result = calculate(operation, equation, variables, point=point)
                        self.assertIs(result["finite"], expected is not None)
                        if operation in LOCAL_NUMBERS:
                            self.assertEqual(result["value"], expected)
                        else:
                            self.assertEqual(result["dimension"], expected)
                            self.assertEqual(result["basis"], None if expected is None else ["1", "z"])

    def test_number_requests_count_without_enumerating_standard_monomials(self):
        x, y = variables = np.symbols("x y")
        degree = 2**64 + 17
        f = (x - Fraction(2, 3))**2 + np.from_terms(variables, {(0, degree): 1})
        with patch("engine.quotient_monomials._box_points", side_effect=AssertionError("enumerated a basis")):
            for operation in LOCAL_NUMBERS:
                with self.subTest(operation=operation):
                    result = calculate(operation, f, variables, point=["2/3", 0])
                    self.assertEqual((result["finite"], result["value"]), (True, degree - 1))
