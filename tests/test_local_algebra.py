"""Kernel-level examples for rational quotients and hypersurface germs at the origin."""
import unittest

from engine import local_algebra
from engine import native_polynomial as np
from engine.polynomial import parse_expression


def calculate(operation, inputs, names=("x", "y"), order="grevlex"):
    ring = {"field": "QQ", "variables": list(names), "order": order}
    symbols = tuple(np.Symbol(name) for name in names)

    def polynomial(text):
        return np.from_terms(symbols, parse_expression(text, list(names)))

    if operation in ("quotient-basis", "local-quotient-basis"):
        operands = {"ideal": [polynomial(f) for f in inputs]}
    else:
        operands = {"polynomial": polynomial(inputs)}
    return local_algebra.handle(operation, ring, operands)


class LocalAlgebraTests(unittest.TestCase):
    def assert_quotient(self, result, scope, basis):
        self.assertEqual(result["type"], "quotient-basis")
        self.assertEqual(result["scope"], scope)
        self.assertIs(result["finite"], basis is not None)
        self.assertEqual(result["dimension"], len(basis) if basis is not None else None)
        self.assertEqual(result["basis"], basis)

    def assert_number(self, result, expected):
        self.assertEqual(result["type"], "integer")
        self.assertEqual(result["scope"], "origin")
        self.assertIs(result["finite"], expected is not None)
        self.assertEqual(result["value"], expected)

    def test_cancellation_between_lowest_parts(self):
        # The lowest part of either generator is x. Subtracting gives y²(1-y),
        # so y² belongs to the ideal at the origin.
        ideal = ["x+y^2", "x+y^3"]
        self.assert_quotient(calculate("local-quotient-basis", ideal), "origin", ["1", "y"])
        # Here their difference gives x². Further reduction by x+y² gives y⁴.
        ideal = ["x+y^2", "x+y^2+x^2"]
        self.assert_quotient(calculate("local-quotient-basis", ideal), "origin", ["1", "y", "y^2", "y^3"])

    def test_invertible_local_factor_discards_positive_dimensional_component(self):
        ideal = ["(x-1)*x^2", "(x-1)*y^3"]
        self.assert_quotient(calculate("quotient-basis", ideal), "global", None)
        self.assert_quotient(calculate("local-quotient-basis", ideal), "origin",
                             ["1", "x", "y", "x*y", "y^2", "x*y^2"])
        self.assert_quotient(calculate("local-quotient-basis", ["x+2"]), "origin", [])

    def test_monomial_representatives_follow_global_and_local_orders(self):
        ideal = ["x-y^2", "x*y"]
        global_bases = {"grevlex": ["1", "x", "y"], "lex": ["1", "y", "y^2"]}
        for order, global_basis in global_bases.items():
            with self.subTest(order=order):
                result = calculate("quotient-basis", ideal, order=order)
                self.assertEqual(result["ring"]["order"], order)
                self.assert_quotient(result, "global", global_basis)
                local = calculate("local-quotient-basis", ideal, order=order)
                self.assert_quotient(local, "origin", ["1", "y", "y^2"])

    def test_unit_zero_and_infinite_quotients(self):
        for op, scope in [("quotient-basis", "global"), ("local-quotient-basis", "origin")]:
            for names, rows, expected in [
                    ((), [], ["1"]), ((), ["5/3"], []), (("x", "y"), [], None),
                    (("x", "y"), ["x*y"], None)]:
                with self.subTest(operation=op, names=names, ideal=rows):
                    result = calculate(op, rows, names)
                    self.assertEqual(result["ring"]["variables"], list(names))
                    self.assert_quotient(result, scope, expected)

    def test_jacobian_algebra_has_the_ordered_derivatives(self):
        # The variables here are in the opposite order to their printed names.
        names = ("y", "x")
        result = calculate("local-jacobian-algebra", "y^2/3-x^3", names)
        self.assertEqual(result["type"], "local-jacobian-algebra")
        self.assertEqual(result["scope"], "origin")
        self.assertIs(result["finite"], True)
        self.assertEqual(result["dimension"], 2)
        self.assertEqual(result["basis"], ["1", "x"])
        self.assertEqual(result["ring"]["variables"], list(names))
        self.assertEqual([parse_expression(f, names) for f in result["generators"]],
                         [parse_expression(f, names) for f in ("2*y/3", "-3*x^2")])

    def test_smooth_germ_and_nonisolated_critical_locus(self):
        # The derivative 1+2*x is a nonconstant local unit, including when a
        # second critical point occurs away from the origin.
        for f, expected in [("x+x^2+y^2", 0), ("x^2", None), ("0", None)]:
            with self.subTest(polynomial=f):
                jacobian = calculate("local-jacobian-algebra", f)
                self.assertIs(jacobian["finite"], expected is not None)
                self.assertEqual(jacobian["dimension"], expected)
                self.assertEqual(jacobian["basis"], [] if expected == 0 else None)
                self.assert_number(calculate("milnor-number", f), expected)
                self.assert_number(calculate("tjurina-number", f), expected)

    def test_smooth_numbers_skip_derivatives_but_jacobian_algebra_returns_them(self):
        names = ("x", "y", "z")
        f = "x^3+x*y^2-5*y/7"
        expected = [parse_expression(g, names) for g in ("3*x^2+y^2", "2*x*y-5/7", "0")]
        for order in ("lex", "grevlex"):
            with self.subTest(order=order):
                for operation in ("milnor-number", "tjurina-number"):
                    result = calculate(operation, f, names, order)
                    self.assert_number(result, 0)
                    self.assertNotIn("generators", result)
                result = calculate("local-jacobian-algebra", f, names, order)
                self.assertEqual((result["finite"], result["dimension"], result["basis"]), (True, 0, []))
                self.assertEqual([parse_expression(g, names) for g in result["generators"]], expected)

    def test_tjurina_quotient_also_imposes_the_equation(self):
        f = "x^4+y^5+x^2*y^3"
        for operation, expected in [("milnor-number", 12), ("tjurina-number", 11)]:
            with self.subTest(operation=operation):
                answer = calculate(operation, f)
                self.assert_number(answer, expected)
                self.assertNotIn("basis", answer)

    def test_wide_exponent_milnor_number_without_enumerating_the_basis(self):
        degree = 2**64 + 17
        self.assert_number(calculate("milnor-number", f"x^{degree}", ("x",)), degree - 1)

    def test_hypersurface_requires_coordinates_and_origin_on_equation(self):
        for operation in ("local-jacobian-algebra", "milnor-number", "tjurina-number"):
            for names, f in [((), "0"), (("x", "y"), "1+y+x^2+y^2")]:
                with self.subTest(operation=operation, variables=names, polynomial=f):
                    with self.assertRaises(ValueError):
                        calculate(operation, f, names)


class PlanarQuotientLengthTests(unittest.TestCase):
    def test_asymmetric_steps_with_redundant_and_outside_generators(self):
        from engine.quotient_monomials import quotient_monomials

        heads = [(9, 0), (0, 8), (0, 12), (2, 7), (2, 3),
                 (2, 3), (3, 4), (5, 2), (8, 1), (10, 1), (1, 8)]
        points = [(x, y) for x in range(9) for y in range(8)
                  if not any(x >= a and y >= b for a, b in heads)]
        points.sort(key=lambda p: (sum(p), -p[0]))
        self.assertEqual(quotient_monomials(2, reversed(heads), include_basis=False), (len(points), None))
        self.assertEqual(quotient_monomials(2, heads, include_basis=True), (len(points), points))

    def test_two_interleaved_planar_components(self):
        from engine.quotient_monomials import quotient_monomials

        # Take (x,z)^d, (y^4,y*w^2,w^3) and (v^5) on disjoint sets of axes.
        d = 256
        heads = [(i, 0, d-i, 0, 0) for i in range(d+1)]
        heads.extend([(0, 4, 0, 0, 0), (0, 1, 0, 2, 0),
                      (0, 0, 0, 3, 0), (0, 0, 0, 0, 5)])
        self.assertEqual(quotient_monomials(5, heads, include_basis=False), (d*(d+1)//2 * 9 * 5, None))


if __name__ == "__main__":
    unittest.main()
