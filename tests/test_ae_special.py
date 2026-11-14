"""Exact special cases for the extended right-left tangent quotient."""
import unittest

from tests.test_local_algebra_singular import calculate


class AeSpecialCases(unittest.TestCase):
    def check(self, variables, components, value, method=None):
        result = calculate("ae-codimension", variables=variables, components=components)
        self.assertIs(result["finite"], value is not None, result)
        self.assertEqual(result["value"], value, result)
        if method:
            self.assertEqual(result["method"], method)

    def test_rectangular_first_jets_and_constant_map(self):
        for names, polynomials in (
                (("x", "y"), ["x", "x+y"]),
                (("x", "y", "z"), ["x+y^2", "y-z"]),
                (("x", "y"), ["x+y", "x-y", "x^2+y^3"]),
                (("t",), ["0", "t^7", "t+t^3"])):
            with self.subTest(names=names, polynomials=polynomials):
                self.check(names, polynomials, 0, "regular_map")
        self.check(("x", "y"), ["0", "0"], None, "constant_map")

    def test_scalar_jacobian_target_action(self):
        # In the Milnor algebra of x^5+y^5+x^3*y^3, f reduces to a nonzero
        # multiple of x^3*y^3 and f²=0. Its quotient by 1,f has length 16-2.
        self.check(("x", "y"), ["x^5+y^5+x^3*y^3"], 14, "jacobian_target_powers")
        self.check(("x", "y"), ["(x+y)^5+y^5+(x+y)^3*y^3"], 14, "jacobian_target_powers")
        self.check(("x", "y"), ["(2/3)*x^5+3*y^5+x^3*y^2"], 15, "jacobian_target_powers")
        # Euler cancellation leaves y^5, divisible by a Jacobian head and requiring division.
        self.check(("x", "y"), ["x^4+y^5+x^2*y^3+3*x^4*y-17*x^2*y^5"],
                   10, "jacobian_target_powers")
        # Here f reduces to -x³y³z³/8. Its square is the top standard monomial.
        self.check(("x", "y", "z"), ["x^8+y^8+z^8+x^3*y^3*z^3"],
                   340, "jacobian_target_powers")
        self.check(("x", "y"), ["(x+y)^3"], None, "nonisolated_critical_locus")
        huge = 2**80 + 9
        self.check(("x", "y", "z"), [f"(-2/3)*x^{huge}+y^4+17*z^5"],
                   (huge - 1) * 3 * 4 - 1, "separated_jacobian")
        self.check(("t",), [f"t^{huge}+t^{huge+1}"], huge - 2, "univariate_order")

    def test_semigroup_gaps_and_redundant_components(self):
        # <3,4,5> has gaps 1,2, both pseudo-Frobenius. Three target
        # components give 5 independent classes, a fourth gives 7.
        self.check(("t",), ["t^3", "t^4", "t^5"], 5, "monomial_curve_semigroup")
        self.check(("t",), ["t^5", "-3*t^3", "0", "t^4"], 7, "monomial_curve_semigroup")
        self.check(("t",), ["t^4", "t^3", "5*t^5", "t^3"], 7, "monomial_curve_semigroup")
        a = 2**70 + 1
        self.check(("t",), [f"t^{a}", f"(-7/2)*t^{a+2}"],
                   (a - 1)*(a + 1)//2, "monomial_curve_semigroup")

    def test_ramification_of_the_common_parameter(self):
        # Both generators have a common polynomial parameter. Only the
        # critical parameter induces a multiple cover at this source point.
        for h in ("t^2+t^3", "t^3-2*t^4"):
            with self.subTest(h=h):
                self.check(("t",), [f"({h})^2", f"({h})^3"], None, "multiple_cover")
                self.check(("t",), [h, f"({h})^3-5*({h})"], None, "multiple_cover")
        self.check(("t",), ["(t+t^2)^2", "(t+t^2)^3"], 1)
        # Coprime orders imply a primitive parameter on the image.
        self.check(("t",), ["t^2+t^3", "t^3"], 1)

    def test_products_of_ramified_factors(self):
        self.check(("x", "y"), ["x^2+x^3", "y^2-y^5"], None, "ramified_axis")
        self.check(("x", "y", "z"), ["y^3+y^4", "x^5", "z+z^2"], None, "ramified_axis")
        self.check(("x", "y"), ["x^2", "(1+x^2)*y^2"], None, "ramified_axis")
        # A lone ramified factor of order two gives a stable fold.
        self.check(("x", "y"), ["x+x^2", "y^2+y^3"], 0)

    def test_multiplicity_two_hypersurfaces(self):
        # In A=QQ[[x,s]], s=y², the residual ideal is (P, ∂P/∂x, s∂P/∂s).
        self.check(("x", "y"), ["x", "y^2", "x^4*y+x^2*y^3+y^5"], 6, "double_cover_hypersurface")
        self.check(("x", "y"), ["x", "y^2", "-3*x^4*y+4*y^7+x*y^5+x^4*y^2"],
                   7, "double_cover_hypersurface")
        self.check(("x", "y"), ["x", "y^2", "x*y^3+y^7"], None, "double_cover_hypersurface")
        self.check(("x", "y", "z"), ["-5*y", "3*z^2", "x^3*z+y^2*z+z^5+3*x*z^6", "(17/2)*x"],
                   4, "double_cover_hypersurface")
        k = 2**84 + 33
        self.check(("x", "y"), ["x", "y^2", f"y^3+(123456789012345678901/7)*x^{k}*y"],
                   k-1, "double_cover_hypersurface")
        # A cubic projection has a different tangent quotient.
        self.check(("x", "y"), ["x", "y^3", "x*y+y^5"], 2)

    def test_free_source_and_unfolding_parameters(self):
        self.check(("u", "v"), ["v^2", "v^3"], None, "unused_source_coordinate")
        self.check(("u", "v"), ["v^2+u^5", "u"], 0, "trivial_unfolding")
        self.check(("u", "v"), ["u", "v^3+u^2"], None, "trivial_unfolding")
        self.check(("u", "v"), ["v^2", "-2*u", "v^3+u^3"], None, "trivial_unfolding")
        # The mixed term makes this a stable cusp, not the trivial unfolding of v^3.
        self.check(("u", "v"), ["u", "v^3+u*v"], 0)
