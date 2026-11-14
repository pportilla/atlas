"""Exact analytic families and infinite graded quotients."""
from itertools import permutations
import unittest

from tests.test_local_algebra_singular import calculate


class AeGradedTests(unittest.TestCase):
    def test_powers_of_three_lines(self):
        # Independent finite-module lengths, including permutations with even c.
        for (a, b, c), expected in [
            *((triple, 115) for triple in permutations((2, 3, 5))),
            ((1, 3, 5), 20), ((3, 4, 5), 529), ((2, 5, 7), 729), ((3, 5, 7), 1710),
        ]:
            with self.subTest(a=a, b=b, c=c):
                result = calculate("ae-codimension", components=[f"x^{a}", f"y^{b}", f"(x+y)^{c}"])
                self.assertEqual((result["finite"], result["value"]), (True, expected))
                if min(a, b, c) > 1:
                    self.assertEqual(result["method"], "graded_tangent_ranks")
                    self.assertGreater(result["last_weight"], 0)
        oblique = ["-3/7*(2*x+y)^2", "5*(x-3*y)^3", "17/13*(5*x-2*y)^5"]
        result = calculate("ae-codimension", components=oblique)
        self.assertEqual((result["finite"], result["value"]), (True, 115))

    def test_mixed_binary_forms_and_non_coprime_powers(self):
        for components, expected in [
            (["x^2", "y^4", "(x+y)^3"], None),  # joint gcd 1, but not pairwise
            (["x^2", "y^4", "(x+y)^3+2*x*y^2"], 69),  # same three degrees, finite
            (["x^2", "y^3", "(65521*x+y)^5"], 115),  # a bad modular image must fall back to QQ
            (["x^2", "y^3", "x^5"], None),      # repeated direction
            (["x^2", "y^3", "x^5+y^5"], None),  # sparse homogeneous nonpower
            (["x^2", "y^3", "(x+y)^5+x^2*y^3"], 115),  # full support, altered interior coefficient
        ]:
            with self.subTest(components=components):
                result = calculate("ae-codimension", components=components)
                self.assertEqual((result["finite"], result["value"]), (expected is not None, expected))

    def test_mixed_forms_with_bad_modular_rank(self):
        # Exact graded graph elimination gives these lengths for the unscaled
        # forms. Rescaling either source axis over QQ preserves the grading
        # and the quotient after compensating in the corresponding target.
        # Modulo 65521 the rescaled third component loses one variable.
        for a, b, form, expected in [
            (3, 7, "{x}^2-3*{x}*{y}-{y}^2", 242),
            (2, 3, "{x}^5+{x}^4*{y}-{x}*{y}^4+{y}^5", 115),
        ]:
            weights = []
            for x, y in [("x", "y"), ("(65521*x)", "y"), ("x", "(65521*y)")]:
                with self.subTest(a=a, b=b, x=x, y=y):
                    components = [f"x^{a}", f"y^{b}", form.format(x=x, y=y)]
                    result = calculate("ae-codimension", components=components)
                    self.assertEqual((result["method"], result["finite"], result["value"]),
                                     ("graded_tangent_ranks", True, expected))
                    weights.append(result["last_weight"])
            self.assertEqual(len(set(weights)), 1)

    def test_plane_unfolding(self):
        # For (x, xy+y^d), eliminate the x component using df/dx.
        # Modulo df/dy, x=-d*y^(d-1). The remaining tangent space has
        # powers y^S and y^(1+S), where S=<d-1,d>. Count their missing powers.
        for d in (3, 4, 5, 7, 12, 18):
            with self.subTest(d=d):
                cutoff = (d-2)*(d-1)
                semigroup = {a*(d-1)+b*d for a in range(d) for b in range(d)}
                expected = sum(k not in semigroup and k-1 not in semigroup for k in range(cutoff))
                result = calculate("ae-codimension", components=["x", f"x*y+y^{d}"])
                self.assertIs(result["finite"], True)
                self.assertEqual(result["value"], expected)

    def test_surface_to_three_space(self):
        # In R^3/df the odd terms remaining in the third component are
        # represented by y,x*y,...,x^(k-2)*y for this S-family.
        for k in (1, 2, 3, 7, 12, 20):
            with self.subTest(k=k):
                result = calculate("ae-codimension", components=["x", "y^2", f"y^3+x^{k}*y"])
                self.assertEqual((result["finite"], result["value"]), (True, k-1))

    def test_graded_infinite_and_regular_changes(self):
        # Over an algebraic closure (x²-y²,2xy) is equivalent to (u²,v²).
        # The normal space has a direction in every odd degree on a branch.
        for components in (["x^2-y^2", "2*x*y"],
                           ["x^2-y^2", "18446744073709551617/7*x*y"],
                           ["x^2+y^2", "x^2-y^2", "x*y"]):
            with self.subTest(components=components):
                result = calculate("ae-codimension", components=components)
                self.assertEqual((result["finite"], result["value"]), (False, None))

    def test_plateau_does_not_certify_jet_stabilization(self):
        # For (u²,u¹¹), missing odd powers 1,3,5,7,9 in the second
        # component give jet dimensions 1,1,2,2,... before reaching 5.
        # An invertible reparametrization preserves each source filtration.
        result = calculate("ae-codimension", variables=("t",),
                           components=["(t+t^2)^2", "(t+t^2)^11"])
        self.assertEqual((result["finite"], result["value"]), (True, 5))
        self.assertEqual(result["method"], "extended_tangent_jets")
        self.assertGreaterEqual(result["jet_degree"], 9)

    def test_affine_singularities_away_from_the_origin(self):
        # y*(1-y) is an invertible change of coordinate at zero. Its
        # additional critical point at y=1/2 has no effect on the local answer.
        result = calculate("ae-codimension", components=["x+x^2", "(y-y^2)^5+(x+x^2)*(y-y^2)"])
        self.assertEqual((result["finite"], result["value"]), (True, 3))
