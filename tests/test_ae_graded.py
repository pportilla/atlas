"""Exact analytic families and infinite graded quotients."""
import unittest

from tests.test_local_algebra_singular import calculate


class AeGradedTests(unittest.TestCase):
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
