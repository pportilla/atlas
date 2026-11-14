"""Compare local Hilbert–Samuel lengths and multiplicities with live Singular."""
from fractions import Fraction
import shutil
import unittest

from tests.test_local_algebra_singular import calculate, singular
from tests.test_local_standard_singular import answer, translated_ring


@unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
class HilbertSamuelSingularTests(unittest.TestCase):
    def check_invariants(self, ideal, variables, order, dimension, multiplicity, *, point=None, degrees=None):
        coords = [0] * len(variables) if point is None else point
        requested = list(range(6)) if degrees is None else degrees
        # Compute each requested length and its predecessor independently in
        # Singular's translated local ring. The graded value is their difference.
        indices = sorted({k for n in requested for k in (n, n - 1) if k >= 0})
        checks = singular("dp", [
            f"ideal original={','.join(ideal) or '0'};",
            *translated_ring(order, coords),
            answer("dim(G)"), answer("mult(G)"),
            *(answer(f"vdim(std(J+maxideal({k + 1})))") for k in indices),
        ], variables)
        self.assertEqual(checks[:2], [dimension, multiplicity])
        lengths = dict(zip(indices, checks[2:], strict=True))
        lengths[-1] = 0

        operands = {"ideal": ideal}
        if point is not None:
            operands["point"] = point
        options = {} if degrees is None else {"degrees": degrees}
        result = calculate("hilbert-samuel-function", variables=variables, order=order, **operands, **options)
        self.assertEqual(result["type"], "hilbert-samuel-function")
        self.assertEqual(result["degrees"], requested)
        self.assertEqual(result["values"], [lengths[n] for n in requested])
        self.assertEqual(result["graded_values"], [lengths[n] - lengths[n - 1] for n in requested])
        self.assertEqual((result["dimension"], result["multiplicity"]), (dimension, multiplicity))

        number = calculate("hilbert-samuel-multiplicity", variables=variables, order=order, **operands)
        self.assertEqual(number["type"], "integer")
        self.assertEqual((number["dimension"], number["value"]), (dimension, multiplicity))
        for record in (result, number):
            self.assertEqual(record["scope"], "origin" if point is None else "point")
            if point is not None:
                self.assertEqual(record["point"], [str(Fraction(q)) for q in point])

    def test_multiplicity_at_four_points_of_the_affine_plane(self):
        u, v = "(x-1/2)", "(y+2/3)"
        curve = [f"{v}^3-{u}^3*({u}-1)^2"]
        for point, dimension, multiplicity in (
            (["2/4", "-4/6"], 1, 3), (["3/2", "-2/3"], 1, 2),
            (["19/2", "106/3"], 1, 1), (["5/2", "-2/3"], -1, 0),
        ):
            for order in ("lex", "grevlex"):
                with self.subTest(point=point, order=order):
                    self.check_invariants(curve, ("x", "y"), order, dimension, multiplicity, point=point)

    def test_ideals_of_different_dimensions(self):
        for variables, ideal, dimension, multiplicity in (
            (("x", "y", "z"), [], 3, 1),          # Regular local ring.
            (("x", "y"), ["1"], -1, 0),           # Zero quotient.
            # A plane and a line meet at the origin. Only the plane contributes
            # to the two-dimensional multiplicity; the line changes the lengths.
            (("x", "y", "z"), ["x*y", "x*z"], 2, 1),
            # The line x=1 is a global component; the origin has finite length 4.
            (("x", "y"), ["(1-x)*(x+y^2)", "(1-x)*(x+y^2+x^2)"], 0, 4),
            (("x", "y", "z"), ["x^2+y^3", "y^3", "z^2", "x*z-y^2+z^3"], 0, 7),
        ):
            for order in ("lex", "grevlex"):
                with self.subTest(ideal=ideal, order=order):
                    self.check_invariants(ideal, variables, order, dimension, multiplicity,
                                          degrees=[5, 0, 3, 1, 5, 8])

    def test_stabilization_beyond_default_degree_samples(self):
        # The associated graded algebra has pure powers x^37 and y^41.
        # Its first six lengths agree with those of a regular surface, but
        # its socle is in degree 76 and its eventual length is 1517.
        for order in ("lex", "grevlex"):
            with self.subTest(order=order):
                self.check_invariants(["x^37+y^38", "y^41"], ("x", "y"), order, 0, 37 * 41,
                                      degrees=[0, 5, 36, 37, 40, 41, 75, 76, 77, 89])
                # The embedded finite piece adds a graded basis element in
                # degrees 42 through 58. The curve's eventual graded value is 42.
                self.check_invariants(["x^43", "x^42*y^17"], ("x", "y"), order, 1, 42,
                                      degrees=[5, 41, 42, 58, 59, 75])
