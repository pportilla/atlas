import unittest
from unittest.mock import patch

import sympy as sp

from engine.constructible_euler import euler_characteristic


x,y,z = sp.symbols("x y z")


class ConstructibleEuler(unittest.TestCase):
    def test_native_euler_never_switches_to_symbolic_arithmetic(self):
        with self.assertRaisesRegex(ValueError, "rational or marked"):
            euler_characteristic((x,), (x,), domain=sp.ZZ)
        with patch("engine.native_library._kernel", return_value=None), patch("engine.kernel.strata", return_value=None):
            with self.assertRaisesRegex(ArithmeticError, "Euler characteristic kernel"):
                euler_characteristic((x,), (x,))

    def test_constant_locus_over_a_marked_field(self):
        from engine.candidate_field import indexed_field
        field, t = indexed_field((1,0,-2),0)
        self.assertEqual(euler_characteristic((), (t,), domain=field), 0)
        self.assertEqual(euler_characteristic((), (t*t-2,), domain=field), 1)
        self.assertEqual(euler_characteristic((), (), ((t*t-2,),), domain=field), 0)

    def test_independent_factors_and_arbitrary_integer_counts(self):
        from engine import native_library
        if not hasattr(native_library._kernel(), "atlas_euler"):
            self.skipTest("Requires the sparse FLINT Euler kernel.")
        # Each coordinate can be chosen independently among the indicated
        # distinct roots of unity. Use the formula even for impractically
        # large dense reference arrays.
        degree = 10**24
        self.assertEqual(euler_characteristic((x, y), (x**degree-1, y**3-1)), 3*degree)
        coordinates = sp.symbols("t:70")
        self.assertEqual(euler_characteristic(coordinates, tuple(v**2-1 for v in coordinates)), 2**70)

    def test_points_empty_loci_multiplicities_and_closed_exclusions(self):
        self.assertEqual(euler_characteristic((),()),1)
        self.assertEqual(euler_characteristic((),(3,)),0)
        self.assertEqual(euler_characteristic((x,),((x**2-1)**3,(x-1)**2*(x+1))),2)
        self.assertEqual(euler_characteristic((x,y),(),((x,y),(x-1,y))),-1)
        self.assertEqual(euler_characteristic((x,y,z),(),((x,y),(y,z))),0)
        self.assertEqual(euler_characteristic((x,y),(),((),)),0)
        self.assertEqual(euler_characteristic((x,y),(),((1,),)),1)

    def test_sparse_fibres_and_substitution_beyond_a_machine_word(self):
        from engine import native_library
        if not hasattr(native_library._kernel(), "atlas_euler"):
            self.skipTest("Requires the sparse FLINT Euler kernel.")
        n = 2**64
        # Projecting the Fermat curve to y gives n sheets off the n
        # roots of unity, and exactly one point over each root.
        for power in (n, 10**2200):
            self.assertEqual(euler_characteristic((x,y),(x**power + y**power - 1,)), power*(2-power))
        # Substitution x = -y has even and odd powers, and x = 0
        # specializes directly to the sparse constant coefficient.
        self.assertEqual(euler_characteristic((x,y),(x+y, x**n + y**n - 2)), n)
        self.assertEqual(euler_characteristic((x,y),(x+y, x**(n+1) + y**(n+1) - 2)), 0)
        self.assertEqual(euler_characteristic((x,y),(x, x**n + y**n - 1)), n)

    def test_squarefree_reduction_when_sparse_integer_gcd_declines(self):
        # At this exponent FLINT's fmpz_mpoly_gcd declines the square and
        # its x-derivative. The reduced zero set is a graph over the (y,z)-plane.
        graph = x + y**(2**64 + 1) + y*z**2
        self.assertEqual(euler_characteristic((x,y,z), (graph**2,)), 1)
        self.assertEqual(euler_characteristic((x,y,z), (graph**2,), ((y,),)), 0)

    def test_ramification_and_a_whole_line_fiber(self):
        curve = y**2-x*(x-1)*(x-2)
        # Over the x-line the degree-two map has three simple branch points.
        self.assertEqual(euler_characteristic((x,y),(curve,)),-1)
        # At x=4 two points become a whole affine line, lowering χ by one.
        self.assertEqual(euler_characteristic((x,y),((x-4)*curve,)),-2)
        # Delete the branch point above 0 and both points above 4.
        self.assertEqual(euler_characteristic((x,y),(curve,),((x,y),(x-4,))),-4)

    def test_quadratic_projection_with_nonconstant_leading_coefficient(self):
        # At x=0 the fibre is a line, a point or empty in these examples.
        # The remaining fibres have generically two roots and one at a
        # simple discriminant zero. The coordinate exclusion removes x=0.
        for equation, total, punctured in (
                (x*y**2+x+1, -1, -1),
                (x*y**2+y+1, 0, -1),
                (x*(y**2+x+1), 0, -1),
                (x*y**2+y, 1, 0)):
            with self.subTest(equation=equation):
                self.assertEqual(euler_characteristic((x,y),(equation,)), total)
                self.assertEqual(euler_characteristic((x,y),(equation,),((x,),)), punctured)

    def test_contact_chart_quadratic_and_coordinate_exclusions(self):
        q = x*y - z**2
        a = 2*q**6 + 3*x**6 + 5*y**6 + 7*z**6 + 1
        h = sp.Symbol("h")
        equation = a*h**2 - (q**3 + z**3 + 1)*h + 1
        self.assertEqual(euler_characteristic((x,y,z,h), (equation,)), -622)
        self.assertEqual(euler_characteristic((x,y,z,h), (), ((h,), (equation,))), 622)

    def test_specialized_gcd_and_nonreduced_fibers(self):
        # The degree of the common zero set jumps at x=0,1,−1. At 0 the
        # double root counts once; each of the other fibers has two points.
        equations = ((y**2-x**2)**2,y**2-x**4)
        self.assertEqual(euler_characteristic((x,y),equations),5)
        self.assertEqual(euler_characteristic((x,y),(x*y-1,)),0)
        self.assertEqual(euler_characteristic((x,y),(x*y,)),1)

    def test_higher_dimensional_fiber_integral(self):
        # Fermat Milnor fiber in C³: χ=1+(2−1)(3−1)(4−1).
        self.assertEqual(euler_characteristic((x,y,z),(x**2+y**3+z**4-1,)),7)
        self.assertEqual(euler_characteristic((x,y,z),(z-x**2-y**2,x*y-1)),0)
        # Product of three affine lines, each missing two points.
        self.assertEqual(euler_characteristic((x,y,z),(),
                         ((x*(x-1),),(y*(y-1),),(z*(z-1),))),-1)

    def test_conjugate_points_over_a_number_field(self):
        r = sp.sqrt(2)
        field = sp.QQ.algebraic_field(r)
        curve = y**2-(x**2-2)*(x-r)
        # The collision at +sqrt(2) also counts as one geometric point.
        self.assertEqual(euler_characteristic((x,y),(curve,),domain=field),0)
        self.assertEqual(euler_characteristic((x,y),(curve,),((x-r,y),),domain=field),-1)
        a = sp.CRootOf(z**3-z-1,0)
        self.assertEqual(euler_characteristic((x,y,z),(x-a,y-z,z**2-2),
                         domain=sp.QQ.algebraic_field(a)),2)


if __name__=="__main__":
    unittest.main()
