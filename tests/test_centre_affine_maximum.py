"""The batched affine descent and the general maximal-contact chart algorithm."""
from fractions import Fraction
from unittest import TestCase
from unittest.mock import patch

from engine import centres, native_polynomial as np
from engine.native_library import replay_scope


class AffineCentreMaximum(TestCase):
    def test_earlier_nonlinear_contact_at_translated_points_and_permuted_axes(self):
        x, y, z = variables = np.symbols("x y z")
        origin = (x-2, y+3, z-5)
        for p, q, r in (origin, origin[1:]+origin[:1], origin[2:]+origin[:2]):
            f = (p+q)**2+2*p**3+3*r**3
            args = variables, (), 3, (f,), [], [origin], {}
            with self.subTest(f=f):
                actual = centres._affine_maximum(*args, ())
                self.assertIsNotNone(actual)
                with patch.object(centres, "_affine_maximum", return_value=None):
                    expected = centres.maximum(*args, [])
                self.assertEqual(actual, expected)
                self.assertEqual(actual[2], origin)

    def test_multiple_contact_levels_and_exceptional_birth(self):
        x, y, z = np.symbols("x y z")
        variables = (x, y, z)
        first = (2, 0, Fraction(3, 2), 0, 1, 0, centres.INFINITE)
        second = (2, 0, Fraction(1, 2), 1, 1, 0, centres.INFINITE)
        for f, divisors, working, records, invariant, ideal in (
            ((x+y)**2+x**3+z**3, [], [variables], {}, first, variables),
            ((y+1)**2+x*(z**3+1), [("E", (x,))], [(x,y+1), (x,y+1,z**3+1)],
             {"E": first}, second, (z**3+1,x,y+1)),
            ((z+2)**2+y*(x**2-2), [("E", (y,))], [(y,z+2)], {"E": first},
             second, (x**2-2,y,z+2)),
        ):
            with self.subTest(f=f):
                args = variables, (), 3, (f,), divisors, working, records
                direct = centres._affine_maximum(*args, ())
                self.assertEqual(direct, (invariant, (), ideal))
                self.assertIs(type(direct[0][1]), int)
                with patch.object(centres, "_affine_maximum", return_value=None):
                    self.assertEqual(direct, centres.maximum(*args, list(records)))

    def test_ownership_of_conjugate_points(self):
        x, y, z = variables = np.symbols("x y z")
        f = (y+1)**2+x*(z**2-1)
        args = variables, (), 3, (f,), [("E", (x,))], [(x, y+1, z-1)], {"E": (2, 0, 2)}
        # The working set contains only one of the two closed components.
        self.assertIsNone(centres._affine_maximum(*args, ()))
        with patch.object(centres, "_affine_maximum", return_value=None):
            expected = centres.maximum(*args, ["E"])
        self.assertEqual(centres.maximum(*args, ["E"]), expected)

    def test_general_chart_handles_wide_exponents_coefficients_and_marked_frame(self):
        x, y, theta = np.symbols("x y theta")
        large = x**(2**64)
        cusp = (2, 0, Fraction(3, 2), 0, centres.INFINITE)
        for variables, ambient, f, working, value, ideal in (
            ((x,y), (), y-large, [(x,y)], (1, 0, centres.INFINITE), (large-y,)),
            ((x,y), (), (2**170+3)*y**2-x**3, [(x,y)], cusp, (x,y)),
            ((x,y,theta), (theta**2-2,), y**2-x**3, [(x,y)], cusp, (theta**2-2,x,y)),
        ):
            with self.subTest(f=f, ambient=ambient):
                args = variables, ambient, 2, (f,), [], working, {}
                self.assertIsNone(centres._affine_maximum(*args, ()))
                self.assertEqual(centres.maximum(*args, []), (value, (), ideal))

    def test_open_sets_general_directions_and_missing_kernel_operation(self):
        x, y, z = variables = np.symbols("x y z")
        args = variables, (), 3, (x*y,), [], [variables], {}
        self.assertIsNone(centres._affine_maximum(*args, ()))
        args = variables, (), 3, ((x+y)**2+x**3+z**3,), [], [variables], {}
        self.assertIsNone(centres._affine_maximum(*args, ((x,),)))
        with replay_scope(compute="gpu"):
            self.assertIsNone(centres._affine_maximum(*args, ()))
        with patch("engine.native_library._kernel", return_value=object()):
            self.assertIsNone(centres._affine_maximum(*args, ()))
