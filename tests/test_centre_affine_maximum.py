"""The batched affine descent and the general maximal-contact chart algorithm."""
from fractions import Fraction
import os
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

    def test_transverse_double_point_and_global_line(self):
        x, y, z = variables = np.symbols("x y z")
        a, b = x - 2, y + 3
        point = (a, b, z)
        for f, birth, expected in (
            (a*b + z**3, None, point),
            (a*b + z**3, (2, 0, 2), point),
            (a*b, (2, 0, 1, 0, centres.INFINITE), (a, b)),
            # A different singular point on the boundary prevents using a
            # single global maximal ideal to represent this contact.
            (a*(a-1)*b + z**3, None, None),
            # A degenerate tangent section must follow the full descent.
            (a**2 + b**3 + z**3, None, None),
        ):
            with self.subTest(f=f, birth=birth):
                args = variables, (), 3, (f,), [("E", (z,))], [point], {"E": birth}
                with patch.dict(os.environ, ATLAS_LINEAR_ALGEBRA="0"):
                    reference = centres.maximum(*args, ["E"])
                self.assertEqual(centres.maximum(*args, ["E"]), reference)
                if expected is not None:
                    direct = centres._affine_maximum(*args, ())
                    self.assertEqual(direct, reference)
                    self.assertEqual(direct[2], expected)
                    self.assertIs(type(direct[0][1]), int)

    def test_ownership_of_conjugate_points(self):
        x, y, z = variables = np.symbols("x y z")
        f = (y+1)**2+x*(z**2-1)
        args = variables, (), 3, (f,), [("E", (x,))], [(x, y+1, z-1)], {"E": (2, 0, 2)}
        # The working set contains only one of the two closed components.
        self.assertIsNone(centres._affine_maximum(*args, ()))
        with patch.object(centres, "_affine_maximum", return_value=None):
            expected = centres.maximum(*args, ["E"])
        self.assertEqual(centres.maximum(*args, ["E"]), expected)

    def test_transverse_higher_order_at_a_thick_working_point(self):
        variables = np.symbols("x y z")
        for i, d in enumerate((3, 4, 5)):
            a, b, e = variables[i:] + variables[:i]
            point = (a**3, b, e)
            # Repeated tangent factors still span two transverse directions.
            f = a**(d-1)*b + e**(d+1)
            for birth, first, second in ((None, 1, 0), ((d, 1, 1), 0, 1)):
                with self.subTest(order=d, birth=birth):
                    args = variables, (), 3, (f,), [("E", (e,))], [point, point], {"E": birth}
                    with patch.dict(os.environ, ATLAS_LINEAR_ALGEBRA="0"):
                        reference = centres.maximum(*args, ["E"])
                    self.assertEqual(reference, ((d, first, 1, second, 1, 0, centres.INFINITE), (), variables))
                    self.assertEqual(centres._affine_maximum(*args, ()), reference)

    def test_transverse_certificate_needs_global_boundary_and_birth(self):
        x, y, z = axes = np.symbols("x y z")
        for f, work, birth in (
            # One tangent direction on E, despite a second normal direction.
            ((x+y)**4 + z**9, axes, None),
            # Full tangent rank at zero, but another order-four point on E.
            ((x*(x-1))**4 + y**4 + z**9, axes, None),
            # The working piece has an unconstrained coordinate.
            (x**4+y**4+z**9, (x, y), None),
            # E stays new. The centre can contain the full z-axis.
            (x**4+y**4+x**4*z**9, axes, (4, 0, 1, 0)),
        ):
            with self.subTest(f=f, work=work):
                args = axes, (), 3, (f,), [("E", (z,))], [work], {"E": birth}
                self.assertIsNone(centres._affine_maximum(*args, ()))

    def test_double_contact_with_a_residual_power_on_the_boundary(self):
        variables = np.symbols("x y z")
        for i, a in enumerate((3, 4, 5)):
            t, s, e = variables[i:] + variables[:i]
            f = (1 + e*s)*t**2 + e**3*s**a + e**2*s**a*t
            args = variables, (), 3, (f,), [("E", (e,))], [(t, s, e**3)], {"E": (2, 1)}
            with self.subTest(residual=Fraction(a, 2)):
                with patch.dict(os.environ, ATLAS_LINEAR_ALGEBRA="0"):
                    reference = centres.maximum(*args, ["E"])
                expected = (2, 0, Fraction(a, 2), 1, 1, 0, centres.INFINITE)
                self.assertEqual(reference, (expected, (), variables))
                self.assertEqual(centres._affine_maximum(*args, ()), reference)

    def test_double_contact_cancellations_and_other_boundary_points(self):
        t, s, e = variables = np.symbols("t s e")
        for f in (
            (t + e*s)**2 + e**3*s**3,       # Leading critical coefficient cancels.
            t**2 + e**2*s**4 + e**3,        # A mixed term lowers the residual order.
            t**2 + e**2*s**2*(s-1)**2,      # Two critical points on the boundary.
        ):
            args = variables, (), 3, (f,), [("E", (e,))], [variables], {"E": (2, 1)}
            with self.subTest(f=f):
                with patch.dict(os.environ, ATLAS_LINEAR_ALGEBRA="0"):
                    reference = centres.maximum(*args, ["E"])
                self.assertEqual(centres.maximum(*args, ["E"]), reference)

    def test_cubic_contact_on_a_working_boundary_line(self):
        t, e, s = axes = np.symbols("t e s")
        f = t**3 + t**2*e**2*s**2 + t*e**2*s**3 + e*s**4 + e**2*s**3
        args = axes, (), 3, (f,), [("E", (e,))], [(e**2, t**3), (e, t)], {"E": (3, 1)}
        with patch.dict(os.environ, ATLAS_LINEAR_ALGEBRA="0"):
            reference = centres.maximum(*args, ["E"])
        self.assertEqual(reference, ((3, 0, Fraction(4, 3), 1, 1, 0, centres.INFINITE), (), axes))
        self.assertEqual(centres._affine_maximum(*args, ()), reference)

    def test_competing_conjugate_roots_of_the_boundary_coefficient(self):
        t, e, s = axes = np.symbols("t e s")
        for multiplicity, expected, accepted in (
            (1, axes, True),
            (2, (s**3+s, t, e), False),
            (3, (s**2+1, t, e), False),
        ):
            with self.subTest(multiplicity=multiplicity):
                f = t**2 + e*s**2*(s**2+1)**multiplicity
                args = axes, (), 3, (f,), [("E", (e,))], [(e, t)], {"E": (2, 1)}
                with patch.dict(os.environ, ATLAS_LINEAR_ALGEBRA="0"):
                    reference = centres.maximum(*args, ["E"])
                self.assertEqual(reference[2], expected)
                self.assertEqual(centres._affine_maximum(*args, ()), reference if accepted else None)

    def test_cubic_contact_requires_all_coefficients_and_global_working_support(self):
        t, e, s = axes = np.symbols("t e s")
        principal = t**3 + e*t**9
        for f, work in (
            (principal + e*s**4 + t*e*s, (e, t)),      # A lower coefficient wins.
            (principal + e*s**4*(s-1)**4, (e, t)),     # A second point on E.
            (principal + e**3*s**4, (e, t)),           # The entire line has order three.
            (principal + e*s**4, (t, s)),              # A line extends off E.
        ):
            with self.subTest(f=f, work=work):
                args = axes, (), 3, (f,), [("E", (e,))], [work], {"E": (3, 1)}
                self.assertIsNone(centres._affine_maximum(*args, ()))

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
