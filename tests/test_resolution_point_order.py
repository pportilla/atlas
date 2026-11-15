"""Exact order on a smooth frame and certification of working point supports."""
from fractions import Fraction
import unittest
from unittest.mock import patch

from engine import native_polynomial as np
from engine.blowup_ideal import ideal_basis
from engine.centre_weights import Presentation
from engine.centres import Frame, maximum
from engine.native_library import ideal_request
from tests.test_centre_working_pieces import grouped
from tests.test_component_regularity import b9_component_requests


def order_request(variables, equations, point, polynomial, cap):
    header = f"smooth_point_order {len(variables)} {len(equations)} {cap}"
    rows = [*(np.row(variables, f) for f in equations), " ".join(map(str, point)),
            np.row(variables, polynomial)]
    return header, rows


def frame_order(variables, equations, point, polynomial, cap=10):
    return ideal_request(*order_request(variables, equations, point, polynomial, cap))


def frame_points(variables, equations, units=()):
    rows = [np.row(variables, f) for f in (*equations, *units)]
    return ideal_request(f"frame_points {len(variables)} {len(equations)} {len(units)}", rows)


def direction_request(variables, equations, point, polynomial, coords, axis, degree):
    _, rows = order_request(variables, equations, point, polynomial, degree)
    return (f"smooth_point_direction {len(variables)} {len(equations)} {degree} {axis}",
            [*rows, " ".join(map(str, coords))])


def finite_order_cases():
    x, y, z, w = np.symbols("x y z w")
    p = Fraction(3, 2)
    a, b, c = x - p, y + 2, z - 1
    # The local equation b + b^2 = a^2 + c^3 gives b order two.
    # Subtract its linear-in-b solution to expose the order-four cancellation.
    implicit = b + b**2 - a**2 - c**3
    yield "curved_translated", (x, y, z), (implicit,), (p, -2, 1), b-a**2-c**3, 4
    yield "mixed_tangent", (x, y, z), (implicit,), (p, -2, 1), a*b*c, 4
    # Two equations with a coupled Jacobian. At the origin they imply
    # z=x^2 and w=y^2, so z^2*w + x*y^6 has order six.
    rows = (z+w-x**2-y**2, z-w-x**2+y**2)
    yield "coupled_codimension_two", (w, y, x, z), rows, (0,)*4, z**2*w+x*y**6, 6
    yield "affine_threefold", (x, y, z), (), (p, -2, 1), a**2*b*c**4+c**8, 7
    # Localizing at (0,0) removes the disjoint sheet x=1.
    yield "separate_sheet", (x, y), (x*(x-1),), (0,0), x+y**5, 5
    yield "unit_on_curve", (x, y), (y+y**2-x**3,), (0,0), 1+x, 0
    target = next(r for r in b9_component_requests() if r["chart"] == 23)
    variables = np.symbols("x y u z a b")
    *equations, residual = (np.from_row(variables, row) for row in target["rows"])
    yield "b9_chart_23", variables, equations, (0, Fraction(-1,4), Fraction(15,26), 0, 0, 0), residual, 6


class ResolutionPointOrder(unittest.TestCase):
    def test_implicit_contact_derivatives_agree_on_the_entire_frame(self):
        x, y, z, t = v = np.symbols("x y z t")
        frame = Frame(v, (x, y), (z*z+z-x*y, t+z*y-x**2), units=(1+2*z,))
        f = t**3 + x*z**2 - y**4
        for axis in frame.coords:
            with self.subTest(axis=axis):
                chain = [f]
                for _ in range(4):
                    chain.append(frame.derivative(chain[-1], axis))
                self.assertEqual(frame.derivative_chain(f, axis, 4), tuple(chain))

    def test_contact_derivative_chain_preserves_wide_sparse_terms(self):
        x, y, z = variables = np.symbols("x y z")
        frame = Frame(variables, (x,y), (z-x,))
        # On z=x the signed implicit field is d/dx + d/dz. The other
        # coordinate remains fixed, including its arbitrary-precision power.
        factor = Fraction(2**310+17, 2**220+5)*y**(2**75+3)
        expected = tuple(factor*c*z**k for c,k in ((1,4),(4,3),(12,2),(24,1),(24,0),(0,0)))
        self.assertEqual(frame.derivative_chain(expected[0], x, 5), expected)

    def test_weighted_threshold_is_attained_at_one_point(self):
        # The two rows attain their highest order at different points.
        # Distinct components can also carry different exceptional exponents.
        table = Presentation(("E",), (({"E":1}, None, Fraction(2)),
                                      ({"E":0}, None, Fraction(3))))
        points = [((), (8,3)), ((), (1,12)), ((0,), (2,6))]
        self.assertEqual(table.point_maximum(points), Fraction(3,2))

    def test_contact_direction_uses_actual_parameter_axes(self):
        x, y, z = variables = np.symbols("x y z")
        rows = (z+z*z-x*x-y**3,)
        for axis, degree, expected in ((0,1,"yes"), (0,2,"unknown"), (1,2,"yes"), (1,3,"unknown")):
            with self.subTest(axis=axis, degree=degree):
                args = direction_request(variables, rows, (0,0,0), z, (0,1), axis, degree)
                self.assertEqual(ideal_request(*args), [expected])
        # x,z are not local parameters here. The vanishing along a putative
        # parameter axis carries no certificate.
        invalid = direction_request(variables, rows, (0,0,0), y, (0,2), 0, 2)
        self.assertEqual(ideal_request(*invalid), ["unknown"])

    def test_exact_degree_with_cancellation_and_rational_point(self):
        for name, variables, equations, point, polynomial, expected in finite_order_cases():
            with self.subTest(name=name):
                self.assertEqual(frame_order(variables, equations, point, polynomial, expected), [str(expected)])
                if expected:
                    self.assertEqual(frame_order(variables, equations, point, polynomial, expected-1), ["unknown"])

    def test_finite_precision_distinguishes_exact_order_from_unknown(self):
        x, y = variables = np.symbols("x y")
        rows = (x*(x-1),)
        for polynomial in (0, x, x*(x-1), y**11):
            with self.subTest(polynomial=polynomial):
                self.assertEqual(frame_order(variables, rows, (0,0), polynomial, 10), ["unknown"])
        self.assertEqual(frame_order(variables, rows, (1,0), x), ["0"])

    def test_invalid_point_or_singular_frame_cannot_certify(self):
        x, y = variables = np.symbols("x y")
        for eq, point in ((y-x**2, (2,0)), (x*y,(0,0)), (y**2-x**3,(0,0))):
            with self.subTest(eq=eq):
                self.assertEqual(frame_order(variables, (eq,), point, 1+x), ["unknown"])

    def test_wide_exponents_and_coefficients(self):
        x, y = variables = np.symbols("x y")
        n = 2**73 + 1
        large = Fraction(2**299 + 17, 2**167 + 3)
        f = large*(x**n-1-n*(x-1))
        self.assertEqual(frame_order(variables, (y-x+1,), (1,0), f, 2), ["2"])
        self.assertEqual(frame_order(variables, (y-x,), (0,0), x**n, 4), ["unknown"])

    def test_support_keeps_all_geometric_points_and_open_constraints(self):
        x, y = variables = np.symbols("x y")
        rows = ((x-2)**2*(x+3), y-2*x)
        self.assertEqual(set(frame_points(variables, rows)), {"points", "2 4", "-3 -6"})
        self.assertEqual(frame_points(variables, rows, (x+3,)), ["points","2 4"])
        self.assertEqual(frame_points(variables, (x+y-1, x-y-3)), ["points","2 -1"])
        self.assertEqual(frame_points(variables, (x-1, y), (y,)), ["points"])

    def test_finite_positive_order_proves_nonzero_germ_without_inverting_it(self):
        x, y = variables = np.symbols("x y")
        frame = Frame(variables, (y,), (x+x*x-y**2,))
        self.assertEqual(frame.working_points((), ((x,y),)), ("0 0",))
        with patch.object(frame, "reduced", side_effect=AssertionError("unnecessary global basis")):
            self.assertFalse(frame.vanishes(x))
        self.assertFalse(frame.known_scalar_unit(x))
        self.assertTrue(frame.inside(x, (x,y)))

    def test_local_zero_at_each_witness_still_checks_global_membership(self):
        x, y = variables = np.symbols("x y")
        frame = Frame(variables, (y,), (x*(x-1),))
        self.assertEqual(frame.working_points((), ((x,y),)), ("0 0",))
        self.assertIsNone(frame.point_order(x, "0 0"))
        self.assertFalse(frame.vanishes(x))
        self.assertTrue(frame.vanishes(x*y*(x-1)))

    def test_distinct_point_germs_can_have_different_active_functions(self):
        x, y = variables = np.symbols("x y")
        frame = Frame(variables, (y,), (x*(x-1),))
        f = x*y**4
        groups = grouped(frame, ((x,y), (x-1,y)), functions=(f,))
        supports = {tuple(q[1] for q in collection):
                    [ideal_basis(variables, (*frame.rows,*w)) for w in pieces]
                    for pieces, collection in groups}
        self.assertEqual(supports, {():[(x,y)], (f,):[(x-1,y)]})

    def test_regular_working_germ_avoids_splitting_a_disjoint_component(self):
        x, y = variables = np.symbols("x y")
        frame = Frame(variables, (y,), (x*(x-1),))
        f = x*y**4
        # Global saturation removes x=0. Near the only working point (1,0)
        # the germ has positive finite order, so the sole collection stays.
        self.assertFalse(frame.graph_regular(f))
        with patch("engine.centres.saturate", side_effect=AssertionError("unnecessary split")):
            result = grouped(frame, ((x-1,y),), functions=(f,))
        self.assertEqual(result, [(((x-1,y),), [({}, f, Fraction(1))])])

    def test_positive_dimension_and_nonrational_components_fall_back(self):
        x, y = variables = np.symbols("x y")
        self.assertEqual(frame_points(variables, (y-x,)), ["unknown"])
        self.assertEqual(frame_points(variables, ((x-1)*(x*x+1), y-x)), ["unknown"])

    def test_exceptional_coordinate_open_filters_the_complete_support(self):
        large = Fraction(2**197+5, 2**87+1)
        points = ["0 1 -2", f"{large} 0 -1", f"2 -7 {large}"]
        self.assertEqual(ideal_request("contact_open_points 3", ["0 1", *points]),
                         ["points", points[-1]])
        self.assertEqual(ideal_request("contact_open_points 3", ["", *points]),
                         ["points", *points])

    def test_working_point_retains_full_curve_centre_on_implicit_frame(self):
        x, y, z, w = variables = np.symbols("x y z w")
        # The germ is a transversal node along x=y=0, w=z^2. Selecting
        # at (0,0,1,1) has to return the entire double curve in the chart.
        ambient = (w-z**2,)
        value, _, centre = maximum(variables, ambient, 3, (x*y,), [],
                                    [(x,y,z-1,w-1)], {}, [])
        self.assertEqual(value, (2,0,1,0,float("inf")))
        self.assertEqual(ideal_basis(variables, centre), ideal_basis(variables, (*ambient,x,y)))
