"""Resolution of non-isolated germs that the quick search used to reject.

The cases exercise rational branches of the input as separate fan components, split
curve centres that miss the owned fibre, prime parts of a centre with
different numerical data, the maximal order rule, and principal
saturation by gcds.
"""
import time
import unittest

from engine import native_polynomial
from engine.affine_kernel import _ring, _request, _sparse_expression, frame_locus
from engine.local_blowups import acampo_over
from engine.monodromy import monodromy_zeta
from engine.origin_backend import resolve_origin
from engine.polynomial import format_poly, parse
from engine.resolve import resolve
from engine.topological import topological_zeta

SURFACE = ["x", "y", "z"]
THREEFOLD = ["x", "y", "z", "t"]


def quick(polynomial, names, mode="embedded"):
    return resolve(dict(polynomial=polynomial, variables=names, mode=mode, search="quick", compute="cpu",
                        monodromy=True))


def blowups(polynomial, names, rules):
    symbols = tuple(map(native_polynomial.Symbol, names))
    equation = native_polynomial.from_terms(symbols, parse(polynomial, names))
    _, _, charts, centres = acampo_over(symbols, [("strict", 1, 1, (equation,))], symbols, rules=rules,
                                        deadline=time.monotonic() + 120, count_strata=False)
    return charts, centres


class RationalBranchesOnNewtonFans(unittest.TestCase):
    def test_coordinate_plane_through_an_a2_surface(self):
        # y*(x^2+x*z+y^3): the plane y=0 meets the A2 surface in two lines.
        # Their union is singular where the fan's strict transforms cross normally.
        record = quick("x^2*y+y^4+z*x*y", SURFACE)
        self.assertEqual((record["status"], record["engine"]), ("resolved", "newton-polyhedron"))
        self.assertEqual(record["monodromy_check"]["status"], "verified")
        zeta, topological = record["monodromy_zeta"]["expression"], record["topological_zeta"]["expression"]
        self.assertEqual((zeta, topological), ("1/(1-T^4)", "(s^2 + 2*s + 4)/(4*(s + 1)^3)"))
        # The smooth-centre blowups give the same zeta functions.
        p = parse("x^2*y+y^4+z*x*y", SURFACE)
        other = resolve_origin(p, SURFACE, "embedded")
        other.update(variables=SURFACE, polynomial=format_poly(p, SURFACE), mode="embedded", origin=[0, 0, 0])
        own = monodromy_zeta(other, polynomial=p)
        self.assertEqual((own["expression"], topological_zeta(other, polynomial=p, monodromy=own)["expression"]),
                         (zeta, topological))

    def test_four_hyperplanes_over_the_gaussian_field(self):
        # x*y*(z+i*t)*(z-i*t): normal crossings over Q(i), with chi(F) = 0.
        record = quick("x*y*(z^2+t^2)", THREEFOLD)
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["monodromy_zeta"]["expression"], "1")
        self.assertEqual(record["topological_zeta"]["expression"], "(s + 1)^(-4)")


class SmoothCentreRules(unittest.TestCase):
    def test_declined_boundary_split_uses_singular_locus(self):
        from unittest.mock import patch
        from engine.local_blowups import Chart, GiveUp, _Order
        x, y, z = variables = tuple(map(native_polynomial.Symbol, SURFACE))
        chart = Chart(variables, (), (), variables, (), (), ())
        with patch("engine.local_blowups._apart", side_effect=GiveUp("No principal generator on the chart.")):
            centres, used_boundary = _Order([])._smooth_centres([chart], [(x*y, z)])
        from engine.blowup_ideal import ideal_basis
        self.assertEqual(ideal_basis(variables, centres[0]), ideal_basis(variables, variables))
        self.assertFalse(used_boundary)

    def test_boundary_tangency_in_a_maximal_order_centre(self):
        # The exceptional curves distinguish the branches of the order-two
        # locus. One later smooth centre is tangent to the old boundary.
        # Resolving its tangency first preserves normal crossings there.
        from engine.resolution_complex import dual_complex
        record = quick("z^2-x*(x^2-y^3)^4", SURFACE)
        self.assertEqual(record["status"], "resolved")
        self.assertIn("maximal order with boundary", {s["rule"] for s in record["steps"]})
        self.assertEqual(dual_complex(record), record["dual_complex"])

    def test_boundary_traces_on_components_of_a_smooth_centre(self):
        from engine.blowup_ideal import ideal_basis
        from engine.centre_weights import centre_rows
        from engine.kernel import exprs
        from engine.kernel_json import kernel_json
        x, y, z = variables = tuple(map(native_polynomial.Symbol, SURFACE))

        def defect(centre, divisors, dimension):
            rows = kernel_json("atlas_graph_wide", "centre_boundary_defect", width=3, dimension=str(dimension),
                               ideal=centre_rows(variables, centre),
                               divisors=[centre_rows(variables, (d,)) for d in divisors])
            return None if rows is None else ideal_basis(variables, exprs(variables, rows))

        # The divisor y=0 contains the sheet z=0. On the sheet z=1 it
        # either crosses transversely or is tangent at x=0.
        sheets = z*(z-1)
        self.assertIsNone(defect((sheets, y-z*x), (y,), 1))
        self.assertEqual(defect((sheets, y-z*x**2), (y,), 1), ideal_basis(variables, (z-1, x, y)))
        # All pairs of traces are transverse, but three lines meet in a plane.
        self.assertEqual(defect((z-x-y,), (x, y, z), 2), ideal_basis(variables, variables))
        self.assertIsNone(defect((y, z), (y-(2**100+1)*z*x**(2**32+1),), 1))

    def test_split_curve_side_off_the_owned_fibre_is_not_chosen(self):
        # The complete rule split a curve centre and blew up the side inside
        # an earlier divisor, which met no owned point. Nothing changed, and
        # the same centre returned until the stage limit of 150.
        _, centres = blowups("z^3-x*(x^2-y^3)^2", SURFACE, ("complete",))
        self.assertLess(len(centres), 100)

    def test_prime_parts_with_different_numerical_data(self):
        # A centre whose components lie on different old divisors gives the
        # new divisor different data on them. Its prime parts with the
        # largest data are blown up first.
        _, centres = blowups("x^2+(y^2+z^2+t^2)^3", THREEFOLD, ("complete",))
        self.assertTrue(centres)

    def test_maximal_order_rule_blows_up_the_line_of_singular_points(self):
        # x*y^4+z^2+t^2 is singular along the x-axis, with transverse type A3.
        # The order-two locus near the fibre is the whole axis: two blowups.
        _, centres = blowups("x*y^4+z^2+t^2", THREEFOLD, ("order",))
        self.assertEqual([(c["kind"], c["rule"]) for c in centres], [("curve", "maximal order")] * 2)
        # The double suspension of x*y^4 has a Milnor fibre homotopic to S^3
        # with trivial monodromy.
        record = quick("x*y^4+z^2+t^2", THREEFOLD)
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["monodromy_zeta"]["expression"], "1")
        self.assertEqual(record["monodromy_zeta"]["milnor_fiber_euler_characteristic"], 0)

    def test_maximal_order_rule_on_a_squared_node_curve(self):
        # x^4+(y^2-z^2)^3: the complete rule finds no hypersurface of maximal contact.
        _, centres = blowups("x^4+(y^2-z^2)^3", SURFACE, ("order", "complete"))
        self.assertTrue(centres)
        self.assertIn("maximal order", {c["rule"] for c in centres})


class PrincipalSaturation(unittest.TestCase):
    def test_gcd_saturation_agrees_with_the_affine_kernel(self):
        x, y, z = variables = tuple(map(native_polynomial.Symbol, SURFACE))
        cases = [((x**2 + y*z)**2 * (y - 1) * (x + z**3), (y - 1) * (x**2 + y*z)),
                 ((x**2 + y*z)**2 * (y - 1) * (x + z**3), y - 1),
                 ((x*y - 1)**3 * (z**2 - x), (x*y - 1) * (y + 2)),
                 (x**3 + y**2*z, y*z + 1)]
        ring = _ring(variables)
        for equation, unit in cases:
            with self.subTest(equation=str(equation)):
                fast = frame_locus(variables, (equation,), (unit,), closure=True)
                lines = ["frame_basis", ring.header, "1", ring.text(equation), ring.text(unit)]
                general = tuple(_sparse_expression(variables, row) for row in _request(lines))
                self.assertEqual(fast, general)


if __name__ == "__main__":
    unittest.main()
