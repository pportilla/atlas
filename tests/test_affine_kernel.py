import unittest
from unittest.mock import patch

import sympy as sp

from engine import affine_kernel
from engine.affine_kernel import (canonical_laurent_terms, chart_data, decode, frame_derivative,
                                 frame_linear, frame_locus, frame_quotient, frame_restrict,
                                 inverse_map, lattice_characters, lattice_dimension, pullbacks,
                                 rational_factors, sparse_terms, transition_pole)


class AffineMaps(unittest.TestCase):
    def test_fan_chart_equation_text_matches_the_exact_sparse_row(self):
        from engine.native_library import affine_request
        from engine.polynomial import format_poly, sparse_row

        charts = ["1 0", "0 1", "1 2", "1 0", "1 1", "1 2"]
        equations = [
            ("-y^6+x^3*y^3/3-x*y^5+x+y-5/3",
             ["1/3*u^3*v^3-u*v^5-v^6+u+v-5/3",
              "1/3*u^3*v^6-u*v^6-v^6+u*v+v-5/3"]),
            ("-3/7", ["-3/7", "-3/7"]),
        ]
        # The exponents and the rational coefficient exceed the Python
        # decimal limit. A graded degree tie still uses chart-variable order.
        wide = "1" + "0" * 4500
        previous = "9" * 4500
        equations.append((f"-y^{wide}+{wide}/3*x^{wide}+x",
                          [f"{wide}/3*u^{wide}-v^{wide}+u",
                           f"{wide}/3*u^{wide}*v^{previous}-v^{previous}+u"]))
        for equation, expected_texts in equations:
            with self.subTest(equation=equation[:28]):
                rows = affine_request(["fan_charts", "x y", "u v", equation, *charts])
                self.assertEqual(len(rows), 8)
                for i, expected in enumerate(expected_texts):
                    header, text, sparse, _ = rows[i*4:i*4+4]
                    terms = sparse_row(sparse)
                    self.assertEqual(text, format_poly(terms, ("u", "v")))
                    self.assertEqual(header.split()[0], str(int(any(any(e) for e in terms))))
                    self.assertEqual(text, expected)

    def test_fan_projection_preserves_wide_exponents_and_signed_coefficients(self):
        from engine.native_library import affine_request

        # The first two ray columns tie in total degree. Both the degree and
        # the coefficient exceed machine integers and the Python decimal limit.
        power = "1" + "0" * 4500
        coefficient = "8" * 4501
        header = ["x y z", "zeta alpha mu", "z"]
        cone = ["1 0 0", "0 1 0", f"{power} {power} 1", "5 6 4"]
        frame = [f"0 -{coefficient} 1", "-1 1 0", f"1 {coefficient} 0"]
        rows = affine_request(["fan_charts_with_projection", *header, *frame, *cone])
        self.assertEqual(rows[:4], affine_request(["fan_charts", *header, *cone]))
        first, second = f"zeta*mu^{power}", f"alpha*mu^{power}"
        self.assertEqual(rows[4:], [
            f"-{coefficient}*{second}+mu",
            f"-{first}+{second}",
            f"{first}+{coefficient}*{second}",
        ])

    def test_native_factor_multiplicities_are_exact_decimal_integers(self):
        from engine.native_library import affine_request

        # Distinct wide powers prevent a truncated order from passing because
        # the total degree or another factor's multiplicity happens to agree.
        x_power = "1" + "0" * 4500
        y_power = str(2**63 + 17)
        for operation in ("factor", "geometric_factors"):
            with self.subTest(operation=operation):
                rows = affine_request([operation, "x y", f"x^{x_power}*y^{y_power}"])
                parts = {row: info.split()[0] for info, row in zip(rows[1::2], rows[2::2])}
                self.assertEqual(parts, {"1 1 0": x_power, "1 0 1": y_power})

        # Direct Python callers keep their interpreter's input digit policy.
        # These integers came from the verified native result.
        from engine import identity_atlas, native_library, weighted_record
        from engine import native_polynomial as np
        from engine.polynomial import format_poly
        n, m = 10**4500, 2**63+17
        names = ("x", "y")
        variables = np.symbols("x y")
        terms = {(n, m): 1}
        with patch.object(affine_kernel.gb, "_MODE", "native"), patch.object(affine_kernel.gb, "CHECK", False), \
                patch.object(native_library, "_MODE", "native"), patch.object(native_library, "CHECK", False):
            _, rows = affine_kernel.rational_factor_rows(variables, np.from_terms(variables, terms))
            self.assertEqual({row: k for row, k, _ in rows}, {"1 1 0": n, "1 0 1": m})
            factors = identity_atlas.rational_factors(terms, names)
            self.assertEqual({format_poly(f, names): k for f, k in factors}, {"x": n, "y": m})
            factors, unit = weighted_record._factor(format_poly(terms, names), names)
            self.assertEqual((dict(factors), unit), ({"x": n, "y": m}, "1"))

    def test_native_input_rows_reuse_encodings_in_chart_order(self):
        from engine import native_polynomial as np
        a, b = np.Dummy("v"), np.Dummy("v")
        variables = [a, b]
        self.assertEqual(np.row(variables, a), "1 1 0")
        with patch.object(np, "_chart_positions", side_effect=AssertionError("row encoded twice")):
            self.assertEqual(np.row(tuple(variables), a), "1 1 0")
        variables.reverse()
        self.assertEqual(np.row(variables, a), "1 0 1")
        self.assertEqual(np.row(variables, b), "1 1 0")
        self.assertEqual(np.row(variables, 1), "1 0 0")
        with self.assertRaises(TypeError):
            np.row(variables, 1.0)

    def test_coordinate_rows_reuse_native_chart_variables(self):
        from engine import native_polynomial as np
        a, b = np.Dummy("row"), np.Dummy("row")
        with patch.object(affine_kernel, "_polynomial", side_effect=AssertionError):
            self.assertIs(affine_kernel._sparse_expression((b, a), "1 0 1"), a)
            self.assertIs(affine_kernel._sparse_expression((a, b), "1 0 1"), b)
        with self.assertRaises(ValueError):
            affine_kernel._sparse_expression((a, a), "1 0 1")

    def test_common_characters_with_arbitrary_signed_coordinates(self):
        # Nonsymmetric, orientation reversing, and too large for a machine integer.
        shear = 2**100
        rays = ((-1, shear), (0, 1))
        self.assertEqual(lattice_characters(rays, ((-2*shear-5, -2), (3*shear+1, 3))),
                         ((5, -2), (-1, 3)))
        self.assertEqual(lattice_characters(rays, ()), ())
        for bad in (((2,0), (0,1)), ((1,0), (-2,0))):
            with self.assertRaisesRegex(ArithmeticError, "unimodular"):
                lattice_characters(bad, ())
        with self.assertRaises(ValueError):
            lattice_characters(rays, ((0,1,2),))
        with self.assertRaises(TypeError):
            lattice_characters(rays, ((1.5,0),))

    def test_laurent_translation_and_affine_rank(self):
        rays = ((1,1), (0,1))
        terms = {(0,0): 3, (4,2): sp.Rational(-2,3), (-1,1): 7, (9,9): 0}
        moved = {(x-3,y+11):c for (x,y),c in terms.items()}
        expected = {(2,0): 3, (4,2): sp.Rational(-2,3), (0,1): 7}
        self.assertEqual(canonical_laurent_terms(rays, terms), expected)
        self.assertEqual(canonical_laurent_terms(rays, moved), expected)
        self.assertEqual(canonical_laurent_terms(rays, {}), {})
        self.assertEqual(lattice_dimension(expected), 2)
        self.assertEqual(lattice_dimension(((0,1), (2,3), (-5,-4))), 1)
        self.assertEqual(lattice_dimension(((-111,321),)), 0)
        with self.assertRaises(ValueError):
            lattice_dimension(())

    def test_rational_coefficients_and_coordinate_orders(self):
        x, y, u, v = sp.symbols("x y u v")
        result = chart_data((u, v), (u, (u*v + 1)/2), (u,), ((2*y - 1)**2 - x**3)/3, (x, y))
        self.assertEqual(result, (((2, 2),), (v**2 - u).expand()/3))

    def test_reduction_before_orders_on_a_finite_cover(self):
        x, y, u, v, a = sp.symbols("x y u v a")
        # The coefficient of u**2 vanishes over the coefficient field.
        result = chart_data((u, v, a), (u, u*(a + u*v), a), (u,), x**2 + y**2,
                            (x, y, a), modulus=a**2 + 1)
        self.assertEqual(result, (((3, 3),), 2*a*v + u*v**2))

    def test_composition_between_spaces_of_different_dimensions(self):
        x, y, a = sp.symbols("x y a")
        self.assertEqual(pullbacks((x, y, a), (a*x + y, a*y), (x, y),
                                   (x*y, 0), modulus=a**2 + 1), (a*y**2 - x*y, 0))

    def test_transition_cancels_common_factors(self):
        x, y, u, v = sp.symbols("x y u v")
        # Overlap with a translated chart of a blowup. The common v cancels.
        inverse = inverse_map((u, v), (u, u*(v + sp.Rational(1, 2))), (x, y))
        self.assertEqual(inverse, ((x, 1), (y - x/2, x)))
        self.assertEqual(transition_pole((u, v), (u*v, v), (x, y), inverse), u)
        self.assertEqual(transition_pole((u, v), (u, u*(v - 1)), (x, y), inverse), 1)

    def test_nonlinear_elimination_and_same_named_rings(self):
        x, y = sp.symbols("x y")
        mapping = (y + x**2, x*y + x**3 + sp.Rational(3, 5))
        inv = inverse_map((x, y), mapping, (x, y))
        self.assertEqual(transition_pole((x, y), mapping, (x, y), inv), 1)
        # Both coordinates have a pole at x=0, with different orders.
        self.assertEqual(transition_pole((x, y), (x, y), (x, y), inv), x)

    def test_distinct_coordinates_with_identical_display_names(self):
        a, b, x, y = (sp.Dummy("v") for _ in range(4))
        variables, base = (a, b), (x, y)
        self.assertEqual(pullbacks(variables, (b, a), base, (x-y,)), (b-a,))
        self.assertEqual(chart_data(variables, (a, a*b), (a,), x**2+y, base),
                         (((1, 2),), a+b))
        inverse = inverse_map(variables, (a, a*b), base)
        self.assertEqual(inverse, ((x, 1), (y, x)))
        self.assertEqual(transition_pole(variables, (a*b, b), base, inverse), a)
        self.assertEqual(frame_linear(variables, a*b+2*b, b), (a+2, 0))
        self.assertEqual(frame_restrict(variables, a+b, a, sp.S.One, b), 0)
        self.assertEqual(frame_quotient(variables, a**2*b, a), a*b)
        self.assertTrue(frame_locus(variables, (a, b-1), (a+b-1,)))
        self.assertEqual(frame_derivative(variables, a**2*b, {b: 2}), 2*a**2)
        p = sp.Poly((a+b)*(a-b)**2, *variables, domain=sp.QQ)
        self.assertEqual(rational_factors(p), p.factor_list())

    def test_nonidentifier_coordinate_and_generated_name_collision(self):
        a, b = sp.Symbol("x*y"), sp.Symbol("atlasVar0")
        self.assertEqual(frame_derivative((a,b), a**2*b, {a:b}), 2*a*b**2)

    def test_nonbirational_map(self):
        x, u = sp.symbols("x u")
        with self.assertRaisesRegex(ArithmeticError, "rational inverse"):
            inverse_map((u,), (u**2,), (x,))

    def test_unbounded_sparse_degree_and_dimension(self):
        variables = sp.symbols("v:34")
        x, y = variables[0], variables[-1]
        degree = 10**22
        self.assertEqual(sparse_terms((x, y), (x**degree + y**degree)**2),
                         (((2*degree, 0), 1), ((degree, degree), 2), ((0, 2*degree), 1)))
        self.assertEqual(decode((x,), f"{x}^{degree}"), x**degree)
        self.assertEqual(pullbacks((x,), (x,), (y,), (y**degree + 1,)), (x**degree + 1,))
        self.assertEqual(chart_data(variables, variables, (x,), x**degree * (1+y), variables),
                         (((degree, 1),), 1+y))
        self.assertEqual(inverse_map(variables, variables, variables), tuple((v, 1) for v in variables))

    def test_wide_dimension_and_chart_contraction(self):
        from engine.local_blowups import _dimension
        variables = sp.symbols("v:70")
        a, b, c = variables[:3]
        self.assertEqual(_dimension(variables, (a*b, b*c, c*a)), 68)
        self.assertEqual(_dimension(variables, ()), 70)
        self.assertEqual(_dimension(variables, (sp.S.One,)), -1)
        self.assertEqual(frame_locus(variables, (a*b,), (b,), closure=True), (a,))
        self.assertTrue(frame_locus(variables, (a*b,), (a, b)))

    def test_dimension_adds_heights_of_disjoint_leading_supports(self):
        from engine.ideal_operations import dimension
        from engine.native_polynomial import symbols

        variables = symbols("x0:31")
        generators = []
        for i in range(10):
            # Interleave the coordinates so components are not adjacent in the ring.
            a, b, c = variables[i], variables[10+i], variables[20+i]
            generators.extend((a*b, b*c, c*a))
        self.assertEqual(dimension(variables, generators), 11)  # ten triangles and a free line
        self.assertEqual(dimension(variables, ()), 31)
        self.assertEqual(dimension(variables, (1,)), -1)

    def test_dimension_groups_transitively_overlapping_supports(self):
        from engine.ideal_operations import dimension
        from engine.native_polynomial import symbols

        a, b, c, d, e, f, g = variables = symbols("x0:7")
        # The bridge b*c joins the disjoint edges a*b and c*d. Two
        # coordinates cover the path and one covers the separate edge.
        self.assertEqual(dimension(variables, (a*b, c*d, e*f, b*c)), 4)
        # One coordinate now covers the bridge and both outer edges.
        self.assertEqual(dimension(variables, (a*b, b*c, b*d, e*f)), 5)

    def test_inverse_wider_than_packed_graph(self):
        local, base = sp.symbols("u:17"), sp.symbols("x:17")
        result = inverse_map(local, (local[0]*local[1], *local[1:]), base)
        self.assertEqual(result, ((base[0], base[1]), *((x, 1) for x in base[1:])))

    def test_large_smooth_germ_and_saved_chart(self):
        from engine.atlas import from_json, to_json
        from engine.resolve import resolve
        variables = [f"x{i}" for i in range(34)]
        result = resolve({"polynomial": "x0 + x33", "variables": variables})
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(to_json(from_json(result["atlas"])), result["atlas"])

    def test_unavailable_binary_and_explicit_reference(self):
        x, u = sp.symbols("x u")
        from engine.polynomial import InputError
        with patch("engine.native_library._kernel", return_value=None):
            affine_kernel._library.cache_clear()
            try:
                with patch.object(affine_kernel.gb, "_MODE", "native"):
                    with self.assertRaises(affine_kernel._AffineUnavailable):
                        decode((x,), "1/3+x")
                with patch.object(affine_kernel.gb, "_MODE", "sympy"):
                    self.assertEqual(decode((x,), "1/3+x"), x+sp.Rational(1, 3))
                    self.assertEqual(frame_linear((x,), x+4, x), (1, 4))
                    p = sp.Poly((x+1)**2, x, domain=sp.QQ)
                    self.assertEqual(rational_factors(p), p.factor_list())
                    with self.assertRaises(InputError):
                        decode((x,), "1/x")
                    with self.assertRaisesRegex(ArithmeticError, "rational inverse"):
                        inverse_map((u,), (u**2,), (x,))
            finally:
                affine_kernel._library.cache_clear()

    def test_native_errors_and_comparison_failures_propagate(self):
        x = sp.Symbol("x")
        with patch.object(affine_kernel.gb, "_MODE", "native"):
            with patch.object(affine_kernel, "_request", side_effect=ArithmeticError("invalid operation")):
                with self.assertRaisesRegex(ArithmeticError, "invalid operation"):
                    frame_linear((x,), x+1, x)
            with patch.object(affine_kernel, "_request", return_value=["none"]), patch.object(affine_kernel.gb, "CHECK", True):
                with self.assertRaises(AssertionError):
                    frame_linear((x,), x+1, x)


if __name__ == "__main__":
    unittest.main()
