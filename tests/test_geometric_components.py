import unittest

import sympy as sp

from engine import groebner as gb
from engine.blowup_ideal import ideal_basis
from engine.geometric_components import _absolute_factor, _shape, geometric_components


class ComponentTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.z = sp.symbols("x y z")
        self.v = (self.x,self.y,self.z)

    def test_graph_unit_and_ambient(self):
        x,y,z = self.v
        self.assertEqual(geometric_components(self.v,[1],-1), [])
        self.assertEqual(geometric_components(self.v,[],3), [()])
        self.assertEqual(geometric_components(self.v,[z-y**2,y-x**2],1),
                         [ideal_basis(self.v,[z-y**2,y-x**2])])

    def test_primes_over_extension(self):
        x,y,z = self.v
        self.assertIsNone(geometric_components(self.v,[x*y-1,z**2+1],1))
        field = sp.QQ.algebraic_field(sp.I)
        pieces = geometric_components(self.v,[x*y-1,z**2+1],1,domain=field)
        self.assertEqual(set(pieces),{(x*y-1,z-sp.I),(x*y-1,z+sp.I)})

    @unittest.skipUnless(gb._MODE == "native", "Tests the native worker's marked polynomial representation.")
    def test_projected_shapes_and_graph_units_over_the_marked_field(self):
        from engine import native_polynomial as np
        from engine.candidate_field import indexed_field
        from engine.field_factors import factors
        from engine.field_polynomial import polynomial
        from engine.geometric_components import _factor_equations

        u, z, t = np.symbols("v0:3")
        field, alpha = indexed_field((1,0,-2), 1)
        # z=(1-u²)/alpha requires inversion in K, rather than Q[alpha].
        graph = ideal_basis((u,z), (u**2+alpha*z-1,), domain=field)
        self.assertEqual(geometric_components((u,z),graph,1,domain=field), [graph])
        # On D(u), z=alpha*t/u over the smooth conic t²-alpha*u²=1.
        # Its closure cannot lose the coefficient parameter in either row.
        variables = (u,z,t)
        ideal = ideal_basis(variables, (u*z-alpha*t, t**2-alpha*u**2-1), domain=field)
        self.assertEqual(_shape(variables, ideal, 1, field, 1, variables, variables), [ideal])
        self.assertEqual([(p.as_expr(), m) for p,m in factors(polynomial((u-alpha)**2*(u+alpha)**3,(u,),field))],
                         [(u-alpha,2),(u+alpha,3)])
        for offset, expected in ((0, False), (1, True)):
            section = polynomial(u**2+alpha*z**2+offset, (u,z), field)
            self.assertIs(_absolute_factor(section,1), expected)
            self.assertIs(_factor_equations(section,1), expected)

    def test_translated_norm_preserves_multiplicities_and_skips_a_bad_shift(self):
        from engine.field_factors import factors
        x,y,_ = self.v
        field = sp.QQ.algebraic_field(sp.I)
        # At a=0 each rational factor is common to both conjugates. At
        # a=1 the factor x-y still has the same translate on both sheets.
        f = (x-y)**3*(x+sp.I*y)**2*(y+1)
        poly = sp.Poly(f,x,y,domain=field)
        result = {(p.as_expr(), m) for p,m in factors(poly)}
        self.assertEqual(result, {(x-y,3), (x+sp.I*y,2), (y+1,1)})

    def test_coefficient_equations_over_an_algebraic_field(self):
        from engine.geometric_components import _factor_equations
        x,y,_ = self.v
        field = sp.QQ.algebraic_field(sp.sqrt(2))
        for f, expected in ((x**2+sp.sqrt(2)*y**2, False), (x**2+sp.sqrt(2)*y**2+1, True)):
            self.assertIs(_factor_equations(sp.Poly(f,x,y,domain=field), 1), expected)

    def test_reduced_support_projection_proves_meeting_primes_and_global_coverage(self):
        from engine.chart_expressions import runtime_expressions
        from engine.geometric_components import _intersection
        from engine.reference import QQ
        x,y,z = coords = runtime_expressions().symbols("x y z")
        curves = ((y-x**2,z-x), (y-x**3,z-2*x))
        ideal = _intersection(coords, *curves, QQ)
        expected = {ideal_basis(coords, curve) for curve in curves}
        self.assertIsNone(_shape(coords, ideal, 1, QQ, 1, coords, coords))
        self.assertEqual(set(_shape(coords, ideal, 1, QQ, 1, coords, coords, reduced_support=True)), expected)
        # An entire component is vertical for this projection. Even the
        # reduced-support test must refuse this incomplete generic shape.
        vertical = (x-1,z)
        larger = _intersection(coords, ideal, vertical, QQ)
        self.assertIsNone(_shape(coords, larger, 1, QQ, 1, coords, coords, reduced_support=True))

    def test_special_fiber_lost_by_generic_projection(self):
        x,y,z = self.v
        line_a = (y,z)
        line_b = (x-1,z-1)
        ideal = ideal_basis(self.v, [a*b for a in line_a for b in line_b])
        # The second line maps to the single first-coordinate value 1.
        self.assertIsNone(_shape(self.v,ideal,1,sp.QQ,1,sp.eye(3)))
        self.assertEqual(set(geometric_components(self.v,ideal,1)), {line_a,line_b})

    def test_shape_over_a_number_field_contracts_the_graph(self):
        u,z,t = self.v
        field = sp.QQ.algebraic_field(sp.sqrt(2))
        # z=t/u on D(u); t^2-2*u^2-1 is an irreducible affine conic
        # and forces u to be invertible where u*z=t.
        ideal = ideal_basis(self.v, (u*z-t, t**2-2*u**2-1), domain=field)
        self.assertEqual(_shape(self.v, ideal, 1, field, 1, sp.eye(3)), [ideal])

    def test_generic_primitive_coordinate_for_four_points(self):
        x,y,_ = self.v
        components = geometric_components((x,y), [x*x-1,y*y-1],0)
        self.assertEqual(set(components), {(x-a,y-b) for a in (-1,1) for b in (-1,1)})

    def test_projection_schedule_inverts_words_in_reverse_order(self):
        from engine.geometric_components import _projections
        x,y,z = self.v
        # Base twelve encodes E_01(+1) followed by E_12(+1) at code 85.
        # The product's inverse has the cross term +z in its first image.
        maps = list(_projections(self.v, 85))
        self.assertEqual(maps[1], ((z,x,y), (y,z,x)))
        self.assertEqual(maps[3], ((x,y,2*x+4*y+z), (x,y,-2*x-4*y+z)))
        self.assertEqual(maps[-1], ((x+y,y+z,z), (x-y+z,y-z,z)))

    def test_absolute_factors_and_saturation(self):
        x,y,z = self.v
        # A smooth affine conic and an irreducible cusp exercise distinct
        # absolute-irreducibility certificates. Smoothness is a caller check.
        for f in [x*x+y*y-1,x**3-y*y]:
            self.assertIsNotNone(geometric_components((x,y),[f],1))
        self.assertIsNone(geometric_components((x,y),[x*x+y*y],1))
        # Returning the coordinate axes as components of a smooth stratum
        # would be incorrect; their intersection must be detected.
        self.assertIsNone(geometric_components((x,y),[x*y],1))
        self.assertEqual(geometric_components(self.v,[x*y-1,x*z-y],1),
                         [ideal_basis(self.v,[x*y-1,x*z-y])])


    def test_absolute_plane_factors_after_nonlinear_change(self):
        x, y, _ = self.v
        # The norm of y + i*x + i*y**2 splits into two conjugate factors.
        split = (x+y**2)**2 + y**2
        cusp = (x-y**2)**5 + y**7
        self.assertFalse(_absolute_factor(sp.Poly(split, x, y, domain=sp.QQ), 1))
        result = _absolute_factor(sp.Poly(cusp, x, y, domain=sp.QQ), 1)
        if gb._MODE == "sympy" or gb._kernel() is None:
            # The bounded coefficient-equation reference may exhaust its effort.
            self.assertIsNot(result, False)
        else:
            self.assertTrue(result)

    def test_rational_projective_criteria(self):
        x, y, z = self.v
        # A rank-two form using all three coordinates; homogenizing the
        # second form gives rank four. The affine cubic has smooth closure.
        for f, expected in (((x+y)**2 + (y+z)**2, False),
                            (x*x + y*y + z*z - 1, True),
                            (x**3 + y**3 + z**3 - 1, True)):
            self.assertEqual(_absolute_factor(sp.Poly(f, x, y, z, domain=sp.QQ), 1), expected)

    def test_primitive_vertices_of_a_sparse_section(self):
        x,y,z = self.v
        # This affine restriction arises on a four-dimensional chart. Its
        # total degree is 17, well beyond the bounded factor equations.
        f = x**2*y**9*z**6 + x**2*y**5*z**2 + x*y + 1
        self.assertEqual(geometric_components(self.v, (f,), 2), [(f,)])
        # xy has odd exponents but lies in the interior of an edge. Only
        # vertices can certify primality: this norm has two absolute factors.
        norm = (x+y)**2 + z**2
        self.assertIsNone(geometric_components(self.v, (norm,), 2))

    def test_sparse_rational_factor_metadata(self):
        from engine.affine_kernel import _sparse_expression, rational_factor_rows
        x, y, z = self.v
        norm = (x+y)**2 + z**2
        irreducible = x**4 + y**9 - z
        polynomial = -sp.Rational(5, 7)*norm**2*(x**4-y*z+1)*irreducible
        scalar, rows = rational_factor_rows(self.v, polynomial)
        observed = [(_sparse_expression(self.v, row), m, certificate) for row, m, certificate in rows]
        self.assertEqual(scalar, -sp.Rational(5,7))
        self.assertEqual(observed, [(sp.expand(norm), 2, False), (x**4-y*z+1, 1, True),
                                    (irreducible, 1, True)])
        self.assertEqual(rational_factor_rows(self.v, sp.Integer(0)), (0, ()))

    def test_sparse_factor_order_against_nested_coefficients(self):
        from engine.affine_kernel import _sparse_expression, rational_factor_rows
        x,y,z = self.v
        # The historical ordering compares lower-variable coefficient sequences
        # from their leading terms, with zero polynomials at skipped powers.
        factors = (x+y**7+z**2+2, x+y**7+z**5-1, x-y**5*z+2, x*y**6+z-2, x*y**4+z)
        poly = sp.prod(factors)
        _, rows = rational_factor_rows(self.v, poly)
        expected = [p.as_expr() for p, _ in sp.Poly(poly, *self.v, domain=sp.QQ).factor_list()[1]]
        self.assertEqual([_sparse_expression(self.v,r) for r, _, _ in rows], expected)
