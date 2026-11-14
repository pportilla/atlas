from contextlib import ExitStack
import unittest
from unittest.mock import patch

import sympy as sp

from engine import groebner as gb
from engine.field_polynomial import lift, lower, modulus


class NumberFieldPolynomials(unittest.TestCase):
    def test_missing_native_field_factorization_on_a_symbolic_input(self):
        from engine import field_factors
        if gb._MODE != "native":
            self.skipTest("Checks the default native dispatch.")
        x = sp.Symbol("external")
        p = sp.Poly(x**2-7, x, domain=sp.QQ.algebraic_field(sp.sqrt(7)))
        with patch.object(gb, "_kernel", return_value=None):
            with self.assertRaisesRegex(ArithmeticError, "make groebner"):
                field_factors.factors(p)

    def test_third_party_scalar_requires_explicit_symbolic_coercion(self):
        from engine.field_coefficient import root_position, UnsupportedExpression
        from engine.native_library import Unavailable
        t,theta = sp.symbols("t theta")
        field = sp.QQ.algebraic_field(sp.sqrt(5))
        poly = sp.PurePoly(t**2-t-1,t,domain=sp.QQ)
        # An algebraic NumberSymbol which the external algebra package
        # understands, but whose private spelling is outside the QQbar reader.
        value = sp.GoldenRatio
        if gb._MODE != "native":
            self.assertEqual(lift(value,(),field,theta),(theta+1)/2)
            self.assertEqual(root_position(poly,value),1)
            return
        root = sp.CRootOf(poly.as_expr(),1)
        over_field = sp.Poly(t+(1+sp.sqrt(5))/2,t,domain=field)
        with patch.object(field,"convert",side_effect=AssertionError("symbolic coercion")), \
             patch.object(field,"to_sympy",side_effect=AssertionError("symbolic expansion")), \
             patch.object(sp.PurePoly,"same_root",side_effect=AssertionError("symbolic isolation")):
            with self.assertRaisesRegex(UnsupportedExpression,"CRootOf.*ATLAS_GROEBNER=sympy"):
                lift(value,(),field,theta)
            with self.assertRaisesRegex(UnsupportedExpression,"CRootOf"):
                root_position(poly,value)
            self.assertEqual(lift(root,(),field,theta),theta/2+sp.S.Half)
            self.assertEqual(lift(over_field,(t,),field,theta),t+theta/2+sp.S.Half)
            with patch("engine.field_coefficient.field_request",return_value=None):
                with self.assertRaisesRegex(Unavailable,"make groebner"):
                    root_position(sp.PurePoly(t**2-19,t),sp.sqrt(19))

    @unittest.skipUnless(gb._MODE == "native" and hasattr(gb._kernel(),"atlas_field_integer_factors"),
                         "the native radical display is not in use")
    def test_additive_generator_display_without_symbolic_expansion(self):
        from engine.field_polynomial import _scalar_expression
        from sympy.ntheory import factor_
        from sympy.polys.numberfields import minpoly
        t,theta = sp.symbols("t theta")
        generators = (sp.sqrt(2)+sp.sqrt(3), sp.sqrt(2)+sp.sqrt(1+sp.sqrt(2)),
                      1/sp.sqrt(1+sp.sqrt(2)), (1+sp.sqrt(3))**sp.Rational(2,3),
                      2**sp.Rational(1,3)+3**sp.Rational(1,3), 12**sp.Rational(1,5),
                      sp.sqrt(2)+sp.CRootOf(t**3+t+1,0), sp.cos(sp.pi/7)+sp.sqrt(2),
                      sp.exp(sp.I*sp.pi/5)+sp.exp(sp.I*sp.pi/3))
        for generator in generators:
            with self.subTest(generator=generator):
                field = sp.QQ.algebraic_field(generator)
                degree = field.mod.degree()
                values = [field([1]+[0]*j) for j in range(1,degree)]
                values.append(field(list(range(degree,0,-1))))
                expected = [field.to_sympy(a) for a in values]
                with ExitStack() as guard:
                    for obj,name in ((field,"to_sympy"),(sp.Expr,"expand"),(sp,"expand"),
                                     (sp.AlgebraicNumber,"as_expr"),(factor_,"factorint"),(sp,"minpoly")):
                        guard.enter_context(patch.object(obj,name,side_effect=AssertionError(name)))
                    guard.enter_context(patch(minpoly.__module__+".minpoly",side_effect=AssertionError("minpoly")))
                    self.assertEqual([_scalar_expression(a,field) for a in values],expected)
                    self.assertEqual(lift(generator, (), field, theta),theta)
                    self.assertEqual(lower(theta**(degree-1),(t,),field,theta),expected[-2])

    def test_indexed_field_output_uses_only_the_powers_present(self):
        from engine.field_coefficient import indexed_field
        from engine.field_polynomial import _line_expression
        x,t = sp.symbols("x t")
        # Eisenstein, and long enough that building all powers for two rational
        # coefficients would do substantially more work than producing the answer.
        for coeffs in ((1, *(0,)*38, 2), (1,0,0,0,512,1024)):
            field, root = indexed_field(coeffs, t, 0)
            with patch.object(field,"to_sympy",side_effect=AssertionError("unexpected field power table")):
                self.assertEqual(_line_expression("1 8 0;2 0 0", (x,), field), x**8+2)
                self.assertEqual(_line_expression("1 1 2;2 1 0", (x,), field), (root**2+2)*x)

    @unittest.skipUnless(gb._MODE == "native" and hasattr(gb._kernel(), "atlas_field_flint_index"),
                         "the native root display permutation is not in use")
    def test_marked_field_root_indices_without_external_isolation(self):
        from engine.field_coefficient import _call, program, root_position
        x = sp.Symbol("t")
        expr = x**4 + 1
        poly = sp.PurePoly(expr, x, domain=sp.QQ)
        roots = [sp.CRootOf(expr, i) for i in range(poly.degree())]
        # The two operations must also agree with each independently boxed
        # historical root, so a mutually inverse but wrong permutation fails.
        reference = []
        for r in roots:
            b = r._get_interval()
            reference.append(("root", "4", "1", "0", "0", "0", "1",
                              *map(str, (b.ax,b.bx,b.ay,b.by))))
        primitive = sp.exp(sp.I*sp.pi/4)
        with patch.object(sp.CRootOf, "_get_interval", side_effect=AssertionError("external root isolation")):
            for i,r in enumerate(roots):
                code = program(r)
                self.assertEqual(code[0], "flintroot")
                coords = _call("atlas_field_coefficient", [" ".join(code), " ".join(reference[i])])
                self.assertEqual(coords, "0 1")
            self.assertEqual(root_position(poly, primitive), 3)

    @unittest.skipUnless(gb._MODE == "native" and hasattr(gb._kernel(), "atlas_field_coefficient"),
                         "the native field factor representation is not in use")
    def test_factor_rows_preserve_field_order_without_symbolic_coercion(self):
        from engine.field_factors import factors
        x,y = sp.symbols("x y")
        a = sp.sqrt(2)
        field = sp.QQ.algebraic_field(a)
        poly = sp.Poly((x+y**2-a)*(x+y**2+a)*(x*y**2+y-a), x,y,domain=field)
        expected = poly.factor_list()[1]
        with patch.object(field, "from_sympy", side_effect=AssertionError("unexpected field coercion")):
            self.assertEqual(factors(poly), expected)

    @unittest.skipUnless(gb._MODE != "sympy" and hasattr(gb._kernel(), "atlas_field_coefficient"),
                         "the native qqbar coefficient bridge is not in use")
    def test_native_power_basis_recognition_and_nonmembership(self):
        from engine.field_coefficient import coefficients
        from sympy.polys.polyerrors import CoercionFailed
        t,theta = sp.symbols("t theta")
        for generator, values in [
                (sp.sqrt(2), (sp.tan(sp.pi/8,evaluate=False), sp.cot(sp.pi/8,evaluate=False),
                              sp.sec(sp.pi/4,evaluate=False), sp.csc(sp.pi/4,evaluate=False))),
                (sp.sqrt(2)+sp.sqrt(3), (sp.sqrt(3), sp.sqrt(6), 1/(sp.sqrt(2)-sp.sqrt(3)))),
                (sp.sqrt(-3), (sp.exp(2*sp.pi*sp.I/3), (-sp.Rational(1,8))**sp.Rational(1,3))),
                (sp.CRootOf(t**3-t+1,0), (1+sp.CRootOf(t**3-t+1,0)**2,)),
                (sp.CRootOf(t**3-t+1,1), (1+sp.CRootOf(t**3-t+1,1)**2,))]:
            field = sp.QQ.algebraic_field(generator)
            for value in values:
                try:
                    converted = field.convert(value)
                except CoercionFailed:
                    converted = field.convert(sp.expand_complex(value))
                expected = tuple(reversed(converted.to_list()))
                with patch.object(field, "convert", side_effect=AssertionError("unexpected field coercion")):
                    self.assertEqual(coefficients(value,field), expected)
                    with patch.object(gb,"CHECK",False):
                        lifted = lift(value, (), field, theta)
                    self.assertEqual(lifted, sum(sp.Rational(c)*theta**i for i,c in enumerate(expected)))
        field = sp.QQ.algebraic_field(sp.sqrt(2))
        with self.assertRaises(CoercionFailed):
            coefficients(sp.sqrt(3), field)

    @unittest.skipUnless(gb._MODE != "sympy" and hasattr(gb._kernel(), "atlas_wide_block"),
                         "the native sparse block-order kernel is not in use")
    def test_sparse_block_orders_and_queries_beyond_packed_range(self):
        from engine.field_wide import wide_block_basis
        x,y,theta = sp.symbols("x y theta")
        a = sp.sqrt(2)
        field = sp.QQ.algebraic_field(a)
        for domain, t, c in [(sp.QQ,None,sp.Rational(1,3)), (field,theta,a)]:
            for order in ("lex","grevlex","first"):
                ideal = (x**2+c*x*y-3, c*y**2+x-1)
                native = wide_block_basis(ideal, (x,y), order, domain, t)
                reference = sp.groebner(ideal,x,y,domain=domain,order=gb._reference_order(order))
                self.assertEqual(native.exprs, reference.exprs)
                self.assertEqual(native.reduce((x+c*y)**6), reference.reduce((x+c*y)**6))
        # Avoid constructing the dense portable counterpart of this sparse
        # monomial. The exact expected answer takes a single term.
        with patch.object(gb, "_MODE", "native"):
            v = sp.symbols("v:32")
            f = v[0]**(10**12)-a*v[-1]
            large = gb.groebner((f,), v, domain=field)
            self.assertEqual(large.exprs, [f])
            self.assertEqual(large.reduce(3*f), ([3],0))
            # A colon passes through one additional coefficient variable and
            # an elimination variable. Both dimensions exceed the packed ABI.
            with patch.object(gb, "CHECK", False):
                quotient = gb.colon(((v[0]-a*v[1])*v[2],), (v[0]-a*v[1],), v, domain=field)
            self.assertEqual(quotient, (v[2],))
            small = gb.groebner((x-a*y,), (x,y), domain=field)
            with patch.object(gb, "CHECK", False):
                self.assertEqual(small.reduce((x-a*y)*y**(10**12)), ([y**(10**12)],0))

    def test_lift_lower_preserves_marked_embedding(self):
        x,t = sp.symbols("x t")
        for a in (sp.sqrt(2), -sp.sqrt(2), sp.CRootOf(t**3-t+1, 0), sp.sqrt(2)+sp.sqrt(3)):
            field = sp.QQ.algebraic_field(a)
            theta = sp.Dummy("theta")
            coeff = sp.sqrt(6) if a == sp.sqrt(2)+sp.sqrt(3) else a
            expr = (x+coeff)**3 + (coeff+1)/(a-1)
            self.assertEqual(sp.Poly(lower(lift(expr, (x,), field, theta), (x,), field, theta),
                                     x, domain=field), sp.Poly(expr, x, domain=field))
            self.assertEqual(lower(modulus(field, theta), (x,), field, theta), 0)

    def test_all_orders_division_and_unit_ideal(self):
        x,y,z = sp.symbols("x y z")
        a = sp.sqrt(2)
        field = sp.QQ.algebraic_field(a)
        variables = (x,y,z)
        for order in ("grevlex", "lex", "first"):
            ideal = (a*x**2+x*y-1, (a+1)*y**2-1, (x+a)*z)
            basis = gb.groebner(ideal, variables, order=order, domain=field)
            reference = sp.groebner(ideal, *variables, order=gb._reference_order(order), domain=field)
            self.assertEqual(basis.polys, reference.polys)
            for f in (sp.S.Zero, (x+a*y)**4+z**3/5, x**3*z*y+(a-1)*y):
                self.assertEqual(basis.reduce(f), reference.reduce(f))
                self.assertEqual(basis.contains(f), reference.contains(f))
            unit = gb.groebner((1,), variables, order=order, domain=field)
            f = (x+a)**5
            self.assertEqual(unit.reduce(f), ([sp.Poly(f, *variables, domain=field).as_expr()],0))
            self.assertEqual(unit.reduce(sp.S.Zero), ([],0))

    def test_prefix_colon_empty_ideal_and_coordinate_identities(self):
        a = sp.sqrt(2)
        field = sp.QQ.algebraic_field(a)
        x,y = sp.Dummy("v"), sp.Dummy("v")
        v = (x,y)
        known = (a*(x-y),)
        basis = gb.groebner((*known, y*y-a), v, domain=field, known=1)
        self.assertEqual(basis.polys, sp.groebner((*known,y*y-a), *v, domain=field, order="grevlex").polys)
        empty = gb.groebner((), v, domain=field)
        self.assertEqual(empty.reduce(x+a*y), ([], x+a*y))
        if gb._MODE != "sympy" and hasattr(gb._kernel(), "atlas_field_division"):
            self.assertEqual(set(gb.colon(((x-a*y)*x, (x-a*y)*y), (x-a*y,), v, domain=field)), {x,y})
            with patch("sympy.groebner", side_effect=AssertionError("unexpected portable basis")):
                with patch.object(gb, "_MODE", "native"):
                    exact = gb.groebner((a*x-y,), v, domain=field)
                    self.assertTrue(exact.contains(a*x-y))
                    self.assertEqual(exact.reduce(a*x), ([a],y))

    def test_reduction_beyond_packed_degree_and_unavailable_binary(self):
        x = sp.Symbol("x")
        field = sp.QQ.algebraic_field(sp.sqrt(2))
        basis = gb.groebner((x,), (x,), domain=field)
        # A remainder query may exceed the packing of an already held basis.
        self.assertTrue(basis.contains(x**32768))
        with patch.object(gb, "_kernel", return_value=None):
            if gb._MODE == "native":
                with self.assertRaisesRegex(ArithmeticError,"make groebner"):
                    gb.groebner((x-1,), (x,), domain=field)
            else:
                self.assertEqual(gb.groebner((x-1,), (x,), domain=field).exprs,[x-1])
