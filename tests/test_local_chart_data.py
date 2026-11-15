"""Native graph blowup data and elimination of graph coordinates."""
import unittest
from unittest.mock import patch

import sympy as sp

from engine.blowup_ideal import ideal_subset
from engine.local_blowups import Chart, _blow_up, _eliminate, _strict_ideal


class LocalGraphChart(unittest.TestCase):
    def test_strict_component_product_in_the_marked_chart_ring(self):
        from engine import native_polynomial as np
        from engine.centre_weights import component_ideal_product
        from engine.local_blowups import _hypersurface

        for symbols in (np, sp):
            with self.subTest(expressions=symbols.__name__):
                x, y, a = symbols.Dummy("u"), symbols.Dummy("u"), symbols.Dummy("a")
                variables = (y, a, x)
                field = (a, a**2 - 2, 2, 0)
                components = (("S1", 1, 1, (3*a*x,)), ("S2", 1, 1, (a*y,)), ("E1", 2, 2, (x,)))
                chart = Chart(variables, (field[1],), components, (), (), (), (), coefficient_field=field)
                self.assertEqual(_hypersurface(chart), (6*x*y,))

                f = x**(2**70) * symbols.Rational(2**130 + 1, 3)
                self.assertEqual(component_ideal_product(variables, [(f, 0, f), (2, y)]),
                                 (2*f, f*y, 0, 0, 2*f, f*y))
                self.assertEqual(component_ideal_product(variables, []), (1,))
                self.assertEqual(component_ideal_product(variables, [(x,), ()]), ())

    def test_unavailable_native_operations_on_external_symbols(self):
        from engine import kernel, local_blowups as local
        from engine.native_library import _MODE
        if _MODE != "native":
            self.skipTest("Checks the default native dispatch.")
        x, = sp.symbols("x", seq=True)
        with patch.object(kernel, "dimension", return_value=None):
            with self.assertRaisesRegex(local.GiveUp, "native dimension"):
                local._dimension((x,), (x,))
        with patch.object(local, "_point_factors", side_effect=ArithmeticError):
            with self.assertRaisesRegex(local.GiveUp, "separate"):
                local._point_components((x,), (x,))
        with patch.object(kernel, "finite_radical", return_value=None), patch.object(
                local, "_point_factor_rows", side_effect=ArithmeticError):
            with self.assertRaisesRegex(local.GiveUp, "reduce"):
                local._radical_points((x,), (x**2,))
        with patch.object(kernel, "strict_parts", return_value=None), patch.object(local, "colon", return_value=None):
            with self.assertRaisesRegex(local.GiveUp, "residual divisor"):
                local._strict_ideal((x,), (), x, (x**2,))

    def test_unpacked_linear_blowup_and_fibre_rank(self):
        from engine.local_blowups import _blow_up_linear, _failures
        variables = sp.symbols("u:33")
        x,y = variables[:2]
        chart = Chart(variables, (), (("strict", 1, 1, (x*y,)),),
                      (x,y), (), (("x",x),("y",y)), ())
        over, inside = _failures(chart, 33)
        self.assertEqual(inside, {("strict",)})
        self.assertEqual(set(over), {("strict",)})
        children = _blow_up_linear(chart, [(x,0),(y,0)], "E1", 1, 2)
        self.assertEqual(len(children), 2)
        for child in children:
            strict, exceptional = child.components
            self.assertEqual(exceptional[:3], ("E1",2,2))
            self.assertEqual(strict[3][0]*exceptional[3][0]**2,
                             dict(child.coordinates)["x"]*dict(child.coordinates)["y"])

    def test_curved_center_keeps_orders_original_map_and_removed_points(self):
        x, y, z = sp.symbols("x y z")
        variables = (x, y, z)
        h = y - x**2
        chart = Chart(variables, (), (("strict", 1, 1, (z-h**2,)), ("E", 2, 2, (z,))),
                      variables, (), tuple(zip(("x", "y", "z"), variables)), (), ((x-2, y),))
        children = _blow_up(chart, (h, z), "B", 1, 3, 2)
        self.assertEqual(len(children), 2)
        for child in children:
            mapping = dict(child.coordinates)
            self.assertEqual(tuple(mapping), ("x", "y", "z"))
            strict, = next(row[3] for row in child.components if row[0] == "strict")
            _, n, nu, pivot = next(row for row in child.components if row[0] == "B")
            self.assertEqual((n, nu), (3, 3))
            image = (z-h**2).xreplace(dict(zip(variables, mapping.values())))
            total = strict*pivot[0]
            self.assertTrue(ideal_subset(child.variables, (image,), (*child.ambient, total)))
            self.assertTrue(ideal_subset(child.variables, (total,), (*child.ambient, image)))
            self.assertEqual(child.removed, ((x-2, mapping["y"]),))

    def test_generators_of_different_orders_give_one_residual_ideal(self):
        x, y, z = sp.symbols("x y z")
        self.assertEqual(_strict_ideal((x, y, z), (z-x*y,), x, (x*y, x**2*z, z-x*y)),
                         (1, (y, z)))

    def test_principal_graph_transform_divides_in_the_chart_ring(self):
        from engine import kernel
        from engine.local_blowups import _generators
        x, y, z = sp.symbols("x y z")
        variables = (x, y, z)
        pivot = x**2+1
        ambient = (z-pivot*y,)
        order, after = _strict_ideal(variables, ambient, pivot, (z**3 + pivot**4,))
        self.assertEqual(order, 3)
        expected = (y**3+pivot,)
        self.assertTrue(ideal_subset(variables, after, (*ambient, *expected)))
        self.assertTrue(ideal_subset(variables, expected, (*ambient, *after)))
        if kernel._MODE != "sympy":
            order, (q,) = kernel.graph_strict_part(variables, ambient, -3*pivot, 21*z**3)
            self.assertEqual(order, 3)
            self.assertEqual(q, -sp.Rational(7,9)*y**3)
        self.assertEqual(len(_generators(variables, ambient, expected)), 1)
        self.assertEqual(_generators(variables, ambient, ambient), ())

    def test_residual_dimension_on_components_of_an_exceptional_cylinder(self):
        from engine import kernel
        if kernel._MODE == "sympy":
            self.skipTest("Exercises the native exceptional-stratum test.")
        x, y, z = sp.symbols("x y z")
        variables = (x, y, z)
        # The base has two irreducible planes. A residual with a constant
        # coefficient of z misses the generic point of each. x*z alone
        # vanishes on an entire plane, which must be detected.
        for equations, threshold, expected in [
                ((z + x*y**5,), 2, True), ((x*z,), 2, False),
                ((x*z, (x-1)*y), 2, True), ((), 2, False),
                ((z,), 1, False), ((x*(x-1),), 3, True)]:
            self.assertEqual(kernel.residual_dimension_below(
                variables, (x*(x-1),), equations, threshold), expected)

    def test_a_zero_map_coordinate_and_empty_removed_ideal_survive(self):
        x, y = sp.symbols("x y")
        chart = Chart((x, y), (y,), (("strict", 1, 1, (x+y,)),), (x, y), (y,),
                      (("x", x+y), ("y", y)), (), ((y,),))
        result = _eliminate(chart)
        self.assertEqual(result.variables, (x,))
        self.assertEqual(result.coordinates, (("x", x), ("y", 0)))
        self.assertEqual(result.fibre, (x,))
        self.assertEqual(result.removed, ((),))
        self.assertEqual(result.region, ())

    def test_blowups_preserve_the_selected_coefficient_field(self):
        x, y, u = sp.symbols("x y u")
        field = (u, u**2-2, 2, "sqrt(2)")
        chart = Chart((x, y, u), (field[1],), (("strict", 1, 1, (y-x,)),),
                      (x, y), (), (("x", x), ("y", y)), (), coefficient_field=field)
        for child in _blow_up(chart, (x, y), "E", 1, 2, 2):
            self.assertEqual(child.coefficient_field, field)
            self.assertIn(field[1], child.ambient)
            self.assertEqual(child.one_sheet(2), 1)


if __name__ == "__main__":
    unittest.main()


class WideNativeChart(unittest.TestCase):
    def test_large_degree_graph_transform_uses_wide_quotients(self):
        from engine import native_polynomial as np
        from engine.local_blowups import _dimension
        from engine.native_library import _MODE
        x, y, z = variables = sp.symbols("x y z") if _MODE == "sympy" else tuple(map(np.Symbol, "xyz"))
        ambient = (z-y,)
        residual = (2**180+1)*z**40000+1
        order, ideal = _strict_ideal(variables, ambient, x, (x*residual,))
        self.assertEqual(order, 1)
        self.assertTrue(ideal_subset(variables, ideal, (*ambient, residual)))
        self.assertTrue(ideal_subset(variables, (residual,), (*ambient, *ideal)))
        self.assertEqual(_dimension(variables, (*ambient, x, *ideal)), 0)

    def test_large_fibre_restriction_and_kernel_limits_without_sympy(self):
        from pathlib import Path
        import subprocess
        import sys
        import os
        code = """
import sys
class Guard:
    def find_spec(self, name, *args):
        if name == 'sympy' or name.startswith('sympy.'):
            raise AssertionError(name)
sys.meta_path.insert(0, Guard())
sys.path.insert(0, sys.argv[1])
from unittest.mock import patch
from engine import native_polynomial as np
from engine.local_blowups import Chart, GiveUp, _dimension, _point_components, _radical_points, _strata
coords = np.symbols('v0:35')
x,y = coords[:2]
charts = [Chart(coords, (), (('E1',2,3,(x,)), ('strict',1,1,(x-y,))), (x,y), (), (), ())]
assert _dimension(coords, (x*y,)) == 34
assert _strata(charts)[1] == {frozenset(('E1','strict')): 1}
with patch('engine.kernel.finite_radical', return_value=None):
    assert _radical_points((x,y), (x*x,y)) == (x,y)
with patch('engine.kernel.strata', return_value=None):
    assert _point_components((x,y), (x,y)) == [(x,y)]
for f, patched in [(_dimension, 'engine.kernel.dimension'),
                   (_radical_points, 'engine.local_blowups._point_factor_rows'),
                   (_point_components, 'engine.local_blowups._point_factors')]:
    with patch('engine.kernel.finite_radical', return_value=None):
        with patch(patched, **({'return_value': None} if f == _dimension else {'side_effect': ArithmeticError})):
            try:
                f((x,y), (x,y))
            except GiveUp:
                pass
            else:
                raise AssertionError(patched)
assert 'sympy' not in sys.modules
"""
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-I", "-c", code, str(root)], capture_output=True,
                                text=True, env={**os.environ, "ATLAS_GROEBNER":"native"})
        self.assertEqual(result.returncode, 0, result.stderr)
