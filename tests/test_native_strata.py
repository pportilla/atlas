import os
import unittest

import sympy as sp

from engine import kernel
from engine.constructible_euler import euler_characteristic
from engine.groebner import _kernel

x, y, z, t = sp.symbols("x y z t")


def native(variables, equations, excluded=()):
    """chi of V(equations) minus the V(I), I in excluded, from the kernel."""
    found = kernel.strata(variables, (), [tuple(equations), *map(tuple, excluded)])
    return found.get((0,), 0)


@unittest.skipUnless(_kernel() and os.getenv("ATLAS_GROEBNER") != "sympy", "the native kernel is not in use")
class NativeEuler(unittest.TestCase):
    """native/euler.cpp against engine/constructible_euler.py and against known values."""

    def test_known_values(self):
        curve = y**2 - x*(x - 1)*(x - 2)
        cases = [
            ((x,), ((x**2 - 1)**3, (x - 1)**2*(x + 1)), (), 2),
            ((x, y), (curve,), (), -1),                       # an affine elliptic curve
            ((x, y), ((x - 4)*curve,), (), -2),
            ((x, y), (curve,), ((x, y), (x - 4,)), -4),
            ((x, y), ((y**2 - x**2)**2, y**2 - x**4), (), 5),
            ((x, y), (x*y - 1,), (), 0),
            ((x, y), (x*y,), (), 1),
            ((x, y, z), (x**2 + y**3 + z**4 - 1,), (), 7),   # a Milnor fibre: 1 + 1*2*3
            ((x, y, z), (z - x**2 - y**2, x*y - 1), (), 0),
            # (C minus two points) x (C minus two points)
            ((x, y, z), (x - y,), ((x*(x - 1),), (y*(y - 1),), (z*(z - 1),)), 1),
        ]
        for variables, equations, excluded, expected in cases:
            with self.subTest(equations=equations, excluded=excluded):
                self.assertEqual(native(variables, equations, excluded), expected)
                self.assertEqual(euler_characteristic(variables, equations, excluded), expected)

    def test_agrees_with_the_reference(self):
        cases = [
            ((x, y, z), (x**3 + y**3 + z**3 - 3*x*y*z - 1,), ((x + y + z,),)),
            ((x, y, z), (x*y*z - 1, x + y + z - 3), ()),
            ((x, y, z), (x**2*y - z**3,), ((x, z), (y - 1,))),
            ((x, y, z, t), (t**2 + 5*t + 7, x**2 - t*y**3 + z, x*y - z**2), ((x, y),)),
            ((x, y, z, t), (x**2 + y**2 + z**2 + t**2 - 1, x*t - y*z), ((x - 1,), (y, z))),
            ((x, y, z), ((x**2 + y**2 - 1)*(z**2 - x), z*(x - y)), ((x*y,),)),
        ]
        for variables, equations, excluded in cases:
            with self.subTest(equations=equations, excluded=excluded):
                self.assertEqual(native(variables, equations, excluded),
                                 euler_characteristic(variables, equations, excluded))

    def test_strata_of_a_chart(self):
        # The coordinate planes: a plane minus two lines and a line minus a
        # point have chi = 0; only the origin counts.
        found = kernel.strata((x, y, z), (), [(x,), (y,), (z,)])
        self.assertEqual(found, {(0, 1, 2): 1})
        # Two lines of the plane z = 0 meeting at a point, and a third closed
        # set that does not meet their point.
        found = kernel.strata((x, y, z), (z,), [(x,), (y,), (x - 1, y - 1)])
        self.assertEqual(found, {(0, 1): 1, (2,): 1})

    def test_strata_on_coordinate_orbits(self):
        # On z=0 the Laurent trinomial on (C*)² covers a punctured
        # line with degree six; its two coordinate axes have 3 and 2 roots.
        axes = [(x,), (y,), (z,)]
        found = kernel.strata((x,y,z), (z**4,), [*axes, (2*x**2 + 3*y**3 - 1,)])
        self.assertEqual(found, {(2,3):-6, (2,):6, (0,2,3):3, (0,2):-3,
                                 (1,2,3):2, (1,2):-2, (0,1,2):1})
        # A repeated Laurent root crossed with a free torus has chi=0.
        found = kernel.strata((x,y,z), (z,), [*axes, ((x*y-1)**2,)])
        self.assertEqual(found, {(0,1,2):1})
        # Four terms on a two-torus can introduce singularities. The
        # general calculation counts the crossing of x=1 and y=1 once.
        found = kernel.strata((x,y,z), (z,), [*axes, ((x-1)*(y-1),)])
        self.assertEqual(found[(2,3)], -1)


if __name__ == "__main__":
    unittest.main()
