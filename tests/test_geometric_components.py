import unittest

import sympy as sp

from engine.blowup_ideal import ideal_basis
from engine.geometric_components import geometric_components, _shape


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

    def test_special_fiber_lost_by_generic_projection(self):
        x,y,z = self.v
        line_a = (y,z)
        line_b = (x-1,z-1)
        ideal = ideal_basis(self.v, [a*b for a in line_a for b in line_b])
        # The second line maps to the single first-coordinate value 1.
        self.assertIsNone(_shape(self.v,ideal,1,sp.QQ,1,sp.eye(3)))
        self.assertEqual(set(geometric_components(self.v,ideal,1)), {line_a,line_b})

    def test_generic_primitive_coordinate_for_four_points(self):
        x,y,_ = self.v
        components = geometric_components((x,y), [x*x-1,y*y-1],0)
        self.assertEqual(set(components), {(x-a,y-b) for a in (-1,1) for b in (-1,1)})

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
