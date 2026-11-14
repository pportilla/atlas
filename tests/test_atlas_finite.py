"""Partial finite-centre identifications must not certify a global divisor."""
import unittest
from dataclasses import replace

import sympy as sp

from engine.atlas_finite import finite_complex
from engine.local_blowups import Chart


class FiniteCentre(unittest.TestCase):
    def test_ownership_is_checked_independently_on_rational_orbits(self):
        t, v, w = sp.symbols("t v w")
        # On this affine modification, y/x=t extends regularly across
        # D=(t³-1). One open omits its rational component t=1.
        p = t**3 - 1
        mapping = (("x", v*p), ("y", t*v*p), ("z", w*p))
        c = Chart((t, v, w), (), (("D", 1, 1, (p,)),), tuple(f for _, f in mapping), (), mapping, (),
                  removed=((t-1,),))
        counts = {frozenset(("D",)): None}
        self.assertIsNone(finite_complex([c], 3, counts))
        other = replace(c, removed=((t**2+t+1,),))
        self.assertEqual(len(finite_complex([c, other], 3, counts)["vertices"]), 3)
        self.assertIsNone(finite_complex([c, other, c], 3, counts))
        away = (("x", sp.S.One), ("y", t), ("z", v))
        self.assertIsNone(finite_complex([replace(c, coordinates=away, fibre=(sp.S.One, t, v), removed=())],
                                         3, counts))
