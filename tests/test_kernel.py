import os
import unittest

import sympy as sp
from sympy import QQ

from engine import blowup_ideal as ideals
from engine import kernel
from engine.general_resolution import _attempt, _charts, _pieces, _workers
from engine.groebner import _kernel
from engine.resolution_search import ResolutionCandidate, _candidate, _projective

x, y, z, w = sp.symbols("x y z w")


def candidate(variables, center, boundary=None):
    u = _projective(variables, len(center))
    pieces = (center,) if boundary is None else boundary(u)
    return ResolutionCandidate(center, u, pieces, QQ, None, "test")


@unittest.skipUnless(_kernel() and os.getenv("ATLAS_GROEBNER") != "sympy", "the native kernel is not in use")
class KernelIdealTests(unittest.TestCase):
    """Each kernel routine against the SymPy route it replaces."""

    def test_saturation_is_the_lexicographic_basis(self):
        cases = [((x, y), (x*y,), x), ((x, y, z), (x*z - y**2, x**3 - y*z), x),
                 ((x, y, z), (x**2*y - z, x*z**2 - y**2, y*z - x), x*y - z),
                 ((x, y), (x**2, x*y), sp.Integer(0)), ((x, y), (x**2*y, x*y**2), sp.Integer(3)), ((x, y), (), x),
                 ((x, y, z), (sp.Rational(2, 3)*x*y - z**2, x*z + sp.Rational(1, 5)), y + 2*z)]
        for variables, ideal, by in cases:
            with self.subTest(ideal=ideal, by=by):
                self.assertEqual(kernel.saturate(variables, ideal, by), ideals._saturate(variables, ideal, by, QQ))

    def test_radical_membership(self):
        cases = [((x, y), (x,), (x**2, y)), ((x, y), (x, y), (x**2,)), ((x, y, z), (x*y, z), (x**2, y**3*z, z**2 - x)),
                 ((x, y), (), (x,)), ((x, y), (x,), ()), ((x, y), (x - 1,), (x**2 - 1, (x - 1)*y))]
        for variables, smaller, larger in cases:
            with self.subTest(smaller=smaller, larger=larger):
                self.assertIs(kernel.radical_subset(variables, smaller, larger),
                              ideals._radical_subset(variables, smaller, larger, QQ))

    def test_smooth_locus_and_normal_crossings(self):
        cases = [((x, y), (x*(x - 1), x*y**2), 1), ((x, y, z), (x*(x - 1), x*y, (x - 1)*z, y*z), 1),
                 ((x, y), (y**2 - x**3,), 1), ((x, y, z), (x*y - z**2,), 2), ((x, y, z), (x, y), 1),
                 ((x, y), (), 2), ((x, y), (), 1), ((x, y), (x,), 3), ((x, y), (sp.Integer(1),), 0),
                 ((x, y, z), (x*z, y*z, z*(z - 1)), 2), ((x, y, z), (z, y, x**2), 1)]
        for variables, ideal, dimension in cases:
            with self.subTest(ideal=ideal, dimension=dimension):
                self.assertEqual(kernel.smooth_locus(variables, ideal, dimension),
                                 ideals._smooth_locus(variables, ideal, dimension, QQ))
        crossings = [((x, y), (), None, ((x,), (x - 1,), (y,)), 2), ((x, y), (), None, ((x,), (y,), (x + y,)), 2),
                     ((x, y, z), (z - x*y,), (x - y**2, z - y**3), ((x,), (y, z)), 2)]
        for arguments in crossings:
            with self.subTest(arguments=arguments):
                self.assertEqual(kernel.verify_snc(*arguments), ideals._verify_snc(*arguments, QQ))

    def test_charts_of_a_blowup(self):
        h = y - x**2
        cases = [((x, y, z), (), (z - h**2,), (h, z), ()), ((x, y), (), (y**2 - x**3,), (x, y), ()),
                 ((x, y), (x,), (), (x, y), ((y,),)), ((x, y), (), (x*y,), (x**3, x*y, y**2), ((x,), (x - y,))),
                 ((x, y, z), (), (x**2 - y**2*z,), (x, sp.Integer(0), y), ())]
        for variables, ambient, strict, center, exceptional in cases:
            with self.subTest(center=center):
                self.assertEqual(ideals._kernel_charts(variables, ambient, strict, center, exceptional),
                                 ideals._blowup_charts(variables, ambient, strict, center, exceptional, QQ))

    def test_unsuitable_input_is_left_to_sympy(self):
        self.assertIsNone(kernel.saturate((x, y), ("x",), x))
        self.assertIsNone(kernel.saturate((x, y), (1/x,), x))
        self.assertIsNone(kernel.radical_subset((x, y), (sp.sqrt(2)*x,), (x,)))
        self.assertIsNone(kernel.smooth_locus((x, y), (x**40000 - y,), 1))
        many = sp.symbols("a:40")
        self.assertIsNone(kernel.smooth_locus(many, (many[0],), 39))


@unittest.skipUnless(_kernel() and os.getenv("ATLAS_GROEBNER") != "sympy", "the native kernel is not in use")
class KernelVerificationTests(unittest.TestCase):
    """The verification of a centre, which uses the rank test wherever the
    dimension is known in advance (native/KERNEL.md, Proposition 4), against
    the route through the exact smoothness test."""

    def assert_same(self, variables, f, c):
        fast = ideals.resolution_charts(variables, f, c.center, c.projective, c.boundary)
        self.assertIsNotNone(fast)
        if fast:
            fast = fast[0], _pieces(c, fast[0]), fast[1]
        self.assertEqual(fast, _charts(variables, f, c))
        return bool(fast)

    def test_candidates_that_pass(self):
        h = y - x**2
        self.assertTrue(self.assert_same((x, y, z), x**2 - y**2*z, candidate((x, y, z), (x, y))))
        self.assertTrue(self.assert_same((x, y, z), h*h - x*z*z, candidate((x, y, z), (h, z))))
        self.assertTrue(self.assert_same((x, y), y**2 - x**4, candidate(
            (x, y), (x**3, x*y, y**2), lambda u: ((x, y, u[0]), (x, y, u[2])))))
        self.assertTrue(self.assert_same((x, y, z), x*y - z**2, candidate((x, y, z), (x, y, z))))

    def test_candidates_that_fail(self):
        # Not in the singular locus; a singular strict transform; a tangency.
        self.assertFalse(self.assert_same((x, y), x, candidate((x, y), (x, y))))
        self.assertFalse(self.assert_same((x, y, z), x**2 - y**2*z, candidate((x, y, z), (x, y, z))))
        self.assertFalse(self.assert_same((x, y), y**2 - x**3, candidate((x, y), (x, y))))
        # A boundary that is not homogeneous, one that misses a component.
        self.assertFalse(self.assert_same((x, y), x*y, candidate((x, y), (x, y), lambda u: ((x, y, x + u[0]*x),))))
        self.assertFalse(self.assert_same((x, y), y**2 - x**4, candidate(
            (x, y), (x**3, x*y, y**2), lambda u: ((x, y, u[0]),))))

    def test_a_piece_of_too_small_dimension_is_rejected(self):
        # The pieces cover the exceptional curve of the blowup of a node, and
        # every tangent space is small enough, but the second piece is a
        # point. Only the exact test sees that it is not a curve.
        c = candidate((x, y), (x, y), lambda u: ((x, y), (x, y, u[0])))
        self.assertFalse(self.assert_same((x, y), x*y, c))
        c = candidate((x, y, z), (x, y), lambda u: ((x, y), (x, y, z, u[1])))
        self.assertFalse(self.assert_same((x, y, z), x**2 - y**2*z, c))

    def test_workers_are_reused_and_replaced_after_a_timeout(self):
        good = _candidate((x, y, z), (x, y), "test")
        f = x**2 - y**2*z
        self.assertEqual(_attempt((x, y, z), f, good, "embedded", 1, 30)[0], "certified")
        with _workers.lock:
            first = [worker.process.pid for worker in _workers.idle]
        self.assertTrue(first)
        self.assertEqual(_attempt((x, y, z), f, _candidate((x, y, z), (x, y, z), "test"), "embedded", 1, 30)[0], "rejected")
        with _workers.lock:
            self.assertEqual([worker.process.pid for worker in _workers.idle], first)
        self.assertEqual(_attempt((x, y, z), f, good, "embedded", 1, 1e-6), ("undecided", None))
        self.assertEqual(_attempt((x, y, z), f, good, "embedded", 1, 30)[0], "certified")
        with _workers.lock:
            self.assertTrue(all(worker.alive() for worker in _workers.idle))


if __name__ == "__main__":
    unittest.main()
