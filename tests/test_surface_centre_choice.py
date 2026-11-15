"""Compare structural smooth-surface choices with the full canonical chart rule."""
import unittest
from unittest.mock import patch

from engine import native_polynomial as np, surface


class SurfaceCentreChoice(unittest.TestCase):
    def setUp(self):
        self.ring = tuple(map(np.Symbol, "uxy"))
        self.u, self.x, self.y = self.ring

    def matches(self, strict, divisors, work, *, holes=(), expected):
        args = (self.ring, (), (strict,), [(k, (v,)) for k, v in divisors], work)
        got = surface._line_centre(*args, holes)
        self.assertEqual(got, expected)
        with patch.object(surface, "_line_centre", return_value=None):
            ordinary = surface.step(*args, away=holes)
        self.assertEqual(surface.step(*args, away=holes), ordinary)

    def test_doubled_trace_has_the_whole_line_as_centre(self):
        u, x, y = self.ring
        # Geometrically split conjugate lines cross the doubled trace.
        self.matches(u-(3*x+2)**2, [("B2", y*y+1), ("B1", u)], [(3*x+2, y*y+1, u)],
                     expected=((2,2,0,("B1",)), "curve", (u,x+np.coerce(2)/3)))

    def test_ownership_and_denominator(self):
        u, x, y = self.ring
        # The graph open omits a vertical fibre above y=-1. The removed
        # centre excludes its intersection with the chosen working line.
        f = (y+1)*u+x*x
        centre = ((2,2,0,("B1",)), "curve", (u,x))
        self.matches(f, [("B1",u)], [(x,u)], holes=[(y+1,x,u)], expected=centre)
        self.matches(f, [("B1",u)], [(x,u)], holes=[(x,u)], expected=None)

    def test_shared_trace_and_last_label_ties(self):
        u, x, y = self.ring
        self.matches(u-x, [("C",y), ("B",u), ("A",x)], [(u,x,y)],
                     expected=((2,1,0,("A","B")), "curve", (u,x)))
        # These are ordinary smooth divisors in the threefold. Each has a
        # double trace, and the last label wins regardless of input order.
        self.matches(u, [("Z",u-(x-1)**2), ("A",u-x*x)], [(u,x), (u,x-1)],
                     expected=((2,2,0,("Z",)), "curve", (u,x-1)))
        # Two disjoint branches share two labels each. Compare label pairs
        # as tuples, independently of the order of factors and divisors.
        self.matches(u, [("A",x), ("D",u+x), ("C",u+x-1), ("B",x-1)], [(u,x), (u,x-1)],
                     expected=((2,1,0,("B","C")), "curve", (u,x-1)))

    def test_ambiguous_or_singular_traces_use_the_general_rule(self):
        u, x, y = self.ring
        for trace in [(x*y)**2, x*y*(x-y), x**3-y*y, x*y]:
            with self.subTest(trace=trace):
                self.matches(u-trace, [("E",u)], [(u,x,y)], expected=None)
        # Removing the denominator could hide another component of the winning
        # closed trace. Leave its closure to the ordinary model selection.
        args = (self.ring, (), ((y+1)*u+x*x,), [("E",((y+1)*u,))], [(u,x,y)], ())
        self.assertIsNone(surface._line_centre(*args))

    def test_large_rational_coefficient_and_wide_speculative_decline(self):
        u, x, y = self.ring
        c = np.coerce(2**100+3)/19
        self.matches(u-(x+c)**2, [("E",u)], [(x+c,u)],
                     expected=((2,2,0,("E",)), "curve", (u,x+c)))
        args = (self.ring, (), (u-x**(2**64),), [("E",(u,))], [(u,x)], ())
        self.assertIsNone(surface._line_centre(*args))
