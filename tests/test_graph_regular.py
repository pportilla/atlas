"""Regular multipliers on the graph frames used by intrinsic centre selection."""
from fractions import Fraction
import gzip
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from engine import centres, native_polynomial as np
from engine.centre_frame import Frame
from engine.native_library import _MODE, ideal_request

FIXTURE = Path(__file__).parent / "fixtures/b8-centre-and-incidence.json.gz"


def b9_frame():
    """Reconstruct the first B9 frame from the saved parent of blowup B8."""
    with gzip.open(FIXTURE, "rt") as stream:
        record = json.load(stream)
    a, b, c = parent = np.symbols("a b c")
    x, y, z, u, v = variables = np.symbols("x y z u v")
    pivot = np.from_row(parent, record["parent"]["chosen"][0]).xreplace({b: x})
    # In the first chart the other centre generators are y*pivot and z*pivot.
    strict = np.from_row(parent, record["function"])
    transformed = strict.xreplace({a: pivot*y-3211264*x-112, b: x, c: pivot*z}) / pivot**2
    equations = (u - (pivot*y**2-6422528*x*y-368*y+10312216477696), v-pivot,
                 x*y*z*v/2 + 23*y*z*v/802816 - 1605632*z*v - z/2048)
    return variables, equations, transformed


def certificate(variables, equations, function):
    rows = [np.row(variables, f) for f in (*equations, function)]
    return ideal_request(f"graph_regular {len(variables)} {len(equations)}", rows)


@unittest.skipUnless(_MODE == "native", "Exercises the native centre selector.")
class GraphRegular(unittest.TestCase):
    def test_triangular_substitution_preserves_thickness(self):
        x, y, u, v = variables = np.symbols("x y u v")
        # Modulo the first two rows, v-u^2 is x and u-y^2 is x^2.
        eq = (u-y*y-x*x, v-u*u-x, (v-u*u)**3*(y-1)**2)
        for function, answer in ((v-u*u+y, "yes"), (u-y*y, "unknown"),
                                 (y-1, "unknown"), (v-u*u-x, "unknown"), (0, "unknown")):
            with self.subTest(function=function):
                self.assertEqual(certificate(variables, eq, function), [answer])

    def test_nonscalar_pivot_and_multiple_residual_relations(self):
        x, y, z = variables = np.symbols("x y z")
        # Dividing xy-x by x would remove the entire x=0 component.
        self.assertEqual(certificate(variables, (x*y-x, z*z), x), ["unknown"])
        self.assertEqual(certificate(variables, (z-y*y, x*x, x*y), x), ["unknown"])

    def test_wide_exponents_and_rational_coefficients(self):
        x, y, u = variables = np.symbols("x y u")
        scale = Fraction(2**151+27, 2**133+1)
        self.assertEqual(certificate(variables, (scale*u-y**2, (y-1)**2), x+u), ["yes"])
        wide = x**(2**35+1)
        self.assertEqual(certificate(variables, (u-y*y,), wide+2), ["yes"])
        self.assertEqual(certificate(variables, (u-y*y, y*y), y*(wide+2)), ["unknown"])

    def test_b9_keeps_the_working_piece_and_skips_the_split(self):
        variables, eq, function = b9_frame()
        self.assertEqual(len(np.terms(variables, function)), 5974)
        frame = Frame(variables, variables[-2:], eq)
        item = ({}, function, Fraction(1))
        work = (variables[:3],)
        state = centres.State(frame, (item,), (), (), (), work, (), ())
        with patch.object(centres, "saturate", side_effect=AssertionError("redundant split")):
            self.assertEqual(list(centres._groups(state)), [(work, [item])])

    def test_regular_germ_at_a_zero_and_empty_open_piece(self):
        x, y, z = variables = np.symbols("x y z")
        # On the two disjoint lines z=y, y^2=1, x is regular even where
        # x=0. The piece x=1 is absent because the frame inverts x-1.
        frame = Frame(variables, (x,), (z-y, y*y-1), units=(x-1,))
        item = ({}, x, Fraction(1))
        state = centres.State(frame, (item,), (), (), (), ((x,), (x-1,)), (), ())
        with patch.object(centres, "saturate", side_effect=AssertionError("redundant split")):
            self.assertEqual(list(centres._groups(state)), [(((x,),), [item])])

    def test_projection_limits_expanded_coefficient_storage(self):
        # Substituting into C*u replicates a 12,001-bit coefficient in 12,000
        # terms. Even though the term count fits, decline the large expansion.
        graph = ";".join(["1 0 1 0", *(f"-1 {i} 0 0" for i in range(2, 12002))])
        coefficient = str(1 << 12000)
        # A repeated graph equation drops out on projection. Without the
        # guard the routine expands C*u before certifying a polynomial ring.
        rows = [graph, graph, coefficient + " 0 1 0"]
        self.assertEqual(ideal_request("graph_regular 3 2", rows), ["unknown"])
