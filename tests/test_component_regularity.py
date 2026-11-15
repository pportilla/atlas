"""Nonzerodivisor certificates on reducible and nonreduced resolution frames."""
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from engine import centres, native_polynomial as np
from engine.centre_frame import Frame
from tests.test_graph_regular import certificate


def b9_component_requests():
    """Packets captured during B9 selection on the normal intrinsic path."""
    path = Path(__file__).parent / "fixtures/b9-component-saturations.json.gz"
    with gzip.open(path, "rt") as source:
        return json.load(source)


class ComponentRegularity(unittest.TestCase):
    def test_embedded_associated_prime(self):
        x, y, z = variables = np.symbols("x y z")
        # I = (x) intersect (x^2,y).  y misses the minimal prime but kills x.
        equations = (x*x, x*y)
        self.assertEqual(certificate(variables, equations, y), ["unknown"])
        self.assertEqual(certificate(variables, equations, y+z), ["yes"])

    def test_thickened_components(self):
        x, y, z = variables = np.symbols("x y z")
        equations = (x*x, y*z)
        for multiplier, expected in ((y+z, "yes"), (x, "unknown"), (y, "unknown")):
            with self.subTest(multiplier=multiplier):
                self.assertEqual(certificate(variables, equations, multiplier), [expected])

    def test_nonzero_on_component_without_rational_points(self):
        x, y, z = variables = np.symbols("x y z")
        # The two primes are (x^2+1, y) and (x^2+1, z).
        self.assertEqual(certificate(variables, (x*x+1, y*z), (y+z)*x), ["yes"])
        self.assertEqual(certificate(variables, (x*x+1, y*z), x*y), ["unknown"])

    def test_uninformative_fibre_still_checks_the_whole_prime(self):
        x, y, z = variables = np.symbols("x y z")
        dense = z**64 - 1 + sum((z**k for k in range(1, 63)), start=0) - 62
        # On the z=1 fibre this evaluates to zero, but it is regular modulo
        # the embedded ideal. Multiplying by y makes it an actual zerodivisor.
        equations = (x*x, x*y)
        self.assertEqual(certificate(variables, equations, dense), ["yes"])
        self.assertEqual(certificate(variables, equations, y*dense), ["unknown"])

    def test_expensive_decomposition_leaves_saturation_to_the_exact_fallback(self):
        x, y, z, u = variables = np.symbols("x y z u")
        equations = (-3*x*x*y + x*x*u + x*y*u + 2*y*y*z + x*y - 3*x*z + 2*x*u - 3*y*u + z*u,
                     x*x*y + 2*x*x*u + x*y*u - 3*x*z*u + 2*y*y*z - 3*x*x - 3*y*z - 3*y*u + 2*z*u)
        # Independent ordinary saturation preserves this ideal. Decomposing
        # these coupled cubics exceeds the optional certificate's work budget.
        self.assertEqual(certificate(variables, equations, x+y), ["unknown"])
        self.assertEqual(certificate(variables[:3], (x*x, y*z), y+z), ["yes"])

    def test_wide_coefficient_and_nonconstant_residues(self):
        x, y, z = variables = np.symbols("x y z")
        scale = Fraction(2**263 + 7, 2**181 + 3)
        self.assertEqual(certificate(variables, (scale*x*x, y*z), y+scale*z), ["yes"])

    def test_wide_exponent_never_discards_an_embedded_component(self):
        x, y, z = variables = np.symbols("x y z")
        multiplier = y*z**(2**75 + 5)
        self.assertEqual(certificate(variables, (x*x, x*y), multiplier), ["unknown"])

    def test_product_section_with_repeated_factors_and_permuted_axes(self):
        x, y, z, u, t = axes = np.symbols("x y z u t")
        # Modulo (x^2,y^2), the product section t^2*z has two thick
        # components. At t=0 the multiplier is u*z. Over z=0 it is t.
        equations = (x*x, y*y, t*t*z)
        for variables in (axes, (t, u, x, z, y)):
            with self.subTest(variables=variables):
                self.assertEqual(certificate(variables, equations, u*z+t), ["yes"])
                self.assertEqual(certificate(variables, equations, t), ["unknown"])

    def test_product_section_keeps_embedded_associated_primes(self):
        x, y, z, u, t = variables = np.symbols("x y z u t")
        # In I=(x^2,xy,t*z) the multiplier y avoids (x,t) and (x,z)
        # but kills the nonzero class of x on both sections.
        self.assertEqual(certificate(variables, (x*x, x*y, t*z), y), ["unknown"])

    def test_b9_frames_skip_the_actual_split_in_centre_selection(self):
        names = ("x", "y", "u", "z", "a", "b")
        for chart in b9_component_requests():
            with self.subTest(chart=chart["chart"]):
                packet = "\n".join([chart["header"], *chart["rows"]]).encode()
                self.assertEqual(hashlib.sha256(packet).hexdigest(), chart["request_sha256"])
                _, width, count = chart["header"].split()
                variables = tuple(map(np.Symbol, names[:int(width)]))
                *equations, multiplier = (np.from_row(variables, row) for row in chart["rows"])
                self.assertEqual(len(equations), int(count))
                frame = Frame(variables, variables[1:3], equations)
                item = ({}, multiplier, Fraction(1))
                work = ((),)
                state = centres.State(frame, (item,), (), (), (), work, (), ())
                with patch.object(centres, "saturate", side_effect=AssertionError("redundant saturation")):
                    self.assertEqual(list(centres._groups(state)), [(work, [item])])
