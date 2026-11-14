"""Membership of defining equations and rational constants on open frames."""
from fractions import Fraction
import gzip
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from engine import centres, native_polynomial as np
from engine.centre_frame import Frame
from engine.native_library import _MODE, ideal_request


def b9_restricted_frame():
    """First restricted maximal-contact frame after the B9 graph certificate."""
    path = Path(__file__).parent / "fixtures/b9-restricted-frame.json.gz"
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    return data


def native_check(width, equations, query, nonempty=False):
    return ideal_request(f"frame_membership {width} {len(equations)} {int(nonempty)}",
                         [*equations, query])


@unittest.skipUnless(_MODE == "native", "Exercises native membership on rational frames.")
class FrameMembership(unittest.TestCase):
    def test_b9_defining_equation_without_basis(self):
        packet = b9_restricted_frame()
        variables = tuple(map(np.Symbol, packet["variables"]))
        equations = tuple(np.from_row(variables, r) for r in packet["level_rows"])
        units = tuple(np.from_row(variables, r) for r in packet["units"])
        frame = Frame(variables, tuple(map(np.Symbol, packet["coords"])), equations, units)
        # Reconstruct through separately decoded terms in the opposite order.
        row = ";".join(reversed(packet["level_rows"][-1].split(";")))
        self.assertEqual(row.count(";") + 1, 1136)
        query = np.from_row(variables, row)
        with patch.object(frame, "_reduced_terms", side_effect=AssertionError("unnecessary basis")):
            self.assertTrue(frame.vanishes(query))

    def test_exact_equality_with_wide_exponents_and_coefficients(self):
        exponent = 2**83 + 23
        coefficient = Fraction(2**223+5, 2**199-1)
        terms = [f"{coefficient} {exponent} 0", "7 0 1"]
        self.assertEqual(native_check(2, [";".join(terms)], ";".join(terms[::-1])), ["yes"])
        self.assertEqual(native_check(2, [";".join(terms)], f"{coefficient} 0 {exponent};7 1 0"),
                         ["unknown"])

    def test_scalar_requires_nonemptiness(self):
        for flag, expected in ((False, "scalar"), (True, "no")):
            for query in ("1 0 0", "-7/8 0 0"):
                with self.subTest(flag=flag, query=query):
                    self.assertEqual(native_check(2, ["1 1 0"], query, flag), [expected])
        self.assertEqual(native_check(2, ["1 1 0"], "", False), ["yes"])

    def test_nonempty_working_slice_proves_the_frame_nonempty(self):
        x, y = variables = np.symbols("x y")
        for extra_unit in (False, True):
            frame = Frame(variables, (x,), (y-x*x,), (x+1,))
            if extra_unit:
                self.assertFalse(frame.inside(x+2, (x-1,)))
            else:
                self.assertFalse(frame.empty(x-1))
            # The point (1,1) satisfies every condition in either check.
            with patch("engine.centre_frame.frame_locus", side_effect=AssertionError("retested nonemptiness")), \
                 patch.object(frame, "_reduced_terms", side_effect=AssertionError("unnecessary basis")):
                self.assertFalse(frame.vanishes(1))

    def test_contact_keeps_a_witness_on_its_defining_equation(self):
        x, y = variables = np.symbols("x y")
        # The known working point for y=0, x=0 meets the new contact equation x.
        item = ({}, x, Fraction(1))
        state = centres.State(Frame(variables, (x,), (y,)), (item,), (), (), (x,), ((),), (), ())
        contact, _ = next(centres._contact(state, centres._Context({}, (), (), None)))
        with patch("engine.centre_frame.frame_locus", side_effect=AssertionError("lost witness")), \
             patch.object(contact.frame, "_reduced_terms", side_effect=AssertionError("unnecessary basis")):
            self.assertFalse(contact.frame.vanishes(1))

    def test_contact_hit_off_the_equation_is_insufficient(self):
        x, y = variables = np.symbols("x y")
        frame = Frame(variables, (x,), (y,), (x,))
        # The parent test finds x=1, but x=0 has no point on D(x).
        state = centres.State(frame, (({}, x, Fraction(1)),), (), (), (), ((x-1,),), (), ())
        contact, _ = next(centres._contact(state, centres._Context({}, (), (), None)))
        self.assertTrue(contact.frame.vanishes(1))

    def test_scalar_proof_reused_for_collection_and_coordinate_orders(self):
        x, y, z = variables = np.symbols("x y z")
        frame = Frame(variables, (x, y), (z-x*y,), (x-2,))
        scalar = Fraction(-7, 11)
        # Both coordinate divisors meet this open graph, including at (0,0,0).
        self.assertFalse(frame.empty(x, y))
        self.assertFalse(frame.vanishes(scalar))
        state = centres.State(frame, (({}, scalar, Fraction(1)),), (), (), (), ((x, y),), (), ())
        with patch.object(frame, "_implicit_input", side_effect=AssertionError("redundant implicit basis")), \
             patch.object(centres, "saturate", side_effect=AssertionError("redundant split")):
            self.assertEqual(list(centres._groups(state)), [(((x, y),), list(state.coll))])
            self.assertEqual(centres._remove_coordinate_powers(frame, scalar, (y, x)), ([0, 0], scalar))

    def test_unchecked_witness_locus_does_not_certify_a_restriction(self):
        x, y = variables = np.symbols("x y")
        frame = Frame(variables, (x,), (y,), (x,))
        self.assertFalse(frame.empty(x-1))
        # The proposed witness locus would lie on x=0, but no successful
        # nonempty check establishes that it has any point on the frame.
        child = frame.restricted(x, x, 1, witness_loci=((x,),))
        self.assertTrue(child.vanishes(1))

    def test_restriction_can_make_a_nonempty_frame_empty(self):
        x, y = variables = np.symbols("x y")
        frame = Frame(variables, (x,), (y,), (x,))
        self.assertFalse(frame.empty(x-1))
        self.assertFalse(frame.vanishes(1))
        # Restricting to x=0 removes every point on the open x!=0.
        child = frame.restricted(x, x, 1)
        self.assertTrue(child.vanishes(1))

    def test_empty_slice_does_not_imply_empty_frame(self):
        x, y = variables = np.symbols("x y")
        frame = Frame(variables, (x,), (y-x*x,), (x,))
        self.assertTrue(frame.empty(x))
        self.assertFalse(frame.vanishes(1))
        # Non-generator membership still uses the ordinary exact algorithm.
        self.assertTrue(frame.vanishes(y*y-x**4))
        self.assertFalse(frame.vanishes(x+y))
