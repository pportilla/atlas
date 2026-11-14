"""Working ideals in the canonical resolution's centre selection."""
from fractions import Fraction
import unittest
from unittest.mock import patch

from engine import centres
from engine.blowup_ideal import ideal_basis
from engine.centres import Frame, State, _groups
from engine.native_polynomial import symbols
from engine.origin_backend import resolve_origin
from engine.polynomial import parse


def grouped(frame, working, locus=(), functions=()):
    collection = tuple(({}, f, Fraction(1)) for f in functions)
    state = State(frame, collection, (), (), locus, working, (), ())
    return list(_groups(state))


class CentreWorkingPieces(unittest.TestCase):
    def test_equivalent_presentations_on_the_locus_keep_the_original_representative(self):
        x, y, z = symbols("x y z")
        frame = Frame((x, y, z), (x, y), (z-x,))
        thick = x**(2**33 + 1)
        # On this frame and y=0 the third equation is a unit multiple of
        # the first. The ideal (x) has the same support, but another thickness.
        working = ((thick,), (thick,), ((2**101 + 3)*thick + y + z-x,), (x,), (x-1,))
        self.assertEqual(grouped(frame, working, locus=(y,)),
                         [(((thick,), (x,), (x-1,)), [])])

    def test_saturation_splits_branches_and_discards_pieces_outside_the_open(self):
        x, z, t = symbols("x z t")
        frame = Frame((x, z, t), (x,), (t*(t-1), z-t), units=(x,))
        # The two disjoint lines have t=0 and t=1. Two generators of
        # the support lead to the very same off-branch at each split.
        functions = (t,) * 8
        with patch.object(centres, "saturate", wraps=centres.saturate) as saturations:
            groups = grouped(frame, ((x-3,), (x,)), functions=functions)
        # Both surviving branches have one working ideal. Saturations per
        # new function stay bounded even though the off-branch has two generators.
        self.assertLessEqual(saturations.call_count, 8 * len(functions))
        self.assertEqual(len(groups), 2)
        branches = {}
        for working, collection in groups:
            self.assertEqual(len(working), 1)
            branches[tuple(item[1] for item in collection)] = ideal_basis(
                frame.variables, (*frame.rows, *working[0]))
        self.assertEqual(branches, {
            (): ideal_basis(frame.variables, (x-3, z, t)),
            functions: ideal_basis(frame.variables, (x-3, z-1, t-1)),
        })

    def test_triple_tangent_cone_resolves_by_the_expected_canonical_centres(self):
        names = ["x", "y", "z"]
        polynomial = parse("x*y*z+x^4+y^4+z^4", names)
        record = resolve_origin(polynomial, names, "intrinsic")
        self.assertIsNotNone(record)
        self.assertEqual((record["status"], record["engine"]), ("resolved", "canonical-blowups"))
        self.assertEqual([step["center"] for step in record["steps"]], ["points", "points", "curve"])
        self.assertEqual(record["steps"][0]["invariant"], ["3", "0", "1", "0", "1", "0", "oo"])
        self.assertEqual(len(record["charts"]), 20)
