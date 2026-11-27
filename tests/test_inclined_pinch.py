"""A line of pinch singularities that no single translated toric fan resolves."""
from os import environ
import unittest
from unittest.mock import patch

from engine import native_library, toric_backend
from engine.kernel_json import kernel_json
from engine.polynomial import encode_sparse_row, parse, parse_expression
from engine.resolve import resolve


class InclinedPinch(unittest.TestCase):
    names = list("xyz")
    example = "(x+y+z)^2-z*(x-y)^2"

    def setUp(self):
        if not hasattr(native_library._kernel(), "atlas_inclined_toric_obstruction_abi"):
            self.skipTest("Build the native toric obstruction certificate.")

    def certificate(self, source):
        p = parse(source, self.names)
        return kernel_json("atlas_resolution_candidate", "inclined_toric_obstruction",
                           width=3, row=encode_sparse_row(p))

    def test_verified_line_and_the_old_toric_proposal(self):
        for source in (self.example, "3*(x+2*y)^2/7+5*(x-3*z)^2*(y-z)/2"):
            with self.subTest(source=source):
                self.assertTrue(self.certificate(source))
                p = parse(source, self.names)
                for mode in ("embedded", "intrinsic"):
                    self.assertIsNone(toric_backend.resolve_toric(p, self.names, mode, compute="cpu"))

    def test_inconclusive_factors_directions_and_caps(self):
        for source in ("x^2-z*y^2", "(x+y)^2-z*(x-y)^2", "(x+y+z)^2-(2*x+z)*(x-y)^2",
                       self.example+"+x^2", self.example+"+x*y*z", self.example+"+x^4",
                       self.example+"+z", "(x+y+z)^2-(x+y+z)^3",
                       str(2**90)+"*(x+y+z)^2-z*(x-y)^2"):
            with self.subTest(source=source):
                self.assertFalse(self.certificate(source))
        with patch.dict(environ, {"ATLAS_LINEAR_ALGEBRA": "0"}):
            self.assertFalse(self.certificate(self.example))

    def test_only_the_preliminary_toric_trial_is_skipped(self):
        request = dict(polynomial=self.example, variables=self.names, mode="embedded", compute="cpu")
        with patch.object(toric_backend, "resolve_toric", wraps=toric_backend.resolve_toric) as trial:
            with patch.dict(environ, {"ATLAS_LINEAR_ALGEBRA": "0"}):
                before = resolve(request)
            self.assertEqual(trial.call_count, 1)
            trial.reset_mock()
            after = resolve(request)
            trial.assert_not_called()
        self.assertEqual(after["engine"], "verified-ideal-blowup")
        before.pop("elapsed_ms")
        after.pop("elapsed_ms")
        self.assertEqual(before, after)


class SingularMonomialCurve(unittest.TestCase):
    example = "-4*x^3*z+3*x^2*y^2+6*x*y*z-4*y^3-z^2"

    def test_weighted_curve_certificate(self):
        def certified(source):
            row = encode_sparse_row(parse(source, list("xyz")))
            return kernel_json("atlas_resolution_candidate", "singular_torus_curve", width=3, row=row)

        self.assertTrue(certified(self.example))
        self.assertTrue(certified(self.example.replace("z", "(-z)")))
        space_curve = "(x*y-z^2)^2+(x^2-y^3)^2"
        self.assertTrue(certified(space_curve))
        self.assertFalse(certified(space_curve + "+z^6"))
        self.assertFalse(certified("(" + space_curve + ")^2"))
        for f in (self.example + "+x^7", "(x*y-z)^2", "(x-y)^2+(x-z)^2",
                  self.example + "+x^65", self.example + "+z^2/3"):
            with self.subTest(f=f):
                self.assertFalse(certified(f))
        with patch.dict(environ, {"ATLAS_LINEAR_ALGEBRA": "0"}):
            self.assertFalse(certified(self.example))

    def test_nonreduced_exceptional_section_and_its_fibre(self):
        def chart(divisor, strict, maps=("x", "y", "z")):
            def rows(sources):
                return [encode_sparse_row(parse_expression(f, list("xyz"))) for f in sources]
            return dict(width=3, maps=rows(maps), owned=[], ambient=[], holes=[], theta=-1, parent=-1,
                        repair=[], components=[["strict", rows([strict]), True], ["B1", rows([divisor]), False]])

        def curve(*charts):
            return kernel_json("atlas_chart_geometry", "single_curve", charts=charts)

        active = chart("x^2-y", "-z^2+4*x*z-4*y")
        away = chart("x", "1+x")
        self.assertEqual(curve(active, away)["vertices"], [{"id": "B1", "dimension": 1}])
        self.assertIsNone(curve(away))
        self.assertIsNone(curve(active, active))
        self.assertIsNone(curve(chart("x*y", "z")))
        self.assertIsNone(curve(chart("x^2-y^3", "z")))
        self.assertIsNone(curve(chart("x-1", "z")))
        self.assertIsNone(curve(dict(active, holes=[["1 1 0 0"]])))

    def test_full_intrinsic_record_matches_original_worker_and_gluing(self):
        q = dict(variables=list("xyz"), polynomial=self.example, mode="intrinsic", compute="cpu")
        with patch.dict(environ, {"ATLAS_LINEAR_ALGEBRA": "0"}):
            original = resolve(q)
        current = resolve(q)
        self.assertEqual(current["engine"], "canonical-blowups")
        for record in (original, current):
            record.pop("elapsed_ms")
        self.assertEqual(original, current)
