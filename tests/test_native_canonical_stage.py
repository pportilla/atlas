"""The canonical blowup loop in one native call per step against its step-by-step route.

native/canonical_blowup.cpp blows up whole stages and checks their charts,
native/centre_maximum.cpp chooses the centres (the invariant of Bierstone and
Milman, and the procedure for a smooth surface). Both compose the native
operations that engine/local_blowups.py, engine/centres.py and
engine/surface.py call one at a time, which remain the route for rings
beyond the packed kernel. The two routes must give the same charts, centres
and errors.
"""
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from engine import centres, local_blowups as lb, surface
from engine import native_polynomial as np
from engine.native_library import _kernel
from engine.polynomial import parse

CASES = ("(x+y)^2+z^3+x^3", "x^2+(y+z)^3+y^4", "(x*y-z^2)^2+x^5+y^5", "x^2*y+y^4+z^3")


def _stepwise():
    """Patches that send every step to the step-by-step route."""
    stack = ExitStack()
    stack.enter_context(patch.object(lb, "_native_charts", lambda charts: False))
    stack.enter_context(patch.object(centres, "_native_maximum", lambda *args: centres._DECLINED))
    stack.enter_context(patch.object(surface, "_native_step", lambda *args: surface._DECLINED))
    return stack


def _resolution(text):
    symbols = np.symbols("x y z")
    equation = np.from_terms(symbols, parse(text, ["x", "y", "z"]))
    return lb.acampo_over(symbols, [("strict", 1, 1, (equation,))], symbols, dimension=3, rules=("complete", "direct"))


def _outcome(function, *args):
    try:
        return "value", function(*args)
    except Exception as error:  # the same error, with the same message, on both routes
        return "error", type(error).__name__, str(error)


@unittest.skipIf(_kernel() is None, "Build the native kernel with make.")
class NativeCanonicalStageTests(unittest.TestCase):
    def test_resolutions_agree_with_the_step_by_step_route(self):
        for text in CASES:
            with self.subTest(equation=text):
                native = _resolution(text)
                with _stepwise():
                    stepwise = _resolution(text)
                data, strata, charts, steps = native
                self.assertEqual(data, stepwise[0])
                self.assertEqual(strata, stepwise[1])
                self.assertEqual(charts, stepwise[2])
                self.assertEqual(steps, stepwise[3])

    def test_every_centre_choice_agrees_with_the_step_by_step_route(self):
        calls = {"maximum": 0, "surface": 0}
        maximum, step = centres.maximum, surface.step

        def checked_maximum(variables, ambient, dimension, strict, divisors, working, records, births,
                            deadline=None, away=()):
            calls["maximum"] += 1
            args = (variables, ambient, dimension, strict, divisors, working, records, births, None, away)
            mine = _outcome(centres._native_maximum, *args)
            self.assertNotEqual(mine, ("value", centres._DECLINED))
            self.assertEqual(mine, _outcome(centres._maximum_stepwise, *args))
            return maximum(*args[:8], deadline, away)

        def checked_step(variables, ambient, strict, divisors, working, deadline=None, away=()):
            calls["surface"] += 1
            mine = _outcome(surface._native_step, variables, ambient, strict, divisors, working, away)
            self.assertNotEqual(mine, ("value", surface._DECLINED))
            self.assertEqual(mine, _outcome(surface._step_stepwise, variables, ambient, strict, divisors, working,
                                            None, away))
            return step(variables, ambient, strict, divisors, working, deadline, away)

        with patch.object(lb, "maximum", checked_maximum), patch.object(surface, "step", checked_step):
            for text in CASES:
                _resolution(text)
        self.assertGreater(calls["maximum"], 5)
        self.assertGreater(calls["surface"], 5)

    def test_rings_beyond_the_packed_kernel_take_the_step_by_step_route(self):
        variables = np.symbols("u0:33")
        x, y = variables[:2]
        chart = lb.Chart(variables, (), (("strict", 1, 1, (x*y,)),), (x, y), (), (("x", x), ("y", y)), ())
        over, inside = lb._failures(chart, 33)
        self.assertEqual(inside, {("strict",)})
        self.assertEqual(set(over), {("strict",)})
        children = lb._blown_up([chart], [(x, y)], "E1", 1, 33, 31, None)
        self.assertEqual(len(children), 2)
        for child in children:
            strict, exceptional = child.components
            self.assertEqual(exceptional[:3], ("E1", 2, 2))
            mapping = dict(child.coordinates)
            self.assertEqual(strict[3][0]*exceptional[3][0]**2, mapping["x"]*mapping["y"])

    def test_retained_charts_keep_the_older_centres_that_still_meet_them(self):
        x, y, z = variables = np.symbols("x y z")
        chart = lb.Chart(variables, (), (("strict", 1, 1, (x**2 + y**3 + z**5,)),), variables, (),
                         tuple((v.name, v) for v in variables), (), ((x, y - 1, z),))
        native = lb._blown_up([chart], [(x, y, z)], "B1", 1, 3, 0, None, True)
        with _stepwise():
            stepwise = lb._blown_up([chart], [(x, y, z)], "B1", 1, 3, 0, None, True)
        self.assertEqual(native, stepwise)
        self.assertEqual(native[-1].removed, ((x, y - 1, z), (x, y, z)))


if __name__ == "__main__":
    unittest.main()
