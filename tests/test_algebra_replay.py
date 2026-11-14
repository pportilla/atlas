"""Replay the shell and script forms emitted by the public algebra endpoint."""

import json
import shlex
import subprocess
import unittest
from unittest.mock import patch

from engine import native_polynomial as native
from engine.algebra import calculate_algebra
from engine.algebra_replay import make_replay
from tests.test_cli_session import ROOT


class AlgebraReplayTests(unittest.TestCase):
    def check_replay(self, names, operation, operands, *, fallback=False):
        request = dict(ring=dict(variables=names, order="lex"), operation=operation,
                       operands=operands, compute="cpu")
        status, record = calculate_algebra(request)
        self.assertEqual(status, 200, record)
        replay = record["replay"]
        self.assertEqual("algebra_request(" in replay["script"], fallback)
        for command, stdin in ((shlex.split(replay["command"]), None),
                               (["./atlas", "run", "-"], replay["script"])):
            proc = subprocess.run(command, cwd=ROOT, input=stdin, text=True,
                                  capture_output=True, timeout=15)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            output = json.loads(proc.stdout)
            self.assertEqual(output["result"], record["result"])
            self.assertEqual(output["execution"]["requested"], "cpu")
        return replay

    def test_argument_shaped_polynomial_and_exact_fractions(self):
        # '--gen' means minus minus the variable 'gen' in the polynomial grammar.
        replay = self.check_replay(["gen", "y"], "normal-form",
                                   dict(ideal=["--gen"], polynomial="-gen+y/2"))
        self.assertIn("--gen=--gen", shlex.split(replay["command"]))

    def test_ideals_with_only_scalar_generators(self):
        self.check_replay([], "normal-form", dict(ideal=[], polynomial="-2/3"))
        self.check_replay(["x"], "intersection", dict(ideal=["1/2"], other_ideal=[]))

    def test_elimination_uses_named_coordinates(self):
        self.check_replay(["y", "x"], "eliminate", dict(ideal=["x+y", "y-1"], variables=["y"]))

    def test_gcd_preserves_constant_ambient_ring_and_reserved_names(self):
        self.check_replay(["x", "y"], "gcd", {"polynomial": "-1/2", "other_polynomial": "0"})
        self.check_replay(["gcd"], "gcd", {"polynomial": "gcd^2", "other_polynomial": "-gcd"},
                          fallback=True)

    def test_conflicting_names_and_large_expansion(self):
        self.check_replay(["class", "I", "result"], "membership",
                          dict(ideal=[], polynomial="class+I+result"), fallback=True)
        self.check_replay(["x"], "normal-form", dict(ideal=["1"], polynomial="10^4500"), fallback=True)

    def test_provably_long_lines_skip_all_polynomial_formatting(self):
        x, y = native.Symbol("x"), native.Symbol("y")
        power = (1+x+y)**25  # 351 terms
        short = (1+x+y)**18  # 190 terms per GCD operand
        cases = [
            ("normal-form", {"ideal": ["1"], "polynomial": "(1+x+y)^25"},
             {"ideal": [native.coerce(1)], "polynomial": power}),
            ("gcd", {"polynomial": "(1+x+y)^18", "other_polynomial": "(1+x+y)^18"},
             {"polynomial": short, "other_polynomial": short}),
            ("intersection", {"ideal": ["x"], "other_ideal": ["y"] * 215},
             {"ideal": [x], "other_ideal": [y] * 215}),
        ]
        for operation, operands, prepared in cases:
            with self.subTest(operation=operation):
                request = dict(operation=operation, operands=operands, compute="cpu",
                               ring=dict(field="QQ", variables=["x", "y"], order="lex"))
                with patch.object(native.Expr, "__str__", side_effect=AssertionError("formatted operand")):
                    replay = make_replay(request, prepared)
                self.assertIn("algebra_request(", replay["script"])

    def test_short_argument_list_on_each_of_two_lines_keeps_typed_replay(self):
        # Each ideal fits on its own line. The combined length exceeds the cutoff.
        rows = ["x"] * 125
        self.check_replay(["x"], "intersection", {"ideal": rows, "other_ideal": rows})
