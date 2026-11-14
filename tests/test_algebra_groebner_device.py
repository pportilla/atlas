"""Opt-in physical-device checks for the Phase 1 rational ideal handlers.

Run with ATLAS_TEST_ALGEBRA_GPU=1 after building the GPU kernels.
"""
from fractions import Fraction
import os
import unittest

from benchmarks.groebner_hard import katsura
from engine import native_polynomial as poly
from engine.algebra_execution import algebra_execution
from engine.algebra_groebner import ideal_membership, normal_form, reduced_basis


@unittest.skipUnless(os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1",
                     "Set ATLAS_TEST_ALGEBRA_GPU=1 to require a physical GPU")
class AlgebraGroebnerDevice(unittest.TestCase):
    def test_certified_basis_and_reusable_reductions(self):
        names, equations = katsura(6)
        variables = tuple(poly.Dummy(name) for name in names)
        # The invertible change u6 -> 2^192*u6 grows the reconstructed
        # coefficients until enough replay primes fit a device-sized batch.
        scale = 2**192
        generators = [
            poly.from_terms(variables, {powers: coefficient * scale**powers[-1]
                                        for powers, coefficient in equation.items()})
            for equation in equations
        ]
        with algebra_execution("cpu"):
            cpu = reduced_basis(generators, variables, order="grevlex")
        with algebra_execution("gpu") as execution:
            gpu = reduced_basis(generators, variables, order="grevlex")
            stages = execution.report()["stages"]
        self.assertEqual(len(cpu.exprs), 41)
        self.assertEqual(gpu.exprs, cpu.exprs)

        for basis in (cpu, gpu):
            self.assertTrue(all(ideal_membership(f, basis) for f in generators))
            self.assertFalse(ideal_membership(variables[0], basis))
            self.assertEqual(normal_form(generators[0] + Fraction(7, 11), basis),
                             Fraction(7, 11))

        physical = [s for s in stages if s["backend"] in ("opencl", "triton")]
        self.assertTrue(any(s["operation"] == "groebner_trace" and s["pairs"] > 0
                            for s in physical), stages)
        self.assertTrue(any(s["operation"] == "groebner_replay" and s["primes"] > 0
                            for s in physical), stages)
        self.assertTrue(any(s["operation"] == "groebner_certification" and s["backend"] == "cpu"
                            and s["bases"] > 0 and s["attempts"] > 0 for s in stages), stages)

        # This reduction has an independent oracle: a term survives precisely
        # when its exponents are below both powers in the monomial ideal.
        x, y = variables[:2]
        f = (1 + x + y)**32 / 7
        expected = poly.from_terms((x, y), {e: c for e, c in poly.terms((x, y), f).items()
                                            if e[0] < 8 and e[1] < 9})
        with algebra_execution("cpu"):
            monomial_basis = reduced_basis((x**8, y**9), (x, y), order="lex")
            self.assertEqual(normal_form(f, monomial_basis), expected)
        with algebra_execution("gpu") as reduction:
            self.assertEqual(normal_form(f, monomial_basis), expected)
            self.assertTrue(ideal_membership(f - expected, monomial_basis))
            stages = reduction.report()["stages"]
        witnesses = [s for s in stages if s["operation"] == "monomial_normal_form"
                     and s["backend"] in ("opencl", "triton") and s["host_verified"]]
        self.assertTrue(any(s["terms"] == 561 and s["removed"] == 489 for s in witnesses), stages)
        self.assertTrue(any(s["terms"] == s["removed"] == 489 for s in witnesses), stages)
