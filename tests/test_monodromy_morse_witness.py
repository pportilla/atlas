"""Certified nearby Morse strata for poles invisible in the origin zeta function."""
from fractions import Fraction
import unittest

from engine.app import calculate_resolution
from engine.monodromy_morse_witness import nearby_morse_witness
from engine.session import new_session


class MorseWitnessTests(unittest.TestCase):
    def test_detects_order_two_on_generic_critical_line_and_plane(self):
        for variables, f, determinant in [
                ("xyzt", "x^2+y^2+t*z^2+z^3", 8),
                ("xyztu", "x^2+y^2+(t+u)*z^2+z^3+x*y*z", 16)]:
            with self.subTest(f=f):
                session = new_session(calculate_resolution)
                names = ",".join(repr(name) for name in variables)
                list(session.run(f"{','.join(variables)} = coordinates({names}, coeffs=QQ)\n"
                                 f"C = monodromy_check({f}, search='quick')"))
                C = session.variables["C"]
                self.assertEqual((C["status"], C["monodromy_zeta"]["expression"]), ("verified", "1"))
                pole = next(p for p in C["poles"] if p["value"] == "-3/2")
                self.assertEqual((pole["cyclotomic_exponent"], pole["witness"]), (0, "nearby transverse Morse"))
                witness = pole["transverse"]
                self.assertEqual(witness["coordinates"], tuple("xyz"))
                self.assertEqual(witness["parameters"], tuple(variables[3:]))
                self.assertEqual(witness["parameter_values"], (1,) * (len(variables) - 3))
                self.assertEqual(witness["transverse_hessian_determinant"], determinant)
                self.assertEqual(witness["cyclotomic_exponent"], -1)

    def test_rational_cross_terms_and_zero_diagonal_pivot(self):
        witness = nearby_morse_witness("x*y/2+t*z^2/3", list("xyzt"))
        self.assertEqual(witness["coordinates"], tuple("xyz"))
        self.assertEqual(witness["transverse_hessian_determinant"], Fraction(-1, 6))

    def test_specialization_cannot_replace_critical_locus_or_parity_checks(self):
        for f in ["x^2+y^2+t*z^2+z^3+x*t*(t-1)",
                  "x^2+y^2+t*z^2+z^3+t^2*(t-1)",
                  "x^2+y^2+t*z^2+z^2*u+u^3",  # degenerate transverse Hessian
                  "x^2+y^2+(t-1)*z^2+z^3"]:  # the chosen Hessian specialization vanishes
            with self.subTest(f=f):
                self.assertIsNone(nearby_morse_witness(f, list("xyztu" if "u" in f else "xyzt")))
        self.assertIsNone(nearby_morse_witness("x^2+y^2+t*z^2+z^2*u^2+u^2", list("xyztu")))


if __name__ == "__main__":
    unittest.main()
