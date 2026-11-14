"""Regression checks for the native classifier and all blowup charts."""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.native_backend import resolve_native, smooth_projective_tangent
from engine.polynomial import parse

def run(source, names, mode, **kwargs):
    sparse = parse(source, names)
    return resolve_native([(p, str(c)) for p, c in sparse.items()], names, mode, **kwargs)

class OrdinaryTests(unittest.TestCase):
    def test_projective_tangent_only_certifies_proven_smooth_sections(self):
        names = ["x","y","z"]
        forms = (
            ("x^3+y^3+z^3+x*y*z/3+x^7", True),
            ("x^2+y+z^3", True),
            ("x^3+y^3+z^3-3*x*y*z+x^7", False),
            ("1000000007*1000000009*(x^2+y^2+z^2+x*y)", False),
            ("x^2+y^2+z^2+x^18446744073709551617", False),
        )
        for equation, expected in forms:
            with self.subTest(equation=equation):
                self.assertEqual(smooth_projective_tangent(len(names), parse(equation, names, bounded=False)), expected)

    def test_four_variable_coordinate_witness(self):
        details = {}
        self.assertIsNone(run("a^2+b^3+c^4+t^5", ["a","b","c","t"], "embedded", diagnostics=details))
        self.assertEqual(details["code"], "singular_tangent_direction")
        self.assertEqual(details["coordinate_index"], 1)
        self.assertEqual(details["point"], [0,1,0,0])
        self.assertEqual(details["tangent_cone"], "a^2")

    def test_exact_jacobian_budget(self):
        details = {}
        self.assertIsNone(run("a^5+b^5+c^5+t^5+a^4*b", ["a","b","c","t"], "embedded", diagnostics=details))
        self.assertEqual((details["degree"], details["rows"], details["columns"], details["column_limit"]),
                         (13,880,560,512))

    def test_inconclusive_modular_test(self):
        # The rational tangent cone is smooth, but scaling by both primes
        # kills the Jacobian after reduction. No singular point can be inferred.
        details = {}
        source = "1000000007*1000000009*(a^2+b^2+c^2+t^2+a*b)"
        self.assertIsNone(run(source, ["a","b","c","t"], "embedded", diagnostics=details))
        self.assertEqual(details["code"], "inconclusive_modular_jacobian")
        self.assertEqual(details["tested_primes"], [1000000007,1000000009])
        self.assertNotIn("point", details)
    def test_all_affine_pullbacks(self):
        source = "2*x*y-z^2+y^3/7"
        result = run(source, ["x","y","z"], "intrinsic")
        self.assertEqual(result["certificate"]["method"], "projective Jacobian rank")
        self.assertEqual(len(result["dual_complex"]["vertices"]), 1)
        for chart in result["charts"]:
            pullback = re.sub("[xyz]", lambda m: "(" + chart["map"][m[0]] + ")", source)
            total = chart["ambient_total_transform"]
            self.assertEqual(parse(pullback, chart["coordinates"]), parse(total, chart["coordinates"]))

    def test_projective_singularity_away_from_coordinate_axes(self):
        self.assertIsNone(run("x^3+y^3+z^3-3*x*y*z", ["x","y","z"], "embedded"))

    def test_smooth_non_diagonal_projective_cubic(self):
        result = run("x^2*y+y^2*z+z^2*x", ["x","y","z"], "embedded")
        self.assertEqual(result["certificate"]["rank"], 15)
        self.assertEqual(result["multiplicity"], 3)

    def test_zero_infinity_and_nonsplit_directions(self):
        source, names = "x*y*(x^2+y^2)", ["x","y"]
        intrinsic = run(source, names, "intrinsic")
        ambient = run(source, names, "embedded")
        self.assertEqual(len(ambient["dual_complex"]["vertices"]), 1)
        dual = intrinsic["dual_complex"]
        self.assertEqual(len(dual["vertices"]), 4)
        self.assertEqual(dual["cells"], [])
        left, right = [set(e["id"] for e in c["exceptional"]) for c in intrinsic["charts"]]
        self.assertEqual(len(left & right), 2)
        self.assertEqual(len(left | right), 4)

    def test_nonreduced_projective_direction_at_infinity(self):
        self.assertIsNone(run("x^2*(x^2+y^2)+y^6", ["x","y"], "embedded"))

    def test_smooth_identity(self):
        result = run("v^7+w/3", ["v","w"], "intrinsic")
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["dual_complex"]["vertices"], [])
        self.assertEqual(result["charts"][0]["map"], {"v":"u1","w":"u2"})

if __name__ == "__main__":
    unittest.main()
