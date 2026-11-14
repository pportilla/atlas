"""Independent normalized-ring and map checks over QQ, including multiple components."""
import shutil
import unittest

from benchmarks.normalization import Case, corpus, normalize, verify


class NormalizationTests(unittest.TestCase):
    def test_scalar_ring_and_empty_scheme(self):
        for gens, expected in (((), 1), (("0",), 1), (("2/3",), 0)):
            result = normalize(Case("scalar", (), gens))
            self.assertTrue(result["input_reduced"])
            self.assertEqual(len(result["components"]), expected)
            for component in result["components"]:
                self.assertEqual(component["images"], [])
                self.assertEqual(component["relations"], [])
                self.assertEqual(component["fractions"], [])

    def test_wide_coefficients_and_exponents(self):
        n = 2**64 + 1
        result = normalize(Case("thick_line", ("x","y"), (f"x^{n}",)))
        self.assertFalse(result["input_reduced"])
        self.assertEqual([c["prime"] for c in result["components"]], [["x"]])
        wide = normalize(Case("wide_cusp", ("x","y"), (f"y^2-x^{n}",)))
        component, = wide["components"]
        parameter, = component["ring"]["variables"]
        self.assertEqual(component["relations"], [])
        self.assertEqual(component["images"], [f"{parameter}^2", f"{parameter}^{n}"])
        self.assertEqual(component["fractions"], [{"numerator": "y", "denominator": f"x^{n//2}"}])
        a = 2**160 + 1
        result = normalize(Case("rational_point", ("x",), (f"x-1/{a}",)))
        self.assertTrue(result["input_reduced"])
        self.assertEqual([c["prime"] for c in result["components"]], [[f"x-1/{a}"]])

    @unittest.skipUnless(shutil.which("Singular"), "Singular is required for the independent comparison")
    def test_normalized_subalgebras_and_maps(self):
        for case in (*corpus(),
                     Case("nonreduced_cusp", ("x","y"), ("(y^2-x^3)^2",)),
                     Case("embedded_at_origin", ("x","y"), ("x^2","x*y")),
                     Case("unit", ("x","y"), ("x","x-1")),
                     Case("polynomial_algebra", ("x","y"), ())):
            with self.subTest(case=case.name):
                verify(case, normalize(case))

    @unittest.skipUnless(shutil.which("Singular"), "Singular is required for the independent comparison")
    def test_both_orders_on_singular_space_curve(self):
        for order in ("lex", "grevlex"):
            c = Case("space_curve", ("x","y","z"), ("y^2-x*z","y*z-x^3","z^2-x^2*y"), order)
            with self.subTest(order=order):
                verify(c, normalize(c))

    @unittest.skipUnless(shutil.which("Singular"), "Singular is required for the independent comparison")
    def test_weighted_content_maps(self):
        for order in ("lex", "grevlex"):
            cases = (
                Case("weighted_factors", ("x", "s", "y"),
                     ("y^3+x*(x+s)^5*(s+1)^2*y+x*(x+s)^7*(s+1)^3",), order),
                Case("nonmonic", ("x", "z", "y"), ("x*y^3+(x+z)^5*y+(x+z)^7",), order),
                Case("constant_blocks_content", ("x", "y"), ("y^3+x*y+1",), order),
                Case("name_collision", ("n1", "y"), ("y^4-n1^3*y^2+n1^6*y-n1^7",), order),
            )
            for case in cases:
                with self.subTest(order=order, case=case.name):
                    verify(case, normalize(case))
