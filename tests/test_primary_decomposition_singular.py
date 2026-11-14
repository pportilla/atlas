"""Check intersections, primaryness and associated primes of rational decompositions."""
import shutil
import unittest

from benchmarks.primary_decomposition import Case, compare_in_singular, corpus
from engine.algebra import dispatch


def decompose(names, generators, order="grevlex"):
    ring = {"variables": list(names), "order": order}
    result = dispatch({"operation": "primary-decomposition", "ring": ring,
                       "operands": {"ideal": list(generators)}, "compute": "cpu"})["result"]
    assert result["type"] == "primary-decomposition"
    assert result["ring"] == dict(ring, field="QQ")
    return result["components"]


class PrimaryDecompositionTests(unittest.TestCase):
    def test_unit_zero_and_scalar_rings(self):
        for names in ((), ("x", "y")):
            for generators in ([], ["0", "0"]):
                self.assertEqual(decompose(names, generators), [{"primary": [], "prime": []}])
            self.assertEqual(decompose(names, ["0", "-2/7"]), [])
        self.assertEqual(decompose(["x"], ["x", "x-1"]), [])

    def test_wide_sparse_embedded_component(self):
        # (x^N, x*y) has the simple line (x) and an embedded origin.
        n = 2**64 + 7
        parts = decompose(("x", "y"), [f"x^{n}", "x*y"])
        self.assertEqual({frozenset(p["prime"]) for p in parts},
                         {frozenset(["x"]), frozenset(["x", "y"])})
        # Do not require a particular choice of the embedded primary component.
        embedded, = [p for p in parts if set(p["prime"]) == {"x", "y"}]
        request = {"operation": "membership", "ring": {"variables": ["x", "y"]},
                   "operands": {"ideal": embedded["primary"]}, "compute": "cpu"}
        for power, expected in ((n, True), (n - 1, False)):
            request["operands"]["polynomial"] = f"x^{power}"
            self.assertIs(dispatch(request)["result"]["value"], expected)

    def test_exact_rational_coefficients(self):
        a = 2**128 + 1
        parts = decompose(("x",), [f"(x-1/{a})^2*(x+1/{a})"], order="lex")
        self.assertEqual({tuple(p["prime"]) for p in parts}, {(f"x-1/{a}",), (f"x+1/{a}",)})
        doubled, = [p for p in parts if p["prime"] == [f"x-1/{a}"]]
        expected = dispatch({"operation": "groebner", "ring": {"variables": ["x"], "order": "lex"},
                             "operands": {"ideal": [f"(x-1/{a})^2"]}})["result"]["generators"]
        self.assertEqual(doubled["primary"], expected)

    @unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
    def test_qq_components_against_singular(self):
        # In the first case neither coordinate alone separates the two generic
        # primes over QQ(t). The second needs a power of the exceptional factor
        # to recover its embedded multiplicity.
        cases = [Case("coupled_quadratics", ("t", "x", "y"), ("x^2-t", "(y^2-t)^2")),
                 Case("thick_exceptional", ("x", "y"), ("(x-y)^2", "(x-y)*(y-1)^3")),
                 Case("same_monomial_prime", ("x", "y"), ("x^4", "x^3*y^2", "x*y^4", "y^6")),
                 *corpus()]
        for order in ("lex", "grevlex"):
            for c in cases:
                with self.subTest(case=c.name, order=order):
                    components = decompose(c.variables, c.generators, order)
                    # Equality of ideals is order independent. Check both
                    # Atlas output orders in the oracle's chosen input order.
                    compare_in_singular(c, components, method="primdecSY")
