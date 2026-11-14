"""Independent checks on the equations exported by the A-series kernel."""
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.chain_backend import resolve_chain
from engine.polynomial import parse


class ChainEquations(unittest.TestCase):
    variables = ["x", "y", "z"]

    def _resolve(self, formula, mode):
        result = resolve_chain(parse(formula, self.variables), self.variables, mode)
        self.assertIsNotNone(result, "Build native/chain.mk before running these checks.")
        return result

    def test_pullbacks_over_coefficient_field(self):
        # Compare sparse polynomials exactly, reducing powers of theta with
        # the declared minimal polynomial of the coefficient field.
        for equation in ["x*y+z^3", "3*y*z-x^4/7", "x^2+y^2+2*z^5",
                         "2*x^2+3*x*y-5*y^2-7*z^8", "x*y+2*y^2-z^5"]:
            for mode in ("intrinsic", "embedded"):
                with self.subTest(equation=equation, mode=mode):
                    result = self._resolve(equation, mode)
                    field = result["coefficient_field"]
                    radical = Fraction(field["minimal_polynomial"].split("(")[1][:-1]) if "generator" in field else 0
                    for chart in result["charts"]:
                        def expand(expression):
                            terms = parse(expression, [*chart["coordinates"], "theta"])
                            reduced = {}
                            for (*coords, d), coeff in terms.items():
                                key = (*coords, d % 2)
                                reduced[key] = reduced.get(key, 0) + coeff * radical**(d//2)
                            return {key: c for key,c in reduced.items() if c}
                        pullback = re.sub(r'[xyz]', lambda m: "("+chart["map"][m[0]]+")", equation)
                        self.assertEqual(expand(pullback), expand(chart["ambient_total_transform"]), chart["id"])
        # Exact classification also handles colliding input/local variable names.
        self.assertIsNotNone(resolve_chain(parse("theta*a+b^3", ["theta","a","b"]),
                                           ["theta","a","b"], "embedded"))

    def test_curve_labels_and_line_center(self):
        intrinsic = self._resolve("x*y+z^5","intrinsic")
        edges = {frozenset(c["vertices"]) for c in intrinsic["dual_complex"]["cells"]}
        self.assertEqual(edges, {frozenset(p) for p in [
            ("E1u","E2u"), ("E2u","E2v"), ("E2v","E1v")]})
        self.assertTrue(all(v["self_intersection"] == -2 for v in intrinsic["dual_complex"]["vertices"]))
        embedded = self._resolve("x*y+z^5","embedded")
        self.assertEqual(len(embedded["charts"]), 7)
        self.assertEqual(embedded["steps"][-1]["type"],"line blowup")
        self.assertEqual(len(embedded["steps"][-1]["center_charts"]),2)
        self.assertEqual([c["dimension"] for c in embedded["dual_complex"]["cells"]], [1,1,1,2])

    def test_embedded_transversality_on_all_coordinate_strata(self):
        # Restrict the binomial to every intersection of exceptional coordinate
        # hyperplanes, including the ambient chart. Its monomials have disjoint
        # supports. A nonzero constant term makes any zero lie on the torus of
        # the other monomial, where its derivative is nonzero. A linear term
        # instead gives a unit derivative. These tests certify SNC in A^3.
        for exponent in range(2, 11):
            result = self._resolve(f"x*y+z^{exponent}", "embedded")
            for chart in result["charts"]:
                coords = chart["coordinates"]
                # An extra variable makes the parser accept constant terms.
                polynomial = parse(f"d*({chart['strict_transform']})", [*coords, "d"])
                powers = [e[:3] for e in polynomial]
                self.assertEqual(len(powers), 2)
                self.assertTrue(all(not (p and q) for p, q in zip(*powers)))
                axes = [coords.index(c["equation"]) for c in chart["exceptional"]]
                for size in range(len(axes)+1):
                    for vanish in combinations(axes, size):
                        terms = [e for e in powers if not any(e[d] for d in vanish)]
                        self.assertTrue(terms and any(sum(e) <= 1 for e in terms),
                                        (exponent, chart["id"], vanish))

    def test_outside_family(self):
        for equation in ["(x+y)^2+z^3", "x*y+z^3+z^4", "x^3+y^5+z^7"]:
            self.assertIsNone(resolve_chain(parse(equation, self.variables), self.variables))


if __name__ == "__main__":
    unittest.main()
