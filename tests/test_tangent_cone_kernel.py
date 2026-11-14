"""Tangent initial ideals, directional coordinates and exact associated graded series."""

from itertools import combinations
from math import comb
from random import Random
import unittest
from unittest.mock import patch

from engine import native_polynomial as np
from engine import tangent_cone
from engine.polynomial import parse_expression


def calculate(ideal, *, series=False, names=("x", "y"), order="grevlex", point=None):
    names = list(names)
    coords = tuple(map(np.Symbol, names))
    operands = {"ideal": [np.from_terms(coords, parse_expression(text, names)) for text in ideal]}
    if point is not None:
        operands["point"] = point
    op = "tangent-cone-hilbert-series" if series else "tangent-cone"
    return tangent_cone.handle(op, {"field": "QQ", "variables": names, "order": order}, operands)


def coefficients(series):
    return {r["degree"]: r["coefficient"] for r in series["numerator"]}


class TangentConeKernelTests(unittest.TestCase):
    def test_tangent_scheme_detects_cancellation_and_nilpotents(self):
        # The two given generators have identical linear part x. Modulo the
        # first row x=-y^2, their difference x^2 supplies the further relation y^4.
        for order in ("lex", "grevlex"):
            ideal = ["x+y^2", "x+y^2+x^2"]
            cone = calculate(ideal, order=order)
            self.assertCountEqual(cone["generators"], ["x", "y^4"])
            self.assertEqual((cone["dimension"], cone["multiplicity"]), (0, 4))
            self.assertEqual(cone["coordinate_system"], "tangent")
            series = calculate(ideal, order=order, series=True)
            self.assertEqual(coefficients(series), {0: 1, 1: -1, 4: -1, 5: 1})
            self.assertEqual(series["denominator_power"], 2)

    def test_shifted_repeated_tangent_line_is_a_quadratic_form(self):
        p = ["2/4", "-6/9"]
        f = "(x-1/2+y+2/3)^2+(y+2/3)^3"
        for order in ("lex", "grevlex"):
            cone = calculate([f], point=p, order=order)
            self.assertEqual(cone["scope"], "point")
            self.assertEqual(cone["point"], ["1/2", "-2/3"])
            self.assertEqual(cone["generators"], ["x^2+2*x*y+y^2"])
            self.assertEqual((cone["dimension"], cone["multiplicity"]), (1, 2))
            series = calculate([f], point=p, series=True, order=order)
            self.assertEqual(coefficients(series), {0: 1, 2: -1})
            self.assertEqual(calculate([f])["generators"], ["1"])

    def test_unit_zero_and_no_variables(self):
        for names in ((), ("x",), ("x", "y", "z")):
            w = len(names)
            for ideal, dim, mult, forms, num in ((["0"], w, 1, [], {0: 1}), (["7"], -1, 0, ["1"], {})):
                with self.subTest(ideal=ideal, names=names):
                    cone = calculate(ideal, names=names)
                    self.assertEqual((cone["dimension"], cone["multiplicity"], cone["generators"]), (dim, mult, forms))
                    self.assertNotIn("point", cone)
                    series = calculate(ideal, names=names, series=True, point=["0"]*w)
                    self.assertEqual(series["scope"], "point")
                    self.assertEqual(series["point"], ["0"]*w)
                    self.assertEqual((series["denominator_power"], coefficients(series)), (w, num))

    def test_embedded_component_contributes_to_the_graded_series(self):
        # The quotient has basis y^i, xy^i, x^2. Its graded dimensions are
        # 1, 2, 3, 2, 2, ... although its top-dimensional multiplicity is 2.
        series = calculate(["x^3", "x^2*y", "x^3*y^7"], series=True)
        num = coefficients(series)
        self.assertEqual(num, {0: 1, 3: -2, 4: 1})
        self.assertEqual((series["dimension"], series["multiplicity"]), (1, 2))
        graded = [sum(c*comb(n-k+1, 1) for k, c in num.items() if k <= n) for n in range(6)]
        self.assertEqual(graded, [1, 2, 3, 2, 2, 2])

    def test_wide_exponents_with_no_degree_enumeration(self):
        k = 2**64+17
        ideal = [f"y^{k}", f"y^{k+1}"]
        for op in (False, True):
            result = calculate(ideal, point=["3/2", 0], series=op)
            self.assertEqual((result["dimension"], result["multiplicity"]), (1, k))
            if op:
                self.assertEqual(coefficients(result), {0: 1, k: -1})
            else:
                self.assertEqual(result["generators"], [f"y^{k}"])

    def test_huge_principal_shift_recovers_all_directional_coefficients(self):
        k = 2**64+7
        ideal = [f"(x^{k}-1+y)^2"]
        with patch("engine.point_germs._translate", side_effect=AssertionError("full expansion")):
            cone = calculate(ideal, point=["1", 0])
            series = calculate(ideal, point=["1", 0], series=True)
        self.assertEqual(coefficients(series), {0: 1, 2: -1})
        self.assertEqual(len(cone["generators"]), 1)
        self.assertEqual(parse_expression(cone["generators"][0], ["x", "y"]),
                         parse_expression(f"(x+y/{k})^2", ["x", "y"]))

    def test_sparse_numerators_match_inclusion_exclusion(self):
        rng = Random(4291)
        names = list("xyz")
        for _ in range(16):
            heads = [tuple(rng.randrange(4) for _ in names) for _ in range(rng.randrange(2, 8))]
            ideal = ["*".join(f"{s}^{n}" for s,n in zip(names, h) if n) or "1" for h in heads]
            # Inclusion-exclusion over translated positive orthants, keeping
            # degrees of their lcm. This is independent of the box partition.
            expected = {0: 1}
            for size in range(1, len(heads)+1):
                for subset in combinations(heads, size):
                    k = sum(map(max, zip(*subset)))
                    expected[k] = expected.get(k, 0) + (-1)**size
            expected = {k:v for k,v in expected.items() if v}
            series = calculate(ideal, series=True, names=names)
            self.assertEqual(coefficients(series), expected)
            self.assertEqual([r["degree"] for r in series["numerator"]], sorted(expected))
