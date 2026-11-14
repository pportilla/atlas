"""Standard bases and certified local remainders at rational points."""
from fractions import Fraction
from math import comb
import shutil
import unittest
from unittest.mock import patch

from engine import local_standard
from engine import native_polynomial as np
from engine.polynomial import parse_expression
from tests.test_local_algebra_singular import singular


def native(text, names):
    return np.from_terms(tuple(map(np.Symbol, names)), parse_expression(text, names))


def translated(text, names, point):
    """Express a polynomial at x=p+a using a as the coordinate variables."""
    coords = tuple(map(np.Symbol, names))
    return native(text, names).xreplace({v: v + Fraction(p) for v, p in zip(coords, point)})


def local_head(polynomial, names, order):
    terms = np.terms(tuple(map(np.Symbol, names)), polynomial)
    if terms:
        degree = min(map(sum, terms))
        lowest = [e for e in terms if sum(e) == degree]
        if order == "lex":
            return max(lowest)
        return min(lowest, key=lambda e: e[::-1])


class LocalStandardKernel(unittest.TestCase):
    def calculate(self, operation, generators, polynomial=None, *, names=("x", "y"), point=None,
                  order="grevlex", known_basis=False):
        ring = {"field": "QQ", "variables": list(names), "order": order}
        operands = {"ideal": [native(f, names) for f in generators]}
        if point is not None:
            operands["point"] = list(point)
        if polynomial is not None:
            operands["polynomial"] = native(polynomial, names)
        result = local_standard.handle(operation, ring, operands, known_basis=known_basis)
        self.assertEqual(result["ring"], ring)
        self.assertEqual(result["scope"], "point")
        self.assertEqual(result["type"], operation)
        self.assertEqual(result["point"], ["0"] * len(names) if point is None else list(point))
        if polynomial is not None:
            unit = translated(result["unit"], names, result["point"])
            self.assertEqual(np.terms(tuple(map(np.Symbol, names)), unit).get((0,) * len(names)), 1)
        return result

    def test_local_unit_and_ideals_over_the_coefficient_field(self):
        for names, ideal, polynomial, member in [
                (("x", "y"), ["2+x+y"], "x*y+3", True),
                ((), ["0"], "-5/7", False), ((), ["-2/3"], "-5/7", True)]:
            with self.subTest(names=names, ideal=ideal):
                result = self.calculate("local-normal-form", ideal, polynomial, names=names)
                self.assertEqual(result["reduction"], "standard")
                self.assertEqual(not native(result["remainder"], names), member)
                basis = self.calculate("local-standard-basis", ideal, names=names)
                heads = [local_head(native(g, names), names, "grevlex") for g in basis["generators"]]
                self.assertEqual((0,) * len(names) in heads, member)

    def test_certificate_in_original_coordinates_at_fractional_point(self):
        # The linear factor vanishes at the selected point and is nonzero at
        # another global zero of the ideal, (3, -2/3).
        point = ["1/2", "-2/3"]
        ideal = ["(x-1/2)*(x-3)", "y+2/3"]
        basis = self.calculate("local-standard-basis", ideal, point=point)
        heads = {local_head(translated(g, ("x", "y"), point), ("x", "y"), "grevlex")
                 for g in basis["generators"]}
        self.assertTrue({(1, 0), (0, 1)} <= heads)
        for f, expected in [("x-1/2", 0), ("2+x-1/2", 2)]:
            result = self.calculate("local-normal-form", ideal, f, point=point)
            self.assertEqual(result["reduction"], "standard")
            self.assertEqual(native(result["remainder"], ("x", "y")), expected)

    def test_standard_heads_use_declared_tie_break_with_private_coordinate_name(self):
        names = ("homogenizer", "y", "z", "w")
        ideal = ["homogenizer*z+y^2+z^3", "w^2"]
        for order, expected in [("lex", (1, 0, 1, 0)), ("grevlex", (0, 2, 0, 0))]:
            with self.subTest(order=order):
                basis = self.calculate("local-standard-basis", ideal, names=names, order=order)
                heads = [local_head(native(g, names), names, order) for g in basis["generators"]]
                minimal = {expected, (0, 0, 0, 2)}
                self.assertTrue(minimal.issubset(heads))
                self.assertTrue(all(any(all(m >= e for m, e in zip(head, divisor)) for divisor in minimal)
                                    for head in heads))

    def test_finite_normal_form_reduces_all_tail_terms(self):
        ideal = ["x+y^2", "x+y^2+x^2"]
        f = "x+y^3"
        for order in ("lex", "grevlex"):
            with self.subTest(order=order):
                result = self.calculate("local-normal-form", ideal, f, order=order)
                self.assertEqual(result["reduction"], "standard")
                coords = tuple(map(np.Symbol, ("x", "y")))
                unit, remainder = (native(result[k], ("x", "y")) for k in ("unit", "remainder"))
                self.assertTrue(remainder)
                self.assertTrue(all(a == 0 and b < 4 for a, b in np.terms(coords, remainder)))
                # The finite local algebra is QQ[y]/(y^4) with x = -y².
                residual = (unit * native(f, ("x", "y")) - remainder).xreplace({coords[0]: -coords[1]**2})
                self.assertTrue(all(b >= 4 for a, b in np.terms(coords, residual)))

    def test_positive_dimensional_monomial_associates_at_rational_point(self):
        names, point = ("x", "y"), ["2/3", "-1/2"]
        a, b = "(x-2/3)", "(y+1/2)"
        # Both generators define the same mixed monomial up to a local unit.
        ideal = [f"{a}^2*{b}*(2+{a})", f"-3*{a}^2*{b}*(1-{b})"]
        standard = f"1+{a}*{b}^3+{a}^4+2*{b}^3"
        f = f"{standard}+{a}^2*{b}+7*{a}^4*{b}^3"
        result = self.calculate("local-normal-form", ideal, f, names=names, point=point)
        self.assertEqual(result["reduction"], "standard")
        self.assertEqual(native(result["unit"], names), 1)
        self.assertEqual(native(result["remainder"], names), native(standard, names))

    def test_associate_rows_remain_polynomials_and_replay_without_a_lift(self):
        names = ("x", "y")
        for point in (["0", "0"], ["1/2", "-2/3"]):
            a, b = (f"({name}-({p}))" for name, p in zip(names, point))
            rows = [f"{a}*{b}*(1+{a})", f"{a}*{b}*(1+{b})", f"{a}^2*{b}*(1-{b})"]
            ideal = [f"2*({rows[0]})", f"-3/5*({rows[0]})", f"-7*({rows[1]})", rows[2]]
            expected = {native(f, names) for f in rows[:2]}
            for order in ("lex", "grevlex"):
                with self.subTest(point=point, order=order):
                    basis = self.calculate("local-standard-basis", ideal, names=names, point=point, order=order)
                    result = basis["generators"]
                    # xy is the only minimal local head. Keep one actual monic
                    # polynomial with that head, including its unit factor.
                    self.assertEqual(len(result), 1)
                    self.assertIn(native(result[0], names), expected)
                    self.assertEqual(self.calculate("local-standard-basis", ideal[::-1], names=names,
                                                    point=point, order=order), basis)
                    self.assertEqual(self.calculate("local-standard-basis", result, names=names,
                                                    point=point, order=order), basis)
                    if shutil.which("Singular"):
                        original = ",".join(str(translated(f, names, point)) for f in ideal)
                        selected = translated(result[0], names, point)
                        self.assertEqual(singular("Ds" if order == "lex" else "ds", [
                            f"ideal I={original}; ideal G=std(I); ideal B={selected};",
                            'print("ANSWER "+string(size(reduce(B,G))==0));',
                            'print("ANSWER "+string(size(reduce(I,std(B)))==0));',
                            'print("ANSWER "+string(size(reduce(lead(G),std(lead(B))))==0));',
                            'print("ANSWER "+string(size(reduce(lead(B),std(lead(G))))==0));',
                        ], names), [1, 1, 1, 1])

    def test_missing_coordinatewise_minimum_uses_general_basis(self):
        names = ("x", "y")
        # x+y² has componentwise minimum (0,0) across its exponents, but no
        # constant term. Together with x*(1+x) it forces y² into the ideal.
        ideal = ["x*(1+x)", "x+y^2"]
        for order in ("lex", "grevlex"):
            with self.subTest(order=order):
                basis = self.calculate("local-standard-basis", ideal, names=names, order=order)
                heads = {local_head(native(g, names), names, order) for g in basis["generators"]}
                self.assertEqual(len(basis["generators"]), 2)
                self.assertEqual(heads, {(1, 0), (0, 2)})

    @unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
    def test_weak_normal_forms_and_leading_ideal_match_singular_at_the_point(self):
        names, point = ("x", "y"), ["1/2", "1/3"]
        a, b = "(x-1/2)", "(y-1/3)"
        ideal = [f"{a}-{a}^2-{b}^2"]
        rows = ",".join(str(translated(g, names, point)) for g in ideal)
        basis = self.calculate("local-standard-basis", ideal, point=point)
        atlas = ",".join(str(translated(g, names, point)) for g in basis["generators"])
        self.assertEqual(singular("ds", [
            f"ideal I={rows}; ideal G=std(I); ideal B={atlas};",
            'print("ANSWER "+string(size(reduce(B,G))==0));',
            'print("ANSWER "+string(size(reduce(I,std(B)))==0));',
            'print("ANSWER "+string(size(reduce(lead(G),std(lead(B))))==0));',
        ], names), [1, 1, 1])
        for f, member, reduction in [(a, False, "standard"), (f"{a}+{a}^3-{b}^2", False, "weak"),
                                     (f"({ideal[0]})*({a}+1)", True, "standard")]:
            result = self.calculate("local-normal-form", ideal, f, point=point)
            self.assertEqual(result["reduction"], reduction)
            u, r = (str(translated(result[k], names, point)) for k in ("unit", "remainder"))
            if f == a:
                # A unit other than 1 is needed in the Mora certificate.
                self.assertNotEqual(native(result["unit"], names), 1)
            self.assertEqual(not native(result["remainder"], names), member)
            self.assertEqual(singular("ds", [
                f"ideal G=std(ideal({rows})); poly f={translated(f, names, point)};",
                f"poly u={u}; poly r={r};",
                'print("ANSWER "+string(reduce(u*f-r,G)==0));',
                'print("ANSWER "+string(reduce(f,G)==0));',
            ], names), [1, int(member)])

    def test_low_degree_remainder_modulo_wide_power(self):
        result = self.calculate("local-normal-form", ["x^18446744073709551617"], "x", names=("x",))
        self.assertEqual(result["reduction"], "standard")
        self.assertEqual(result["remainder"], "x")

    def test_finite_query_jet_uses_the_mixed_standard_basis(self):
        # In u=x-1, v=y+1, v²=-u³ and v³=0 imply u³v=u⁶=0.
        n = 10**21
        ideal = ["(x-1)^3+(y+1)^2", "(y+1)^3"]
        query = f"x^{n}*y^2"
        expected = "+".join(
            [f"{comb(n, i)}*(x-1)^{i}" for i in range(6)] +
            [f"(-{2*comb(n, i)})*(x-1)^{i}*(y+1)" for i in range(3)] +
            [f"(-{comb(n, i)})*(x-1)^{i+3}" for i in range(3)])
        for order in ("lex", "grevlex"):
            options = dict(order=order, point=["1", "-1"])
            result = self.calculate("local-normal-form", ideal, query, **options)
            self.assertEqual((result["unit"], result["reduction"]), ("1", "standard"))
            self.assertEqual(native(result["remainder"], ("x", "y")), native(expected, ("x", "y")))
            rows = self.calculate("local-standard-basis", ideal, **options)["generators"]
            self.assertEqual(self.calculate("local-normal-form", rows, query,
                                            known_basis=True, **options), result)

    def test_finite_query_jet_at_reciprocal_and_zero_coordinates(self):
        n = 2**65 + 1
        u, v = "(x+2)", "(y-1/2)"
        result = self.calculate("local-normal-form", [f"{u}^2", f"{v}^2"], f"(x*y)^{n}",
                                point=["-2", "1/2"])
        # (-2+u)^n*(1/2+v)^n = -(1-n*u/2)*(1+2*n*v) modulo (u²,v²).
        expected = f"-(1-{n}*{u}/2)*(1+{2*n}*{v})"
        self.assertEqual(native(result["remainder"], ("x", "y")), native(expected, ("x", "y")))
        self.assertEqual((result["unit"], result["reduction"]), ("1", "standard"))
        result = self.calculate("local-normal-form", ["(x-2)^2", "y"], f"x^{n}*y^2+1",
                                point=["2", "0"])
        self.assertEqual(result["remainder"], "1")

    def test_large_jet_coefficients_cancel_across_distinct_rational_coordinates(self):
        n = 2**64+7
        a, b = "(x-2)", "(y-1/4)"
        result = self.calculate("local-normal-form", [f"{a}^2", f"{b}^2"],
                                f"x^{2*n}*y^{n}", point=["2","1/4"])
        # x^(2n) and y^n cannot be written out individually at the point.
        expected = f"(1+{n}*{a})*(1+{4*n}*{b})"
        self.assertEqual(native(result["remainder"], ("x","y")), native(expected, ("x","y")))
        self.assertEqual((result["unit"],result["reduction"]), ("1","standard"))

    def test_coordinate_power_remainder_groups_reciprocal_coordinates_before_projection(self):
        n = 2**65 + 1
        names, point = ("x", "y", "z"), ["-2", "1/2", "0"]
        for u_bound, v_bound in [(1, 1), (1, 2), (2, 33)]:
            with self.subTest(bounds=(u_bound, v_bound)):
                ideal = [f"(x+2)^{u_bound}", f"(y-1/2)^{v_bound}", f"z^{n}"]
                query = f"(x*y)^{n} + x^{n}*z^{n}"
                result = self.calculate("local-normal-form", ideal, query, point=point, names=names)
                u_series = "+".join(f"{comb(n, k)}*(-(x+2)/2)^{k}" for k in range(u_bound))
                v_series = "+".join(f"{comb(n, k)}*(2*y-1)^{k}" for k in range(v_bound))
                expected = native(f"-({u_series})*({v_series})", names)
                self.assertEqual(native(result["remainder"], names), expected)
                self.assertEqual((result["unit"], result["reduction"]), ("1", "standard"))

    def test_large_query_transfers_to_a_free_axis_without_dense_expansion(self):
        n = 2**64+7
        result = self.calculate("local-normal-form", ["x-y", "z^2"], f"x^{n}",
                                names=("x","y","z"), point=["1","1","0"])
        self.assertEqual((result["unit"],result["reduction"],result["remainder"]),
                         ("1","standard",f"y^{n}"))

    def test_query_centering_uses_ordinary_translation_when_cheaper_or_unbounded(self):
        cases = [
            (["(x-1)^2", "y"], "x^3", "3*x-2"),
            (["(x-1)^2", "y^18446744073709551617"], "x^3", "3*x-2"),
            (["(y-1)^2", "(y-1)*(x-1)"], "x^130", "x^130"),
        ]
        for ideal, query, expected in cases:
            point = ["1", "1"] if query == "x^130" else ["1", "0"]
            with self.subTest(ideal=ideal):
                result = self.calculate("local-normal-form", ideal, query, point=point)
                self.assertEqual(result["remainder"], expected)

    def test_trivial_rational_point_results_do_not_translate_wide_powers(self):
        power = "x^18446744073709551617"
        names, point = ("x",), ["2"]
        # Replacing x with x+2 in the wide power would need 2^64+2 terms.
        with patch.object(local_standard, "_translate", side_effect=AssertionError("unexpected translation")):
            for ideal, query, expected in [(["0"], power, power), (["1"], power, "0"),
                                          ([power], "0", "0"), ([power], "x+1", "0")]:
                with self.subTest(ideal=ideal, query=query):
                    result = self.calculate("local-normal-form", ideal, query, names=names, point=point)
                    self.assertEqual(result["reduction"], "standard")
                    self.assertEqual(result["unit"], "1")
                    self.assertEqual(native(result["remainder"], names), native(expected, names))
            basis = self.calculate("local-standard-basis", [power], names=names, point=point)
            self.assertEqual(basis["generators"], ["1"])

    def test_same_sign_evaluation_of_a_sparse_wide_local_unit(self):
        # x^N is negative at this point. Hence both terms of 1-x^N are
        # positive even though the rational point's denominator is enormous
        # when raised to N. Guard against materializing that rational number.
        def bounded_power(base, exponent):
            if exponent > 64:
                raise AssertionError("unexpected expansion of a wide rational power")
            return original_power(base, exponent)

        original_power = Fraction.__pow__
        with patch.object(Fraction, "__pow__", bounded_power):
            result = self.calculate("local-standard-basis", ["1-x^18446744073709551617"],
                                    names=("x",), point=["-1/1000000007"])
        self.assertEqual(result["generators"], ["1"])

    def test_project_monomial_ideal_at_mixed_point_without_expanding_wide_powers(self):
        names, point = ("x", "y", "z"), ["2/3", "0", "-3/4"]
        n = 2**64 + 1
        ideal = [f"x^{n}*y^2", "z^3*y^3"]
        query = f"x^{n}+x^{n-1}*y^2"
        # x and z are units here. Only the y-exponents can cut out the ideal.
        with patch.object(local_standard, "_translate", side_effect=AssertionError("unexpected translation")):
            basis = self.calculate("local-standard-basis", ideal, names=names, point=point)
            result = self.calculate("local-normal-form", ideal, query, names=names, point=point)
        rows = [parse_expression(g, names) for g in basis["generators"]]
        self.assertIn({(0, 2, 0): 1}, rows)
        for row in rows:
            self.assertEqual(len(row), 1)
            ((a, b, c), coefficient), = row.items()
            self.assertEqual((a, c, coefficient), (0, 0, 1))
            self.assertGreaterEqual(b, 2)
        self.assertEqual((result["reduction"], result["unit"]), ("standard", "1"))
        self.assertEqual(parse_expression(result["remainder"], names), {(n, 0, 0): 1})

    def test_saved_standard_basis_at_its_point_skips_basis_computation(self):
        names, point = ("x", "y"), ["1/2", "-1/3"]
        a, b = "(x-1/2)", "(y+1/3)"
        ideal, query = [f"{a}+{b}^2", f"{a}^2"], f"{a}+{b}^3"
        basis = self.calculate("local-standard-basis", ideal, names=names, point=point)["generators"]
        with patch.object(local_standard, "_standard_basis", wraps=local_standard._standard_basis) as rebuild:
            fresh = self.calculate("local-normal-form", basis, query, names=names, point=point)
            rebuild.assert_called_once()
            rebuild.reset_mock()
            saved = self.calculate("local-normal-form", basis, query, names=names, point=point, known_basis=True)
            rebuild.assert_not_called()
        self.assertEqual(saved, fresh)
        self.assertEqual(native(saved["remainder"], names), native(f"-{b}^2+{b}^3", names))

    def test_basis_is_idempotent_and_saved_normal_form_matches_rebuilding(self):
        names, order = ("x", "y", "z"), "lex"
        # The sum of the input rows gives y. Then y²+z gives z, so the
        # quotient class of x-y is x even though x is a free coordinate.
        ideal = ["-y^2+y-z", "y^2+z"]
        basis = self.calculate("local-standard-basis", ideal, names=names, order=order)
        rows = basis["generators"]
        self.assertEqual(self.calculate("local-standard-basis", rows, names=names, order=order), basis)
        fresh = self.calculate("local-normal-form", ideal, "x-y", names=names, order=order)
        unit, remainder = (native(fresh[k], names) for k in ("unit", "remainder"))
        residual = unit * native("x-y", names) - remainder
        self.assertFalse(residual.xreplace({np.Symbol("y"): 0, np.Symbol("z"): 0}))
        for known in (False, True):
            with self.subTest(known_basis=known):
                replayed = self.calculate("local-normal-form", rows, "x-y", names=names,
                                          order=order, known_basis=known)
                self.assertEqual(replayed, fresh)


if __name__ == "__main__":
    unittest.main()
