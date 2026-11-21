import unittest
from fractions import Fraction
from unittest.mock import patch

from engine import native_library, polynomial
from engine.polynomial import InputError, LimitError, parse_expression


class PublicInputArithmetic(unittest.TestCase):
    def test_public_whitespace_and_operator_boundaries(self):
        source = "-(x\u00a0-\ny)**2/2\r+\tx^2"
        for bounded in (True, False):
            self.assertEqual(parse_expression(source, ["x","y"], bounded=bounded),
                             {(2,0): Fraction(1,2), (1,1): 1, (0,2): Fraction(-1,2)})
            for invalid in ("x * * 2", "2 1*x", "x y", "x^2^3", "x\0+y", "x/y"):
                with self.subTest(source=invalid, bounded=bounded), self.assertRaises(InputError):
                    parse_expression(invalid, ["x","y"], bounded=bounded)

    def test_quick_lexical_and_depth_limits(self):
        for source, names, reason in (
                ("x", ["x"]*9, "8 variables"),
                ("x"+" "*8192, ["x"], "8192"),
                ("+".join(["x"]*1025), ["x"], "tokens"),
                ("("*65+"x"+")"*65, ["x"], "nesting"),
                ("x^0001", ["x"], "exponents"),
                ("x^129", ["x"], "exponents"),
                ("1"*257+"*x", ["x"], "literal")):
            with self.subTest(reason=reason), self.assertRaisesRegex(LimitError,reason):
                parse_expression(source, names, bounded=True)
        self.assertEqual(parse_expression("("*64+"x"+")"*64, ["x"], bounded=True), {(1,): 1})
        self.assertEqual(parse_expression("x^128", ["x"], bounded=True), {(128,): 1})

    def test_quick_intermediate_arithmetic_budgets(self):
        for source, variables, reason in (
                ("x^128*x-x^128*x", ["x"], "degree"),
                ("(2^128)^32/(2^128)^32", ["x"], "coefficient"),
                ("1/((2^128)^31)/2^128", ["x"], "coefficient"),
                ("(a+b+c+d+e+f+g+h)^8", list("abcdefgh"), "term"),
                ("((x+y+z+w)^14)^2", list("xyzw"), "product")):
            with self.subTest(source=source), self.assertRaisesRegex(LimitError,reason):
                parse_expression(source, variables, bounded=True)
        self.assertEqual(parse_expression("((x+y)^5-(x+y)^5)^128", ["x","y"], bounded=True), {})
        with patch.multiple(polynomial, MAX_PRODUCTS=4, MAX_TERMS=3, MAX_BITS=2, MAX_DEGREE=2):
            self.assertEqual(parse_expression("(x+y)^2", ["x","y"], bounded=True),
                             {(2,0): 1, (1,1): 2, (0,2): 1})
            with self.assertRaisesRegex(LimitError, "product"):
                parse_expression("(x+y)^2*(x+y)", ["x","y"], bounded=True)

    def test_unbounded_sizes_and_large_computed_coefficient(self):
        self.assertEqual(parse_expression("2^20000*x^10000000000000", ["x"]),
                         {(10**13,): Fraction(2**20000)})
        self.assertEqual(parse_expression("1/2^20000", ["x"]), {(0,): Fraction(1, 2**20000)})
        self.assertEqual(parse_expression("2^20000*x", ["x"]), {(1,): Fraction(2**20000)})
        names = [f"x{i}" for i in range(34)]
        self.assertEqual(parse_expression("x33", names), {(0,)*33 + (1,): 1})

    def test_unbounded_flat_sum_cancels_across_many_distinct_monomials(self):
        source = "-x^2+3*y/5"
        for k in range(1, 514):
            source += f"+x^{k}/3-x^{k}*y/2"
        for k in range(513, 0, -1):
            source += f"-x^{k}/3--x^{k}*y/2"
        self.assertEqual(parse_expression(source, ["x", "y"]),
                         {(2, 0): -1, (0, 1): Fraction(3, 5)})

    def test_unbounded_grouped_sum_as_power_product_and_divisor(self):
        source = "-(x-y)^2--(x+y)^2/2+(x-y+y)*(x+y-y)-x/(y+3-y)+x/3"
        self.assertEqual(parse_expression(source, ["x", "y"]),
                         {(2, 0): Fraction(1, 2), (1, 1): 3, (0, 2): Fraction(-1, 2)})

    def test_quick_flat_sum_checks_limits_before_cancellation(self):
        for source, budget, reason, expected in (
                ("x+y+1-1", {"MAX_TERMS": 2}, "term", {(1, 0): 1, (0, 1): 1}),
                ("3*x+x-x", {"MAX_BITS": 2}, "coefficient", {(1, 0): 3})):
            with self.subTest(source=source), patch.multiple(polynomial, **budget):
                self.assertEqual(parse_expression(source, ["x", "y"]), expected)
                with self.assertRaisesRegex(LimitError, reason):
                    parse_expression(source, ["x", "y"], bounded=True)

    def test_exhaustive_nesting(self):
        depth = 12_000
        cases = (
            ("(-" * depth + "x" + ")" * depth + "^2", {(2,): 1}),
            ("1-(" * depth + "x" + ")" * depth, {(1,): 1}),
            ("-" * (depth+1) + "x**2*3", {(2,): -3}),
        )
        for source, expected in cases:
            self.assertEqual(parse_expression(source, ["x"]), expected)
        with self.assertRaises(InputError):
            parse_expression("(" * depth + "x" + ")" * (depth-1), ["x"])

    def test_wide_power_of_a_sparse_monomial_and_an_algebraic_coefficient(self):
        n = 2**64 + 1
        depth = 12_001
        self.assertEqual(
            parse_expression("-" * depth + "(" * depth + "-x*y" + ")" * depth + f"**{n}", ["x", "y"]),
            {(n, n): 1},
        )
        # In QQ[t]/(1+t+t²), 1+t = -t², t³=1 and n=5 modulo 6.
        for expression, expected in (
            (f"(t+1)^{n}", {(0, 1): -1}),
            (f"((t+1)*x)^{n}", {(n, 1): -1}),
            (f"(t^2+t+1)^{n}", {}),
            ("(t^2+t+1)^0", {(0, 0): 1}),
        ):
            with self.subTest(expression=expression):
                self.assertEqual(parse_expression(expression, ["x", "t"], coefficient_modulus="t^2+t+1"),
                                 expected)

    def test_stale_binary(self):
        with patch.object(native_library, "affine_request", side_effect=native_library.Unavailable("build kernel")):
            with self.assertRaisesRegex(native_library.Unavailable, "build kernel"):
                parse_expression("x^5+x^5", ["x"])
        if hasattr(native_library._kernel(), "atlas_input_abi"):
            with patch.object(polynomial, "_parse_tokens", side_effect=AssertionError("unexpected reference")):
                with self.assertRaises(LimitError):
                    parse_expression("x^129", ["x"], bounded=True)

