"""Check quotient representatives, arithmetic identities, and unit decisions."""
from dataclasses import replace
import json
import shutil
import unittest

from benchmarks.quotient_ring import Case, corpus, native_wire, validate
from engine.native_library import affine_request


def compute(case):
    line, = affine_request(native_wire(case))
    return json.loads(line)


class QuotientArithmeticTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("Singular"), "Singular is required")
    def test_all_operations_in_both_orders(self):
        for case in corpus():
            for order in ("lex", "grevlex"):
                with self.subTest(name=case.name, order=order):
                    query = replace(case, order=order)
                    validate(query, compute(query), 15)

    def test_division_requires_a_unit_even_for_zero_numerator(self):
        for relations, denominator in [
            ((), "x"), (("x*y",), "x"), (("x^2",), "x"), (("x*y-1",), "x+y"),
            (("y-x^2",), "y-x^2"), (("x^3", "y^4"), "x+y")
        ]:
            nonunit = Case("nonunit", ("x", "y"), relations, "unit", (denominator,))
            with self.subTest(relations=relations, denominator=denominator):
                self.assertIs(compute(nonunit)["value"], False)
                for op in (replace(nonunit, action="inverse"),
                           replace(nonunit, action="power", exponent=-1),
                           replace(nonunit, action="divide", operands=("0", denominator))):
                    self.assertEqual(compute(op)["error"]["code"], "invalid_request")

    def test_unbounded_input_and_exponents(self):
        n = 2**79 + 7
        big = 2**211 + 31
        for exponent in (n, -n):
            query = Case("order_two", ("x",), ("x^2-1",), "power", ("x",), exponent)
            self.assertEqual(compute(query)["representative"], "x")
        scale = Case("wide", ("x","y"), ("x^2",), "reduce", (f"x^{n}+{big}*y^{n}",))
        self.assertEqual(compute(scale)["representative"], f"{big}*y^{n}")

    def test_packed_exponent_overflow_keeps_exact_result(self):
        # The lex substitution x -> y^3 overflows an internal packed exponent
        # although the input exponent itself fits. The unreduced product also
        # crosses the bound independently in the second query.
        cases = [
            (Case("tail", ("x", "y"), ("x-y^3",), "reduce", ("x^110+y",), order="lex"), "y^330+y"),
            (Case("product", ("x", "y"), ("y^2",), "multiply", ("x^80+y", "x^70")), "x^150+x^70*y"),
        ]
        for query, expected in cases:
            with self.subTest(name=query.name):
                self.assertEqual(compute(query)["representative"], expected)

    def test_component_nonunit_and_positive_dimensional_inverse(self):
        # Neither branch may infer that a nonunit on one component is a unit
        # in the whole ring. This includes a mixed monomial/binomial box.
        for relations, element in [(("x^3-x", "y^3-y"), "x+y"),
                                   (("x^2", "y^2-1"), "x+y+1")]:
            query = Case("nonunit", ("x", "y"), relations, "unit", (element,))
            self.assertIs(compute(query)["value"], False)
            self.assertEqual(compute(replace(query, action="inverse"))["error"]["code"], "invalid_request")
        # The unit emerges from the interaction of two positive-dimensional
        # relations, with z supplying a free coordinate.
        query = Case("graph", ("x", "y", "z"), ("y^2-x", "x^2*y-1"),
                     "inverse", ("x+y",))
        for order in ("lex", "grevlex"):
            q = replace(query, order=order)
            product = replace(q, action="multiply", operands=("x+y", compute(q)["representative"]))
            self.assertEqual(compute(product)["representative"], "1")

    def test_zero_ring_and_empty_relations(self):
        for action, operands, n in (("unit",("0",),0), ("inverse",("0",),0),
                                     ("divide",("3","0"),0), ("power",("0",),-17)):
            query = Case("zero_ring", ("x",), ("1",), action, operands, n)
            with self.subTest(action=action):
                result = compute(query)
                if action == "unit":
                    self.assertIs(result["value"], True)
                else:
                    self.assertEqual(result["representative"], "0")
        query = Case("QQ", (), (), "divide", ("7/3", "2/5"))
        self.assertEqual(compute(query)["representative"], "35/6")
