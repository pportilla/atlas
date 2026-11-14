"""Phase 1 rational ideal operations through the shared algebra dispatcher."""
import os
import shutil
import subprocess
import unittest
from unittest.mock import patch

from benchmarks.groebner_hard import katsura
from engine.algebra import dispatch
from engine.gpu_devices import available_devices
from engine.polynomial import format_poly, parse_expression


def calculate(operation, operands, variables=("x", "y"), order="grevlex", compute="cpu"):
    return dispatch({
        "operation": operation,
        "ring": {"field": "QQ", "variables": list(variables), "order": order},
        "operands": operands,
        "compute": compute,
    })


def polynomials(expressions, variables):
    return {frozenset(parse_expression(f, variables, _reference=True).items()) for f in expressions}


CASES = [
    ("quotient", {"ideal": ["x^2*y"], "other_ideal": ["x"]}, ["x*y"]),
    ("saturation", {"ideal": ["x^2*y"], "polynomial": "x"}, ["y"]),
    ("intersection", {"ideal": ["x"], "other_ideal": ["y"]}, ["x*y"]),
    ("eliminate", {"ideal": ["x^2-1", "y-x"], "variables": ["x"]}, ["y^2-1"]),
    ("dimension", {"ideal": ["0"]}, 2),
    ("dimension", {"ideal": ["1"]}, -1),
]


class IdealOperations(unittest.TestCase):
    def test_ordered_ideal_trace_closes_after_a_basis_exception(self):
        from contextlib import contextmanager
        from engine import native_polynomial as np
        from engine.ideal_operations import intersection

        active = []
        @contextmanager
        def observe(name, **counts):
            active.append(name)
            try:
                yield
            finally:
                self.assertEqual(active.pop(), name)

        def unavailable(label, variables, generators, order):
            self.assertEqual(label, "intersection_basis")
            self.assertEqual(order, "first")
            raise ArithmeticError("test basis callback failed")

        x, y = map(np.Symbol, ("x", "y"))
        with self.assertRaisesRegex(ArithmeticError, "test basis callback failed"):
            intersection((x,y), (x,), (y,), basis=unavailable, stage=observe)
        self.assertEqual(active, [])
        self.assertEqual(intersection((x,y), (x,), (y,)).generators, (x*y,))

    def test_phase_one_examples(self):
        for policy in ("cpu", "gpu"):
            for op, operands, expected in CASES:
                with self.subTest(op=op, operands=operands, compute=policy):
                    record = calculate(op, operands, compute=policy)
                    self.assertEqual(record["status"], "ok", record)
                    result = record["result"]
                    names = ["y"] if op == "eliminate" else ["x", "y"]
                    self.assertEqual(result["ring"]["variables"], names)
                    self.assertEqual(result["ring"]["field"], "QQ")
                    self.assertEqual(record["execution"]["requested"], policy)
                    if isinstance(expected, int):
                        self.assertEqual((result["type"], result["value"]), ("integer", expected))
                    else:
                        self.assertEqual(result["type"], "ideal")
                        self.assertEqual(polynomials(result["generators"], names), polynomials(expected, names))

    def test_nonprincipal_divisor_and_zero_operands(self):
        cases = [
            ("quotient", {"ideal": ["x*y"], "other_ideal": ["x", "y"]}, ["x*y"]),
            ("quotient", {"ideal": ["x"], "other_ideal": ["0"]}, ["1"]),
            ("saturation", {"ideal": ["x"], "polynomial": "0"}, ["1"]),
        ]
        for op, operands, expected in cases:
            with self.subTest(op=op, operands=operands):
                record = calculate(op, operands)
                self.assertEqual(record["status"], "ok", record)
                self.assertEqual(polynomials(record["result"]["generators"], ["x", "y"]),
                                 polynomials(expected, ["x", "y"]))

    def test_dense_regular_saturation_preserves_order_and_embedded_components(self):
        from engine import kernel, native_polynomial as np
        variables = ("x", "y", "z")
        x, y, z = coords = np.symbols("x y z")
        # x^2, xy has an embedded associated prime (x,y). The first
        # multiplier misses both primes, the second lies in the embedded one.
        dense = sum(z**i for i in range(32))
        for by, formula, expected, regular in (
                (y+dense, "y+" + "+".join(f"z^{i}" for i in range(32)), (x*x, x*y), True),
                (y*dense, "y*(" + "+".join(f"z^{i}" for i in range(32)) + ")", (x,), False)):
            self.assertEqual(kernel.saturate(coords, (x*x, x*y), by), expected)
            for order in ("lex", "grevlex"):
                record = calculate("saturation", {"ideal": ["x^2", "x*y"], "polynomial": formula},
                                   variables, order)
                self.assertEqual(record["status"], "ok", record)
                self.assertEqual(record["result"]["ring"]["order"], order)
                self.assertEqual(polynomials(record["result"]["generators"], variables),
                                 polynomials(["x^2", "x*y"] if regular else ["x"], variables))
                stages = [s["operation"] for s in record["execution"]["stages"]]
                self.assertEqual("adjoin_inverse" in stages, not regular)

    def test_quotient_reduces_and_normalizes_divided_rows(self):
        # x+y is not monomial. Removing it from a reduced intersection basis
        # can leave a reducible tail and a nonunit leading coefficient.
        for divisor in ("x+y", "(3*x+3*y)/7"):
            with self.subTest(divisor=divisor):
                rows = [f"({divisor})*x", f"({divisor})*y"]
                result = calculate("quotient", {"ideal": rows, "other_ideal": [divisor]},
                                   order="lex")["result"]
                self.assertEqual(result["ring"]["order"], "lex")
                self.assertEqual(polynomials(result["generators"], ["x", "y"]),
                                 polynomials(["x", "y"], ["x", "y"]))

    def test_principal_monomial_quotient_divides_inputs_then_takes_one_basis(self):
        # After removing x^2*y, the two input rows still need reduction.
        generators = ["0", "3*x^2*y*(x*y-1)", "-4*x^2*y*(y^2-x)", "0"]
        divisor = ["0", "-2/3*x^2*y", "0"]
        expected = {"lex": ["x-y^2", "y^3-1"], "grevlex": ["x^2-y", "x*y-1", "y^2-x"]}
        for order, basis in expected.items():
            with self.subTest(order=order):
                record = calculate("quotient", {"ideal": generators, "other_ideal": divisor}, order=order)
                result = record["result"]
                self.assertEqual(result["ring"]["order"], order)
                self.assertEqual(polynomials(result["generators"], ["x", "y"]), polynomials(basis, ["x", "y"]))
                stages = record["execution"]["stages"]
                self.assertIn("quotient_divide_inputs", [s["operation"] for s in stages])
                self.assertNotIn("quotient_by_1_lift", [s["operation"] for s in stages])
                if os.getenv("ATLAS_GROEBNER") != "sympy":
                    self.assertEqual([s["label"] for s in stages if s["operation"] == "groebner_basis"],
                                     ["quotient_basis"])

    def test_principal_monomial_quotient_checks_all_terms_before_dividing(self):
        # Both leading terms have an x factor, but dividing x^2-y is inexact.
        record = calculate("quotient", {"ideal": ["x*y", "x^2-y"], "other_ideal": ["x"]})
        self.assertEqual(polynomials(record["result"]["generators"], ["x", "y"]),
                         polynomials(["x^2", "y"], ["x", "y"]))
        stages = record["execution"]["stages"]
        operations = [s["operation"] for s in stages]
        self.assertLess(operations.index("quotient_divide_inputs"), operations.index("quotient_by_1_lift"))
        if os.getenv("ATLAS_GROEBNER") != "sympy":
            self.assertEqual([s["label"] for s in stages if s["operation"] == "groebner_basis"],
                             ["quotient_by_1_basis"])

    def test_wide_quotient_preserves_legacy_coordinate_types(self):
        from engine import native_polynomial
        from engine.reference import sp
        from engine.wide_basis import wide_colon

        for symbolic in (native_polynomial, sp):
            coordinates = tuple(symbolic.Symbol(f"v{i}") for i in range(34))
            x, y = coordinates[:2]
            with self.subTest(symbolic=symbolic):
                self.assertEqual(wide_colon((x,), (0,), coordinates), (1,))
                self.assertEqual(wide_colon((x*y,), (0, 2), coordinates), (x*y,))

    def test_returned_basis_matches_its_reported_order(self):
        # These two reduced bases differ in size. Saturating by a nonzero
        # constant preserves the ideal, and Atlas may choose its output order.
        bases = {"lex": ["x-y^2", "y^3-1"], "grevlex": ["x^2-y", "x*y-1", "y^2-x"]}
        for order in bases:
            record = calculate("saturation", {"ideal": ["x*y-1", "y^2-x"], "polynomial": "2"}, order=order)
            self.assertEqual(record["status"], "ok", record)
            result = record["result"]
            self.assertEqual(polynomials(result["generators"], ["x", "y"]),
                             polynomials(bases[result["ring"]["order"]], ["x", "y"]))

    def test_eliminate_middle_variable_then_reuse_ring(self):
        record = calculate("eliminate", {"ideal": ["z-y", "y-x^2"], "variables": ["y"]},
                           variables=("x", "y", "z"))
        self.assertEqual(record["status"], "ok", record)
        result = record["result"]
        self.assertEqual(result["ring"]["variables"], ["x", "z"])
        self.assertEqual(calculate("dimension", {"ideal": result["generators"]},
                                   variables=result["ring"]["variables"])["result"]["value"], 1)

    def test_eliminate_all_variables(self):
        for generators, expected, dimension in [
                (["x-y", "y^2-1"], [], 0), (["x", "1-x"], ["1"], -1)]:
            with self.subTest(generators=generators):
                result = calculate("eliminate", {"ideal": generators, "variables": ["y", "x"]})["result"]
                self.assertEqual(result["ring"]["variables"], [])
                self.assertEqual(result["generators"], expected)
                self.assertEqual(calculate("dimension", {"ideal": expected}, variables=())["result"]["value"],
                                 dimension)

    @unittest.skipUnless(os.getenv("ATLAS_GROEBNER", "native") == "native",
                         "The diagnostic dense polynomial representation cannot store this degree")
    def test_lex_elimination_with_a_large_surviving_exponent(self):
        # Substitution gives y^(8*2^61). The intermediate Gröbner basis and
        # the projected generator both require more than a 64-bit exponent.
        from engine.algebra import calculate_algebra
        status, record = calculate_algebra({
            "operation": "eliminate", "ring": {"variables": ["x", "y"], "order": "lex"},
            "operands": {"ideal": ["x-y^2305843009213693952", "x^8"], "variables": ["x"]},
            "compute": "cpu",
        })
        self.assertEqual(status, 200, record)
        self.assertEqual(record["result"], {
            "type": "ideal", "ring": {"field": "QQ", "variables": ["y"], "order": "lex"},
            "generators": ["y^18446744073709551616"],
        })

    @unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
    def test_singular_phase_one_oracle(self):
        script = """
LIB "elim.lib";
ring R=0,(x,y),dp;
short=0;
option(redSB);
ideal D=x^2*y; ideal X=x; ideal Y=y; ideal E=x^2-1,y-x;
print(string(std(quotient(D,X))));
print(string(std(sat(D,X))));
print(string(std(intersect(X,Y))));
print(string(std(eliminate(E,x))));
print(dim(std(ideal(0))));
print(dim(std(ideal(1))));
quit;
"""
        reference = subprocess.run(["Singular", "-q"], input=script, capture_output=True,
                                   text=True, check=True, timeout=30)
        lines = reference.stdout.strip().splitlines()
        self.assertEqual(len(lines), len(CASES), reference.stdout + reference.stderr)
        for (op, operands, expected), line in zip(CASES, lines):
            with self.subTest(op=op, operands=operands):
                result = calculate(op, operands)["result"]
                if isinstance(expected, int):
                    self.assertEqual(result["value"], int(line))
                else:
                    names = result["ring"]["variables"]
                    self.assertEqual(polynomials(result["generators"], names), polynomials(line.split(","), names))

    @unittest.skipUnless(os.getenv("ATLAS_GROEBNER", "native") == "native",
                         "Physical-device checks use the production native backend")
    def test_katsura_operations_on_physical_device(self):
        devices = available_devices()
        if not devices:
            self.skipTest("No physical Triton or OpenCL device")
        names, rows = katsura(6)
        ideal = [format_poly(row, names) for row in rows]
        extended = [*names, "z"]
        product = [f"z*({g})" for g in ideal]
        cases = [
            ("eliminate", {"ideal": [*ideal, "z"], "variables": ["z"]}, extended),
            ("intersection", {"ideal": ideal, "other_ideal": ["z"]}, extended),
            ("quotient", {"ideal": product, "other_ideal": ["z"]}, extended),
            ("saturation", {"ideal": product, "polynomial": "z"}, extended),
            ("dimension", {"ideal": ideal}, names),
        ]
        for op, operands, coordinates in cases:
            baseline = calculate(op, operands, variables=coordinates)
            self.assertEqual(baseline["execution"]["backend"], "cpu")
            self.assertTrue(all(s["backend"] == "cpu" for s in baseline["execution"]["stages"]))
            for device in devices:
                backend = device["backend"]
                with self.subTest(operation=op, device=device), patch.dict(os.environ, {
                    "RESOLUTION_GPU_BACKEND": backend, "RESOLUTION_GPU_INDEX": str(device["index"])
                }):
                    record = calculate(op, operands, variables=coordinates, compute="gpu")
                    self.assertEqual(record["result"], baseline["result"])
                    stages = record["execution"]["stages"]
                    device_stages = [s for s in stages if s["backend"] == backend]
                    self.assertTrue(any(s.get("pairs", 0) or s.get("primes", 0) for s in device_stages), stages)
                    for stage in device_stages:
                        self.assertTrue(stage["label"].endswith("_basis"), stage)
                        self.assertEqual(stage["device"]["index"], device["index"])
                    self.assertTrue(any(s["operation"] == "groebner_certification" and s["backend"] == "cpu"
                                        and s["bases"] > 0 for s in stages), stages)


if __name__ == "__main__":
    unittest.main()
