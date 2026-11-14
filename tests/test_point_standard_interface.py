"""Point-local standard bases and weak normal forms across the public entry points."""
from fractions import Fraction
import json
import shlex
import subprocess
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from engine.polynomial import parse_expression
from tests.test_cli_session import ROOT, run_script
from tests.test_http_basis_refs import RING, endpoint, post


def substitute_x(text, x, *, derivative=False):
    """Evaluate at x, retaining the exact coefficients of powers of y."""
    coefficients = {}
    for (i, j), c in parse_expression(text, ["x", "y"], _reference=True).items():
        if derivative:
            if i == 0:
                continue
            c *= i
            i -= 1
        coefficients[j] = coefficients.get(j, 0) + c * x**i
    return {j: c for j, c in coefficients.items() if c}


def at_point(text, x, y, *, derivative=False):
    return sum(c * y**j for j, c in substitute_x(text, x, derivative=derivative).items())


def shell(args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)


class PointStandardInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def request(self, operation, operands, *, ring=RING, compute="cpu"):
        status, record = post(self.url, operation, operands, ring=ring, compute=compute)
        self.assertEqual(status, 200, record)
        self.assertEqual((record["status"], record["operation"]), ("ok", operation))
        self.assertEqual(record["result"]["type"], operation)
        self.assertEqual(record["result"]["scope"], "point")
        self.assertEqual(record["result"]["ring"], {**RING, **ring})
        if operation == "local-normal-form":
            self.assertIn(record["result"]["reduction"], ("standard", "weak"))
        self.assertEqual(record["execution"]["requested"], compute)
        return record

    def assert_replays(self, record):
        replay = record["replay"]
        self.assertIn(record["operation"].replace("-", "_") + "(", replay["script"])
        args = shlex.split(replay["command"])
        self.assertEqual(args[:3], ["./atlas", "algebra", record["operation"]])
        for repeated in (run_script(replay["script"]), shell(args)):
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            loaded = json.loads(repeated.stdout)
            self.assertEqual(loaded["result"], record["result"])
            self.assertEqual(loaded["execution"]["requested"], record["execution"]["requested"])

    def test_basis_at_a_nonzero_rational_point_uses_original_coordinates(self):
        a, b = Fraction(-1, 2), 2
        record = self.request("local-standard-basis", {"ideal": ["x+1/2"], "point": ["-3/6", b]})
        result = record["result"]
        self.assertEqual(result["point"], ["-1/2", "2"])
        # Every generator vanishes along x=a, and one generates (x-a) locally.
        generators = result["generators"]
        self.assertTrue(generators)
        for g in generators:
            self.assertEqual(substitute_x(g, a), {})
        self.assertTrue(any(at_point(g, a, b, derivative=True) for g in generators))
        self.assert_replays(record)

    def test_weak_reduction_at_a_rational_point_and_negative_cli_argument(self):
        op = "local-normal-form"
        operands = {"ideal": ["x+1/2"], "polynomial": "x+y", "point": ["-3/6", 2]}
        record = self.request(op, operands)
        result = record["result"]
        self.assertEqual(result["point"], ["-1/2", "2"])
        a = Fraction(-1, 2)
        unit, remainder = result["unit"], result["remainder"]
        self.assertEqual(at_point(unit, a, 2), 1)
        # Modulo this principal ideal the exact condition u*f-r in I is
        # equivalent to the polynomial in y vanishing after x=-1/2.
        self.assertEqual(substitute_x(f"({unit})*(x+y)-({remainder})", a), {})
        if result["reduction"] == "standard":
            self.assertTrue(all(i == 0 for i, _ in parse_expression(remainder, ["x", "y"], _reference=True)))
        self.assert_replays(record)
        explicit = shell(["./atlas", "algebra", op, "--vars", "x,y", "--order", "lex",
                          "--gen", "x+1/2", "--poly", "x+y", "--point", "-3/6,2", "--json"])
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        self.assertEqual(json.loads(explicit.stdout)["result"], result)

    def test_weak_reduction_and_a_nonconstant_multiplier(self):
        # In the local ring x*(1-y)=y gives x=y/(1-y).
        f = "x*(1-y)-y"
        result = self.request("local-normal-form", {"ideal": [f], "polynomial": "x"})["result"]
        self.assertEqual(result["reduction"], "weak")
        self.assertTrue(parse_expression(result["remainder"], ["x", "y"], _reference=True))
        self.assertEqual(at_point(result["unit"], 0, 0), 1)
        # Changing the constant part of the relation to y² requires a multiplier
        # to express x with a polynomial in y. Here (1-y)*x-y² is the input generator.
        data = {"ideal": ["x*(1-y)-y^2"], "polynomial": "x"}
        record = self.request("local-normal-form", data)
        result = record["result"]
        self.assertEqual(result["reduction"], "standard")
        self.assertEqual(parse_expression(result["unit"], ["x", "y"], _reference=True),
                         {(0, 0): 1, (0, 1): -1})
        self.assertEqual(parse_expression(result["remainder"], ["x", "y"], _reference=True), {(0, 2): 1})
        self.assert_replays(record)

    def test_unquoted_large_integer_coordinate_replays_without_decimal_literal_limits(self):
        digits = "9" * 4501
        # Build the wire bytes directly so this client does not convert the coordinate.
        body = (b'{"operation":"local-normal-form","ring":{"variables":["x"],"order":"lex"},'
                b'"operands":{"ideal":[],"polynomial":"1","point":[' + digits.encode() +
                b']},"compute":"cpu"}')
        req = Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            response = urlopen(req, timeout=30)
        except HTTPError as error:
            response = error
        with response:
            record = json.load(response)
            self.assertEqual(response.status, 200, record.get("error"))
        result = record["result"]
        self.assertEqual(result["point"], [digits])
        self.assertEqual((result["remainder"], result["unit"]), ("1", "1"))
        replay = record["replay"]
        self.assertIn("algebra_request(", replay["script"])
        for process in (run_script(replay["script"]), shell(shlex.split(replay["command"]))):
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout)["result"], result)

    def test_implicit_origin_and_scalar_ring(self):
        operands = {"ideal": ["x+1/2"], "polynomial": "y"}
        for point in ({}, {"point": [0, "0/7"]}):
            with self.subTest(point=point):
                result = self.request("local-normal-form", {**operands, **point})["result"]
                self.assertEqual(result["point"], ["0", "0"])
                self.assertEqual(result["remainder"], "0")
                self.assertEqual(at_point(result["unit"], 0, 0), 1)
        scalar = {**RING, "variables": []}
        for ideal, expected in (([], "5/7"), (["-2/3"], "0")):
            with self.subTest(ideal=ideal):
                data = {"ideal": ideal, "point": []}
                basis = self.request("local-standard-basis", data, ring=scalar)
                self.assertEqual(basis["result"]["point"], [])
                self.assertEqual(basis["result"]["generators"], ["1"] if ideal else [])
                self.assert_replays(basis)
                form = self.request("local-normal-form", {**data, "polynomial": "5/7"}, ring=scalar)
                self.assertEqual(form["result"]["point"], [])
                self.assertEqual((form["result"]["unit"], form["result"]["remainder"]), ("1", expected))
                self.assert_replays(form)

    def test_gpu_request_preserves_normal_form_and_unit(self):
        data = {"ideal": ["x+1/2"], "polynomial": "x+y", "point": ["-1/2", 2]}
        cpu = self.request("local-normal-form", data)["result"]
        record = self.request("local-normal-form", data, compute="gpu")
        self.assertEqual(record["result"], cpu)
        self.assertTrue(record["execution"]["device"])
        self.assertTrue(record["execution"]["stages"])
        self.assert_replays(record)

    def test_saved_basis_restored_only_inside_the_session(self):
        cases = (
            # A translated zero-dimensional ideal gives a nonzero exact normal form.
            (["x", "y"], ["(x+1/2)^2+y^3", "(x+1/2)*y+y^3"], ["-1/2", "0"],
             "1+(x+1/2)+(x+1/2)^4+y", "standard"),
            # Rebuilding this basis used to change the weak representative of x-y.
            (["x", "y", "z"], ["-y^2+y-z", "y^2+z"], ["0"]*3, "x-y", None),
            # These two genuinely weak reductions used to produce different remainders.
            (["x", "y", "z"], ["-x^3+y^2-y*z+x", "x^2+x*z"], ["0"]*3, "x*z", "weak"),
        )
        for names, generators, point, query, reduction in cases:
            with self.subTest(generators=generators, point=point):
                source = (
                    f"{', '.join(names)} = coordinates({', '.join(map(repr, names))})\n"
                    f"I = ideal({', '.join(generators)}, order='lex')\n"
                    f"B = local_standard_basis(I, point={point!r}, compute='cpu')\n"
                    "print(B.json)\n"
                    f"h = local_normal_form({query}, B, compute='cpu')\n"
                    "print(h.json)\n"
                )
                session = run_script(source)
                self.assertEqual(session.returncode, 0, session.stderr)
                basis, saved = map(json.loads, session.stdout.splitlines())
                def constructed(stages):
                    return any(stage.get("label") == "local_standard_basis" for stage in stages)

                self.assertTrue(constructed(basis["execution"]["stages"]))
                saved_stages = saved["execution"]["stages"]
                operations = [stage["operation"] for stage in saved_stages]
                self.assertIn("restore_local_basis", operations)
                self.assertIn("local_normal_form", operations)
                self.assertFalse(constructed(saved_stages))
                result = saved["result"]
                self.assertEqual(result["point"], point)
                self.assertNotEqual(result["remainder"], "0")
                if reduction:
                    self.assertEqual(result["reduction"], reduction)
                data = {"ideal": basis["result"]["generators"], "point": point, "polynomial": query}
                raw = self.request("local-normal-form", data, ring={**RING, "variables": names})
                records = [raw]
                replay = saved["replay"]
                for repeated in (shell(shlex.split(replay["command"])), run_script(replay["script"])):
                    self.assertEqual(repeated.returncode, 0, repeated.stderr)
                    records.append(json.loads(repeated.stdout))
                for record in records:
                    self.assertEqual(record["result"], result)
                    stages = record["execution"]["stages"]
                    self.assertTrue(constructed(stages))
                    self.assertNotIn("restore_local_basis", [stage["operation"] for stage in stages])

    def test_public_requests_cannot_mark_generators_as_a_known_basis(self):
        data = {"ideal": ["x*y", "x^2+y"], "point": [0, 0], "polynomial": "y^2"}
        for operands, options in ((data, {"known_basis": True}), ({**data, "known_basis": True}, {})):
            status, error = post(self.url, "local-normal-form", operands, **options)
            self.assertEqual(status, 400, error)
            self.assertEqual(error["error"]["code"], "invalid_request")
            self.assertNotIn("result", error)

    def test_point_validation_across_transports(self):
        op = "local-normal-form"
        args = {"ideal": ["x"], "polynomial": "x+y"}
        for point in (None, [], [1], [1, 2, 3], "1,2", [True, 0], [0.5, 0],
                      ["1+2", 0], ["x", 0], ["1/0", 2]):
            with self.subTest(point=point):
                status, error = post(self.url, op, {**args, "point": point})
                self.assertEqual(status, 400)
                self.assertEqual(error["error"]["code"], "invalid_request")
                self.assertTrue(error["error"]["message"])
                self.assertNotIn("result", error)
        status, error = post(self.url, "local-standard-basis", {"ideal": ["x"], "point": ["1/0", 2]})
        self.assertEqual(status, 400)
        invalid = shell(["./atlas", "algebra", "local-standard-basis", "--vars", "x,y",
                         "--gen", "x", "--point", "1/0,2", "--json"])
        self.assertEqual(invalid.returncode, 2, invalid.stderr)
        self.assertEqual(json.loads(invalid.stdout), error)
