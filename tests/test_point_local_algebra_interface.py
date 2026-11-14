"""Explicit rational points and the legacy origin through public local-algebra routes."""
import json
import shlex
import unittest

from engine.polynomial import parse_expression
from tests.test_cli_session import run_prompt, run_script
from tests.test_http_basis_refs import RING, endpoint, post
from tests.test_point_standard_interface import shell


OPERATIONS = ("local-quotient-basis", "local-jacobian-algebra", "milnor-number", "tjurina-number")
CUSP = "(y-2)^2-(x+1/2)^3"
POINT = ["-3/6", 2]
NORMALIZED = ["-1/2", "2"]


def terms(text):
    return parse_expression(text, RING["variables"], _reference=True)


def operands(operation, polynomial, generators):
    return {"ideal": generators} if operation == "local-quotient-basis" else {"polynomial": polynomial}


class PointLocalAlgebraInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def calculate(self, operation, inputs, *, compute="cpu"):
        status, record = post(self.url, operation, inputs, compute=compute)
        self.assertEqual(status, 200, record)
        self.assertEqual((record["status"], record["operation"]), ("ok", operation))
        self.assertEqual(record["result"]["ring"], RING)
        self.assertEqual(record["execution"]["requested"], compute)
        return record

    def assert_portable(self, record, *, prompt=False):
        replay = record["replay"]
        script = replay["script"]
        self.assertIn(record["operation"].replace("-", "_") + "(", script)
        command = shlex.split(replay["command"])
        self.assertEqual(command[:3], ["./atlas", "algebra", record["operation"]])
        explicit_point = "point" in record["result"]
        self.assertEqual(any(arg.partition("=")[0] == "--point" for arg in command), explicit_point)
        outputs = []
        for process in (shell(command), run_script(script)):
            self.assertEqual(process.returncode, 0, process.stderr)
            outputs.append(json.loads(process.stdout))
        if prompt:
            code, out, err = run_prompt(script + "quit\n")
            self.assertEqual((code, err), (0, ""))
            outputs.append(json.loads(out.replace("atlas> ", "").strip()))
        for copy in outputs:
            self.assertEqual(copy["result"], record["result"])
            self.assertEqual(copy["execution"]["requested"], record["execution"]["requested"])

    def test_translated_cusp_and_local_length_through_all_routes(self):
        # At p, set u=x+1/2 and v=y-2. Both algebras have ideal (u²,v),
        # since u-1 is invertible in the first and f belongs to its Jacobian.
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                data = operands(operation, CUSP, ["(x+1/2)^2*(x-1/2)", "y-2"])
                record = self.calculate(operation, {**data, "point": POINT})
                result = record["result"]
                self.assertEqual((result["scope"], result["point"]), ("point", NORMALIZED))
                self.assertIs(result["finite"], True)
                if operation in ("milnor-number", "tjurina-number"):
                    self.assertEqual((result["type"], result["value"]), ("integer", 2))
                else:
                    self.assertEqual(result["dimension"], 2)
                    self.assertCountEqual(list(map(terms, result["basis"])), list(map(terms, ["1", "x+1/2"])))
                if operation == "local-jacobian-algebra":
                    self.assertEqual(list(map(terms, result["generators"])),
                                     list(map(terms, ["-3*(x+1/2)^2", "2*(y-2)"])))
                self.assert_portable(record, prompt=True)
        # Exercise a dash-leading value passed as a separate argv element.
        manual = shell(["./atlas", "algebra", "milnor-number", "--vars", "x,y", "--order", "lex",
                        "--point", "-3/6,2", "--poly", CUSP, "--compute", "cpu", "--json"])
        self.assertEqual(manual.returncode, 0, manual.stderr)
        self.assertEqual(json.loads(manual.stdout)["result"]["point"], NORMALIZED)

    def test_omitted_origin_and_explicit_zero_survive_replay(self):
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                inputs = operands(operation, "y^2-x^3", ["x^2", "y"])
                origin = self.calculate(operation, inputs)
                explicit = self.calculate(operation, {**inputs, "point": ["-0/5", 0]})
                self.assertEqual(origin["result"]["scope"], "origin")
                self.assertNotIn("point", origin["result"])
                self.assertEqual(explicit["result"], {**origin["result"], "scope": "point", "point": ["0", "0"]})
                self.assert_portable(origin)
                self.assert_portable(explicit)

    def test_finite_zero_and_infinite_at_the_point(self):
        for operation in OPERATIONS:
            smooth = operands(operation, "y-2", ["x-1"])
            nonisolated = operands(operation, "(x+1/2)^2", ["x+1/2"])
            for inputs, finite in ((smooth, True), (nonisolated, False)):
                with self.subTest(operation=operation, finite=finite):
                    result = self.calculate(operation, {**inputs, "point": POINT})["result"]
                    self.assertEqual(result["point"], NORMALIZED)
                    self.assertIs(result["finite"], finite)
                    if operation in ("milnor-number", "tjurina-number"):
                        self.assertEqual(result["value"], 0 if finite else None)
                    else:
                        self.assertEqual(result["dimension"], 0 if finite else None)
                        self.assertEqual(result["basis"], [] if finite else None)

    def test_invalid_points_equations_and_global_requests_have_structured_errors(self):
        for operation in OPERATIONS:
            inputs = operands(operation, CUSP, ["x+1/2"])
            for point in (None, [-0.5, 2], ["-1/0", 2], [1], [False, 2], ["1+1", 0]):
                with self.subTest(operation=operation, point=point):
                    status, error = post(self.url, operation, {**inputs, "point": point})
                    self.assertEqual(status, 400, error)
                    self.assertEqual(error["error"]["code"], "invalid_request")
                    self.assertNotIn("result", error)
            if operation != "local-quotient-basis":
                with self.subTest(operation=operation, condition="off hypersurface"):
                    status, error = post(self.url, operation, {"polynomial": "x", "point": POINT})
                    self.assertEqual(status, 400, error)
                    self.assertEqual(error["error"]["code"], "invalid_request")
                    bad_cli = shell(["./atlas", "algebra", operation, "--vars", "x,y", "--poly", "x",
                                     "--point", "-3/6,2", "--json"])
                    self.assertEqual(bad_cli.returncode, 2, bad_cli.stderr)
                    self.assertEqual(json.loads(bad_cli.stdout), error)
        for operation in ("quotient-basis", "dimension"):
            with self.subTest(global_operation=operation):
                status, error = post(self.url, operation, {"ideal": ["x"], "point": [0, 0]})
                self.assertEqual(status, 400)
                self.assertEqual(error["error"]["code"], "invalid_request")
