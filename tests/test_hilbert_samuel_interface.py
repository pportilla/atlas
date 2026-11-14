"""Hilbert–Samuel functions and multiplicities across HTTP, CLI and script replay."""
import json
import shlex
import unittest
from urllib.request import Request, urlopen

from engine.polynomial import integer_from_decimal
from tests.test_cli_session import run_script
from tests.test_http_basis_refs import RING, endpoint, post
from tests.test_point_standard_interface import shell


FUNCTION = "hilbert-samuel-function"
MULTIPLICITY = "hilbert-samuel-multiplicity"


class HilbertSamuelInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def calculate(self, operation, operands):
        status, record = post(self.url, operation, operands)
        self.assertEqual(status, 200, record)
        self.assertEqual((record["operation"], record["result"]["ring"]), (operation, RING))
        return record

    def assert_replays(self, record):
        replay = record["replay"]
        args = shlex.split(replay["command"])
        self.assertEqual(args[:3], ["./atlas", "algebra", record["operation"]])
        for process in (shell(args), run_script(replay["script"])):
            self.assertEqual(process.returncode, 0, process.stderr)
            copy = json.loads(process.stdout, parse_int=integer_from_decimal)
            self.assertEqual(copy["result"], record["result"])
            self.assertEqual(copy["execution"]["requested"], record["execution"]["requested"])

    def test_cusp_default_degrees_and_origin_scope(self):
        # The initial form y² leaves 1 standard monomial in degree 0 and 2 in each positive degree.
        ideal = ["y^2-x^3"]
        record = self.calculate(FUNCTION, {"ideal": ideal})
        result = record["result"]
        self.assertEqual((result["type"], result["scope"]), (FUNCTION, "origin"))
        self.assertNotIn("point", result)
        self.assertEqual((result["dimension"], result["multiplicity"]), (1, 2))
        self.assertEqual(result["degrees"], list(range(6)))
        self.assertEqual(result["values"], [1, 3, 5, 7, 9, 11])
        self.assertEqual(result["graded_values"], [1, 2, 2, 2, 2, 2])
        self.assert_replays(record)
        command = shlex.split(record["replay"]["command"])
        command.remove("--json")
        shown = shell(command)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn(["5", "11", "2"], [line.split() for line in shown.stdout.splitlines()])

        origin = self.calculate(MULTIPLICITY, {"ideal": ideal})
        scalar = origin["result"]
        self.assertEqual((scalar["type"], scalar["value"], scalar["dimension"], scalar["scope"]),
                         ("integer", 2, 1, "origin"))
        self.assertNotIn("point", scalar)
        explicit = self.calculate(MULTIPLICITY, {"ideal": ideal, "point": [0, "-0/5"]})
        self.assertEqual(explicit["result"], {**scalar, "scope": "point", "point": ["0", "0"]})
        self.assert_replays(explicit)
        command = shlex.split(explicit["replay"]["command"])
        command.remove("--json")
        shown = shell(command)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("multiplicity", shown.stdout.lower())
        self.assertIn("dimension", shown.stdout.lower())

    def test_translated_ideal_order_duplicates_and_empty_degrees(self):
        operands = {"ideal": ["(x+1/2)^2", "y-2"], "point": ["-3/6", 2]}
        for degrees, values, graded in (([2, "0", 2, 1], [2, 1, 2, 2], [0, 1, 0, 1]),
                                        ([], [], [])):
            with self.subTest(degrees=degrees):
                record = self.calculate(FUNCTION, {**operands, "degrees": degrees})
                result = record["result"]
                self.assertEqual((result["scope"], result["point"]), ("point", ["-1/2", "2"]))
                self.assertEqual((result["dimension"], result["multiplicity"]), (0, 2))
                self.assertEqual(result["degrees"], list(map(int, degrees)))
                self.assertEqual((result["values"], result["graded_values"]), (values, graded))
                self.assert_replays(record)

    def test_zero_ring_has_zero_function_and_multiplicity(self):
        for operation in (FUNCTION, MULTIPLICITY):
            with self.subTest(operation=operation):
                result = self.calculate(operation, {"ideal": ["x-1"]})["result"]
                self.assertEqual(result["dimension"], -1)
                if operation == FUNCTION:
                    self.assertEqual((result["values"], result["graded_values"]), ([0]*6, [0]*6))
                    self.assertEqual(result["multiplicity"], 0)
                else:
                    self.assertEqual(result["value"], 0)

    def test_large_json_integer_and_decimal_string_replay_exactly(self):
        # A line has one graded monomial per degree and cumulative length n+1.
        # Use an unquoted integer beyond the host's decimal conversion guard and
        # the equivalent string a browser can send without rounding its value.
        digits = "9" * 4501
        data = ('{"operation":"hilbert-samuel-function","ring":{"variables":["x","y"]},'
                '"operands":{"ideal":["y"],"degrees":[' + digits + ',"' + digits + '"]}}')
        req = Request(self.url, data=data.encode(), headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=30) as response:
            self.assertEqual(response.status, 200)
            record = json.load(response, parse_int=integer_from_decimal)
        n = 10**4501-1
        result = record["result"]
        self.assertEqual(result["degrees"], [n, n])
        self.assertEqual(result["values"], [n+1, n+1])
        self.assertEqual(result["graded_values"], [1, 1])
        self.assert_replays(record)

    def test_reject_invalid_degree_types_and_unsupported_degrees_operand(self):
        for degrees in (None, "1,2", [True], [1.0], [-1], ["-1"], ["+1"],
                        ["1.0"], [" 1"], ["２"], ["1_0"], [""], [[1]]):
            with self.subTest(degrees=degrees):
                status, record = post(self.url, FUNCTION, {"ideal": [], "degrees": degrees})
                self.assertEqual(status, 400, record)
                self.assertEqual(record["error"]["code"], "invalid_request")
                self.assertNotIn("result", record)
        status, record = post(self.url, MULTIPLICITY, {"ideal": [], "degrees": []})
        self.assertEqual(status, 400, record)
        self.assertEqual(record["error"]["code"], "invalid_request")
        for text in ("-1", "0,,2", "1.5"):
            with self.subTest(text=text):
                process = shell(["./atlas", "algebra", FUNCTION, "--vars", "x,y", "--degrees", text, "--json"])
                self.assertEqual(process.returncode, 2)
