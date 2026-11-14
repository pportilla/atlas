"""Exercise Zariski tangent queries through the public HTTP and terminal interfaces."""
import json
import shlex
import unittest

from tests.cli_support import run_script, shell
from tests.http_support import RING, endpoint, post


SPACE = "zariski-tangent-space"
EMBEDDING = "embedding-dimension"


class ZariskiTangentInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def check_replays(self, operation, ring, operands, fields):
        status, record = post(self.url, operation, operands, ring=ring)
        self.assertEqual(status, 200, record)
        self.assertEqual((record["status"], record["operation"], record["execution"]["requested"]),
                         ("ok", operation, "cpu"))
        expected = {"type": SPACE if operation == SPACE else "integer", "ring": ring, **fields}
        self.assertEqual(record["result"], expected)
        command = shlex.split(record["replay"]["command"])
        self.assertEqual(command[:3], ["./atlas", "algebra", operation])
        if "point" in expected:
            self.assertEqual(command[command.index("--point") + 1], ",".join(expected["point"]))
        else:
            self.assertNotIn("--point", command)
        for execution in (shell(command), run_script(record["replay"]["script"])):
            self.assertEqual(execution.returncode, 0, execution.stderr)
            copy = json.loads(execution.stdout)
            self.assertEqual((copy["operation"], copy["result"], copy["execution"]["requested"]),
                             (operation, expected, "cpu"))
        return command

    def test_translated_oblique_line_wire_and_direction_display(self):
        ring = {**RING, "variables": ["u", "v", "w"]}
        u, v, w = "(u+1/2)", "(v-2)", "(w+1/3)"
        operands = {"ideal": [f"2*{u}+3*{v}+{w}+{v}^2", f"{u}-2*{v}+3*{w}"],
                    "point": ["-3/6", 2, "-2/6"]}
        common = {"scope": "point", "point": ["-1/2", "2", "-1/3"], "ambient_dimension": 3, "jacobian_rank": 2}
        for operation, result in (
            (SPACE, {"dimension": 1, "coordinate_system": "tangent", "equations": ["u+11/7*w", "v-5/7*w"],
                     "basis": [["-11/7", "5/7", "1"]]}),
            (EMBEDDING, {"value": 1}),
        ):
            with self.subTest(operation=operation):
                command = self.check_replays(operation, ring, operands, {**common, **result})
                command.remove("--json")
                shown = shell(command)
                self.assertEqual(shown.returncode, 0, shown.stderr)
                lines = shown.stdout.splitlines()
                for line in ("Point: (-1/2, 2, -1/3)", "Jacobian rank: 2", "Ambient dimension: 3",
                             "Embedding dimension: 1"):
                    self.assertIn(line, lines)
                if operation == SPACE:
                    self.assertIn("Tangent equations: Δu+11/7*Δw, Δv-5/7*Δw", lines)
                    self.assertIn("Basis of tangent directions: (-11/7, 5/7, 1)", lines)

    def test_implicit_and_explicit_origin_survive_portable_replay(self):
        for args, scope in (({}, {"scope": "origin"}), ({"point": ["-0/5", 0]},
                                                       {"scope": "point", "point": ["0", "0"]})):
            common = {**scope, "jacobian_rank": 0, "ambient_dimension": 2}
            for operation, fields in (
                (SPACE, {"dimension": 2, "coordinate_system": "tangent", "equations": [],
                         "basis": [["1", "0"], ["0", "1"]]}),
                (EMBEDDING, {"value": 2}),
            ):
                with self.subTest(operation=operation, scope=scope):
                    self.check_replays(operation, RING, {"ideal": ["y^2-x^3"], **args}, {**common, **fields})

    def test_off_scheme_and_invalid_inputs_have_structured_errors(self):
        for operation in (SPACE, EMBEDDING):
            for operands, ring, message, options in (
                ({"ideal": ["x"], "point": [1, 0]}, RING, "scheme", ["--gen", "x", "--point", "1,0"]),
                ({"ideal": [], "point": ["1/0", 0]}, RING, "denominator", ["--point", "1/0,0"]),
                ({"ideal": [], "point": [1]}, RING, "point", ["--point", "1"]),
                ({"ideal": [], "degrees": [0, 1]}, RING, "degrees", None),
                ({"ideal": []}, {**RING, "field": "GF(3)"}, "field", ["--field", "GF(3)"]),
            ):
                with self.subTest(operation=operation, message=message):
                    status, record = post(self.url, operation, operands, ring=ring)
                    self.assertEqual((status, record["status"], record["error"]["code"]),
                                     (400, "error", "invalid_request"))
                    self.assertIn(message, record["error"]["message"].lower())
                    self.assertNotIn("result", record)
                    if options is not None:
                        cli = shell(["./atlas", "algebra", operation, "--vars", "x,y", "--order", "lex",
                                     "--json", *options])
                        self.assertEqual(cli.returncode, 2)
                        self.assertEqual(json.loads(cli.stdout), record)
