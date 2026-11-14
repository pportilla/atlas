"""Public tangent cone requests, terminal displays, and portable replays."""
import json
import shlex
import unittest

from tests.cli_support import run_script, shell
from tests.http_support import RING, endpoint, post


CONE = "tangent-cone"
SERIES = "tangent-cone-hilbert-series"
CUSP = {"ideal": ["((y+2/3)-(x+1/2))^2-(x+1/2)^3"], "point": ["-3/6", "-4/6"]}


class TangentConeInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def calculate(self, operation, operands, compute="cpu"):
        status, record = post(self.url, operation, operands, compute=compute)
        self.assertEqual(status, 200, record)
        self.assertEqual((record["operation"], record["result"]["type"]), (operation, operation))
        self.assertEqual(record["result"]["ring"], RING)
        self.assertEqual(record["execution"]["requested"], compute)
        self.assertTrue(record["execution"]["device"])
        self.assertTrue(record["execution"]["stages"])
        return record

    def display(self, record):
        args = shlex.split(record["replay"]["command"])
        args.remove("--json")
        shown = shell(args)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        return shown.stdout.splitlines()

    def test_translated_cusp_in_tangent_coordinates_and_replays(self):
        for operation in (CONE, SERIES):
            with self.subTest(operation=operation):
                record = self.calculate(operation, CUSP)
                result = record["result"]
                self.assertEqual((result["scope"], result["point"]), ("point", ["-1/2", "-2/3"]))
                self.assertEqual((result["dimension"], result["multiplicity"]), (1, 2))
                lines = self.display(record)
                self.assertIn("Point: (-1/2, -2/3)", lines)
                self.assertIn("Dimension: 1", lines)
                self.assertIn("Multiplicity: 2", lines)
                if operation == CONE:
                    self.assertEqual(result["coordinate_system"], "tangent")
                    self.assertEqual(result["generators"], ["x^2-2*x*y+y^2"])
                    self.assertIn("Tangent coordinates: Δx = x - (-1/2), Δy = y - (-2/3)", lines)
                    self.assertIn("Tangent cone generators: Δx^2-2*Δx*Δy+Δy^2", lines)
                else:
                    self.assertEqual(result["numerator"], [{"degree": 0, "coefficient": 1},
                                                          {"degree": 2, "coefficient": -1}])
                    self.assertEqual(result["denominator_power"], 2)
                    self.assertIn("H(t) = (1 - t^2)/(1 - t)^2", lines)

                replay = record["replay"]
                self.assertIn(operation.replace("-", "_") + "(", replay["script"])
                for copy in (shell(shlex.split(replay["command"])), run_script(replay["script"])):
                    self.assertEqual(copy.returncode, 0, copy.stderr)
                    loaded = json.loads(copy.stdout)
                    self.assertEqual(loaded["result"], result)
                    self.assertEqual(loaded["execution"]["requested"], "cpu")

    def test_omitted_and_explicit_origin_device_request(self):
        for operation in (CONE, SERIES):
            with self.subTest(operation=operation):
                operands = {"ideal": ["y^2-x^3"]}
                origin = self.calculate(operation, operands)["result"]
                self.assertEqual(origin["scope"], "origin")
                self.assertNotIn("point", origin)
                explicit = self.calculate(operation, {**operands, "point": [0, "-0/5"]}, "gpu")
                self.assertEqual(explicit["result"], {**origin, "scope": "point", "point": ["0", "0"]})
                record = explicit["execution"]
                self.assertIn(f"Device: {record['device']} (requested gpu)", self.display(explicit))

    def test_series_of_wide_monomial_complete_intersection_stays_sparse(self):
        degree = 2**64 + 1
        record = self.calculate(SERIES, {"ideal": [f"x^{degree}", f"y^{degree}"]})
        result = record["result"]
        self.assertEqual(result["numerator"], [{"degree": 0, "coefficient": 1},
                                              {"degree": degree, "coefficient": -2},
                                              {"degree": 2 * degree, "coefficient": 1}])
        self.assertEqual((result["denominator_power"], result["dimension"], result["multiplicity"]),
                         (2, 0, degree**2))
        self.assertIn(f"H(t) = (1 - 2*t^{degree} + t^{2 * degree})/(1 - t)^2", self.display(record))

    def test_invalid_point_and_request_options(self):
        for operation in (CONE, SERIES):
            for operands in ({"ideal": [], "point": [0]}, {"ideal": [], "point": ["1/0", 0]},
                             {"ideal": [], "degrees": [0, 1]}):
                with self.subTest(operation=operation, operands=operands):
                    status, error = post(self.url, operation, operands)
                    self.assertEqual((status, error["error"]["code"]), (400, "invalid_request"))
                    self.assertNotIn("result", error)
                    if operands.get("point") == ["1/0", 0]:
                        invalid = shell(["./atlas", "algebra", operation, "--vars", "x,y", "--point", "1/0,0",
                                         "--order", "lex", "--json"])
                        self.assertEqual(invalid.returncode, 2, invalid.stderr)
                        self.assertEqual(json.loads(invalid.stdout), error)
            status, error = post(self.url, operation, {"ideal": []}, ring={**RING, "field": "GF(3)"})
            self.assertEqual((status, error["error"]["code"]), (400, "invalid_request"))
