"""Local and global quotient lengths through the public shell, scripts and API."""
import json
import shlex
import subprocess
import unittest

from tests.cli_support import ROOT, run_script
from tests.http_support import RING, endpoint, post


class LocalAlgebraInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def run_shell(self, argv):
        return subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=30)

    def request(self, operation, operands, *, compute="cpu"):
        status, record = post(self.url, operation, operands, compute=compute)
        self.assertEqual(status, 200, record)
        self.assertEqual((record["operation"], record["status"]), (operation, "ok"))
        self.assertEqual(record["result"]["ring"], RING)
        self.assertEqual(record["execution"]["requested"], compute)
        return record

    def round_trip(self, operation, operands, *, compute="cpu"):
        record = self.request(operation, operands, compute=compute)
        replay = record["replay"]
        args = shlex.split(replay["command"])
        self.assertEqual(args[:3], ["./atlas", "algebra", operation])
        self.assertIn(operation.replace("-", "_") + "(", replay["script"])
        for process in (self.run_shell(args), run_script(replay["script"])):
            self.assertEqual(process.returncode, 0, process.stderr)
            repeated = json.loads(process.stdout)
            self.assertEqual((repeated["operation"], repeated["result"]), (operation, record["result"]))
            self.assertEqual(repeated["execution"]["requested"], compute)
        return record

    def assert_basis(self, result, kind, scope, expected):
        self.assertEqual((result["type"], result["scope"]), (kind, scope))
        self.assertIs(result["finite"], expected is not None)
        if expected is None:
            self.assertIsNone(result["dimension"])
            self.assertIsNone(result["basis"])
        else:
            self.assertEqual(result["dimension"], len(expected))
            self.assertCountEqual(result["basis"], expected)

    def test_origin_length_ignores_the_other_point(self):
        # x-1 is a unit at the origin. Globally the simple point at x=1 also counts.
        operands = {"ideal": ["x^2*(x-1)", "y"]}
        for op, scope, basis in (("quotient-basis", "global", ["1", "x", "x^2"]),
                                 ("local-quotient-basis", "origin", ["1", "x"])):
            with self.subTest(operation=op):
                self.assert_basis(self.round_trip(op, operands)["result"], "quotient-basis", scope, basis)

    def test_local_jacobian_records_the_actual_derivatives(self):
        record = self.round_trip("local-jacobian-algebra", {"polynomial": "(2*y^2-2*x^3)/2"})
        self.assert_basis(record["result"], "local-jacobian-algebra", "origin", ["1", "x"])
        self.assertEqual(record["result"]["generators"], ["-3*x^2", "2*y"])

    def test_milnor_and_tjurina_numbers_have_distinct_values_and_labels(self):
        # Singular vdim(std(jacob(f))) and vdim(std(jacob(f)+ideal(f))) at the origin.
        f = "x^5+y^5+x^3*y^3"
        for op, label, expected in (("milnor-number", "Milnor number", 16),
                                    ("tjurina-number", "Tjurina number", 15)):
            with self.subTest(operation=op):
                record = self.round_trip(op, {"polynomial": f})
                result = record["result"]
                self.assertEqual((result["type"], result["scope"], result["value"]),
                                 ("integer", "origin", expected))
                self.assertIs(result["finite"], True)
                args = shlex.split(record["replay"]["command"])
                args.remove("--json")
                displayed = self.run_shell(args)
                self.assertEqual(displayed.returncode, 0, displayed.stderr)
                self.assertIn("Scope: origin", displayed.stdout)
                self.assertIn(f"{label}: {expected}", displayed.stdout)

    def test_finite_zero_and_infinite_quotients(self):
        # The ideal (x-1)*(x,y) defines a line away from the isolated origin.
        mixed = {"ideal": ["(x-1)*x", "(x-1)*y"]}
        cases = (("quotient-basis", mixed, "global", None),
                 ("local-quotient-basis", mixed, "origin", ["1"]),
                 ("local-quotient-basis", {"ideal": ["x-1"]}, "origin", []),
                 ("local-quotient-basis", {"ideal": []}, "origin", None),
                 ("quotient-basis", {"ideal": ["1"]}, "global", []))
        for op, operands, scope, basis in cases:
            with self.subTest(operation=op, operands=operands):
                self.assert_basis(self.request(op, operands)["result"], "quotient-basis", scope, basis)
        infinite_jacobian = self.request("local-jacobian-algebra", {"polynomial": "x^2"})["result"]
        self.assert_basis(infinite_jacobian, "local-jacobian-algebra", "origin", None)
        self.assertEqual(infinite_jacobian["generators"], ["2*x", "0"])
        for op in ("milnor-number", "tjurina-number"):
            for f, expected in (("x+y", 0), ("x^2", None), ("0", None)):
                with self.subTest(operation=op, polynomial=f):
                    result = self.request(op, {"polynomial": f})["result"]
                    self.assertIs(result["finite"], expected is not None)
                    self.assertEqual(result["value"], expected)

    def test_gpu_request_returns_exact_local_length_and_observed_execution(self):
        operands = {"ideal": ["x*(x-1)", "y^2"]}
        cpu = self.request("local-quotient-basis", operands)["result"]
        gpu = self.round_trip("local-quotient-basis", operands, compute="gpu")
        self.assertEqual(gpu["result"], cpu)
        self.assert_basis(cpu, "quotient-basis", "origin", ["1", "y"])
        execution = gpu["execution"]
        self.assertTrue(execution["device"])
        self.assertTrue(execution["stages"])
        if execution["backend"] == "cpu":
            self.assertEqual(execution["device"], "CPU")
        else:
            self.assertTrue(any(s["backend"] in ("opencl", "triton") for s in execution["stages"]))

    def test_operand_errors_are_explicit(self):
        for op, operands in (("local-quotient-basis", {"polynomial": "x^2"}),
                             ("milnor-number", {})):
            with self.subTest(operation=op):
                status, error = post(self.url, op, operands)
                self.assertEqual(status, 400)
                self.assertEqual(error["error"]["code"], "invalid_request")
                self.assertTrue(error["error"]["message"])
                self.assertNotIn("result", error)
        status, error = post(self.url, "tjurina-number", {"polynomial": "z+1"})
        self.assertEqual(status, 400)
        self.assertEqual(error["error"]["code"], "invalid_request")
        shell = self.run_shell(["./atlas", "algebra", "tjurina-number", "--vars", "x,y",
                                "--poly", "z+1", "--json"])
        self.assertEqual(shell.returncode, 2, shell.stderr)
        self.assertEqual(json.loads(shell.stdout), error)

    def test_hypersurface_germ_preconditions_reach_shell_and_api(self):
        for op in ("local-jacobian-algebra", "milnor-number", "tjurina-number"):
            for variables, polynomial in ((["x", "y"], "1+x^2+y^2"), ([], "0")):
                with self.subTest(operation=op, variables=variables, polynomial=polynomial):
                    status, error = post(self.url, op, {"polynomial": polynomial},
                                         ring={**RING, "variables": variables})
                    self.assertEqual(status, 400, error)
                    self.assertEqual(error["error"]["code"], "invalid_request")
                    self.assertTrue(error["error"]["message"])
                    shell = self.run_shell(["./atlas", "algebra", op, "--vars", ",".join(variables),
                                            "--order", "lex", "--compute", "cpu", "--poly", polynomial, "--json"])
                    self.assertEqual(shell.returncode, 2, shell.stderr)
                    self.assertEqual(json.loads(shell.stdout), error)
