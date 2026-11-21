"""The public algebra routes must agree on the exact result and its ring."""

import json
import os
import shlex
import subprocess
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from engine.app import AtlasServer
from engine.polynomial import integer_from_decimal
from tests.cli_support import ROOT, run_prompt, run_script
from tests.http_support import QuietHandler


RING = {"field": "QQ", "variables": ["x", "y"], "order": "lex"}
IDEAL = ["x^2-1", "y-x"]


def request(ideal, polynomial, *, order="lex", compute="cpu"):
    return {"operation": "normal-form", "ring": {**RING, "order": order},
            "operands": {"ideal": ideal, "polynomial": polynomial}, "compute": compute}


def command(ideal, polynomial):
    args = ["algebra", "normal-form", "--vars", "x,y", "--order", "lex",
            "--compute", "cpu", "--json"]
    for generator in ideal:
        args.extend(["--gen", generator])
    return [*args, "--poly", polynomial]


class AlgebraInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = AtlasServer(("127.0.0.1", 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      kwargs={"poll_interval": 0.02}, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/api/algebra"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, data):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        req = Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            response = urlopen(req, timeout=30)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response, parse_int=integer_from_decimal)

    def test_public_gpu_arithmetic_batches_keep_exact_json_results(self):
        from engine.algebra import dispatch
        cases = [
            ("polynomial-arithmetic", ["x","y"],
             {"action":"substitute","polynomial":"x^2+y","images":["-y","x"]}),
            ("modular-arithmetic", [], {"action":"hensel","values":["3","4"],"modulus":"7","other":["-2","0","1"]}),
            ("modular-arithmetic", [], {"action":"multiply","values":[[0,1]],"other":[[0,1]],
                                        "modulus":"2","field_polynomial":["1","1","1"]}),
        ]
        for operation, names, operands in cases:
            request = {"operation":operation,"ring":{"variables":names},"operands":operands}
            cpu = dispatch(dict(request,compute="cpu"))
            status, gpu = self.post(dict(request,compute="gpu"))
            self.assertEqual(status,200,gpu)
            self.assertEqual(gpu["result"],cpu["result"])
            self.assertEqual(gpu["execution"]["requested"],"gpu")


    def shell(self, *args):
        return subprocess.run([str(ROOT / "atlas"), *args],
                              capture_output=True, text=True, timeout=30)

    def test_normal_form_through_shell_prompt_script_and_http(self):
        expression = "x^2+y/2"
        args = command(IDEAL, expression)
        shell = self.shell(*args)
        self.assertEqual(shell.returncode, 0, shell.stderr)
        status, api = self.post(request(IDEAL, expression))
        self.assertEqual(status, 200, api)

        source = (
            'x, y = coordinates("x", "y", coeffs=QQ)\n'
            'I = ideal(x^2-1, y-x, order="lex")\n'
            'R = normal_form(x^2+y/2, I, compute="cpu")\n'
            'print(R.json)\n'
            'print(R.polynomial == 1+y/2)\n'
        )
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        record, equality = script.stdout.splitlines()
        self.assertEqual(equality, "True")

        code, out, err = run_prompt(source + shlex.join(args) + "\nquit\n")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        output = out.replace("atlas> ", "").strip()
        prompt_record, equality, prompt_json = output.split("\n", 2)
        self.assertEqual(equality, "True")

        for result in (json.loads(shell.stdout), api, json.loads(record),
                       json.loads(prompt_record), json.loads(prompt_json)):
            with self.subTest(result=result):
                self.assertEqual((result["status"], result["operation"]), ("ok", "normal-form"))
                self.assertEqual(result["result"], {
                    "type": "polynomial", "ring": RING, "polynomial": "1/2*y+1",
                })
                self.assertEqual(result["execution"]["requested"], "cpu")
                self.assertEqual(result["execution"]["device"], "CPU")

    def test_request_errors_agree_and_invalid_json_is_reported(self):
        status, record = self.post(request(IDEAL, "unknown+1"))
        shell = self.shell(*command(IDEAL, "unknown+1"))
        self.assertEqual(status, 400)
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error"]["code"], "invalid_request")
        self.assertTrue(record["error"]["message"])
        self.assertNotIn("result", record)
        self.assertEqual(shell.returncode, 2, shell.stderr)
        self.assertEqual(json.loads(shell.stdout), record)
        for data in (b"{", ["normal-form"], request("x", "1")):
            with self.subTest(data=data):
                status, error = self.post(data)
                self.assertEqual(status, 400)
                self.assertEqual(error["status"], "error")
                self.assertTrue(error["error"]["message"])

    def test_resultant_changes_with_the_selected_variable_and_replays(self):
        # Eliminating the parameter t gives the equation of the parametrized cusp.
        ring = {**RING, "variables": ["t", "x", "y"]}
        for variable, remaining, answer in (("t", ["x", "y"], "-x^3+y^2"),
                                            ("x", ["t", "y"], "t^3-y")):
            with self.subTest(variable=variable):
                operands = {"polynomial": "t^2-x", "other_polynomial": "t^3-y", "variables": [variable]}
                status, record = self.post(dict(operation="resultant", ring=ring, operands=operands))
                self.assertEqual(status, 200, record)
                result = record["result"]
                self.assertEqual(result, {"type": "polynomial", "ring": {**ring, "variables": remaining},
                                          "polynomial": answer})
                args = shlex.split(record["replay"]["command"])[1:]
                self.assertEqual(args[args.index("--eliminate") + 1], variable)
                for run in (self.shell(*args), run_script(record["replay"]["script"])):
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertEqual(json.loads(run.stdout)["result"], result)
                args.remove("--json")
                code, output, error = run_prompt(shlex.join(args) + "\nquit\n")
                self.assertEqual((code, error), (0, ""))
                self.assertIn("Resultant: " + answer, output)

    def test_resultant_requires_one_declared_variable(self):
        operands = {"polynomial": "x", "other_polynomial": "x+y"}
        for selection in ({}, {"variables": []}, {"variables": ["x", "y"]}, {"variables": ["z"]}):
            with self.subTest(selection=selection):
                status, record = self.post(dict(operation="resultant", ring=RING, operands=operands | selection))
                self.assertEqual((status, record["error"]["code"]), (400, "invalid_request"))
                self.assertNotIn("result", record)

    def test_factorization_through_shell_prompt_script_and_http(self):
        operand = {"polynomial": "-3/5*(2*x-y)^2*(x^2+1)"}
        status, record = self.post(dict(operation="factor", ring=RING, operands=operand))
        self.assertEqual(status, 200, record)
        result = record["result"]
        self.assertEqual((result["type"], result["unit"], result["ring"]),
                         ("factorization", "-12/5", RING))
        self.assertCountEqual(result["factors"], [
            {"polynomial": "x-1/2*y", "multiplicity": 2},
            {"polynomial": "x^2+1", "multiplicity": 1},
        ])

        command = shlex.split(record["replay"]["command"])[1:]
        code, output, error = run_prompt(shlex.join(command) + "\nquit\n")
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(json.loads(output.removeprefix("atlas> ").split("atlas>")[0])["result"], result)
        for run in (self.shell(*command), run_script(record["replay"]["script"])):
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)["result"], result)
        command.remove("--json")
        display = self.shell(*command)
        self.assertEqual(display.returncode, 0, display.stderr)
        for term in ("-12/5", "(x-1/2*y)^2", "(x^2+1)"):
            self.assertIn(term, display.stdout)

        status, error = self.post(dict(operation="factor", ring=RING, operands={"polynomial": "x-x"}))
        self.assertEqual((status, error["error"]["code"]), (400, "invalid_request"))
        self.assertNotIn("result", error)

    def test_http_preserves_a_factor_multiplicity_beyond_the_decimal_limit(self):
        source = "x^" + "9" * 4501
        status, record = self.post(dict(operation="factor", ring=RING, operands={"polynomial": source}))
        self.assertEqual(status, 200, record.get("error"))
        self.assertEqual(record["result"]["factors"], [{"polynomial": "x", "multiplicity": 10**4501 - 1}])

    def test_absolute_factorization_http_worker_cli_and_typed_replay(self):
        operands = {"polynomial": "-3/5*(x^2+y^2)^2"}
        status, record = self.post(dict(operation="absolute-factor", ring=RING, operands=operands))
        self.assertEqual(status, 200, record)
        result = record["result"]
        self.assertEqual(result["type"], "absolute-factorization")
        self.assertEqual((result["unit"], [f["multiplicity"] for f in result["factors"]]), ("-3/5", [2, 2]))
        self.assertEqual(result["ring"]["field"]["degree"], 2)
        command = shlex.split(record["replay"]["command"])[1:]
        for run in (self.shell(*command), run_script(record["replay"]["script"])):
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)["result"], result)
        code, out, err = run_prompt(record["replay"]["script"] + "\nquit\n")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(json.loads(out.replace("atlas> ", "").strip())["result"], result)
        command.remove("--json")
        display = self.shell(*command)
        self.assertEqual(display.returncode, 0, display.stderr)
        self.assertIn("-3/5", display.stdout)
        self.assertIn(result["ring"]["field"]["minimal_polynomial"], display.stdout)

    def test_hensel_factor_lift_through_http_worker_cli_prompt_and_script(self):
        operands = {"polynomial": "y^2-2", "factors": ["y-3", "y+3"], "modulus": "7"}
        status, record = self.post(dict(operation="hensel-factors", ring=RING, operands=operands, compute="cpu"))
        expected = {"type": "hensel-factors", "ring": RING, "left": "y+39", "right": "y+10", "modulus": "49"}
        self.assertEqual((status, record.get("result")), (200, expected), record)
        for run in (self.shell(*shlex.split(record["replay"]["command"])[1:]),
                    run_script(record["replay"]["script"])):
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)["result"], expected)
        code, out, error = run_prompt(record["replay"]["script"] + "\nquit\n")
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(json.loads(out.replace("atlas> ", "").strip())["result"], expected)
        for malformed in ({**operands, "factors": ["y-3"]}, {**operands, "modulus": "7/2"}):
            status, record = self.post(dict(operation="hensel-factors", ring=RING, operands=malformed))
            self.assertEqual((status, record["error"]["code"]), (400, "invalid_request"))

    def test_division_bezout_fraction_and_squarefree_replay(self):
        requests = [
            ("polynomial-division", "x^3+y", "x-1", {"quotient": "x^2+x+1", "remainder": "y+1"}),
            ("extended-gcd", "x^2-1", "x^2-x", {"gcd": "x-1", "left_coefficient": "1", "right_coefficient": "-1"}),
            ("cancel-fraction", "(x-y)*(x+1)", "-2*(x-y)*(y+1)",
             {"numerator":"-1/2*x-1/2", "denominator":"y+1"}),
            ("squarefree-factor", "-2*(x-y)^2*(x+y)^2", None, {"unit": "-2",
             "factors": [{"polynomial":"x^2-y^2", "multiplicity":2}]}),
        ]
        for op, left, right, expected in requests:
            with self.subTest(op=op):
                operands = {"polynomial":left}
                if right is not None:
                    operands["other_polynomial"] = right
                status, api = self.post(dict(operation=op, ring=RING, compute="cpu", operands=operands))
                self.assertEqual(status, 200, api)
                for key, value in expected.items():
                    self.assertEqual(api["result"][key], value)
                cli = self.shell(*shlex.split(api["replay"]["command"])[1:])
                script = run_script(api["replay"]["script"])
                for result in (cli, script):
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout)["result"], api["result"])

    def test_gcd_replays_both_polynomials_through_shell_prompt_and_script(self):
        operands = {"polynomial": "(2*x+y)*(x+1)", "other_polynomial": "-(2*x+y)*(y-1)"}
        expected = {"type": "polynomial", "ring": RING, "polynomial": "x+1/2*y"}
        status, record = self.post(dict(operation="gcd", ring=RING, operands=operands, compute="cpu"))
        self.assertEqual(status, 200, record)
        self.assertEqual(record["result"], expected)
        shell = self.shell(*shlex.split(record["replay"]["command"])[1:])
        script = run_script(record["replay"]["script"])
        for repeated in (shell, script):
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertEqual(json.loads(repeated.stdout)["result"], expected)
        code, text, err = run_prompt(record["replay"]["script"] + "quit\n")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(json.loads(text.replace("atlas> ", "").strip())["result"], expected)
        for invalid in ({"polynomial": "x"}, {**operands, "ideal": ["x"]},
                        {**operands, "other_polynomial": ["y"]}):
            status, record = self.post(dict(operation="gcd", ring=RING, operands=invalid))
            self.assertEqual(status, 400, record)
            self.assertEqual(record["error"]["code"], "invalid_request")

    def test_zero_constants_and_empty_ideal(self):
        for ideal, polynomial, expected in (([], "1/2", "1/2"), (["0"], "x-1", "x-1"),
                                             (["-3/5"], "x+y", "0"), (IDEAL, "0", "0")):
            with self.subTest(ideal=ideal, polynomial=polynomial):
                status, record = self.post(request(ideal, polynomial))
                self.assertEqual(status, 200, record)
                self.assertEqual(record["result"]["polynomial"], expected)

    def test_ideal_declaration_preserves_order_and_multiplicity(self):
        ring = {**RING, "variables": ["y", "x"]}
        operands = {"ideal": ["(x+y)/2", "0", "2*x", "2*x"]}
        status, record = self.post(dict(operation="ideal", ring=ring, operands=operands, compute="cpu"))
        self.assertEqual(status, 200, record)
        self.assertEqual(record["result"], {"type": "ideal", "ring": ring,
                                            "generators": ["1/2*y+1/2*x", "0", "2*x", "2*x"]})
        self.assertNotIn("basis_ref", record)
        shell = self.shell(*shlex.split(record["replay"]["command"])[1:])
        script = run_script(record["replay"]["script"])
        for repeated in (shell, script):
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertEqual(json.loads(repeated.stdout)["result"], record["result"])

    def test_ideal_sum_and_product_replay_wide_inputs(self):
        degree, coeff = 2**69 + 3, 2**142 + 1
        left, right = f"{coeff}*x^{degree}", "x^2"
        for operation, expected in (
            ("ideal-sum", [left, right]),
            ("ideal-product", [f"{coeff}*x^{degree+2}"]),
        ):
            with self.subTest(operation=operation):
                status, record = self.post({"operation": operation, "ring": RING,
                                            "operands": {"ideal": [left], "other_ideal": [right]}})
                self.assertEqual(status, 200, record)
                self.assertEqual(record["result"], {"type": "ideal", "ring": RING, "generators": expected})
                self.assertIn(f"result = {operation.replace('-', '_')}(I, J", record["replay"]["script"])
                shell = self.shell(*shlex.split(record["replay"]["command"])[1:])
                script = run_script(record["replay"]["script"])
                for run in (shell, script):
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertEqual(json.loads(run.stdout)["result"], record["result"])

    def test_roadmap_operations_dispatch_with_their_own_operands_and_result_rings(self):
        cases = [
            ("groebner", {"ideal": IDEAL}, "basis", "generators", ["x-y", "y^2-1"]),
            ("membership", {"ideal": IDEAL, "polynomial": "y^2-1"}, "boolean", "value", True),
            ("membership", {"ideal": IDEAL, "polynomial": "y"}, "boolean", "value", False),
            ("eliminate", {"ideal": IDEAL, "variables": ["x"]}, "ideal", "generators", ["y^2-1"]),
            ("intersection", {"ideal": ["x"], "other_ideal": ["y"]}, "ideal", "generators", ["x*y"]),
            ("quotient", {"ideal": ["x^2*y"], "other_ideal": ["x"]}, "ideal", "generators", ["x*y"]),
            ("saturation", {"ideal": ["x^2*y"], "polynomial": "x"}, "ideal", "generators", ["y"]),
            ("dimension", {"ideal": []}, "integer", "value", 2),
            ("dimension", {"ideal": ["1"]}, "integer", "value", -1),
        ]
        for operation, operands, kind, key, expected in cases:
            with self.subTest(operation=operation, operands=operands):
                status, record = self.post({"ring": RING, "operation": operation, "operands": operands,
                                            "compute": "cpu"})
                self.assertEqual(status, 200, record)
                self.assertEqual((record["status"], record["operation"]), ("ok", operation))
                result = record["result"]
                self.assertEqual(result["type"], kind)
                self.assertEqual(result[key], expected)
                self.assertEqual(result["ring"]["variables"], ["y"] if operation == "eliminate" else ["x", "y"])
                self.assertIn(result["ring"]["order"], ("lex", "grevlex"))
                self.assertEqual(result["ring"]["field"], "QQ")

    def test_gpu_selection_reaches_modular_replay(self):
        # Large exact coefficients force a modular basis even for these two generators.
        # The measured replay counts also exercise the CPU fallback on hosts without OpenCL.
        data = request([f"x^2+{2**260+1}*y", "x*y+1"], "x*y", order="grevlex", compute="gpu")
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="opencl"):
            status, record = self.post(data)
        self.assertEqual(status, 200, record)
        self.assertEqual(record["result"]["polynomial"], "-1")
        execution = record["execution"]
        self.assertEqual(execution["requested"], "gpu")
        replays = [s for s in execution["stages"] if s["operation"] == "groebner_replay"]
        self.assertTrue(replays, execution)
        for stage in replays:
            self.assertGreater(stage["batches"], 0)
            self.assertGreater(stage["primes"], 0)
        if any(stage["backend"] == "opencl" for stage in execution["stages"]):
            self.assertIn("opencl", execution["device"].lower())
        else:
            self.assertEqual(execution["device"], "CPU")
            self.assertTrue(any("opencl" in reason.lower() for reason in execution["fallback_reasons"]))
