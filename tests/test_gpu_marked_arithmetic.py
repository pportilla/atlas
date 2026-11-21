"""Check marked scalar arithmetic, geometric-coordinate ownership and public replay."""
import json
import shlex
import subprocess
import threading
import unittest
from urllib.request import Request, urlopen

from engine.algebra import dispatch, calculate_algebra
from engine.app import AtlasServer
from engine.gpu_devices import triton_device
from engine.polynomial import parse_expression
from tests.cli_support import ROOT, run_prompt, run_script
from tests.http_support import QuietHandler


def request(action, polynomial, *, compute="cpu", field=None, names=("x", "y"), **operands):
    return {"operation": "polynomial-arithmetic", "ring": {"field": field or {"generator":"a","embedding":"I"},
            "variables": list(names)}, "compute": compute,
            "operands": {"action": action, "polynomial": polynomial, **operands}}


CASES = (
    ("add", "a*x+1", {"other_polynomial": "2*x-a"}, "a*x+2*x-a+1"),
    ("subtract", "a*x+1", {"other_polynomial": "2*x-a"}, "a*x-2*x+a+1"),
    ("multiply", "a*x+y", {"other_polynomial": "a*x-y"}, "-x^2-y^2"),
    ("divide", "a*x+y", {"other_polynomial": "1+a"}, "(1+a)*x/2+(1-a)*y/2"),
    ("differentiate", "a*x^2+3*y*x-a", {"variable": "x"}, "2*a*x+3*y"),
    ("power", "a*x+y", {"exponent": "3"}, "-a*x^3-3*x^2*y+3*a*x*y^2+y^3"),
    ("substitute", "x^2+a*y", {"images": ["a*y", "a*x"]}, "-y^2-x"),
)


class MarkedArithmeticTests(unittest.TestCase):
    def check_actions(self, mode):
        for action, poly, operands, expected in CASES:
            with self.subTest(action=action):
                data = dispatch(request(action, poly, compute=mode, **operands))
                result = data["result"]
                self.assertEqual(result["ring"]["field"]["minimal_polynomial"], "a^2+1")
                self.assertEqual(parse_expression(result["polynomial"], ["x", "y", "a"]),
                                 parse_expression(expected, ["x", "y", "a"]))
                self.assertEqual(data["execution"]["backend"], "cpu" if mode=="cpu" else "triton")

    def test_exact_actions(self):
        self.check_actions("cpu")

    def test_physical_actions(self):
        if triton_device() is None:
            self.skipTest("Requires a physical Triton device")
        self.check_actions("gpu")
        for exponent, expected in ((2**80+3, "-a"), (2**80+4, "1")):
            for policy in ("cpu", "gpu"):
                data = dispatch(request("power", "a", exponent=exponent, names=(), compute=policy))
                self.assertEqual(data["result"]["polynomial"], expected)

    def test_nonmonomial_images_and_wide_powers(self):
        for policy in ("cpu", "gpu"):
            req = request("substitute", "a*x^2+y^2", images=["1+a*y", "0"], compute=policy)
            result = dispatch(req)["result"]["polynomial"]
            self.assertEqual(parse_expression(result, ["x","y","a"]),
                             parse_expression("a-2*y-a*y^2", ["x","y","a"]))
            req = request("substitute", "x^"+str(2**70+1), names=("x",), images=["a"], compute=policy)
            self.assertEqual(dispatch(req)["result"]["polynomial"], "a")
            req = request("multiply", "2*a*x", other_polynomial="a/3", compute=policy,
                          field={"generator":"a","embedding":"sqrt(2)/2", "minimal_polynomial":"a^2-1/2"})
            result = dispatch(req)["result"]
            self.assertEqual((result["ring"]["field"]["minimal_polynomial"], result["polynomial"]),
                             ("2*a^2-1", "1/3*x"))

    def test_scalar_ring_wide_coefficients_and_selected_embedding(self):
        coefficient = 2**320+3
        variants = []
        for mark in ("sqrt(2)", "-sqrt(2)"):
            data = request("multiply", f"{coefficient}*b", names=(), other_polynomial="b/3",
                           field={"generator":"b","embedding":mark})
            for mode in ("gpu", "cpu"):
                actual = dispatch(dict(data, compute=mode))["result"]
                self.assertEqual(actual["polynomial"], f"{2*coefficient//3}" if coefficient % 3 == 0
                                 else f"{2*coefficient}/3")
            variants.append(actual["ring"]["field"])
        self.assertNotEqual(variants[0]["root_index"], variants[1]["root_index"])
        self.assertEqual(variants[0]["minimal_polynomial"], variants[1]["minimal_polynomial"])

    def test_catalogued_parent_and_failed_independent_input(self):
        program = """
            x=coordinates("x")
            F=absolute_factor(x^2+1)
            K=F.ambient
            g=F.factors[0].polynomial
            u=coordinates("u",coeffs=K.coeffs)
            U=u.parent
            p=polynomial("a+u",ring=U)
            h=polynomial_arithmetic("multiply",p,p,compute="gpu")
            print(h.parent == U)
            print(h == p^2)
            u=coordinates("u",coeffs=K.coeffs,fresh=True)
        """
        from textwrap import dedent
        body=dedent(program).strip()+"\n"
        cmd='h=polynomial_arithmetic("multiply",p,u,compute="cpu")\n'
        shell = run_script(body)
        self.assertEqual((shell.returncode, shell.stdout.splitlines()), (0, ["True","True"]), shell.stderr)
        failed = run_script(body+cmd)
        self.assertNotEqual(failed.returncode, 0)
        status, output, error = run_prompt(body+cmd+'print(h.parent == U)\nquit\n')
        self.assertEqual(status, 0)
        self.assertIn("different", error.lower())
        self.assertEqual(output.count("True"), 3)
        self.assertNotIn("Traceback", error)

    def test_cli_http_and_saved_script(self):
        req=request("divide","a*x",other_polynomial="2+a")
        server=AtlasServer(("127.0.0.1",0), QuietHandler)
        thread=threading.Thread(target=server.serve_forever, kwargs={"poll_interval":0.02}, daemon=True)
        thread.start()
        try:
            post = Request(f"http://127.0.0.1:{server.server_port}/api/algebra", data=json.dumps(req).encode(),
                           headers={"Content-Type":"application/json"})
            with urlopen(post, timeout=30) as received:
                record=json.load(received)
        finally:
            server.shutdown(); server.server_close(); thread.join()
        shell=subprocess.run(shlex.split(record["replay"]["command"]), cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(shell.returncode, 0, shell.stderr)
        self.assertEqual(json.loads(shell.stdout)["result"],record["result"])
        script=run_script(record["replay"]["script"])
        self.assertEqual(script.returncode,0,script.stderr)
        self.assertEqual(json.loads(script.stdout)["result"],record["result"])

    def test_invalid_field_or_divisor(self):
        for extra in ({"minimal_polynomial":"a^2-1"}, {"minimal_polynomial":"0"}, {"root_index":2}, {"degree":5}):
            status, _ = calculate_algebra(request("power","a", exponent=3,
                                           field={"generator":"a","embedding":"I", **extra}))
            self.assertEqual(status,400)
        for g in ("a^2+1","x"):
            status, data=calculate_algebra(request("divide", "x", other_polynomial=g))
            self.assertNotEqual(status,200,data)
