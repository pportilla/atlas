"""Public arithmetic actions, ring ownership, and replay on both backends."""
import json
import shlex
import subprocess
import unittest
from engine.algebra import calculate_algebra, dispatch
from engine.session import new_session
from tests.cli_support import ROOT, run_prompt, run_script


class PolynomialArithmeticModes(unittest.TestCase):
    def test_exact_actions_and_live_parent(self):
        session = new_session(None)
        list(session.run('x,y=coordinates("x","y")\nR=x.parent'))
        cases = [
            ('"add",x/3,y/5', "x/3+y/5"),
            ('"subtract",x/3,y/5', "x/3-y/5"),
            ('"multiply",(2^330+1)*x^37,y^3/7', "(2^330+1)*x^37*y^3/7"),
            ('"divide",x-y,7/13', "(x-y)*13/7"),
            ('"power",x,exponent=2^80', "x^(2^80)"),
            ('"differentiate",x^7*y^2,variable=x', "7*x^6*y^2"),
            ('"substitute",x^2+y,images=[y,x]', "y^2+x"),
            ('"substitute",x*y,images=[2,3]', "6"),
        ]
        for args, expected in cases:
            for mode in ("cpu", "gpu"):
                with self.subTest(args=args, mode=mode):
                    list(session.run(f'F=polynomial_arithmetic({args},compute="{mode}")\n'
                                     f'matches=F==({expected})\nparent_ok=F.parent==R'))
                    self.assertTrue(session.variables["matches"])
                    self.assertTrue(session.variables["parent_ok"])
        list(session.run('old=x\nx,y=coordinates("x","y",fresh=True)'))
        old, x, y = (session.variables[name] for name in ("old", "x", "y"))
        for args, options in [(("multiply", old, x), {}), (("differentiate", old), {"variable":x}),
                              (("substitute", old), {"images":[x, y]})]:
            with self.subTest(action=args[0]), self.assertRaises(ValueError):
                session.builtins["polynomial_arithmetic"](*args, **options)

    def test_cli_worker_script_and_replay(self):
        args = {"operation":"polynomial-arithmetic","ring":{"variables":["x","y"]},
                "operands":{"action":"substitute","polynomial":"x^2+y","images":["-y","x"]},"compute":"cpu"}
        reference = dispatch(args)
        status, worker = calculate_algebra(args)
        self.assertEqual(status, 200, worker)
        self.assertEqual(reference["result"], worker["result"])
        shell = subprocess.run(shlex.split(worker["replay"]["command"]),text=True,capture_output=True,cwd=ROOT,check=True)
        script = run_script(worker["replay"]["script"])
        self.assertEqual(script.returncode,0,script.stderr)
        for output in (shell.stdout, script.stdout):
            self.assertEqual(json.loads(output)["result"],reference["result"])
        command = ('x,y=coordinates("x","y")\na=x\nx,y=coordinates("x","y",fresh=True)\n'
                   'F=polynomial_arithmetic("add",x,y)\nF=polynomial_arithmetic("add",x,a)\n'
                   'print(F==x+y)\nquit\n')
        status, output, error = run_prompt(command)
        self.assertEqual(status,0)
        self.assertTrue(error)
        self.assertIn("True",output)
