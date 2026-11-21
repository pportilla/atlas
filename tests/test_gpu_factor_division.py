"""Exact CPU, native reconstruction with injected witnesses, and physical Triton parity."""
from pathlib import Path
import os
import unittest
from unittest.mock import patch

from engine.algebra import dispatch
from engine.gpu_devices import triton_device
from engine.session import new_session
from engine.session_help import algebra_value_help
from tests.cli_support import run_prompt, run_script


def request(op, f, g=None, names=("x",), order="grevlex", compute="cpu"):
    operands = {"polynomial": f}
    if op == "normal-form":
        operands["ideal"] = [g]
    elif g is not None:
        operands["other_polynomial"] = g
    return {"operation": op, "ring": {"variables": list(names), "order": order}, "operands": operands, "compute": compute}


def requests():
    common = "(x-1)^3*(x+3)^2*(2*x-1)*(3*x+2)*(x+2)*(x^2+1)"
    one = f"({common})*(x^2+x+1)"
    two = f"({common})*(x^3+2)"
    f = "(x+y+2)^12*(2^219+3)/7"
    divisor = "2*x*y^2-3*y^4"
    return [request(op, one) for op in ("factor", "squarefree-factor", "absolute-factor")] + [
        request("gcd", one, two), request("extended-gcd", one, two), request("cancel-fraction", one, two),
        *(request(op, f, divisor, ("x", "y"), order) for op in ("polynomial-division", "normal-form")
          for order in ("lex", "grevlex"))]


def python_witnesses(name, parameters, buffers, count, output, extra):
    if name == "poly_root_screen":
        coeff, n = buffers[0]
        for i in range(count):
            acc = 0
            for c in reversed(coeff[:n]):
                acc = (acc * i + c) % count
            output[i] = int(acc == 0)
    else:
        width, = parameters
        (ptr, n), = buffers
        head, tail, exponents = ptr[:width], ptr[width:2*width], ptr[2*width:n]
        for i in range(count):
            e = exponents[i*width:(i+1)*width]
            k = 0
            while all(a >= b for a, b in zip(e, head)):
                e = [a-u+v for a, u, v in zip(e, head, tail)]
                k += 1
            output[i] = k
    return 1


class TritonFactorDivision(unittest.TestCase):
    def test_public_results_and_ring_ownership(self):
        code = ('x, y = coordinates("x", "y")\nP=x.parent\nD=polynomial_division(x^3+y,x-1)\n'
                'E=extended_gcd(x^2-1,x^2-x)\n'
                'print(D.quotient*(x-1)+D.remainder==x^3+y, D.quotient.parent==P)\n'
                'print(E.left_coefficient*(x^2-1)+E.right_coefficient*(x^2-x)==E.gcd, E.gcd.parent==P)\n')
        cli = run_script(code)
        self.assertEqual((cli.returncode, cli.stdout.splitlines()), (0, ["True True", "True True"]), cli.stderr)
        exitcode, text, errors = run_prompt(code + 'X,Y=coordinates("x","y",fresh=True)\n'
                                            'D=polynomial_division(x^3+y,X-1)\nprint(D.ambient==P)\nquit\n')
        self.assertEqual(exitcode, 0)
        self.assertIn("True True", text)
        self.assertIn("True", text.splitlines()[-1] if not text.endswith("atlas> ") else text[-30:])
        self.assertIn("different", errors+text)
        session = new_session(None)
        list(session.run(code.replace('print(', 'check = (')))
        for key, fields in (("D", ("quotient", "remainder")), ("E", ("gcd", "left_coefficient", "right_coefficient"))):
            value = session.variables[key]
            with patch("engine.algebra.dispatch", side_effect=AssertionError("Help ran a solver")):
                card = algebra_value_help(key, value, session.variables)
            for field in fields:
                self.assertIn(f"{key}.{field}", card)
                list(session.run(f"shown={key}.{field}"))
            self.assertIn("docs/tutorial/polynomials-and-ideals.md#divide-polynomials-and-find-bezout-coefficients", card)
        guide = Path("docs/tutorial/polynomials-and-ideals.md").read_text()
        self.assertIn("## Divide polynomials and find Bezout coefficients", guide)

    def test_reduced_fraction_zero_wide_and_ownership(self):
        for mode in ("cpu", "gpu"):
            for f,g,variables,pair in [
                ("0","x^2",("x",),("0","1")), ("-6/5","-9/2",(),("4/15","1")),
                ("(x^18446744073709551619+y)*(x-2)","-5*(x^18446744073709551619+y)*(y-3)",
                 ("x","y"),("-1/5*x+2/5","y-3"))]:
                data = dispatch(request("cancel-fraction",f,g,variables,compute=mode))["result"]
                self.assertEqual((data["numerator"],data["denominator"]), pair)
            for order, expected in (("lex", ("-3/7", "2*y^2+x")), ("grevlex", ("-3/14", "y^2+1/2*x"))):
                case = request("cancel-fraction","-3/7*(x-y)","(x-y)*(x+2*y^2)",("x","y"),order,mode)
                data = dispatch(case)["result"]
                self.assertEqual((data["numerator"],data["denominator"]), expected)
            with self.assertRaisesRegex(ValueError, "nonzero denominator"):
                dispatch(request("cancel-fraction","1","0",compute=mode))
        session = new_session(None)
        list(session.run('x, y=coordinates("x","y")\nP=x.parent\nuse_ring(P)\n'
                         'C=cancel_fraction(3,2)\nC2=cancel_fraction(x^2-y^2,2*x+2*y)\n'
                         'X, Y=coordinates("x","y",fresh=True)'))
        self.assertIs(session.variables["C"].ambient, session.variables["P"])
        with self.assertRaisesRegex(Exception,"different"):
            list(session.run('C=cancel_fraction(x,Y)'))
        self.assertIs(session.variables["C"].denominator.parent, session.variables["P"])
        card = algebra_value_help("C",session.variables["C"],session.variables)
        self.assertIn("C.denominator",card)
        self.assertIn("docs/tutorial/polynomials-and-ideals.md#cancel-a-rational-function",card)
        list(session.run('y1=C.numerator\ny2=C.denominator'))

    def test_live_coefficients_beyond_public_decimal_limit(self):
        session = new_session(None)
        list(session.run('x=coordinates("x")\nc=2^17000+1\n'
            'C=cancel_fraction(c*(x+1), (x+1)*(x-1), compute="gpu")\n'
            'D=polynomial_division(c*x^2+c,x,compute="gpu")\n'
            'F=squarefree_factor(c*(x+1)^2,compute="gpu")\n'
            'checks=(C.numerator==c,C.denominator==x-1,D.quotient==c*x,D.remainder==c,F.unit==c)'))
        self.assertEqual(session.variables["checks"], (True,)*5)

    def test_squarefree_groups_multiplicities(self):
        case = request("squarefree-factor", "-3*(x-1)^2*(x+1)^2*(x^2+1)^3")
        expected = [{"polynomial": "x^2-1", "multiplicity": 2}, {"polynomial": "x^2+1", "multiplicity": 3}]
        for mode in ("cpu", "gpu"):
            result = dispatch(dict(case, compute=mode))["result"]
            self.assertEqual(result["unit"], "-3")
            self.assertEqual(result["factors"], expected)
            case2 = request("squarefree-factor", "(x*y)^18446744073709551618", names=("x","y"), compute=mode)
            self.assertEqual(dispatch(case2)["result"]["factors"], [{"polynomial":"x*y","multiplicity":18446744073709551618}])

    def test_zero_constants_wide_powers_and_multivariate_domain(self):
        for compute in ("cpu","gpu"):
            d = dispatch(request("polynomial-division", "x^18446744073709551619+x", "x^18446744073709551618", compute=compute))["result"]
            self.assertEqual((d["quotient"], d["remainder"]), ("x", "x"))
            d = dispatch(request("polynomial-division", "-3/7", "5/9", names=(), compute=compute))["result"]
            self.assertEqual((d["quotient"],d["remainder"]), ("-27/35","0"))
            e = dispatch(request("extended-gcd", "x^18446744073709551619", "x^18446744073709551618", compute=compute))["result"]
            self.assertEqual((e["gcd"], e["left_coefficient"],e["right_coefficient"]), ("x^18446744073709551618","0","1"))
            e = dispatch(request("extended-gcd","0","0",names=(),compute=compute))["result"]
            self.assertEqual([e[k] for k in ("gcd","left_coefficient","right_coefficient")], ["0"]*3)
            with self.assertRaisesRegex(ValueError, "one common variable"):
                dispatch(request("extended-gcd","x","y",("x","y"),compute=compute))
            with self.assertRaisesRegex(ValueError, "nonzero divisor"):
                dispatch(request("polynomial-division","x","0",compute=compute))

    def test_root_denominator_divisible_by_screen_prime_and_constants(self):
        f = "(8191*x-1)^2*(x+1)^6*(x^2+1)"
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton"), patch(
                "engine.gpu_factor_division._run", side_effect=python_witnesses) as screen:
            for op in ("factor", "absolute-factor", "squarefree-factor"):
                a = dispatch(request(op, f, compute="cpu"))["result"]
                b = dispatch(request(op, f, compute="gpu"))["result"]
                self.assertEqual(a,b)
            case = request("cancel-fraction", f, "(8191*x-1)^2*(x+1)^7", compute="gpu")
            reduced = dispatch(case)["result"]
            self.assertEqual((reduced["numerator"],reduced["denominator"]), ("x^2+1","x+1"))
            self.assertGreaterEqual(screen.call_count, 4)
        for mode in ("cpu", "gpu"):
            c = dispatch(request("squarefree-factor", "-11/9", names=(), compute=mode))["result"]
            self.assertEqual((c["unit"],c["factors"]), ("-11/9",[]))
            with self.assertRaisesRegex(ValueError, "zero polynomial"):
                dispatch(request("squarefree-factor", "0", compute=mode))

    def test_exact_reconstruction_and_fallback_with_injected_witnesses(self):
        for case in requests():
            reference = dispatch(case)["result"]
            with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton"), patch(
                "engine.gpu_factor_division._run", side_effect=python_witnesses) as launch:
                result = dispatch(dict(case, compute="gpu"))
            with self.subTest(op=case["operation"],order=case["ring"]["order"]):
                self.assertGreater(launch.call_count, 0)
                self.assertEqual(result["result"], reference)
        def corrupt(name, params, buffers, count, output, extra):
            # False positive roots and incorrect divisibility counts cannot change the result.
            if name == "poly_root_screen":
                output[10] = 1
            else:
                for i in range(count):
                    output[i] = 42
            return 1
        for case in requests()[::2]:
            with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton"), patch("engine.gpu_factor_division._run", side_effect=corrupt):
                self.assertEqual(dispatch(dict(case, compute="gpu"))["result"], dispatch(case)["result"])

    def test_physical_triton_matches_cpu(self):
        if not triton_device():
            self.skipTest("Requires a working physical Triton device")
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton"):
            for case in requests():
                gpu = dispatch(dict(case, compute="gpu"))
                with self.subTest(op=case["operation"], order=case["ring"]["order"]):
                    self.assertEqual(gpu["result"], dispatch(case)["result"])
                    self.assertTrue(any(s["backend"]=="triton" and s["operation"].startswith("poly_")
                                        for s in gpu["execution"]["stages"]), gpu["execution"])
