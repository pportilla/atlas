"""Architecture regressions for the C++ computation boundary defined in AGENTS.md."""
import ast
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

from engine.session_expressions import Evaluator, SessionError


ENGINE = Path(__file__).resolve().parents[1] / "engine"

# Python may exchange symbolic expressions with existing callers and compare
# them in diagnostic modes. New production modules must use the native APIs.
SYMBOLIC_INTERFACES = {"field_coefficient"}
ARRAY_TRANSPORT = {"gpu_rank", "gpu_replay", "gpu_trace"}
MATH_PACKAGES = {"math", "sympy", "numpy", "scipy", "gmpy2", "flint", "mpmath"}
ARITHMETIC_FREE = {
    "hilbert_series", "hilbert_samuel", "kernel_json", "local_kernel", "point_geometry",
    "point_initial_form", "point_order", "quotient_monomials",
}


def scalar_formula(function):
    """Identify standalone formulas even when they use no math package."""
    body = function.body
    if ast.get_docstring(function) is not None:
        body = body[1:]
    if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
        return False
    operators = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.UAdd, ast.USub)
    expressions = [body[-1].value]
    calculation = False
    for stmt in body[:-1]:
        if isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) for t in stmt.targets):
            expressions.append(stmt.value)
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name) and isinstance(stmt.op, operators):
            expressions.append(stmt.value)
            calculation = True
        else:
            return False
    for expression in expressions:
        for node in ast.walk(expression):
            if isinstance(node, operators):
                calculation = True
            elif isinstance(node, (ast.Name, ast.Load, ast.BinOp, ast.UnaryOp)):
                continue
            elif not (isinstance(node, ast.Constant) and type(node.value) in (int, float, complex)):
                return False
    return calculation


class NativeMathBoundaryTests(unittest.TestCase):
    def test_computational_dependencies(self):
        violations = []
        for path in sorted(ENGINE.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and scalar_formula(node):
                    violations.append(f"{path.name}:{node.lineno} defines a Python arithmetic formula")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"divmod", "pow"}:
                    violations.append(f"{path.name}:{node.lineno} calls {node.func.id}")
                if isinstance(node, ast.ImportFrom) and not node.level:
                    modules = [node.module or ""]
                    if node.module == "math" and all(a.name == "isfinite" for a in node.names):
                        continue
                elif isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                else:
                    continue
                for module in modules:
                    package = module.split(".")[0]
                    if package == "sympy" and path.stem in SYMBOLIC_INTERFACES:
                        continue
                    if package == "numpy" and path.stem in ARRAY_TRANSPORT:
                        continue
                    if package in MATH_PACKAGES or module == "triton.language":
                        violations.append(f"{path.name}:{node.lineno} imports {module}")
        self.assertEqual(violations, [], "Implement new mathematics in native/.")

    def test_hilbert_and_point_adapters_contain_only_transport(self):
        for name in sorted(ARITHMETIC_FREE):
            with self.subTest(module=name):
                tree = ast.parse((ENGINE / (name + ".py")).read_text())
                for node in ast.walk(tree):
                    self.assertNotIsInstance(node, (ast.BinOp, ast.AugAssign), ast.dump(node))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        self.assertNotIn(node.func.id, ("divmod", "pow", "round", "sum"))


class NativeScalarTests(unittest.TestCase):
    def test_prompt_scalar_arithmetic_uses_the_kernel_and_preserves_types(self):
        evaluator = Evaluator({})
        from engine.native_library import affine_request
        with patch("engine.native_polynomial.affine_request", wraps=affine_request) as native:
            expressions = ["2^100", "(5/7)^-3", "3/1", "-4", "7/3 > 2"]
            results = [value for _, value in evaluator.run("\n".join(expressions))]
            self.assertEqual(results, [2**100, Fraction(343, 125), Fraction(3), -4, True])
            self.assertIs(type(results[0]), int)
            self.assertIs(type(results[2]), Fraction)
            self.assertTrue(all(call.args[0][0] == "scalar_arithmetic" for call in native.call_args_list))
            self.assertIn("compare", [call.args[0][1] for call in native.call_args_list])
        with self.assertRaisesRegex(SessionError, "Division by zero"):
            list(evaluator.run("0^-1"))
