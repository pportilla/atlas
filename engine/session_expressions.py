"""Evaluate Atlas expressions against explicitly supplied operations and session values."""

import ast
from fractions import Fraction
from io import StringIO
import operator
from subprocess import TimeoutExpired
import tokenize

from .session_polynomial import Polynomial


class SessionError(ValueError):
    def __init__(self, message, line):
        super().__init__(message)
        self.line = line


def _parse(source):
    # Change the exponent token before parsing so powers bind more tightly than
    # signs and products. Quoted polynomials and comments keep their contents.
    try:
        tokens = tokenize.generate_tokens(StringIO(source).readline)
        normalized = tokenize.untokenize(
            token._replace(string="**") if token.type == tokenize.OP and token.string == "^"
            else token for token in tokens
        )
        return ast.parse(normalized).body
    except tokenize.TokenError as error:
        message, (line, _) = error.args
        raise SessionError(message, line) from None
    except SyntaxError as error:
        raise SessionError(error.msg, error.lineno or 1) from None


_COMPARISONS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
}
_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}


def _numeric(value):
    return type(value) is int or isinstance(value, (Fraction, Polynomial))


def _arithmetic(op, left, right):
    operation = _OPERATORS.get(type(op))
    if operation is None or not (_numeric(left) and _numeric(right)):
        raise ValueError("Arithmetic requires rational numbers or polynomials and +, -, *, / or ^.")
    if isinstance(op, ast.Div) and not isinstance(left, Polynomial):
        left = Fraction(left)
    if isinstance(op, ast.Pow):
        if isinstance(right, Fraction) and right.denominator == 1:
            right = int(right)
        if type(right) is not int:
            raise ValueError("Use an integer exponent.")
        if type(left) is int and right < 0:
            left = Fraction(left)
    return operation(left, right)


def _field(value, name):
    if name.startswith("_"):
        raise ValueError("Only public Atlas fields are available.")
    if isinstance(value, dict) and name in value:
        return value[name]
    getter = getattr(type(value), "session_attribute", None)
    if getter is None:
        raise ValueError(f"This value has no Atlas field {name!r}.")
    return getter(value, name)


class Evaluator:
    def __init__(self, builtins, *, reserved=()):
        self.builtins = dict(builtins)
        self.reserved = frozenset(reserved)
        self.variables = {}

    def run(self, source, *, transcript=None):
        """Execute in this session; yield (True, value) for top-level expression previews."""
        statements = _parse(source)
        if transcript is not None:
            entries = transcript.entries(source, len(statements))
        for index, statement in enumerate(statements):
            # A failed block can assign bindings before the line that fails.
            # Restore them so later saved statements replay in the same state.
            compound = isinstance(statement, (ast.For, ast.If))
            before = self.variables.copy() if transcript is not None and compound else None
            try:
                value = self._statement(statement)
            except (SessionError, KeyboardInterrupt):
                if before is not None:
                    self.variables = before
                raise
            if transcript is not None:
                transcript.statements.append(entries[index])
            if value is not None:
                yield True, value

    def _statement(self, statement):
        try:
            if isinstance(statement, ast.Expr):
                return self._value(statement.value)
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                self._bind(statement.targets[0], self._value(statement.value))
            elif isinstance(statement, ast.For) and not statement.orelse:
                items = self._value(statement.iter)
                if not isinstance(items, (tuple, list, dict)):
                    raise ValueError("A for loop requires a list, tuple, or record.")
                for item in items:
                    self._bind(statement.target, item)
                    for child in statement.body:
                        self._statement(child)
            elif isinstance(statement, ast.If):
                suite = statement.body if self._value(statement.test) else statement.orelse
                for child in suite:
                    self._statement(child)
            else:
                raise ValueError("Use an expression, assignment, for loop, or if statement.")
        except SessionError:
            raise
        except ZeroDivisionError:
            raise SessionError("Division by zero.", statement.lineno) from None
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError,
                OSError, RuntimeError, TimeoutExpired) as error:
            raise SessionError(str(error), statement.lineno) from None

    def _bind(self, target, value):
        if isinstance(target, ast.Name):
            targets, values = [target], [value]
        elif isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, (tuple, list)) or len(value) != len(target.elts):
                raise ValueError(f"Expected {len(target.elts)} values to unpack.")
            targets, values = target.elts, value
        else:
            raise ValueError("Assign to a name or a list of names.")

        names = []
        for item in targets:
            if not isinstance(item, ast.Name) or item.id.startswith("_"):
                raise ValueError("Assign to public variable names.")
            if item.id in self.builtins or item.id in self.reserved:
                raise ValueError(f"{item.id!r} is a reserved Atlas name.")
            if item.id in names:
                raise ValueError(f"Repeated assignment target {item.id!r}.")
            names.append(item.id)
        self.variables.update(zip(names, values))

    def _value(self, node):
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, str, bool) or node.value is None:
                return node.value
            raise ValueError("Use integers or fractions such as 1/2 for exact coefficients.")
        if isinstance(node, ast.Name):
            if node.id in self.variables:
                return self.variables[node.id]
            if node.id in self.builtins:
                return self.builtins[node.id]
            raise ValueError(f"Unknown name {node.id!r}.")
        if isinstance(node, ast.List):
            return [self._value(element) for element in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._value(element) for element in node.elts)
        if isinstance(node, ast.Attribute):
            return _field(self._value(node.value), node.attr)
        if isinstance(node, ast.Subscript):
            value, index = self._value(node.value), self._value(node.slice)
            if isinstance(value, dict):
                if index not in value:
                    raise ValueError(f"This record has no key {index!r}.")
            elif not isinstance(value, (tuple, list, str)) or type(index) is not int:
                raise ValueError("Index a list or tuple by integer, or a record by key.")
            return value[index]
        if isinstance(node, ast.UnaryOp):
            value = self._value(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            if _numeric(value) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return +value if isinstance(node.op, ast.UAdd) else -value
            raise ValueError("A sign requires a rational number or polynomial.")
        if isinstance(node, ast.BinOp):
            return _arithmetic(node.op, self._value(node.left), self._value(node.right))
        if isinstance(node, ast.Compare):
            left = self._value(node.left)
            for op, next_node in zip(node.ops, node.comparators):
                compare = _COMPARISONS.get(type(op))
                if compare is None:
                    raise ValueError("Use ==, !=, <, <=, > or >= for comparisons.")
                right = self._value(next_node)
                if not compare(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                raise ValueError("Call an Atlas function or method by name.")
            function = self._value(node.func)
            if not callable(function):
                raise ValueError("This value is not an Atlas function.")
            args = [self._value(arg) for arg in node.args]
            kwargs = {}
            for keyword in node.keywords:
                if not keyword.arg or keyword.arg.startswith("_"):
                    raise ValueError("Use public, named keyword arguments.")
                if keyword.arg in kwargs:
                    raise ValueError(f"Repeated keyword {keyword.arg!r}.")
                kwargs[keyword.arg] = self._value(keyword.value)
            return function(*args, **kwargs)
        raise ValueError(f"Unsupported Atlas expression: {type(node).__name__}.")
