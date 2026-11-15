"""Evaluate Atlas expressions against explicitly supplied operations and session values."""

import ast
from fractions import Fraction
from io import StringIO
from itertools import count
import operator
from subprocess import TimeoutExpired
import tokenize

from .native_polynomial import scalar_arithmetic
from .session_polynomial import Polynomial
from .session_quotient import QuotientElement
from .session_values import Divisor, DualComplex


_EVALUATION_ERRORS = (ValueError, TypeError, KeyError, IndexError, AttributeError,
                      ArithmeticError, OSError, RuntimeError, TimeoutExpired)


class SessionError(ValueError):
    def __init__(self, message, line, *, unknown_name=None, function=None, method=None):
        super().__init__(message)
        self.line = line
        self.unknown_name = unknown_name
        self.function = function
        self.method = method


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


_SCALAR_OPERATORS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^"}


def _numeric(value):
    return type(value) is int or isinstance(value, (Fraction, Polynomial, QuotientElement))


def _arithmetic(op, left, right):
    operation = _OPERATORS.get(type(op))
    if operation is None or not (_numeric(left) and _numeric(right)):
        raise ValueError("Arithmetic requires rational numbers or polynomials and +, -, *, / or ^.")
    if isinstance(op, ast.Pow):
        if isinstance(right, Fraction) and right.denominator == 1:
            right = right.numerator
        if type(right) is not int:
            raise ValueError("Use an integer exponent.")
    if isinstance(left, (Polynomial, QuotientElement)) or isinstance(right, (Polynomial, QuotientElement)):
        return operation(left, right)
    result = scalar_arithmetic(_SCALAR_OPERATORS[type(op)], left, right)
    if isinstance(left, Fraction) or isinstance(right, Fraction) or isinstance(op, ast.Div):
        return result
    if isinstance(op, ast.Pow) and right < 0:
        return result
    return result.numerator


def _field(value, name):
    if name.startswith("_"):
        raise ValueError("Only public Atlas fields are available.")
    if isinstance(value, dict) and name in value:
        return value[name]
    getter = getattr(type(value), "session_attribute", None)
    if getter is None:
        raise ValueError(f"This value has no Atlas field {name!r}.")
    return getter(value, name)


def _same_coordinates(previous, value):
    # A scalar, residue class, or computed result can compare equal to a
    # generator while carrying a different parent or additional result data.
    if type(previous) is not type(value):
        return False
    if isinstance(value, tuple):
        return len(previous) == len(value) and all(_same_coordinates(a, b) for a, b in zip(previous, value))
    return previous.parent is value.parent and previous._value == value._value


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
                value, declared = self._assignment_value(statement.value)
                self._bind(statement.targets[0], value, declaration=statement.value if declared else None)
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
        except _EVALUATION_ERRORS as error:
            raise SessionError(str(error), statement.lineno) from None

    def _assignment_value(self, node):
        """Track a direct coordinate declaration through indexing, without evaluating twice."""
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Subscript):
            value, declared = self._assignment_value(node.value)
            return self._index(value, self._value(node.slice)), declared
        return self._value(node), False

    def _bind(self, target, value, *, declaration=None):
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
        if declaration is not None:
            self._check_coordinate_bindings(names, values, declaration)
        self.variables.update(zip(names, values))

    def _check_coordinate_bindings(self, names, values, declaration):
        conflicts = {}
        for name, value in zip(names, values):
            if name in self.variables and not _same_coordinates(self.variables[name], value):
                previous = self.variables[name]
                label = repr(name)
                if isinstance(previous, (Polynomial, QuotientElement)):
                    label += f" in {previous.parent}"
                conflicts[name] = label
        if conflicts:
            used = set(names) | self.variables.keys() | self.builtins.keys() | self.reserved
            aliases = []
            for name in names:
                if name in conflicts:
                    name = next(f"{name}{i}" for i in count(2) if f"{name}{i}" not in used)
                    used.add(name)
                aliases.append(name)
            raise ValueError(f"This declaration would replace {', '.join(conflicts.values())}. "
                             f"Use {', '.join(aliases)} = {ast.unparse(declaration)}.")

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
            raise SessionError(f"Unknown name {node.id!r}.", node.lineno, unknown_name=node.id)
        if isinstance(node, ast.List):
            return [self._value(element) for element in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._value(element) for element in node.elts)
        if isinstance(node, ast.Attribute):
            return _field(self._value(node.value), node.attr)
        if isinstance(node, ast.Subscript):
            return self._index(self._value(node.value), self._value(node.slice))
        if isinstance(node, ast.UnaryOp):
            value = self._value(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            if _numeric(value) and isinstance(node.op, (ast.UAdd, ast.USub)):
                if isinstance(node.op, ast.UAdd):
                    return value
                if isinstance(value, QuotientElement):
                    return -value
                return _arithmetic(ast.Sub(), 0, value)
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
                if type(left) in (int, Fraction) and type(right) in (int, Fraction):
                    matches = compare(scalar_arithmetic("compare", left, right).numerator, 0)
                else:
                    matches = compare(left, right)
                if not matches:
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            return self._call(node)[0]
        raise ValueError(f"Unsupported Atlas expression: {type(node).__name__}.")

    @staticmethod
    def _index(value, index):
        # Scripts can keep indexing the original complex and vertex records.
        if isinstance(value, (DualComplex, Divisor)):
            value = value._record
        if isinstance(value, dict):
            if index not in value:
                raise ValueError(f"This record has no key {index!r}.")
        elif not isinstance(value, (tuple, list, str)) or type(index) is not int:
            raise ValueError("Index a list or tuple by integer, or a record by key.")
        return value[index]

    def _call(self, node):
        if not isinstance(node.func, (ast.Name, ast.Attribute)):
            raise ValueError("Call an Atlas function or method by name.")
        function = self._value(node.func)
        try:
            if not callable(function):
                raise ValueError("This value is not an Atlas function.")
            args, kwargs = self._arguments(node)
            value = function(*args, **kwargs)
        except SessionError as error:
            # Preserve a more specific failure from a nested call.
            if error.function is None and isinstance(node.func, ast.Name):
                error.function = node.func.id
            raise
        except _EVALUATION_ERRORS as error:
            # Methods can share a name with a different top-level operation.
            message = "Division by zero." if isinstance(error, ZeroDivisionError) else str(error)
            if isinstance(node.func, ast.Attribute):
                raise SessionError(message, node.lineno, function=ast.unparse(node.func), method=function) from None
            raise SessionError(message, node.lineno, function=node.func.id) from None
        declared = function is self.builtins.get("coordinates") and not kwargs.get("fresh", False)
        return value, declared

    def _arguments(self, node):
        args = [self._value(arg) for arg in node.args]
        kwargs = {}
        for keyword in node.keywords:
            if not keyword.arg or keyword.arg.startswith("_"):
                raise ValueError("Use public, named keyword arguments.")
            if keyword.arg in kwargs:
                raise ValueError(f"Repeated keyword {keyword.arg!r}.")
            kwargs[keyword.arg] = self._value(keyword.value)
        return args, kwargs
