"""Evaluate Atlas expressions against explicitly supplied operations and session values."""

import ast
from fractions import Fraction
from functools import lru_cache
from io import StringIO
from itertools import count
from keyword import iskeyword
import operator
import re
from subprocess import TimeoutExpired
import tokenize

from .native_polynomial import scalar_arithmetic
from .session_algebra import Ideal, _binary_ideals, ideal_power
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
        if "^" in source:
            tokens = tokenize.generate_tokens(StringIO(source).readline)
            source = tokenize.untokenize(
                token._replace(string="**") if token.type == tokenize.OP and token.string == "^"
                else token for token in tokens
            )
        statements = ast.parse(source).body
        for statement in statements:
            _prepare_assignments(statement)
        return statements
    except tokenize.TokenError as error:
        message, (line, _) = error.args
        raise SessionError(message, line) from None
    except SyntaxError as error:
        raise SessionError(error.msg, error.lineno or 1) from None


# Only short commands can be retained. A decimal literal in such a command
# fits below Python's smallest configurable digit limit (640).
_short_statements = lru_cache(maxsize=256)(_parse)


# A flat binary assignment has an unambiguous spelling. Only recognize a
# complete line of public ASCII names and an optional unsigned decimal argument.
# Quotes, comments, other literals, keywords and larger expressions use _parse.
_NAME = r"[a-zA-Z][a-zA-Z_0-9]*"
_WS = r"[ \t]*"
_OPERAND = rf"({_NAME}|0|[1-9][0-9]*)"
_CALL = re.compile(rf"{_WS}({_NAME}){_WS}\({_WS}({_NAME}){_WS},{_WS}{_OPERAND}{_WS}\){_WS}")
_INFIX = re.compile(rf"{_WS}({_NAME}){_WS}(\*\*|[-+*/^]){_WS}{_OPERAND}{_WS}")
_INFIX_OPERATORS = {"+": ast.Add(), "-": ast.Sub(), "*": ast.Mult(), "/": ast.Div(), "^": ast.Pow(), "**": ast.Pow()}


@lru_cache(maxsize=256)
def _pair_arguments(source):
    if match := _CALL.fullmatch(source):
        operation, left, right = match.groups()
        if iskeyword(operation):
            return None
    elif match := _INFIX.fullmatch(source):
        left, token, right = match.groups()
        operation = _INFIX_OPERATORS[token]
    else:
        return None
    if iskeyword(left) or iskeyword(right):
        return None
    return operation, left, int(right) if right[0].isdigit() else right


@lru_cache(maxsize=256)
def _pair_syntax(source):
    target, equals, rhs = source.partition("=")
    target = target.rstrip(" \t")
    if not equals or not target.isascii() or not target.isidentifier() or target[0] == "_" or iskeyword(target):
        return None
    args = _pair_arguments(rhs)
    return (target, *args) if args is not None else None


def assignment_syntax(source):
    """Return the target, operation and operands of a short, complete binary assignment."""
    return _pair_syntax(source) if len(source) <= 512 else None


def _prepare_assignments(statement):
    if isinstance(statement, (ast.For, ast.If)):
        for child in (*statement.body, *statement.orelse):
            _prepare_assignments(child)
        return
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return
    target, expression = statement.targets[0], statement.value
    if not isinstance(target, ast.Name) or target.id.startswith("_"):
        return
    if (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)
            and not expression.keywords and len(expression.args) == 2):
        operation = expression.func.id
        left, right = expression.args
    elif isinstance(expression, ast.BinOp) and type(expression.op) in _OPERATORS:
        operation, left, right = expression.op, expression.left, expression.right
    else:
        return
    if not isinstance(left, ast.Name):
        return
    if isinstance(right, ast.Name):
        statement._atlas_pair = target.id, operation, left.id, right.id
    elif isinstance(right, ast.Constant) and type(right.value) is int:
        statement._atlas_pair = target.id, operation, left.id, right.value


def _assign_pair(session, pair, line):
    target, operation, first, second = pair
    variables, builtins = session.variables, session.builtins
    try:
        left = variables[first]
        right = second if type(second) is int else variables[second]
    except KeyError:
        return False  # An argument may be built-in, or require an unknown-name diagnostic.
    if type(operation) is str:
        function = variables.get(operation, builtins.get(operation))
        if not callable(function) or function is builtins.get("coordinates"):
            return False
        try:
            result = function(left, right)
        except _EVALUATION_ERRORS as error:
            raise _call_error(operation, line, error) from None
    else:
        try:
            result = _arithmetic(operation, left, right)
        except SessionError:
            raise
        except _EVALUATION_ERRORS as error:
            message = "Division by zero." if isinstance(error, ZeroDivisionError) else str(error)
            raise SessionError(message, line) from None
    if target in session.builtins or target in session.reserved:
        raise SessionError(f"{target!r} is a reserved Atlas name.", line)
    session.variables[target] = result
    return True


def _call_error(name, line, error, method=None):
    if isinstance(error, SessionError):
        if error.function is None and method is None:
            error.function = name
        return error
    message = "Division by zero." if isinstance(error, ZeroDivisionError) else str(error)
    return SessionError(message, line, function=name, method=method)


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
    kind = type(op)
    if isinstance(left, Ideal) or isinstance(right, Ideal):
        if kind is ast.Mult:
            return _binary_ideals("ideal-product", left, right, "auto", infix=True)
        if kind is ast.Add:
            return _binary_ideals("ideal-sum", left, right, "auto", infix=True)
        if kind is ast.Pow and isinstance(left, Ideal):
            return ideal_power(left, right)
        raise TypeError("Use I + J or I * J for two ideals, or I^n for an ordinary ideal power.")
    operation = _OPERATORS.get(kind)
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

    def assign(self, pair, source, transcript):
        """Run a recognized binary assignment and record the original text on success."""
        if not _assign_pair(self, pair, 1):
            return False
        if transcript is not None:
            transcript.statements.append(source)
        return True

    def run(self, source, *, transcript=None):
        """Execute in this session; yield (True, value) for top-level expression previews."""
        if len(source) <= 512:
            pair = _pair_syntax(source)
            if pair is not None and self.assign(pair, source, transcript):
                return
            statements = _short_statements(source)
        else:
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
                pair = getattr(statement, "_atlas_pair", None)
                if pair is not None and _assign_pair(self, pair, statement.lineno):
                    return
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
        except _EVALUATION_ERRORS as error:
            if isinstance(node.func, ast.Attribute):
                raise _call_error(ast.unparse(node.func), node.lineno, error, method=function) from None
            raise _call_error(node.func.id, node.lineno, error) from None
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
