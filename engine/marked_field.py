"""Read marked algebraic scalars from resolution records without symbolic expressions."""
import ast
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
import re

from . import native_library
from .polynomial import InputError, format_poly, integer_from_decimal, integer_text, kernel_fraction, parse_expression, rational_text
from .rational_point import coordinate_fraction


def _rational(node):
    """Evaluate rational exponent arithmetic in the same native ring as chart inputs."""
    from . import native_polynomial as np
    return _rational_value(node, np).scalar


def _rational_value(node, api):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return api.coerce(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        q = _rational_value(node.operand, api)
        return -q if isinstance(node.op, ast.USub) else q
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Rational":
        if not node.keywords and len(node.args) in (1,2):
            return _rational_value(node.args[0], api) / (_rational_value(node.args[1], api) if len(node.args) == 2 else 1)
    if isinstance(node, ast.BinOp):
        a, b = _rational_value(node.left, api), _rational_value(node.right, api)
        if isinstance(node.op, ast.Div): return a / b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Pow) and b.is_Integer:
            return api.coerce(api.scalar_arithmetic("^", a.scalar, b.scalar))
    raise InputError("Expected a rational algebraic exponent or angle.")


def _marked_scalar(request):
    try:
        row = native_library.field_request("atlas_marked_scalar", [request])
    except ArithmeticError as exc:
        raise InputError(str(exc)) from exc
    if row is None:
        raise native_library.Unavailable("Rebuild the native marked scalar kernel with make groebner.")
    return row


def _angle_tokens(node):
    """Translate the public angle syntax into rational operations with pi and I."""
    if isinstance(node, ast.Name) and node.id in ("pi", "I"):
        return node.id
    if isinstance(node, ast.BinOp):
        token = {ast.Mult: "*", ast.Div: "/", ast.Add: "+", ast.Sub: "-"}.get(type(node.op))
        if token:
            return f"{_angle_tokens(node.left)} {_angle_tokens(node.right)} {token}"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _angle_tokens(node.operand) + (" neg" if isinstance(node.op, ast.USub) else "")
    return "q " + rational_text(_rational(node))


def _program(node):
    if isinstance(node, ast.Name) and node.id == "I":
        return "i"
    if isinstance(node, ast.Constant):
        return "q " + rational_text(_rational(node))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub,ast.UAdd)):
        return _program(node.operand) + (" q -1 *" if isinstance(node.op, ast.USub) else "")
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            return f"{_program(node.left)} pow {rational_text(_rational(node.right))}"
        suffix = {ast.Add:"+", ast.Sub:"q -1 * +", ast.Mult:"*", ast.Div:"pow -1 *"}.get(type(node.op))
        if suffix:
            return f"{_program(node.left)} {_program(node.right)} {suffix}"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name, args = node.func.id, node.args
        if name == "CRootOf" and len(args) == 2:
            symbols = {s.id for s in ast.walk(args[0]) if isinstance(s,ast.Name)}
            if len(symbols) != 1:
                raise InputError("Expected one variable in an indexed root polynomial.")
            terms = parse_expression(ast.unparse(args[0]), list(symbols))
            coefficients = " ".join(f"{integer_text(j)} {rational_text(c)}" for (j,), c in terms.items())
            return _marked_scalar(f"indexedroot {rational_text(_rational(args[1]))} {coefficients}")
        if name == "Rational":
            return "q " + rational_text(_rational(node))
        if len(args) == 1 and name == "sqrt":
            return _program(args[0]) + " pow 1/2"
        if len(args) == 1 and name in ("exp", "sin", "cos", "tan", "cot", "sec", "csc"):
            return _marked_scalar(f"trig {name} {_angle_tokens(args[0])}")
    raise InputError("Unsupported expression for a marked algebraic coefficient.")


@lru_cache(maxsize=512)
def program(exact):
    if not isinstance(exact, str):
        raise InputError("Expected a marked algebraic coefficient as text.")
    flint = re.fullmatch(r"PolynomialRootIndexed\(List\(([-\d, ]+)\),\s*(\d+)\)", exact)
    if flint:
        values = flint[1].split(",")
        if not all(re.fullmatch(r"-?\d+", c.strip()) for c in values):
            raise InputError("Invalid coefficients in a marked FLINT root.")
        return _marked_scalar(f"flintroot {flint[2]} {' '.join(values)}")
    try:
        return _program(ast.parse(exact.replace("^", "**"), mode="eval").body)
    except (SyntaxError, TypeError, ZeroDivisionError) as exc:
        raise InputError("Invalid marked algebraic coefficient.") from exc


@lru_cache(maxsize=512)
def description(exact):
    """Primitive integral minimal polynomial (descending) and one-based FLINT root index."""
    try:
        row = native_library.field_request("atlas_field_description", [program(exact)])
    except ArithmeticError as exc:
        raise InputError(str(exc)) from exc
    if row is None:
        raise native_library.Unavailable("Native marked root selection is unavailable.")
    index,*coeffs = map(integer_from_decimal,row.split())
    return tuple(reversed(coeffs)),index


def flint_literal(coefficients, index):
    return f"PolynomialRootIndexed(List({', '.join(map(integer_text,reversed(coefficients)))}), {index})"


@lru_cache(maxsize=128)
def common_fields(exact, variable):
    """Return one marked primitive field and every scalar's image as a polynomial in variable."""
    from .plane_backend import coefficient_field
    marked = tuple(flint_literal(*description(value)) for value in exact)
    common = coefficient_field(marked)
    images = tuple(format_poly(parse_expression(e, ["T"]), (variable,)) for e in common["values"])
    relation = format_poly(parse_expression(common["minimal_polynomial"], ["T"]), (variable,))
    return common["exact"], relation, common["degree"], images


def _coefficient(value):
    if type(value) is int:
        return Fraction(value)
    word = str(value).strip()
    digits = r"\d(?:_?\d)*"
    fraction = re.fullmatch(rf"([+-]?{digits})(?:\s*/\s*([+-]?{digits}))?", word)
    if fraction:
        parts = [part.replace("_", "") for part in fraction.groups() if part is not None]
        return coordinate_fraction("/".join(str(Decimal(part)) if not part.isascii() else part for part in parts))
    if not re.fullmatch(rf"[+-]?(?:{digits}(?:\.(?:{digits})?)?|\.{digits})(?:[eE][+-]?{digits})?", word):
        raise InputError("Invalid rational coefficient in the marked field.")
    sign, mantissa, exponent = Decimal(word).as_tuple()
    coefficient = ("-" if sign else "") + "".join(map(str, mantissa))
    row, = native_library.affine_request(["scalar_arithmetic", "decimal_coefficient", coefficient, integer_text(exponent)])
    return kernel_fraction(row)


def record_field(field, named, texts, parameter):
    """Check the literal mark and recover all named constants actually present on a chart."""
    declared = field.get("generator")
    names = []
    exact = []
    used = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", " ".join(texts)))
    if declared:
        coeffs = field.get("minimal_polynomial_coefficients")
        mark = field.get("exact_root")
        if mark:
            actual, _ = description(mark)
            if coeffs:
                row = tuple(map(_coefficient, coeffs))
                pairs = " ".join(f"{rational_text(a)} {integer_text(b)}" for a,b in zip(row, actual))
                if len(row) != len(actual) or _marked_scalar(f"same {len(row)} {pairs}") != "1":
                    raise InputError("The graph's field polynomial does not define its marked root.")
        elif coeffs:
            row = tuple(map(_coefficient, coeffs))
            source = format_poly({(j,): c for j, c in enumerate(reversed(row)) if c}, ("x",))
            mark = f"CRootOf({source}, 0)"
            actual,_ = description(mark)
            if len(actual) != len(coeffs):
                raise InputError("The graph's field polynomial is reducible.")
        else:
            raise InputError("A graph coefficient field needs an exact marked root.")
        names.append(declared)
        exact.append(mark)
    for k, definition in named.items():
        if k in used:
            names.append(k)
            exact.append(definition["exact"])
    if not names:
        return None, {}
    mark, relation, degree, values = common_fields(tuple(exact), parameter)
    field = dict(generator=parameter, minimal_polynomial=relation, degree=degree, embedding=mark)
    return field, dict(zip(names, values))


@lru_cache(maxsize=256)
def display(mark):
    """Historical exact root spelling for weighted chart output."""
    coeff,index = description(mark)
    if len(coeff) in (2,3):
        # QQbar numbers the positive square-root choice first. The quadratic
        # formula uses index one for its plus sign and zero for its minus.
        formula_index = int(len(coeff) == 3 and index == 1)
        payload = " ".join(integer_text(i) for i in (len(coeff)-1, formula_index, *coeff))
        row = native_library.field_request("atlas_field_low_degree_root", [payload])
        if row is None:
            raise native_library.Unavailable("Rebuild the native marked root kernel with make groebner.")
        if len(coeff) == 2:
            return row
        shift, n, d, value, _ = row.split()
        scalar = kernel_fraction(n + "/" + d)
        offset = coordinate_fraction(shift)
        rad = integer_from_decimal(value)
        root = (f"sqrt({abs(rad)})" if abs(rad) != 1 else "") + ("*I" if rad < 0 else "")
        root = root.lstrip("*")
        terms = []
        if offset: terms.append(str(offset))
        if scalar:
            text = (str(abs(scalar.numerator))+"*" if abs(scalar.numerator)!=1 and root else "")
            text += root or str(abs(scalar.numerator))
            if scalar.denominator != 1: text += "/"+str(scalar.denominator)
            if terms: terms.append((" - " if scalar < 0 else " + ")+text)
            else: terms.append(("-" if scalar<0 else "")+text)
        return "".join(terms) or "0"
    row = native_library.field_request("atlas_field_normalize",[f"{len(coeff)-1} "+ " ".join(map(str,reversed(coeff)))])
    scale,*normalized = map(int,row.split())
    rows = native_library.field_request("atlas_field_display_order",[f"{len(coeff)-1} "+" ".join(map(str,reversed(normalized)))])
    position = list(map(int,rows.split()))[index-1]
    expression = format_poly({(i,):c for i,c in enumerate(reversed(normalized)) if c}, ("x",)).replace("^","**")
    return (str(scale)+"*" if scale != 1 else "")+f"CRootOf({expression}, {position})"
