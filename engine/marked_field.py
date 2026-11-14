"""Read marked algebraic scalars from resolution records without symbolic expressions."""
import ast
from fractions import Fraction
from functools import lru_cache
import re

from . import native_library
from .polynomial import InputError, format_poly, parse_expression


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
            n = int(b)
            return a**n if n >= 0 else 1/(a**(-n))
    raise InputError("Expected a rational algebraic exponent or angle.")


def _angle(node):
    """Separate powers of pi and I from the rational coefficient of an angle."""
    from . import native_polynomial as np
    if isinstance(node, ast.Name) and node.id in ("pi", "I"):
        return np.S.One, int(node.id == "pi"), int(node.id == "I")
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult,ast.Div)):
        a,p,i = _angle(node.left)
        b,q,j = _angle(node.right)
        if isinstance(node.op, ast.Div):
            return a/b,p-q,i-j
        return a*b,p+q,i+j
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        a,p,i = _angle(node.operand)
        return (-a if isinstance(node.op, ast.USub) else a),p,i
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add,ast.Sub)):
        a,p,i = _angle(node.left)
        b,q,j = _angle(node.right)
        if (p,i%4) == (q,j%4):
            return (a-b if isinstance(node.op, ast.Sub) else a+b),p,i
    return _rational_value(node, np), 0, 0


def _program(node):
    if isinstance(node, ast.Name) and node.id == "I":
        return "i"
    if isinstance(node, ast.Constant):
        return "q " + str(_rational(node))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub,ast.UAdd)):
        return _program(node.operand) + (" q -1 *" if isinstance(node.op, ast.USub) else "")
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            return f"{_program(node.left)} pow {_rational(node.right)}"
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
            degree = max((p[0] for p in terms), default=0)
            idx = _rational(args[1])
            if idx.denominator != 1 or not -degree <= idx < degree:
                raise InputError("Index outside the root polynomial.")
            coeff = " ".join(str(terms.get((i,),0)) for i in range(degree+1))
            return f"indexedroot {degree} {coeff} {idx}"
        if name == "Rational":
            return "q " + str(_rational(node))
        if len(args) == 1 and name == "sqrt":
            return _program(args[0]) + " pow 1/2"
        if len(args) == 1 and name in ("exp", "sin", "cos", "tan", "cot", "sec", "csc"):
            q,pi,i = _angle(args[0])
            if not q:
                if name in ("cot", "csc"):
                    raise InputError("Undefined trigonometric scalar.")
                return "q " + ("1" if name in ("exp","cos","sec") else "0")
            if pi != 1 or i%2 != int(name == "exp"):
                raise InputError("Expected a rational multiple of pi in the algebraic scalar.")
            q = -q if i%4 >= 2 else q
            a,b = f"q -1 pow {q}", f"q -1 pow {-q}"
            if name == "exp": return a
            cosine = f"{a} {b} + q 1/2 *"
            sine = f"{a} {b} q -1 * + i * q -1/2 *"
            return {"cos":cosine, "sin":sine, "tan":f"{sine} {cosine} pow -1 *",
                    "cot":f"{cosine} {sine} pow -1 *", "sec":f"{cosine} pow -1",
                    "csc":f"{sine} pow -1"}[name]
    raise InputError("Unsupported expression for a marked algebraic coefficient.")


@lru_cache(maxsize=512)
def program(exact):
    if not isinstance(exact, str):
        raise InputError("Expected a marked algebraic coefficient as text.")
    flint = re.fullmatch(r"PolynomialRootIndexed\(List\(([-\d, ]+)\),\s*(\d+)\)", exact)
    if flint:
        try:
            values = tuple(int(c) for c in flint[1].split(","))
        except ValueError as exc:
            raise InputError("Invalid coefficients in a marked FLINT root.") from exc
        index = int(flint[2])-1
        if not values[-1] or not 0 <= index < len(values)-1:
            raise InputError("Invalid index or polynomial for the marked root.")
        return f"flintroot {len(values)-1} {' '.join(map(str,values))} {index}"
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
    index,*coeffs = map(int,row.split())
    return tuple(reversed(coeffs)),index


def flint_literal(coefficients, index):
    return f"PolynomialRootIndexed(List({', '.join(map(str,reversed(coefficients)))}), {index})"


@lru_cache(maxsize=128)
def common_fields(exact, variable):
    """Return one marked primitive field and every scalar's image as a polynomial in variable."""
    from .plane_backend import coefficient_field
    marked = tuple(flint_literal(*description(value)) for value in exact)
    common = coefficient_field(marked)
    images = tuple(format_poly(parse_expression(e, ["T"]), (variable,)) for e in common["values"])
    relation = format_poly(parse_expression(common["minimal_polynomial"], ["T"]), (variable,))
    return common["exact"], relation, common["degree"], images


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
                row = tuple(Fraction(str(c)) for c in coeffs)
                if not row or not row[0] or tuple(c/row[0] for c in row) != tuple(Fraction(c,actual[0]) for c in actual):
                    raise InputError("The graph's field polynomial does not define its marked root.")
        elif coeffs:
            source = format_poly({(j,): Fraction(str(c)) for j,c in enumerate(reversed(coeffs)) if Fraction(str(c))}, ("x",))
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
    if len(coeff) == 2:
        return str(Fraction(-coeff[1], coeff[0]))
    if len(coeff) == 3:
        a,b,c = coeff
        radical = native_library.field_request("atlas_field_radical",[str(b*b-4*a*c)])
        n,d,rad,_ = map(int,radical.split())
        scalar = Fraction((1 if index == 1 else -1)*n, 2*a*d)
        offset = Fraction(-b,2*a)
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
