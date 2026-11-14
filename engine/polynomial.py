"""Small bounded arithmetic grammar over Q. No evaluation of user code."""
from fractions import Fraction
import re

MAX_DEGREE = 128
MAX_TERMS = 4096
MAX_BITS = 4096
MAX_PRODUCTS = 250_000


class InputError(ValueError):
    pass


class NonReducedError(InputError):
    pass


class LimitError(ValueError):
    pass


def checked(p):
    if len(p) > MAX_TERMS or any(sum(e) > MAX_DEGREE or
            max(c.numerator.bit_length(), c.denominator.bit_length()) > MAX_BITS
            for e, c in p.items()):
        raise LimitError("Expanded polynomial exceeds the degree, term, or coefficient limit.")
    return p


def add(p, q, bounded=True):
    r = p.copy()
    for e, c in q.items():
        r[e] = r.get(e, 0) + c
        if not r[e]:
            del r[e]
    return checked(r) if bounded else r


def mul(p, q, bounded=True):
    if bounded and len(p) * len(q) > MAX_PRODUCTS:
        raise LimitError("Polynomial product exceeds the expansion budget.")
    r = {}
    for e, a in p.items():
        for f, b in q.items():
            k = tuple(x + y for x, y in zip(e, f))
            r[k] = r.get(k, 0) + a * b
    result={k:v for k,v in r.items() if v}
    return checked(result) if bounded else result


def parse(source, variables, *, bounded=True):
    if not isinstance(variables, list) or not variables or any(
        not isinstance(s, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,23}', s)
        for s in variables
    ) or len(set(variables)) != len(variables):
        raise InputError("Choose distinct variable names that start with a letter and contain letters, digits, or underscores.")
    p = parse_expression(source, variables, bounded=bounded)
    if not p:
        raise InputError("The zero polynomial does not define a hypersurface.")
    if (0,) * len(variables) in p:
        raise InputError("The hypersurface does not pass through the origin.")
    return p


def parse_expression(source, variables, *, bounded=False):
    """Read a polynomial in known names, also allowing constants in chart equations."""
    if not isinstance(source, str) or not source.strip():
        raise InputError("Enter a nonempty polynomial.")
    if bounded and (len(variables)>8 or len(source)>8192):
        raise LimitError("Quick parsing accepts up to 8 variables and 8192 characters.")
    tokens = re.findall(r'\*\*|[A-Za-z][A-Za-z0-9_]*|[0-9]+|[^\s]', source)
    if bounded and len(tokens) > 2048:
        raise LimitError("Expression has too many tokens.")
    tokens.append(None)
    pos = 0
    zero = (0,) * len(variables)

    def expr(binding=0, depth=0):
        nonlocal pos
        if bounded and depth > 64:
            raise LimitError("Expression nesting is too deep.")
        t = tokens[pos]
        pos += 1
        if t == '(':
            a = expr(0, depth+1)
            if tokens[pos] != ')':
                raise InputError("Expected a closing parenthesis.")
            pos += 1
        elif t in ('-', '+'):
            a = expr(25, depth+1)
            if t == '-':
                a = {k: -c for k, c in a.items()}
        elif t in variables:
            k = tuple(int(v == t) for v in variables)
            a = {k: Fraction(1)}
        elif t is not None and t.isascii() and t.isdigit():
            if bounded and len(t) > 256:
                raise LimitError("Integer literal is too long.")
            a = {zero: Fraction(int(t))} if int(t) else {}
        else:
            raise InputError(f"Expected a number, declared variable, or parenthesis; found {t!r}.")
        while True:
            op = tokens[pos]
            prec = {'+': 10, '-': 10, '*': 20, '/': 20, '^': 30, '**': 30}.get(op, -1)
            if prec < binding:
                break
            pos += 1
            if prec == 30:
                t = tokens[pos]
                if not t or not t.isascii() or not t.isdigit():
                    raise InputError("Use a nonnegative integer literal for the exponent.")
                if bounded and (len(t) > 3 or int(t) > MAX_DEGREE):
                    raise LimitError(f"Quick parsing accepts exponents up to {MAX_DEGREE}.")
                pos += 1
                if tokens[pos] in ('^','**'):
                    raise InputError("Parenthesize chained powers explicitly.")
                n = int(t)
                b = {zero: Fraction(1)}
                while n:
                    if n % 2:
                        b = mul(b, a, bounded)
                    n //= 2
                    if n:
                        a = mul(a, a, bounded)
                a = b
                continue
            b = expr(prec+1, depth+1)
            if op == '+':
                a = add(a, b, bounded)
            elif op == '-':
                a = add(a, {k:-v for k,v in b.items()}, bounded)
            elif op == '*':
                a = mul(a, b, bounded)
            else:
                if set(b) != {zero}:
                    raise InputError("Division is allowed only by a nonzero rational constant.")
                a = {k:v/b[zero] for k,v in a.items()}
                if bounded: checked(a)
        return a

    p = expr()
    if tokens[pos] is not None:
        raise InputError(f"Unexpected token {tokens[pos]!r}. Use * for multiplication.")
    return p


def format_poly(p, names, scalar=str):
    words = []
    for e, c in sorted(p.items(), key=lambda pair: (sum(pair[0]), pair[0]), reverse=True):
        if not c:
            continue
        coeff = scalar(c)
        negative = coeff.startswith('-')
        if negative:
            coeff = coeff[1:]
        factors = [s if k == 1 else f'{s}^{k}' for s, k in zip(names, e) if k]
        if coeff != '1' or not factors:
            factors.insert(0, coeff)
        words.append(('-' if negative else '+' if words else '') + '*'.join(factors))
    return ''.join(words) or '0'
