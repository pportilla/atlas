"""Load the optional comparison engine only for symbolic callers and reference runs."""
import importlib
import sys


class _Reference:
    def __getattr__(self, name):
        return getattr(importlib.import_module("sympy"), name)


sp = _Reference()


class _Rationals(str):
    """SymPy accepts the domain name in its public polynomial constructors."""
    is_AlgebraicField = False

    def __eq__(self, other):
        if type(other).__module__ == "sympy.polys.domains.rationalfield" and type(other).__name__ == "RationalField":
            return True
        return str.__eq__(self, other)

    __hash__ = str.__hash__

    def __call__(self, *args):
        return sp.QQ(*args)

    def __getattr__(self, name):
        return getattr(sp.QQ, name)


QQ = sys.modules["sympy"].QQ if hasattr(sys.modules.get("sympy"), "QQ") else _Rationals("QQ")


def parse_tokens(tokens, variables, bounded):
    """Independent sparse arithmetic for explicit parser comparisons."""
    from fractions import Fraction
    from .polynomial import InputError, LimitError, MAX_BITS, MAX_DEGREE, MAX_PRODUCTS, MAX_TERMS

    def checked(p):
        if len(p) > MAX_TERMS or any(sum(e) > MAX_DEGREE or
                max(c.numerator.bit_length(), c.denominator.bit_length()) > MAX_BITS
                for e, c in p.items()):
            raise LimitError("Expanded polynomial exceeds the degree, term, or coefficient limit.")
        return p


    def add(p, q):
        r = p.copy()
        for e, c in q.items():
            r[e] = r.get(e, 0) + c
            if not r[e]:
                del r[e]
        return checked(r) if bounded else r


    def mul(p, q):
        if bounded and len(p) * len(q) > MAX_PRODUCTS:
            raise LimitError("Polynomial product exceeds the expansion budget.")
        r = {}
        for e, a in p.items():
            for f, b in q.items():
                k = tuple(x + y for x, y in zip(e, f))
                r[k] = r.get(k, 0) + a * b
        result={k:v for k,v in r.items() if v}
        return checked(result) if bounded else result

    pos = 0
    zero = (0,) * len(variables)

    def expr(binding):
        nonlocal pos
        t = tokens[pos]
        pos += 1
        if t == '(':
            a = yield 0
            if tokens[pos] != ')':
                raise InputError("Expected a closing parenthesis.")
            pos += 1
        elif t in ('-', '+'):
            a = yield 25
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
                        b = mul(b, a)
                    n //= 2
                    if n:
                        a = mul(a, a)
                a = b
                continue
            b = yield prec+1
            if op == '+':
                a = add(a, b)
            elif op == '-':
                a = add(a, {k:-v for k,v in b.items()})
            elif op == '*':
                a = mul(a, b)
            else:
                if set(b) != {zero}:
                    raise InputError("Division is allowed only by a nonzero rational constant.")
                a = {k:v/b[zero] for k,v in a.items()}
                if bounded: checked(a)
        return a

    # Each yielded precedence starts a subexpression. Feed its polynomial back
    # to the waiting parent without recursing through the Python call stack.
    waiting = [expr(0)]
    p = None
    while waiting:
        try:
            binding = waiting[-1].send(p)
        except StopIteration as done:
            p = done.value
            waiting.pop()
        else:
            if bounded and len(waiting) > 64:
                raise LimitError("Expression nesting is too deep.")
            waiting.append(expr(binding))
            p = None
    if tokens[pos] is not None:
        raise InputError(f"Unexpected token {tokens[pos]!r}. Use * for multiplication.")
    return p
