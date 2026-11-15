"""Public polynomial grammar over Q, with optional intermediate expansion budgets."""
from decimal import Decimal
from fractions import Fraction
from numbers import Rational
from collections import namedtuple
import re
import sys

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


def _expression_tokens(source, variables, bounded):
    if not isinstance(source, str) or not source.strip():
        raise InputError("Enter a nonempty polynomial.")
    if bounded and (len(variables)>8 or len(source)>8192):
        raise LimitError("Quick parsing accepts up to 8 variables and 8192 characters.")
    tokens = re.findall(r'\*\*|[A-Za-z][A-Za-z0-9_]*|[0-9]+|[^\s]', source)
    if bounded and len(tokens) > 2048:
        raise LimitError("Expression has too many tokens.")
    return tokens


def _check_literal_digits(tokens, digit_limit=None):
    if digit_limit is None:
        digit_limit = sys.get_int_max_str_digits() if hasattr(sys, "get_int_max_str_digits") else 0
    if digit_limit:
        for t in tokens:
            if len(t) > digit_limit and t.isdigit():
                raise InputError(f"Integer literal has {len(t)} digits; the active limit is {digit_limit} digits.")


def check_expression_digits(expressions, variables):
    """Apply the caller's decimal-literal policy before sending source strings to a worker."""
    limit = sys.get_int_max_str_digits() if hasattr(sys, "get_int_max_str_digits") else 0
    if limit:
        for source in expressions:
            if len(source) > limit:
                _check_literal_digits(_expression_tokens(source, variables, False), limit)


def native_expression_source(source, variables):
    """Encode an unrestricted polynomial in validated names for a native request."""
    tokens = _expression_tokens(source, variables, False)
    _check_literal_digits(tokens)
    return " ".join(tokens)


def parse_expression(source, variables, *, bounded=False, coefficient_modulus=None, as_row=False, _reference=False):
    """Read a chart polynomial. An optional modulus defines its last generator as a field coefficient."""
    tokens = _expression_tokens(source, variables, bounded)
    from . import native_library

    def reference():
        return _parse_tokens([*tokens, None], variables, bounded)

    if _reference:
        terms = reference()
        return encode_sparse_row(terms) if as_row else terms

    # Standalone callers can declare unusual names or the same name twice.
    # Retain their coordinate columns by translating the token for a generator
    # to the product of all its matching columns, as in the original grammar.
    kernel_names = variables
    kernel_tokens = tokens
    if (not variables or len(set(variables)) != len(variables) or
            any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', v) for v in variables)):
        kernel_names = [f"v{j}" for j in range(len(variables))] or ["v0"]
        aliases = {v: "(" + "*".join(kernel_names[j] for j,w in enumerate(variables) if w == v) + ")"
                   for v in variables}
        kernel_tokens = [aliases.get(t, t) for t in tokens]
        if any(t in kernel_names and t not in aliases for t in tokens):
            raise InputError("Unknown chart variable.")
    if not bounded:
        _check_literal_digits(tokens)

    def native_row():
        # Preserve Unicode whitespace and separate tokens such as '* *'.
        # The input grammar and source/token budgets above see the original text.
        if coefficient_modulus is not None:
            if bounded or not variables:
                raise ValueError("Coefficient-field parsing requires an unrestricted ring with a named parameter.")
            lines = ["parse_field_input", " ".join(kernel_names), coefficient_modulus]
        elif bounded:
            lines = ["parse_quick", " ".join(kernel_names)]
            lines.append(f"{MAX_DEGREE} {MAX_TERMS} {MAX_BITS} {MAX_PRODUCTS}")
        else:
            lines = ["parse_input", " ".join(kernel_names)]
        try:
            row, = native_library.affine_request([*lines, " ".join(kernel_tokens)])
        except native_library.Unavailable:
            raise
        except ArithmeticError as error:
            message = str(error)
            if message.startswith("Input budget: "):
                raise LimitError(message.removeprefix("Input budget: ")) from error
            if not bounded and message in ("The chart expression is nested too deeply.", "Chart exponent exceeds FLINT's limits."):
                raise LimitError(message) from error
            raise InputError(message) from error

        return row

    if as_row and variables and native_library._MODE == "native" and not native_library.CHECK:
        return native_row()

    def native():
        parsed = sparse_row(native_row())
        return parsed if variables else {(): value for value in parsed.values()}

    terms = native() if coefficient_modulus is not None else checked_native(native, reference)
    return encode_sparse_row(terms) if as_row else terms


def checked_native(native, reference):
    from . import native_library
    if native_library._MODE == "sympy":
        return reference()
    value = native()
    if native_library.CHECK and value != reference():
        raise AssertionError("The native polynomial operation differs from its independent calculation.")
    return value


def integer_from_decimal(text):
    """Decode an exact kernel integer, even beyond Python's input-literal limit."""
    if len(text) < 640:  # Below Python's smallest configurable guard.
        return int(text)
    digit_limit = sys.get_int_max_str_digits() if hasattr(sys, "get_int_max_str_digits") else 0
    return int(Decimal(text)) if digit_limit and len(text) > digit_limit else int(text)


def encode_sparse_row(terms):
    """Encode coefficients and exponent tuples in the supplied coordinate order."""
    return ";".join(rational_text(c) + " " + " ".join(map(integer_text, e)) for e, c in terms.items() if c)


# Fraction accepts any Rational's already canonical numerator and denominator.
_KernelRatio = namedtuple("_KernelRatio", ("numerator", "denominator"))
Rational.register(_KernelRatio)


def kernel_fraction(text):
    """Transport a canonical native coefficient into Python's immutable rational type."""
    n, _, d = text.partition("/")
    n = integer_from_decimal(n)
    if not d:
        return Fraction(n)
    d = integer_from_decimal(d)
    if d <= 0:
        raise ValueError("Invalid denominator in a native coefficient.")
    return Fraction(_KernelRatio(n, d))


def sparse_row(row):
    """Read FLINT's rational coefficient and exponent tuples without a dense representation."""
    terms = {}
    for term in row.split(";") if row else ():
        coefficient, *powers = term.split()
        terms[tuple(map(integer_from_decimal, powers))] = kernel_fraction(coefficient)
    return terms


def _parse_tokens(tokens, variables, bounded):
    from tests.polynomial_parser_oracle import parse_tokens
    return parse_tokens(tokens, variables, bounded)


def integer_text(n):
    """Format computed coefficients and exponents beyond the decimal literal budget."""
    if type(n) is not int:
        n = int(n)
    return str(n) if n.bit_length() < 2000 else str(Decimal(n))


def rational_text(value):
    den = value.denominator
    return integer_text(value.numerator) if den == 1 else integer_text(value.numerator) + "/" + integer_text(den)


def format_poly(p, names, scalar=rational_text):
    words = []
    for e, c in sorted(p.items(), key=lambda pair: (sum(pair[0]), pair[0]), reverse=True):
        if not c:
            continue
        coeff = scalar(c)
        negative = coeff.startswith('-')
        if negative:
            coeff = coeff[1:]
        factors = [s if k == 1 else f'{s}^{integer_text(k)}' for s, k in zip(names, e) if k]
        if coeff != '1' or not factors:
            factors.insert(0, coeff)
        words.append(('-' if negative else '+' if words else '') + '*'.join(factors))
    return ''.join(words) or '0'
