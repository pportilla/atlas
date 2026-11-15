"""Immutable polynomial values over Q for chart orchestration.

An element is the nonzero coefficient/exponent rows in its actual variables.
Coordinate names identify public symbols. A private coordinate has a distinct
identity even when another has the same printed name. Aligning or permuting
coordinates only relabels columns. FLINT performs arithmetic, exact division
and simultaneous substitution before returning a canonical sparse value.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property, lru_cache
from itertools import count

from .native_library import affine_request
from .polynomial import format_poly, integer_text, kernel_fraction, rational_text, sparse_row


@dataclass(frozen=True, order=True)
class Coordinate:
    name: str
    identity: int = 0


class Expr:
    __slots__ = ('_coords', '_terms', '_hash', '__dict__')
    is_commutative = True

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError("Polynomial values are immutable.")
        object.__setattr__(self, name, value)

    def __init__(self, coords, entries):
        self._coords = coords
        self._terms = entries
        self._hash = hash((coords, entries)) if coords else hash(self.scalar)

    @property
    def scalar(self):
        if self._coords:
            raise TypeError('Expected a constant polynomial.')
        return self._terms[0][1] if self._terms else Fraction(0)

    @property
    def p(self):
        return self.scalar.numerator
    @property
    def q(self):
        return self.scalar.denominator
    @property
    def is_number(self):
        return not self._coords
    is_Rational = is_number
    @property
    def is_Integer(self):
        return self.is_number and self.scalar.denominator == 1
    @property
    def is_Symbol(self):
        return len(self._coords) == 1 and self._terms == (((1,), 1),)
    @property
    def name(self):
        if not self.is_Symbol:
            raise AttributeError('Only a coordinate has a name.')
        return self._coords[0].name

    @cached_property
    def free_symbols(self):
        return frozenset(_symbol(k) for k in self._coords)
    def __hash__(self):
        return self._hash
    def __bool__(self):
        return bool(self._terms)
    def __int__(self):
        return int(self.scalar)
    def __eq__(self, other):
        if isinstance(other, Expr):
            return self._coords == other._coords and self._terms == other._terms
        if isinstance(other, (int,Fraction)):
            return self.is_number and self.scalar == other
        return NotImplemented
    def __lt__(self, other):
        return scalar_arithmetic("compare", self.scalar, coerce(other).scalar) < 0
    def __le__(self, other):
        return scalar_arithmetic("compare", self.scalar, coerce(other).scalar) <= 0
    def __gt__(self, other):
        return scalar_arithmetic("compare", self.scalar, coerce(other).scalar) > 0
    def __ge__(self, other):
        return scalar_arithmetic("compare", self.scalar, coerce(other).scalar) >= 0
    def __neg__(self):
        return _binary('-', coerce(0), self)
    def __add__(self, other):
        return _binary('+', self, coerce(other))
    __radd__ = __add__
    def __sub__(self, other):
        return _binary('-', self, coerce(other))
    def __rsub__(self, other):
        return _binary('-', coerce(other), self)
    def __mul__(self, other):
        return _binary('*', self, coerce(other))
    __rmul__ = __mul__
    def __truediv__(self, other):
        return _binary('/', self, coerce(other))
    def __rtruediv__(self, other):
        return _binary('/', coerce(other), self)
    def __pow__(self, n):
        if int(n) != n or n < 0:
            raise ValueError("Expected a nonnegative integral exponent.")
        return _binary('^', self, coerce(n))

    def __str__(self):
        return format_poly(dict(self._terms), [k.name for k in self._coords])
    __repr__ = __str__

    def xreplace(self, replacements):
        if self in replacements:
            return coerce(replacements[self])
        if not self._coords or not any(v in replacements for v in self.free_symbols):
            return self
        variables = tuple(map(_symbol, self._coords))
        mapping = tuple(coerce(replacements.get(v,v)) for v in variables)
        # A bijective relabeling only permutes exponent columns.
        if all(f.is_Symbol for f in mapping) and len(set(mapping)) == len(mapping):
            keys = tuple(f._coords[0] for f in mapping)
            if keys == self._coords:
                return self
            if keys == tuple(sorted(keys)):
                return Expr(keys, self._terms)
            return from_terms(mapping, dict(self._terms))
        target_keys, request, source_names = _substitution_request(self._coords, mapping)
        lines = affine_request((*request, self.text(source_names)))
        return _decode(target_keys, lines[0])

    def text(self, names):
        if self._coords and not self.free_symbols.issubset(names):
            raise ValueError('Polynomial uses coordinates outside the declared ring.')
        return format_poly(dict(self._terms), [names[_symbol(k)] for k in self._coords])

    def has(self, *symbols):
        return not self.free_symbols.isdisjoint(symbols)
    def as_expr(self):
        return self


@lru_cache(maxsize=2048)
def _symbol(key):
    return Expr((key,), (((1,), Fraction(1)),))

def Symbol(name):
    return _symbol(Coordinate(name))
_dummy_indices = count(1)
def Dummy(name='t'):
    return _symbol(Coordinate(name, next(_dummy_indices)))


@lru_cache(maxsize=2048)
def _constant(value):
    q = Fraction(value)
    return Expr((), (((), q),) if q else ())


def coerce(value):
    if isinstance(value, Expr):
        return value
    if isinstance(value, (int, Fraction)):
        return _constant(value)
    raise TypeError('Expected a native polynomial or rational number.')


def from_terms(variables, entries):
    if len(set(variables)) != len(variables) or any(not v.is_Symbol for v in variables):
        raise ValueError('Expected distinct polynomial coordinates.')
    data = tuple((e, c if type(c) is Fraction else Fraction(c)) for e,c in entries.items() if c)
    if any(len(e) != len(variables) for e,c in data):
        raise ValueError('Invalid polynomial exponent vector.')
    active = sorted((v._coords[0], i) for i,v in enumerate(variables) if any(e[i] for e,c in data))
    if not active:
        return coerce(data[0][1] if data else 0)
    rows = tuple(sorted(((tuple(e[i] for _,i in active), c) for e,c in data), reverse=True))
    keys = tuple(k for k,i in active)
    if len(keys)==1 and rows == (((1,),1),):
        return _symbol(keys[0])
    return Expr(keys, rows)


def from_row(variables, line):
    return from_terms(variables, sparse_row(line))


def from_kernel_row(variables, line):
    value = from_row(variables, line)
    if value._coords and not value.is_Symbol:
        # A kernel result can supply its original ambient row when it becomes
        # a session polynomial. Interned scalars and generators span many rings.
        value.__dict__["_source_row"] = tuple(variables), line
    return value


def terms(variables, value):
    f = coerce(value)
    positions = {v._coords[0]: i for i,v in enumerate(variables)}
    indices = [positions[k] for k in f._coords]
    result = {}
    for e,c in f._terms:
        powers = [0]*len(variables)
        for j,k in zip(indices,e): powers[j] = k
        result[tuple(powers)] = c
    return result


def row(variables, value):
    """Encode an immutable polynomial in the requested chart coordinate order."""
    return _encoded_row(tuple(variables), coerce(value))


def generator_row(variables, value):
    """A native generator's exact sparse bytes in this ambient coordinate order."""
    source = value.__dict__.pop("_source_row", None)
    # The native polynomial owns the row. Keeping another copy in the global
    # transport cache for every short-lived session value can retain large inputs.
    line = source[1] if source is not None and source[0] == variables else _encode_row(variables, value)
    return line.encode("ascii")


def _encode_row(variables, value):
    columns = _chart_positions(variables)
    indices = tuple(columns[key] for key in value._coords)
    lines = []
    for powers, coefficient in value._terms:
        exponents = ['0'] * len(variables)
        for index, power in zip(indices, powers):
            exponents[index] = integer_text(power)
        lines.append(rational_text(coefficient) + ' ' + ' '.join(exponents))
    return ';'.join(lines)


_encoded_row = lru_cache(maxsize=16384)(_encode_row)


@lru_cache(maxsize=1024)
def _chart_positions(variables):
    return {v._coords[0]: i for i, v in enumerate(variables)}


@lru_cache(maxsize=1024)
def _names(keys):
    names = {v:f'v{j}' for j,v in enumerate(map(_symbol,keys))}
    return ' '.join(names.values()) or 'v0', names


@lru_cache(maxsize=1024)
def _substitution_request(source_keys, images):
    """Reuse the native map description across distinct polynomial operands."""
    target_keys = tuple(sorted({key for image in images for key in image._coords}))
    header, target_names = _names(target_keys)
    source, source_names = _names(source_keys)
    request = ('pullbacks', header, source, *(f.text(target_names) for f in images), '0')
    return target_keys, request, source_names


def _decode(keys, line):
    if keys:
        return from_kernel_row(tuple(map(_symbol,keys)),line)
    return coerce(next(iter(sparse_row(line).values()),0))


@lru_cache(maxsize=16384)
def _binary(op, left, right):
    if op == '/' and not right:
        raise ZeroDivisionError
    keys = tuple(sorted(set(left._coords).union(right._coords)))
    header, names = _names(keys)
    source = (f'({left.text(names)})^{integer_text(int(right))}' if op == '^' else
              f'({left.text(names)}){op}({right.text(names)})')
    result, = affine_request(['function', header, source])
    return _decode(keys, result)


@lru_cache(maxsize=4096)
def count_ops(value):
    from .point_geometry import encode_terms, request
    f = coerce(value)
    return int(request("expression_size", width=len(f._coords), terms=encode_terms(dict(f._terms))))


def default_sort_key(value):
    if isinstance(value, str):
        return (3, value)
    if isinstance(value,(tuple,list)):
        return (2,tuple(map(default_sort_key,value)))
    f = coerce(value)
    return (int(bool(f._coords)), f._coords, f._terms)


def _combine(values, operation):
    values = tuple(map(coerce, values))
    if not values:
        return coerce(scalar_arithmetic("empty_product" if operation == "*" else "empty_sum", 0, 0))
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return _binary(operation, *values)
    keys = tuple(sorted({key for value in values for key in value._coords}))
    header, names = _names(keys)
    expression = operation.join(f"({v.text(names)})" for v in values)
    result, = affine_request(["function", header, expression])
    return _decode(keys, result)


def product(values):
    return _combine(values, "*")

def scalar_arithmetic(op, left, right):
    row, = affine_request(["scalar_arithmetic", op, rational_text(left), rational_text(right)])
    return kernel_fraction(row)


def Rational(a, b=None):
    numerator = a.scalar if isinstance(a, Expr) else Fraction(a)
    if b is None:
        return coerce(numerator)
    denominator = b.scalar if isinstance(b, Expr) else Fraction(b)
    if not denominator:
        raise ZeroDivisionError
    return coerce(scalar_arithmetic("/", numerator, denominator))


def symbols(pattern):
    """Numbered coordinates in the half-open range prefixJ:K."""
    import re
    match = re.fullmatch(r"(.*?)([0-9]+):([0-9]+)", pattern)
    if match:
        return tuple(Symbol(match[1]+str(j)) for j in range(int(match[2]), int(match[3])))
    return tuple(Symbol(name) for name in pattern.replace(",", " ").split())


def Add(*values):
    return _combine(values, "+")
def Mul(*values):
    return product(values)
def Integer(value):
    return coerce(int(value))
sympify = coerce
prod = product

class S:
    Zero = coerce(0)
    One = coerce(1)

def native_ring(variables):
    return bool(variables) and isinstance(variables[0], Expr)
