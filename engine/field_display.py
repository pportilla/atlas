"""Render an external field element in its original symbolic generators.

For an additive primitive element, work first in the tensor product of its
summands' simple presentations. Their univariate minimal polynomials have
coprime leading monomials, so ordinary FLINT division gives the reduced
power table without doing a symbolic expansion. Displaying those terms
can introduce products and nested radicals; expand them in a second
triangular presentation and assemble the resulting symbolic leaves.
"""
from collections import defaultdict
from fractions import Fraction
from functools import cmp_to_key, lru_cache
from math import gcd

from .native_library import affine_request, field_request
from .polynomial import format_poly, rational_text, sparse_row
from .reference import sp


def _q(value):
    return Fraction(int(value.numerator), int(value.denominator))


def _held(operation, args):
    if not args:
        return sp.S.Zero if operation is sp.Add else sp.S.One
    if len(args) == 1:
        return args[0]
    args.sort(key=cmp_to_key(sp.Basic.compare))
    return operation(*args, evaluate=False)


def _pow(base, n):
    return base if n == 1 else sp.Pow(base, n, evaluate=False)


def _polynomial(coefficients, variable):
    return format_poly({(j,): _q(c) for j,c in enumerate(reversed(coefficients)) if c}, (variable,))


@lru_cache(maxsize=1024)
def _summand_relation(atom):
    from .field_coefficient import UnsupportedExpression, program
    if isinstance(atom, sp.CRootOf):
        return tuple(atom.poly.all_coeffs())
    if atom.is_Pow and atom.base.is_Rational and atom.exp.is_Rational:
        # QQbar is needed when the displayed root is not of full degree,
        # e.g. a principal power of a negative number.
        if atom.base > 0 and atom.exp == sp.S.Half:
            return (1, 0, -_q(atom.base))
    try:
        description = field_request("atlas_field_description", [" ".join(program(atom))])
    except UnsupportedExpression:
        return ()
    return tuple(reversed(tuple(map(int, description.split()))[1:])) if description else ()


@lru_cache(maxsize=1024)
def _prime_factors(n):
    row = field_request("atlas_field_integer_factors", [rational_text(n)])
    if row is None:
        return None
    return tuple(tuple(map(int, pair.split())) for pair in row.splitlines())


@lru_cache(maxsize=4096)
def _numeric_power(base, exponent):
    """Rational scale and positive rational-power factors for one atom."""
    integer, exponent = divmod(exponent, 1)
    scalar = base**integer
    if not exponent or base == 1:
        return scalar, []
    if exponent == Fraction(1, 2):
        result = field_request("atlas_field_radical", [rational_text(base)])
        if result:
            a,b,d,_ = map(int, result.split())
            return scalar * Fraction(a,b), [(d,exponent)] if d != 1 else []
    powers = {}
    for n,sign in ((base.numerator,1),(base.denominator,-1)):
        if n == 1:
            continue
        factors = _prime_factors(n)
        if factors is None:
            return scalar, [(base,exponent)]
        for prime,power in factors:
            k,r = divmod(sign*power*exponent.numerator,exponent.denominator)
            scalar *= Fraction(prime)**k
            if r:
                powers[prime] = r
    # A radical with residual exponents of common denominator q retains
    # their gcd in the exponent: 12^(2/5) uses one atom, not two.
    groups = defaultdict(dict)
    for p,r in powers.items():
        groups[exponent.denominator//gcd(r,exponent.denominator)][p] = r
    radicals = []
    for group in groups.values():
        common = gcd(*group.values())
        radicand = 1
        for p,r in group.items():
            radicand *= p**(r//common)
        radicals.append((radicand,Fraction(common,exponent.denominator)))
    return scalar, radicals


@lru_cache(maxsize=4096)
def _numeric_product(powers):
    """Combine equal exponents and separate common factors of different bases."""
    scalar = Fraction(1)
    grouped = defaultdict(lambda: Fraction(1))
    for base, exponent in powers:
        c, radicals = _numeric_power(base,exponent)
        scalar *= c
        for b,e in radicals:
            grouped[e] *= b
    pending = [(b,e) for e,b in grouped.items()]
    for i,(b,e) in enumerate(pending):
        for j in range(i):
            a,f = pending[j]
            # Native normalization returns integral residual radicands.
            if a.denominator != 1 or b.denominator != 1:
                continue
            g = gcd(a.numerator,b.numerator)
            if g > 1:
                pending[j] = (a//g,f)
                b //= g
                pending.append((g,e+f))
        pending[i] = (b,e)
    grouped.clear()
    for b,e in pending:
        if b == 1:
            continue
        c,radicals = _numeric_power(Fraction(b),e)
        scalar *= c
        for b,e in radicals:
            grouped[e] *= b
    return scalar, tuple((e,b) for e,b in grouped.items() if b != 1)


def _monomial(atoms, powers):
    if sum(powers) == 1:
        # A single existing leaf needs neither root normalization nor a
        # factorization of a potentially enormous rational radicand.
        return Fraction(1), atoms[powers.index(1)]
    factors, numeric = [], []
    phase = Fraction(0)
    exponential_angle = Fraction(0)
    for atom,n in zip(atoms,powers):
        if not n:
            continue
        if atom == sp.I:
            phase += Fraction(n,2)
        elif atom.is_Pow and atom.base.is_Rational and atom.exp.is_Rational:
            if atom.base < 0:
                phase += n*_q(atom.exp)
            numeric.append((abs(_q(atom.base)),n*_q(atom.exp)))
        elif atom.is_Pow and atom.exp.is_Rational:
            exponent = n*_q(atom.exp)
            if -1 < exponent < 1:
                factors.append(_pow(atom.base,sp.Rational(exponent)))
            else:
                # Negative powers retain a displayed denominator. Distribute
                # its positive power in the same triangular presentation.
                positive = _pow(atom.base,sp.Rational(abs(exponent)))
                denominator = _expanded_display((positive,),{(1,): Fraction(1)})
                factors.append(_pow(denominator,-1))
        elif atom.func is sp.exp:
            angle = atom.args[0].as_coefficient(sp.I*sp.pi)
            if angle is not None and angle.is_Rational:
                exponential_angle += n*_q(angle)
            else:
                factors.append(_pow(atom,n))
        else:
            factors.append(_pow(atom,n))
    angle = exponential_angle % 2
    if (2*angle).denominator == 1:
        phase += angle
    else:
        if angle > 1:
            angle -= 2
        factors.append(sp.exp(sp.I*sp.pi*sp.Rational(angle),evaluate=False))
    scale, numeric_factors = _numeric_product(tuple(numeric))
    radicals = dict(numeric_factors)
    whole, phase = divmod(phase,1)
    scale *= (-1)**whole
    if phase == Fraction(1,2):
        factors.append(sp.I)
    elif phase:
        radicals[phase] = -radicals.get(phase,1)
    for exponent,base in radicals.items():
        factors.append(sp.Pow(sp.Rational(base),sp.Rational(exponent),evaluate=False))
    return scale, _held(sp.Mul,factors)


def _combine(terms):
    result = []
    for m,c in terms.items():
        if c:
            factors = list(sp.Mul.make_args(m)) if m != 1 else []
            if c != 1 or not factors:
                factors.insert(0,sp.Rational(c))
            result.append(_held(sp.Mul,factors))
    return _held(sp.Add,result)


class _DisplayRing:
    """Leaf variables and triangular equations for nested radical powers."""
    def __init__(self):
        self.names = {}
        self.relations = {}

    def text(self, value):
        if value.is_Rational:
            return str(value)
        if value.is_Add or value.is_Mul:
            op = '+' if value.is_Add else '*'
            return '(' + op.join(self.text(a) for a in value.args) + ')'
        if value.is_Pow and value.exp.is_Integer and value.exp >= 0:
            return f"({self.text(value.base)})^{value.exp}"
        if (value.is_Pow and not value.base.is_Rational
                and value.exp.is_Rational and value.exp.p > 1):
            root = sp.Pow(value.base,sp.Rational(1,value.exp.q),evaluate=False)
            return f"({self.text(root)})^{value.exp.p}"
        if value not in self.names:
            # Introduce the radicand first. Reversing this order places
            # the radical itself ahead of all atoms on its relation's RHS.
            relation = None
            if (value.is_Pow and value.exp.is_Rational and value.exp > 0
                    and not value.base.is_Rational):
                relation = f"({self.text(value.base)})^{value.exp.p}"
            name = self.names[value] = f"d{len(self.names)}"
            if relation:
                self.relations[name] = f"{name}^{value.exp.q}-({relation})"
        return self.names[value]


def _expanded_display(atoms, terms):
    ring = _DisplayRing()
    substitutions = [ring.text(a) for a in atoms]
    # The coefficients already account for the primitive element's additive
    # relations. Only distribute products arising inside its display leaves.
    formula = format_poly(terms,[f"({a})" for a in substitutions])
    leaves = tuple(reversed(ring.names))
    header = " ".join(ring.names[a] for a in leaves) or "d0"
    if ring.relations:
        row = affine_request(["wide_division",header,"lex",formula,*ring.relations.values()])[-1]
    else:
        row, = affine_request(["function",header,formula])
    result = defaultdict(Fraction)
    for m,c in sparse_row(row).items():
        scale, leaf = _monomial(leaves,m)
        result[leaf] += c*scale
    return _combine(result)


@lru_cache(maxsize=8192)
def scalar(coefficients, generator):
    """Map descending rational coordinates to the caller's original expression."""
    if len(coefficients) == 1:
        return sp.Rational(coefficients[0])
    if not generator.is_Add:
        terms = {(j,):_q(c) for j,c in enumerate(reversed(coefficients)) if c}
        return _expanded_display((generator,),terms)

    summands = dict(generator.as_coefficients_dict())
    constant = _q(summands.pop(sp.S.One,0))
    atoms = tuple(summands)
    names = tuple(f"a{i}" for i in range(len(atoms)))
    header = " ".join(names)
    linear = "+".join([rational_text(constant),*(f"({summands[a]})*{v}" for a,v in zip(atoms,names))])
    relations = [_polynomial(p,v) for a,v in zip(atoms,names) if (p := _summand_relation(a))]
    # Horner reduction at each step bounds the degree in each summand.
    current = rational_text(_q(coefficients[0]))
    for c in coefficients[1:]:
        value = f"({current})*({linear})+({c})"
        if relations:
            row = affine_request(["wide_division",header,"lex",value,*relations])[-1]
        else:
            row, = affine_request(["function",header,value])
        terms = sparse_row(row)
        current = format_poly(terms,names)
    return _expanded_display(atoms,terms)
