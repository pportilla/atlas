"""Render an external field element in its original symbolic generators.

For an additive primitive element, work first in the tensor product of its
summands' simple presentations. Their univariate minimal polynomials have
coprime leading monomials, so ordinary FLINT division gives the reduced
power table without doing a symbolic expansion. Displaying those terms
can introduce products and nested radicals; expand them in a second
triangular presentation and assemble the resulting symbolic leaves.
"""
from fractions import Fraction
from functools import cmp_to_key, lru_cache

from .native_library import Unavailable, affine_request, field_request
from .polynomial import format_poly, integer_from_decimal, integer_text, rational_text, sparse_row
from .rational_point import coordinate_fraction
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
    return tuple(reversed(tuple(map(integer_from_decimal, description.split()))[1:])) if description else ()


def _request_display(operation, rows):
    reply = field_request("atlas_field_display_terms", [operation, *rows])
    if reply is None:
        raise Unavailable("Rebuild the marked-field display kernel with make groebner.")
    return reply.splitlines()


@lru_cache(maxsize=4096)
def _monomial(atoms, powers):
    occupied = [(i,n) for i,n in enumerate(powers) if n]
    if not occupied:
        return Fraction(1), sp.S.One
    if len(occupied) == 1 and occupied[0][1] == 1:
        return Fraction(1), atoms[occupied[0][0]]
    rows = []
    for i,n in occupied:
        atom = atoms[i]
        parameters = f"{i} {integer_text(n)}"
        if atom == sp.I:
            rows.append(f"i {parameters}")
        elif atom.is_Pow and atom.exp.is_Rational:
            if atom.base.is_Rational:
                rows.append(f"rational {parameters} {rational_text(atom.base)} {rational_text(atom.exp)}")
            else:
                rows.append(f"power {parameters} {rational_text(atom.exp)}")
        elif atom.func is sp.exp and (angle := atom.args[0].as_coefficient(sp.I*sp.pi)) is not None and angle.is_Rational:
            rows.append(f"exp {parameters} {rational_text(angle)}")
        else:
            rows.append(f"atom {parameters}")
    scale, *leaves = _request_display("monomial",rows)
    factors = []
    for leaf in leaves:
        kind, *parts = leaf.split()
        if kind == "i":
            factors.append(sp.I)
        elif kind == "exp":
            angle = sp.Rational(coordinate_fraction(parts[0]))
            factors.append(sp.exp(sp.I*sp.pi*angle,evaluate=False))
        elif kind == "radical":
            exponent, base = map(coordinate_fraction,parts)
            factors.append(_pow(sp.Rational(base),sp.Rational(exponent)))
        else:
            atom = atoms[int(parts[0])]
            exponent = sp.Rational(coordinate_fraction(parts[1]))
            if kind == "atom":
                factors.append(_pow(atom,exponent))
            elif kind == "power":
                factors.append(_pow(atom.base,exponent))
            else:
                positive = _pow(atom.base,exponent)
                denominator = _expanded_display((positive,),{(1,): Fraction(1)})
                factors.append(_pow(denominator,-1))
    return coordinate_fraction(scale), _held(sp.Mul,factors)


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
    indices, rows = {}, []
    for m,c in sparse_row(row).items():
        scale, leaf = _monomial(leaves,m)
        index = indices.setdefault(leaf,len(indices))
        rows.append(f"{index} {rational_text(c)} {rational_text(scale)}")
    leaves = tuple(indices)
    result = {}
    for entry in _request_display("combine",rows):
        i,c = entry.split()
        result[leaves[int(i)]] = coordinate_fraction(c)
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
