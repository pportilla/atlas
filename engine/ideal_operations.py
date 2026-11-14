"""Exact ideal operations on ordered rational coordinates (KERNEL.md §3).

`basis(label, variables, generators, order)` computes a reduced basis. Passing
this stage through the algebra caller keeps its execution policy and counters
on every elimination, including the intermediate intersections of a quotient.
"""
from contextlib import nullcontext
from dataclasses import dataclass

from .affine_kernel import _request, _ring, _sparse_expression, sparse_terms
from .algebra_groebner import reduced_basis
from .chart_expressions import chart_expressions
from .groebner import groebner
from .native_polynomial import native_ring


@dataclass(frozen=True)
class IdealBasis:
    variables: tuple
    generators: tuple
    order: str


def _basis(label, variables, generators, order):
    if native_ring(variables):
        return reduced_basis(generators, variables, order=order)
    return groebner(generators, variables, order=order)


def _stage(label, **counts):
    return nullcontext()


def _constant_ideal(variables, unit, order):
    one = chart_expressions(variables).S.One
    return IdealBasis(variables, (one,) if unit else (), order)


def _reduce(variables, generators, order, basis, label):
    if not variables:
        return _constant_ideal(variables, any(generators), order)
    result = basis(label, variables, generators, order)
    return IdealBasis(variables, tuple(result.exprs), order)


def _contract(removed, surviving, generators, order, basis, stage, operation):
    # A lex basis with an entire prefix eliminated induces lex on the suffix.
    # The first-variable block order induces grevlex on its suffix.
    extended = (*removed, *surviving)
    elimination_order = {"lex": "lex", "grevlex": "first"}[order]
    rows = basis(f"{operation}_basis", extended, generators, elimination_order).exprs
    with stage(f"{operation}_contract", generators=len(rows), variables_removed=len(removed),
               surviving_variables=len(surviving)):
        contracted = tuple(row for row in rows if row.free_symbols.isdisjoint(removed))
    return IdealBasis(surviving, contracted, order)


def eliminate(variables, generators, removed, *, order="grevlex", basis=_basis, stage=_stage):
    """Contract to the subring on the surviving variables in their original order."""
    variables, generators, removed = tuple(variables), tuple(generators), tuple(removed)
    if not removed or len(set(removed)) != len(removed) or not set(removed).issubset(variables):
        raise ValueError("Elimination needs distinct coordinates of the input ring.")
    if order == "lex":
        surviving = tuple(v for v in variables if v not in removed)
        return _contract(removed, surviving, generators, order, basis, stage, "elimination")
    for index, variable in enumerate(removed, start=1):
        surviving = tuple(v for v in variables if v != variable)
        operation = "elimination" if len(removed) == 1 else f"elimination_{index}"
        result = _contract((variable,), surviving, generators, order, basis, stage, operation)
        variables, generators = surviving, result.generators
    return result


def intersection(variables, left, right, *, order="grevlex", basis=_basis, stage=_stage):
    """The contraction of tA + (1-t)B equals A intersect B."""
    return _intersect(tuple(variables), tuple(left), tuple(right), order, basis, stage, "intersection")


def _intersect(variables, left, right, order, basis, stage, operation):
    if not variables:
        return _constant_ideal(variables, any(left) and any(right), order)
    t = chart_expressions(variables).Dummy("intersection")
    with stage(f"{operation}_lift", left_generators=len(left), right_generators=len(right)):
        lifted = (*(t*f for f in left if f != 0), *((1-t)*g for g in right if g != 0))
    return _contract((t,), variables, lifted, order, basis, stage, operation)


def saturate(variables, generators, polynomial, *, order="grevlex", basis=_basis, stage=_stage):
    """Contract I + (t*f - 1), the presentation of I after inverting f."""
    variables, generators = tuple(variables), tuple(generators)
    if polynomial == 0:
        return _constant_ideal(variables, True, order)
    if not variables or not chart_expressions(variables).sympify(polynomial).free_symbols:
        return _reduce(variables, generators, order, basis, "saturation_basis")
    t = chart_expressions(variables).Dummy("inverse")
    with stage("adjoin_inverse", generators=len(generators)):
        lifted = (*generators, t*polynomial - 1)
    return _contract((t,), variables, lifted, order, basis, stage, "saturation")


def _divide_exact(variables, generators, divisor):
    """Divide every generator by the divisor, returning None on the first remainder."""
    ring = _ring(variables)
    result = []
    denominator = ring.text(divisor)
    for generator in generators:
        q, r = _request(["wide_division", ring.header, "grevlex", ring.text(generator), denominator])
        if r:
            return None
        result.append(_sparse_expression(variables, q))
    return tuple(result)


def quotient(variables, left, right, *, order="grevlex", basis=_basis, stage=_stage):
    """Intersect the principal quotients (I intersect (g))/g for nonzero g in J."""
    variables, left = tuple(variables), tuple(left)
    divisors = tuple(g for g in right if g != 0)
    if not divisors or not variables:
        return _constant_ideal(variables, not divisors or any(left), order)
    if len(divisors) == 1 and len(sparse_terms(variables, divisors[0])) == 1:
        # In QQ[x], (d*f1, ..., d*fk) : (d) = (f1, ..., fk) for d != 0.
        with stage("quotient_divide_inputs", input_generators=len(left)):
            divided = _divide_exact(variables, left, divisors[0])
        if divided is not None:
            return _reduce(variables, divided, order, basis, "quotient_basis")
    result = None
    for index, divisor in enumerate(divisors, start=1):
        operation = f"quotient_by_{index}"
        shared = _intersect(variables, left, (divisor,), order, basis, stage, operation)
        with stage(f"{operation}_divide", generators=len(shared.generators)):
            terms = sparse_terms(variables, divisor)
            monomial = len(terms) == 1
            if monomial:
                # A scalar unit does not change the principal ideal. Removing
                # it makes each quotient of a monic multiple monic as well.
                _, coefficient = terms[0]
                if coefficient != 1:
                    divisor = chart_expressions(variables).sympify(divisor) / coefficient
            part = _divide_exact(variables, shared.generators, divisor)
            if part is None:
                raise ArithmeticError("An intersection generator was not divisible by its principal generator.")
        if not part:
            return _constant_ideal(variables, False, order)
        if len(divisors) == 1:
            # Dividing by one monomial preserves order and divisibility of
            # every term, hence reducedness. With a polynomial, tails of the
            # quotients can need autoreduction, even when the factor is monic.
            if monomial:
                return IdealBasis(variables, part, order)
            return _reduce(variables, part, order, basis, "quotient_basis")
        if result is None:
            result = part
        else:
            joined = _intersect(variables, result, part, order, basis, stage, f"quotient_intersection_{index}")
            result = joined.generators
    return IdealBasis(variables, result, order)


def dimension(variables, generators, *, basis=_basis, stage=_stage):
    """Dimension from leading supports, with -1 for the empty variety."""
    variables, generators = tuple(variables), tuple(generators)
    if not variables:
        return -1 if any(generators) else 0
    result = basis("dimension_basis", variables, generators, "grevlex")
    with stage("leading_support_dimension", generators=len(result.exprs), variables=len(variables)):
        ring = _ring(variables)
        value, = _request(["basis_dimension", ring.header, "grevlex", *map(ring.text, result.exprs)])
    return int(value)
