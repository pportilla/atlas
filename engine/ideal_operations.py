"""Exact ideal operations on ordered rational coordinates (KERNEL.md §3).

`basis(label, variables, generators, order)` computes a reduced basis. Passing
this stage through the algebra caller keeps its execution policy and counters
on every elimination, including the intermediate intersections of a quotient.
"""
from contextlib import nullcontext
from dataclasses import dataclass

from .affine_kernel import _request, _ring
from .ordered_ideal_kernel import ordered_ideal
from .algebra_groebner import reduced_basis
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


def eliminate(variables, generators, removed, *, order="grevlex", basis=_basis, stage=_stage):
    """Contract to the subring on the surviving variables in their original order."""
    variables, removed = tuple(variables), tuple(removed)
    if not removed or len(set(removed)) != len(removed) or not set(removed).issubset(variables):
        raise ValueError("Elimination needs distinct coordinates of the input ring.")
    return IdealBasis(*ordered_ideal("eliminate", variables, generators, (), order, basis, stage, removed=removed))


def intersection(variables, left, right, *, order="grevlex", basis=_basis, stage=_stage):
    return IdealBasis(*ordered_ideal("intersection", variables, left, right, order, basis, stage))


def saturate(variables, generators, polynomial, *, order="grevlex", basis=_basis, stage=_stage):
    """Contract I + (t*f - 1), the presentation of I after inverting f."""
    return IdealBasis(*ordered_ideal("saturate", variables, generators, (), order, basis, stage, by=polynomial))


def quotient(variables, left, right, *, order="grevlex", basis=_basis, stage=_stage):
    return IdealBasis(*ordered_ideal("quotient", variables, left, right, order, basis, stage))


def dimension(variables, generators, *, basis=_basis, stage=_stage):
    """Dimension from leading supports, with -1 for the empty variety."""
    variables, generators = tuple(variables), tuple(generators)
    if not variables:
        from .ordered_ideal_kernel import scalar_ideal
        return scalar_ideal("dimension", generators)
    result = basis("dimension_basis", variables, generators, "grevlex")
    with stage("leading_support_dimension", generators=len(result.exprs), variables=len(variables)):
        ring = _ring(variables)
        value, = _request(["basis_dimension", ring.header, "grevlex", *map(ring.text, result.exprs)])
    return int(value)
