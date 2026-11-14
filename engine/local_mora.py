"""Exact weak division at the coordinate origin by a local standard basis.

Finite length gives a fully reduced representative of f. In positive
dimension Mora division gives u(0)=1 and u*f-r in the localized ideal.
A nonzero r has a local leading monomial outside the initial ideal.
"""

from dataclasses import dataclass

from . import native_polynomial as np
from .algebra_execution import current_execution, stage_label
from .native_library import Unavailable
from .wide_basis import wide_basis_from_rows


def _divides(divisor, multiple):
    return all(a <= b for a, b in zip(divisor, multiple))


def finite_degree_cutoff(width, heads):
    """First degree known to vanish in the local quotient, or None."""
    pure = [None] * width
    for head in heads:
        axes = [i for i, exponent in enumerate(head) if exponent]
        if len(axes) == 1:
            i = axes[0]
            if pure[i] is None or head[i] < pure[i]:
                pure[i] = head[i]
    if any(bound is None for bound in pure):
        return None
    # The associated graded quotient vanishes from N onward. Therefore
    # m^N=m*m^N in the local quotient, and Nakayama gives m^N=0.
    return 1 + sum(bound - 1 for bound in pure)


@dataclass
class _Reducer:
    terms: dict
    head: tuple
    ecart: int
    # The coefficient of the input f in this row's representation.
    unit: dict


def _subtract_multiple(target, source, shift, scale, cutoff=None):
    for monomial, c in source.items():
        power = tuple(i + j for i, j in zip(monomial, shift))
        if cutoff is not None and sum(power) >= cutoff:
            continue
        coefficient = target.get(power, 0) - scale * c
        if coefficient:
            target[power] = coefficient
        else:
            target.pop(power, None)


def weak_normal_form(variables, generators, f, order):
    """Use centered QQ coordinates; break lowest-degree ties by lex or grevlex."""
    variables = tuple(variables)
    zero = (0,) * len(variables)
    if order == "lex":
        def key(m):
            return -sum(m), m
    elif order == "grevlex":
        def key(m):
            return -sum(m), tuple(-i for i in reversed(m))
    else:
        raise ValueError("Local division requires a lex or grevlex tie order.")

    reducers = []
    for generator in generators:
        if not generator:
            continue
        terms = np.terms(variables, generator)
        if zero in terms:
            # A nonzero constant term makes this generator a local unit.
            return np.S.One, np.S.Zero, "standard"
        head = max(terms, key=key)
        if all(_divides(head, m) for m in terms):
            # g=x^head*a with a(0) nonzero. Use its monomial associate in
            # the local ring, even when the other exponents are enormous.
            terms = {head: 1}
        ecart = max(map(sum, terms)) - sum(head)
        reducers.append(_Reducer(terms, head, ecart, {}))
    if not reducers or not f:
        return np.S.One, f, "standard"

    if all(len(r.terms) == 1 for r in reducers):
        # Cancellation of one divisible term cannot affect another in a
        # monomial ideal, including a basis of monomials times local units.
        terms = np.terms(variables, f)
        execution = current_execution()
        if execution is not None and execution.replay.compute == "gpu":
            from .gpu_normal_form import monomial_witnesses
            with stage_label("local_normal_form"):
                witnesses = monomial_witnesses(len(variables), tuple(terms), [r.head for r in reducers])
            if witnesses is not None:
                standard = {m: c for (m, c), w in zip(terms.items(), witnesses) if w == -1}
                return np.S.One, np.from_terms(variables, standard), "standard"
        standard = {m: c for m, c in terms.items()
                    if not any(_divides(r.head, m) for r in reducers)}
        return np.S.One, np.from_terms(variables, standard), "standard"

    if all(r.ecart == 0 for r in reducers):
        # Every cancellation stays in one total degree. The global tie order
        # therefore gives the same normal monomials in each homogeneous piece,
        # and ordinary ordered QQ division applies to the entire polynomial.
        # Its sparse remainder kernel can replace x^N modulo x-y by y^N
        # directly, without constructing the N-term division witness.
        rows = tuple(np.row(variables, np.from_terms(variables, r.terms)) for r in reducers)
        basis = wide_basis_from_rows(rows, variables, order)
        if basis is None or not hasattr(basis.library, "atlas_wide_remainder_abi"):
            raise Unavailable("Rebuild the native kernel for sparse remainder-only division.")
        return np.S.One, basis.remainder(f), "standard"

    cutoff = finite_degree_cutoff(len(variables), (r.head for r in reducers))
    finite = cutoff is not None
    pending = np.terms(variables, f)
    remainder = {}
    if finite:
        # Division is linear in this finite quotient. Keep already standard
        # terms out of the reduction queue, even if other input terms reduce.
        initial = pending
        pending = {}
        for m, coefficient in initial.items():
            if sum(m) >= cutoff:
                continue
            if any(_divides(r.head, m) for r in reducers):
                pending[m] = coefficient
            else:
                remainder[m] = coefficient
        if len(remainder) == len(initial):
            return np.S.One, f, "standard"
    unit = {zero: 1}
    reduction = "standard"
    while pending:
        head = max(pending, key=key)
        reducer = min((r for r in reducers if _divides(r.head, head)),
                      key=lambda r: r.ecart, default=None)
        if reducer is None:
            if finite:
                coefficient = remainder.get(head, 0) + pending.pop(head)
                if coefficient:
                    remainder[head] = coefficient
                else:
                    remainder.pop(head, None)
                continue
            remainder = pending
            if any(_divides(r.head, m) for m in remainder for r in reducers if not r.unit):
                reduction = "weak"
            break
        if not finite:
            ecart = max(map(sum, pending)) - sum(head)
            if reducer.ecart > ecart:
                # Mora inserts this dividend precisely when the next reduction
                # raises its homogenized degree. Its homogenized head is outside
                # the current reducer head ideal, so only finitely many insertions
                # occur. Between insertions, reduction in bounded degree terminates.
                reducers.append(_Reducer(pending.copy(), head, ecart, unit.copy()))
        shift = tuple(i - j for i, j in zip(head, reducer.head))
        scale = pending[head] / reducer.terms[reducer.head]
        _subtract_multiple(pending, reducer.terms, shift, scale, cutoff)
        if reducer.unit:
            # Saved dividends have strictly earlier leading monomials. Their
            # head can divide the new head only with positive-degree shift.
            # Consequently subtraction preserves the constant coefficient 1.
            _subtract_multiple(unit, reducer.unit, shift, scale)
    return np.from_terms(variables, unit), np.from_terms(variables, remainder), reduction
