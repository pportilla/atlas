"""Radicals in QQ[x] by separable coordinate eliminants and localization."""

from . import native_polynomial as np
from .affine_kernel import _request, _ring
from .ideal_operations import IdealBasis, intersection, saturate
from .local_algebra import _basis_heads


def _squarefree(variables, f):
    """Retain all factors, including factors involving only parameters."""
    terms = np.terms(variables, f)
    if len(terms) == 1:
        powers, = terms
        return np.from_terms(variables, {tuple(int(e > 0) for e in powers): 1})
    ring = _ring(variables)
    row, = _request(["squarefree_part", ring.header, ring.text(f)])
    return np.from_row(variables, row)


def _coordinate_relations(variables, generators, heads, basis):
    """Find independent parameters U and, for each remaining x, a relation in QQ[U,x]."""
    params, dependent, relations = [], [], []
    head_supports = [{v for v, power in zip(variables, powers) if power} for powers in heads]
    for x in reversed(variables):
        keep = {x, *params}
        relation = next((g for g in generators if g.free_symbols <= keep), None)
        if relation is None and any(support <= keep for support in head_supports):
            # A relation in the subring must have a head divisible by a
            # global basis head in the same subring. When that is possible,
            # eliminate the other variables to decide independence exactly.
            block = tuple(v for v in variables if v not in keep) + (x, *params)
            rows = basis("radical_coordinate_elimination", block, generators, "lex").exprs
            relation = next((g for g in rows if g.free_symbols <= keep), None)
        if relation is None:
            params.append(x)
        else:
            dependent.append(x)
            relations.append(relation)
    return tuple(dependent), tuple(params), relations


def _exceptional_polynomial(dependent, params, block_basis):
    # A lex basis in (dependent,params) is a Gröbner basis over QQ(params)
    # after its leading coefficients in the dependent variables are inverted.
    block = dependent + params
    n = len(dependent)
    coefficients = []
    for f in block_basis:
        terms = np.terms(block, f)
        head = max(terms)[:n]
        lc = np.from_terms(params, {e[n:]: c for e, c in terms.items() if e[:n] == head})
        if not lc.is_number:
            coefficients.append(lc)
    if not coefficients:
        return np.S.One
    return _squarefree(params, np.product(dict.fromkeys(coefficients)))


def radical(variables, generators, *, order, basis, stage):
    """Return the reduced basis of the global radical in the requested order."""
    variables = tuple(variables)
    generators = tuple(f for f in generators if f)
    pieces = []
    while True:
        if not generators or any(f.is_number for f in generators):
            result = IdealBasis(variables, (np.S.One,) if generators else (), order)
            break
        # Each squarefree part belongs to the radical of its principal ideal.
        # This also reduces arbitrarily large monomial powers before any basis.
        with stage("radical_squarefree_generators", generators=len(generators)):
            squarefree = tuple(dict.fromkeys(_squarefree(variables, f) for f in generators))
        rows = tuple(basis("radical_basis", variables, squarefree, order).exprs)
        heads = _basis_heads(variables, rows, order, "radical_initial_monomials")
        if len(squarefree) == 1 or all(all(e <= 1 for e in h) for h in heads):
            # If the initial ideal is radical, the ideal itself is radical:
            # the head of a nonzero normal form cannot have a power in it.
            result = IdealBasis(variables, rows, order)
            break

        dependent, params, relations = _coordinate_relations(variables, rows, heads, basis)
        with stage("radical_squarefree_eliminants", coordinates=len(relations)):
            extra = tuple(_squarefree(variables, f) for f in relations)
        augmented = tuple(dict.fromkeys((*rows, *extra)))
        # Over QQ(params), each dependent coordinate satisfies a separable
        # univariate relation. The quotient is therefore a quotient of a
        # finite product of fields, hence reduced. The added equations also
        # preserve the original variety on every exceptional fiber.
        if params:
            block = dependent + params
            generic = basis("radical_localization_basis", block, augmented, "lex")
            with stage("radical_leading_coefficients", generators=len(generic.exprs)):
                h = _exceptional_polynomial(dependent, params, generic.exprs)
        else:
            h = np.S.One
        if h.is_number:
            final = rows if augmented == rows else basis("radical_result_basis", variables, augmented, order).exprs
            result = IdealBasis(variables, tuple(final), order)
            break

        # The monic localization basis divides using only powers of h in its
        # denominators. Thus saturation gives the contraction of the reduced
        # generic ideal. Cover the remaining locus h=0 in the next iteration.
        pieces.append(saturate(variables, augmented, h, order=order, basis=basis, stage=stage))
        generators = (*augmented, h)
        # h is a nonzero polynomial in independent parameters. Each exceptional
        # ideal strictly enlarges the preceding ideal, so this loop terminates.

    for piece in reversed(pieces):
        result = intersection(variables, piece.generators, result.generators, order=order, basis=basis, stage=stage)
    return result
