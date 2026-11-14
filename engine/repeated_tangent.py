"""Smooth centres for a repeated projective tangent hypersurface.

After blowing up the origin, the reduced intersection of the strict
transform and the exceptional divisor is the projectivized tangent cone.
When its support is smooth and occurs with multiplicity, blow it up and
then blow up the intersection of the first two exceptional divisors.
The further centres, when needed, use the Jacobian rank-defect scheme of
the strict transform on the second exceptional divisor. All these schemes
are intrinsic to the indicated components, so their chart ideals glue.
"""
from functools import lru_cache
from itertools import combinations

from .affine_kernel import _polynomial, _sparse_expression, rational_factor_rows, sparse_terms
from .blowup_ideal import empty_locus, ideal_basis
from .centres import GiveUp
from .chart_expressions import chart_expressions
from .groebner import groebner


def _minors(variables, matrix, rank, basis):
    """Jacobian rank minors modulo a fixed base ideal, stopping at a unit."""
    api = chart_expressions(variables)

    @lru_cache(maxsize=None)
    def minor(rows, columns):
        if not rows:
            return api.S.One
        total = api.S.Zero
        for index, column in enumerate(columns):
            entry = matrix[rows[-1]][column]
            if entry:
                sign = (-1)**(len(rows)-1+index)
                total += sign * entry * minor(rows[:-1], columns[:index] + columns[index+1:])
        return basis.reduce(total)[1]

    nonzero = set()
    try:
        for rows in combinations(range(len(matrix)), rank):
            for columns in combinations(range(len(variables)), rank):
                value = minor(rows, columns)
                if value and not value.free_symbols:
                    return (api.S.One,)
                if value:
                    nonzero.add(value)
        return tuple(nonzero)
    finally:
        minor.cache_clear()


def certify_fibre(chart, dimension):
    """Certify the divisor ranks on the owned fibre of a smooth chart.

    Split a factored fibre or region equation into closed pieces. Reduce
    derivatives modulo the piece *after differentiating*. This avoids
    computing a Gröbner basis with the high-degree strict transform before
    the simpler exceptional equations have been used.
    """
    from .affine_kernel import frame_derivative
    from .local_blowups import _generators, _is_strict, _present

    variables = chart.variables
    ambient = _generators(variables, (), chart.ambient) if chart.ambient else ()
    base = tuple(dict.fromkeys((*chart.region, *chart.fibre)))
    pieces = [base]
    nonzero = [g for g in base if g != 0]
    if nonzero:
        shortest = min(nonzero, key=lambda g: len(sparse_terms(variables, g)))
        _, factors = rational_factor_rows(variables, shortest)
        # A cover of its zero set over the algebraic closure. A rational
        # irreducible factor may represent several geometric branches.
        pieces = [(*base, _sparse_expression(variables, row)) for row, _, _ in factors]

    parts = [(label, ideal) for label, _, _, ideal in chart.components]
    api = chart_expressions(variables)
    one = api.S.One
    gradients = {g: tuple(frame_derivative(variables, g, {v: one}) for v in variables)
                 for g in {*ambient, *(g for _, ideal in parts for g in ideal)}}
    for piece in pieces:
        for size in range(1, len(parts) + 1):
            for subset in combinations(parts, size):
                boundary = [g for label, ideal in subset if not _is_strict(label) for g in ideal]
                strict = [g for label, ideal in subset if _is_strict(label) for g in ideal]
                basis = groebner((*piece, *ambient, *boundary), variables)
                if tuple(basis.exprs) == (1,):
                    continue
                locus = (*basis.exprs, *(basis.reduce(g)[1] for g in strict))
                if not _present(variables, locus, chart.removed):
                    continue
                rank = len(variables) - dimension + size
                jacobian = [tuple(basis.reduce(d)[1] for d in gradients[g])
                            for g in (*ambient, *boundary, *strict)]
                minors = _minors(variables, jacobian, rank, basis)
                if minors != (one,) and _present(variables, (*locus, *minors), chart.removed):
                    raise GiveUp("The total transform fails normal crossings on the owned fibre.")


def _power_support(variables, polynomial):
    _, factors = rational_factor_rows(variables, polynomial)
    if len(factors) != 1 or factors[0][1] < 2:
        raise GiveUp("The tangent section is not a power of one reduced hypersurface.")
    return _sparse_expression(variables, factors[0][0])


def _intersection(chart, first, second):
    parts = {label: ideal for label, _, _, ideal in chart.components}
    if first in parts and second in parts:
        centre = (*parts[first], *parts[second])
        if not empty_locus(chart.variables, (*chart.ambient, *centre)):
            return centre
    return None


class RepeatedTangent:
    """A finite trial before the general resolution procedure."""
    name, stages, retains = "repeated tangent", 6, False
    # These centres are prescribed by the geometry. The other stages test
    # for SNC and compute rank-defect ideals. In particular stage six always
    # checks all charts after the last blowup.
    def needs_failures(self):
        return len(self.exceptional) == 3

    def final_check(self):
        return len(self.exceptional) == 5

    @property
    def keep_complement(self):
        # Off the nodal centre the preceding chart was already smooth.
        # Its old equations are cheaper there; later charts need only own
        # their new exceptional divisor.
        return len(self.exceptional) >= 3

    def __init__(self):
        self.exceptional = []

    def created(self, label, detail):
        self.exceptional.append(label)

    def choose(self, current, failures, dimension, deadline):
        stage = len(self.exceptional)
        if stage == 0:
            chart, = current
            if (chart.ambient or chart.fibre != chart.variables or len(chart.components) != 1
                    or chart.components[0][0] != "strict" or len(chart.components[0][3]) != 1):
                raise GiveUp("The repeated-tangent rule starts with a hypersurface at the origin.")
            terms = sparse_terms(chart.variables, chart.components[0][3][0])
            order = min(sum(m) for m, _ in terms)
            tangent = _polynomial({m: c for m, c in terms if sum(m) == order}, chart.variables)
            _power_support(chart.variables, tangent)
            chosen = [chart.variables]
        elif stage == 1:
            chosen = []
            for chart in current:
                parts = {label: ideal for label, _, _, ideal in chart.components}
                if "strict" not in parts:
                    chosen.append(None)
                    continue
                radial, = parts[self.exceptional[0]]
                strict, = parts["strict"]
                section = groebner((radial,), chart.variables).reduce(strict)[1]
                if not section.free_symbols:
                    if section == 0:
                        raise GiveUp("The tangent section contains the whole exceptional chart.")
                    chosen.append(None)
                    continue
                chosen.append((radial, _power_support(chart.variables, section)))
        elif stage == 2:
            chosen = [_intersection(chart, *self.exceptional) for chart in current]
        elif stage == 3:
            # Because the second exceptional divisor lies over the origin,
            # its rank-defect locus with the strict transform is closed
            # over the fibre. _failures gives the same scheme on overlaps.
            pair = {"strict", self.exceptional[1]}
            chosen = [next((ideal_basis(chart.variables, ideal) for labels, ideal in over.items()
                            if set(labels) == pair), None)
                      for chart, (over, _) in zip(current, failures)]
        elif stage == 4:
            chosen = [_intersection(chart, self.exceptional[1], self.exceptional[3]) for chart in current]
        else:
            raise GiveUp("The repeated-tangent blowups have not yet resolved this germ.")
        return chosen, {"rule": self.name}
