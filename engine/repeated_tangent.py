"""Smooth centres for a repeated projective tangent hypersurface.

After blowing up the origin, the reduced intersection of the strict
transform and the exceptional divisor is the projectivized tangent cone.
When its support is smooth and occurs with multiplicity, blow it up and
then blow up the intersection of the first two exceptional divisors.
The further centres, when needed, use the Jacobian rank-defect scheme of
the strict transform on the second exceptional divisor. All these schemes
are intrinsic to the indicated components, so their chart ideals glue.
"""
from .affine_kernel import _request, _ring, _sparse_expression, rational_factor_rows
from .blowup_ideal import empty_locus, ideal_basis
from .centres import GiveUp
from .groebner import groebner


def certify_fibre(chart, dimension):
    """Check Jacobian ranks on all owned strata with the native ideal kernel."""
    from .kernel import _lines
    from .local_blowups import _is_strict
    from .native_library import Unavailable, affine_request, ideal_request

    variables = chart.variables
    base = tuple(dict.fromkeys((*chart.region, *chart.fibre)))
    sizes = []
    for label, _, _, generators in chart.components:
        sizes.extend((int(_is_strict(label)), len(generators)))
    header = ["owned_crossings", len(variables), dimension, len(chart.ambient), len(chart.components),
              len(base), len(chart.removed), *sizes, *(len(ideal) for ideal in chart.removed)]
    equations = [*chart.ambient, *base, *(g for _, _, _, ideal in chart.components for g in ideal),
                 *(g for ideal in chart.removed for g in ideal)]
    try:
        result, = ideal_request(" ".join(map(str, header)), _lines(variables, equations))
    except Unavailable:
        # Packed degree and variable limits do not restrict the exhaustive
        # verifier. The sparse FLINT kernel uses arbitrary-precision exponents.
        ring = _ring(variables)

        def ideal(generators):
            return [str(len(generators)), *(ring.text(g) for g in generators)]

        request = ["wide_owned_crossings", ring.header, str(dimension),
                   *ideal(chart.ambient), *ideal(base), str(len(chart.components))]
        for _, _, _, generators in chart.components:
            request.extend(ideal(generators))
        request.append(str(len(chart.removed)))
        for generators in chart.removed:
            request.extend(ideal(generators))
        result, = affine_request(request)
    if result != "yes":
        raise GiveUp("The total transform fails normal crossings on the owned fibre.")


def _power_support(variables, polynomial):
    _, factors = rational_factor_rows(variables, polynomial)
    if len(factors) != 1 or factors[0][1] < 2:
        raise GiveUp("The tangent section is not a power of one reduced hypersurface.")
    return _sparse_expression(variables, factors[0][0])


def tangent_power_support(variables, polynomial):
    ring = _ring(variables)
    row, = _request(["lowest_form", ring.header, ring.text(polynomial)])
    return _power_support(variables, _sparse_expression(variables, row))


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
            tangent_power_support(chart.variables, chart.components[0][3][0])
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
