"""Order-two monodromy at nearby points with a nondegenerate transverse quadratic form."""
from fractions import Fraction
from itertools import combinations
from math import lcm, prod

from .fan_geometry import _determinant
from .polynomial import LimitError, parse


def _transverse_determinant(terms, normal):
    """Certify a critical coordinate stratum and specialize its Hessian at parameters = 1."""
    size = len(normal)
    hessian = [[0] * size for _ in normal]
    for powers, coefficient in terms.items():
        transverse_degree = sum(powers[i] for i in normal)
        if transverse_degree < 2:
            # The whole stratum must lie in the critical locus. Cancellation
            # only at the sampled parameter value would not establish that.
            return 0
        if transverse_degree == 2:
            active = [j for j, i in enumerate(normal) if powers[i]]
            if len(active) == 1:
                j, = active
                hessian[j][j] += 2 * coefficient
            else:
                j, k = active
                hessian[j][k] += coefficient
                hessian[k][j] += coefficient

    # The shared lattice determinant takes integer rows.
    scales = [lcm(*(entry.denominator for entry in row)) for row in hessian]
    integral = [[int(entry * scale) for entry in row] for row, scale in zip(hessian, scales)]
    return Fraction(_determinant(integral), prod(scales))


def nearby_morse_witness(polynomial, variables):
    """Try positive-dimensional coordinate strata of odd codimension for the eigenvalue -1.

    A nonzero specialization proves the transverse Hessian is nonsingular
    on a dense open subset of the stratum, hence at points arbitrarily close
    to 0. The holomorphic Morse lemma with parameters identifies the germ
    there with a sum of k squares independent of the parameters. Monodromy
    on its vanishing sphere S^(k-1) is antipodal, with degree (-1)^k.
    The search uses the bounded Atlas input range and a single specialization.
    """
    try:
        terms = parse(polynomial, list(variables), bounded=True)
    except LimitError:
        return None
    n = len(variables)
    for codimension in range(1, n, 2):
        for normal in combinations(range(n), codimension):
            determinant = _transverse_determinant(terms, normal)
            if determinant:
                parameters = tuple(v for i, v in enumerate(variables) if i not in normal)
                return {"model": "sum of squares", "coordinates": tuple(variables[i] for i in normal),
                        "parameters": parameters, "parameter_values": (1,) * len(parameters),
                        "transverse_hessian_determinant": determinant, "cyclotomic_exponent": -1}
    return None
