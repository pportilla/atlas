"""Order-two monodromy at nearby points with a nondegenerate transverse quadratic form."""
from .monodromy import _request
from .point_geometry import encode_terms
from .polynomial import LimitError, parse
from .rational_point import coordinate_fraction


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
    witness = _request("morse", terms=encode_terms(terms), width=len(variables))
    if witness is None:
        return None
    axes = witness["axes"]
    parameters = tuple(v for i, v in enumerate(variables) if i not in axes)
    return {"model": "sum of squares", "coordinates": tuple(variables[i] for i in axes),
            "parameters": parameters, "parameter_values": (1,) * len(parameters),
            "transverse_hessian_determinant": coordinate_fraction(witness["determinant"]),
            "cyclotomic_exponent": -1}
