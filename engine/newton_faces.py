"""Certify Newton nondegeneracy of one face in the torus over Qbar.

The caller enumerates faces. A face is nondegenerate iff its polynomial and
all logarithmic derivatives have no common torus zero. Projection onto pivot
coordinates of the difference lattice preserves that condition: the
coordinate subtorus maps onto the character torus of that lattice. In
characteristic zero this is a smooth surjective map, even if the projected
lattice has finite index. Multiplication by a Laurent monomial is a unit.
"""
from fractions import Fraction
from math import gcd, lcm
from pathlib import Path
import subprocess

from .fan_geometry import lattice_pivots as _pivot_axes
from .binomial_kernel import _newton, edge_parameters, projected_support, _integer
from .native_library import CHECK

_KERNEL = Path(__file__).resolve().parents[1] / "native" / "build" / "face_gb"


def _edge(terms, support, axis):
    # Primitive in the ambient character lattice. For every face-normal ray
    # the difference is orthogonal to that ray, hence is also primitive in
    # the torus orbit's character lattice.
    begin = support[0]
    length, direction, powers = edge_parameters(tuple(support), axis)
    polynomial = {k:terms[e] for e,k in zip(support, powers)}
    return {
        "start": list(begin), "direction": list(direction), "degree": length,
        "root_polynomial": {str(k):str(c) for k,c in sorted(polynomial.items())},
    }, {(k,):v for k,v in polynomial.items()}


def _primitive_reference(coefficients):
    denominator = 1
    for c in coefficients:
        denominator = lcm(denominator,c.denominator)
        if denominator.bit_length() > 8192:
            return "denominator"
    integers = [c.numerator * (denominator//c.denominator) for c in coefficients]
    content = gcd(*integers)
    primitive = [v//content for v in integers]
    return "coefficient" if any(abs(c).bit_length()>8192 for c in primitive) else primitive


def _kernel(terms, dimension, timeout):
    # Clearing rational denominators changes the generator by a torus unit.
    coefficients = list(terms.values())
    rows = _newton("coefficients", dimension, (len(coefficients),),
                   *((c.numerator,c.denominator) for c in coefficients))
    if rows is None:
        primitive = _primitive_reference(coefficients)
    else:
        primitive = rows[0] if rows[0] in ("coefficient", "denominator") else list(map(_integer, rows))
        if CHECK and primitive != _primitive_reference(coefficients):
            raise AssertionError("Primitive Newton coefficients differ from their reference.")
    if primitive == "denominator":
        return "undetermined", {"reason":"Clearing the face denominators exceeds the coefficient budget."}
    if primitive == "coefficient" or any(max(e)>4096 for e in terms):
        return "undetermined", {"reason":"A face coefficient or exponent exceeds the Gröbner input budget."}
    rows = [f"{dimension} {len(terms)}"]
    rows.extend(f"{c} " + " ".join(map(str,e)) for e,c in zip(terms,primitive))
    if not _KERNEL.is_file():
        raise RuntimeError("Native Newton-face checker is not built. Run make.")
    try:
        proc = subprocess.run([str(_KERNEL)], input="\n".join(rows)+"\n", capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "undetermined", {"reason":"The exact Newton-face calculation exceeded its time budget."}
    data = proc.stdout.split()
    if proc.returncode or len(data)!=2 or data[0] not in {"nondegenerate","degenerate","undetermined"}:
        return "undetermined", {"reason":"The native Newton-face calculation did not finish."}
    status, value = data
    key = "gcd_degree" if dimension==1 else "basis_size"
    info = {key:int(value)}
    if dimension>1:
        info["unit_ideal"] = True if status=="nondegenerate" else False if status=="degenerate" else None
    if status=="undetermined":
        info["reason"] = "The Gröbner basis exceeded its polynomial, term, or coefficient-size budget."
    return status, info


def certify_face(face_terms, timeout=5.0):
    """Check the supplied nonempty face-term dictionary over Qbar.

    Return JSON-safe status, method and dimensions. For an edge, edge.direction
    is primitive and edge.root_polynomial maps powers to rational coefficients
    in its character. Its simple roots label the geometric components.
    """
    terms = {tuple(e):Fraction(c) for e,c in face_terms.items() if c}
    if not terms:
        raise ValueError("A Newton face must contain a nonzero monomial.")
    support = sorted(terms)
    n = len(support[0])
    if not 1<=n<=8 or any(len(e)!=n or any(type(x)!=int for x in e) for e in support):
        raise ValueError("Face exponents must be integer vectors of a common dimension from 1 to 8.")
    axes = _pivot_axes(support)
    dimension = len(axes)
    result = {"face_dimension": dimension, "term_count": len(terms), "projection_axes": axes}
    if dimension==1:
        result["edge"], reduced = _edge(terms,support,axes[0])
    if len(terms)==dimension+1:
        return dict(result, status="nondegenerate", method="affinely independent support")
    # A rational linear combination of f and one logarithmic partial can
    # isolate a single monomial, which is invertible on the torus.
    for axis in axes:
        counts = {}
        for e in support:
            counts[e[axis]] = counts.get(e[axis],0)+1
        if len(counts)==2 and 1 in counts.values():
            return dict(result, status="nondegenerate", method="monomial in logarithmic Jacobian", axis=axis)
    if dimension>1:
        reduced = dict(zip(projected_support(support, axes), (terms[e] for e in support)))
    method = "univariate square-free gcd" if dimension==1 else "torus Jacobian Gröbner basis over Q"
    status, details = _kernel(reduced,dimension,timeout)
    return dict(result, status=status, method=method, **details)
