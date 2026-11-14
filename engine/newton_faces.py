"""Certify Newton nondegeneracy of one face in the torus over Qbar.

The caller enumerates faces. A face is nondegenerate iff its polynomial and
all logarithmic derivatives have no common torus zero. Projection onto pivot
coordinates of the difference lattice preserves that condition: the
coordinate subtorus maps onto the character torus of that lattice. In
characteristic zero this is a smooth surjective map, even if the projected
lattice has finite index. Multiplication by a Laurent monomial is a unit.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction
from math import gcd, lcm
from pathlib import Path
import subprocess

from .fan_geometry import lattice_pivots as _pivot_axes
from .binomial_kernel import _newton, edge_parameters, projected_support, _integer
from .native_library import CHECK, current_replay_scope
from .polynomial import integer_text, rational_text, sparse_row

_KERNEL = Path(__file__).resolve().parents[1] / "native" / "build" / "face_gb"
_EXHAUSTIVE = ContextVar("atlas_exhaustive_faces", default=False)


@contextmanager
def face_search(exhaustive):
    """Use the request's face-check policy during resolution and atlas verification."""
    token = _EXHAUSTIVE.set(exhaustive)
    try:
        yield
    finally:
        _EXHAUSTIVE.reset(token)


def exhaustive_faces():
    return _EXHAUSTIVE.get()


def _edge(terms, support, axis):
    # Primitive in the ambient character lattice. For every face-normal ray
    # the difference is orthogonal to that ray, hence is also primitive in
    # the torus orbit's character lattice.
    begin = support[0]
    length, direction, powers = edge_parameters(tuple(support), axis)
    polynomial = {k:terms[e] for e,k in zip(support, powers)}
    return {
        "start": list(begin), "direction": list(direction), "degree": length,
        "root_polynomial": {integer_text(k):rational_text(c) for k,c in sorted(polynomial.items())},
    }, {(k,):v for k,v in polynomial.items()}


def edge_polynomial(edge):
    """Decode the recorded rational polynomial in its primitive character."""
    return sparse_row(";".join(f"{coefficient} {power}"
                               for power, coefficient in edge["root_polynomial"].items()))


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
    rows.extend(integer_text(c) + " " + " ".join(map(str,e)) for e,c in zip(terms,primitive))
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


def _exact_kernel(terms, dimension):
    from . import native_polynomial as np
    from .algebra_groebner import reduced_basis

    # The coordinatewise power map is finite étale on the torus in
    # characteristic zero. In one variable it multiplies the gcd degree by
    # its degree, since the translated edge has a nonzero constant term.
    strides = tuple(gcd(*powers) for powers in zip(*terms))
    terms = {tuple(e[i] // strides[i] for i in range(dimension)): c for e,c in terms.items()}
    axes = tuple(np.Symbol(f"face{i}") for i in range(dimension))
    f = np.from_terms(axes, terms)
    derivatives = []
    for i in range(dimension):
        partial = {e[:i] + (e[i]-1,) + e[i+1:]: c*e[i] for e,c in terms.items() if e[i]}
        derivatives.append(np.from_terms(axes, partial))
    if dimension == 1:
        from .polynomial_gcd import monic_gcd
        common = monic_gcd(axes, f, derivatives[0])
        degree = max(e[0] for e in np.terms(axes, common)) * strides[0]
        return ("degenerate" if degree else "nondegenerate"), {"gcd_degree": degree}

    # On a torus each x_i is invertible, so ordinary partials cut out the
    # same ideal as logarithmic partials. Inverting the coordinate product
    # excludes every point on an axis, including in the proper-ideal case.
    ring = (*axes, np.Symbol("faceInverse"))
    inverse = np.from_terms(ring, {(1,)*len(ring): 1, (0,)*len(ring): -1})
    basis = reduced_basis((f, *derivatives, inverse), ring)
    unit = basis.exprs == (1,)
    return ("nondegenerate" if unit else "degenerate"), {"basis_size": len(basis.exprs), "unit_ideal": unit}


def _exact_face(terms, dimension):
    from .algebra_execution import algebra_execution, current_execution
    replay = current_replay_scope()
    if replay is None or replay.compute != "gpu" or current_execution() is not None:
        return _exact_kernel(terms, dimension)
    # Resolution tracks device replay across every stage. Give its requested
    # device to the shared gcd and basis operations, then merge their counts.
    with algebra_execution(**replay.settings()) as run:
        result = _exact_kernel(terms, dimension)
    replay.merge(run.replay.report())
    for reason in run.reasons:
        if reason not in replay.reasons:
            replay.reasons.append(reason)
    return result


def certify_face(face_terms, timeout=5.0, *, exhaustive=None):
    """Check the supplied nonempty face-term dictionary over Qbar.

    Return JSON-safe status, method and dimensions. For an edge, edge.direction
    is primitive and edge.root_polynomial maps powers to rational coefficients
    in its character. Its simple roots label the geometric components.
    An exhaustive search retries an inconclusive bounded test using exact
    rational algebra with arbitrary-precision coefficients and exponents.
    """
    exhaustive = exhaustive_faces() if exhaustive is None else exhaustive
    terms = {tuple(e):Fraction(c) for e,c in face_terms.items() if c}
    if not terms:
        raise ValueError("A Newton face must contain a nonzero monomial.")
    support = sorted(terms)
    n = len(support[0])
    if not n or any(len(e)!=n or any(type(x)!=int for x in e) for e in support):
        raise ValueError("Face exponents must be integer vectors of a common positive dimension.")
    if not exhaustive and n>8:
        raise ValueError("The bounded face checker accepts at most eight coordinates.")
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
    if dimension <= 8 and (not exhaustive or _KERNEL.is_file()):
        status, details = _kernel(reduced,dimension,timeout)
        if status != "undetermined" or not exhaustive:
            return dict(result, status=status, method=method, **details)
    status, details = _exact_face(reduced, dimension)
    return dict(result, status=status, method=method, **details)
