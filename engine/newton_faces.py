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
from pathlib import Path
import socket
import subprocess
from time import monotonic

from .fan_geometry import lattice_pivots as _pivot_axes
from .binomial_kernel import _newton, edge_parameters, projected_support, _integer
from .native_library import current_replay_scope
from .polynomial import integer_text, rational_text, sparse_row

_KERNEL = Path(__file__).resolve().parents[1] / "native" / "build" / "face_gb"
_WORKER = ContextVar("atlas_bounded_face_worker", default=None)
_EXHAUSTIVE = ContextVar("atlas_exhaustive_faces", default=False)


class _FaceWorker:
    """One killable native face checker shared by the trials of a resolution."""

    def __init__(self):
        self.proc = None
        self.sock = None

    def close(self):
        if self.proc is not None:
            self.sock.close()
            self.proc.kill()
            self.proc.wait()
            self.proc = self.sock = None

    def query(self, payload, timeout):
        end = None if timeout is None else monotonic() + timeout
        if self.proc is None:
            sock, child = socket.socketpair()
            try:
                with child:
                    self.proc = subprocess.Popen([str(_KERNEL), "--serve"], stdin=child,
                                                 stdout=child, stderr=subprocess.DEVNULL)
            except BaseException:
                sock.close()
                raise
            self.sock = sock
        def remaining():
            if end is None:
                return None
            seconds = end - monotonic()
            if seconds <= 0:
                raise TimeoutError
            return seconds

        try:
            self.sock.settimeout(remaining())
            self.sock.sendall(payload.encode("ascii") + b"\n")
            reply = bytearray()
            while b"\n" not in reply:
                self.sock.settimeout(remaining())
                chunk = self.sock.recv(128)
                if not chunk or len(reply) + len(chunk) > 128:
                    self.close()
                    return ""
                reply.extend(chunk)
            return reply.decode("ascii")
        except TimeoutError:
            self.close()
            raise
        except OSError:
            self.close()
            return ""


@contextmanager
def face_search(exhaustive):
    """Use the request's face-check policy during resolution and atlas verification."""
    token = _EXHAUSTIVE.set(exhaustive)
    worker = _FaceWorker()
    worker_token = _WORKER.set(worker)
    try:
        yield
    finally:
        _WORKER.reset(worker_token)
        _EXHAUSTIVE.reset(token)
        worker.close()


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


def _kernel(terms, dimension, timeout):
    # Clearing rational denominators changes the generator by a torus unit.
    coefficients = list(terms.values())
    rows = _newton("coefficients", dimension, (len(coefficients),),
                   *((c.numerator,c.denominator) for c in coefficients))
    primitive = rows[0] if rows[0] in ("coefficient", "denominator") else list(map(_integer, rows))
    if primitive == "denominator":
        return "undetermined", {"reason":"Clearing the face denominators exceeds the coefficient budget."}
    if primitive == "coefficient" or any(max(e)>4096 for e in terms):
        return "undetermined", {"reason":"A face coefficient or exponent exceeds the Gröbner input budget."}
    rows = [f"{dimension} {len(terms)}"]
    rows.extend(integer_text(c) + " " + " ".join(map(str,e)) for e,c in zip(terms,primitive))
    if not _KERNEL.is_file():
        raise RuntimeError("Native Newton-face checker is not built. Run make.")
    worker = _WORKER.get()
    try:
        if worker is None:
            proc = subprocess.run([str(_KERNEL)], input="\n".join(rows)+"\n", capture_output=True,
                                  text=True, timeout=timeout)
            reply = proc.stdout if proc.returncode == 0 else ""
        else:
            reply = worker.query(" ".join(rows), timeout)
    except (subprocess.TimeoutExpired, TimeoutError):
        return "undetermined", {"reason":"The exact Newton-face calculation exceeded its time budget."}
    data = reply.split()
    if len(data)!=2 or data[0] not in {"nondegenerate","degenerate","undetermined"}:
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
    from .affine_kernel import _frame_polys
    header, *rows = _newton("compress", dimension, (len(terms),), *terms)
    strides = tuple(map(_integer, header.split()))
    terms = {tuple(map(_integer, row.split())): c for row, c in zip(rows, terms.values(), strict=True)}
    axes = tuple(np.Symbol(f"face{i}") for i in range(dimension))
    f = np.from_terms(axes, terms)
    derivatives = [_frame_polys("derive", axes, (axis,), (f, np.S.One))[0] for axis in axes]
    if dimension == 1:
        from .polynomial_gcd import monic_gcd
        common = monic_gcd(axes, f, derivatives[0])
        powers = np.terms(axes, common)
        row, = _newton("scaled_degree", len(powers), (strides[0],), *powers)
        degree = _integer(row)
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
    isolated, = _newton("isolate", n, (len(support), len(axes)), axes, *support)
    if isolated != "none":
        return dict(result, status="nondegenerate", method="monomial in logarithmic Jacobian", axis=int(isolated))
    if dimension>1:
        reduced = dict(zip(projected_support(support, axes), (terms[e] for e in support)))
    method = "univariate square-free gcd" if dimension==1 else "torus Jacobian Gröbner basis over Q"
    if dimension <= 8 and (not exhaustive or _KERNEL.is_file()):
        status, details = _kernel(reduced,dimension,timeout)
        if status != "undetermined" or not exhaustive:
            return dict(result, status=status, method=method, **details)
    status, details = _exact_face(reduced, dimension)
    return dict(result, status=status, method=method, **details)
