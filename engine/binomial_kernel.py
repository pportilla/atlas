"""Transport the exact fan trace and generic orbit incidences from the native kernel."""
import ctypes
from decimal import Decimal
from functools import lru_cache
from math import gcd

from .native_library import _MODE, _kernel


def _integer(text):
    # Decimal's exact integer conversion also accepts coefficients beyond
    # Python's per-process limit for decimal integer strings.
    return int(text) if len(text)<640 else int(Decimal(text))


def _encode(value):
    return str(value) if value.bit_length()<2000 else str(Decimal(value))


def run(operation, *rows):
    lib = _kernel()
    if _MODE == "sympy" or lib is None or not hasattr(lib, "atlas_binomial"):
        return None
    call = lib.atlas_binomial
    call.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    call.restype = ctypes.c_void_p
    payload = (operation + "\n" + "\n".join(" ".join(map(_encode,row)) for row in rows)).encode("ascii")
    ptr = call(payload, len(payload))
    if not ptr:
        lib.atlas_groebner_error.restype = ctypes.c_char_p
        raise ValueError(lib.atlas_groebner_error().decode("utf8"))
    try:
        return ctypes.string_at(ptr).decode("ascii").splitlines()
    finally:
        lib.atlas_groebner_release(ptr)


def fan(rows, n, names):
    if rows == ["none"]:
        return None
    lines = iter(rows)
    nr, nc, steps = map(int, next(lines).split())
    rays = [list(map(_integer, next(lines).split())) for _ in range(nr)]
    cones, certificates, trace = [], [], []
    for _ in range(nc):
        criterion, *values = map(_integer, next(lines).split())
        cone, orders = values[:n], values[n:]
        cones.append(cone)
        cert = {"criterion": "unit term"} if criterion < 0 else {
            "criterion": "linear nonexceptional coordinate", "coordinate_index": criterion, "linear_ray_index": cone[criterion]}
        certificates.append({**cert, "ray_indices": cone.copy(), "signed_orders": orders})
    for i in range(steps):
        k, *values = map(_integer, next(lines).split())
        face, count, remaining = values[:k], values[k], values[k+1:]
        trace.append({"type":"blowup", "id":f"blowup-{i+1}", "center_ray_indices":face, "new_ray_index":n+i,
                      "image_zero_coordinates":[names[j] for j in remaining[:count]], "image_monomial_orders":remaining[count:]})
    return {"rays":rays, "cones":cones, "terminal_certificates":certificates, "steps":trace}


def incidence(rows):
    signs = tuple(map(_integer, rows[0].split()))
    pieces, charts = [], []
    for row in rows[1:]:
        if row.startswith("c"):
            values = iter(map(int, row.split()[1:]))
            charts.append(dict(zip(values,values)))
        else:
            k, *values = map(int,row.split()[1:])
            subset, nz, rest = tuple(values[:k]), values[k], values[k+1:]
            pieces.append((subset, tuple(rest[:nz]), rest[nz:nz+k], rest[nz+k:]))
    return signs, pieces, charts


def pairings(rays, forms):
    """Integer order of each monomial or Laurent character on every ray."""
    from .native_library import CHECK
    rows = run(f"pairings {len(forms[0])} {len(rays)} {len(forms)}", *forms, *rays)
    if rows is not None:
        answer = tuple(tuple(map(_integer, row.split())) for row in rows)
        if not CHECK:
            return answer
    reference = tuple(tuple(sum(a*b for a,b in zip(ray,f)) for f in forms) for ray in rays)
    if rows is not None and answer != reference:
        raise AssertionError("The binomial monomial orders differ from their reference.")
    return reference


def _newton(operation, dimension, *rows):
    library = _kernel()
    if _MODE == "sympy" or not hasattr(library, "atlas_newton_support_abi"):
        return None
    return run(f"newton_{operation} {dimension}", *rows)


@lru_cache(maxsize=4096)
def edge_parameters(support, axis):
    """Length, primitive signed character and powers of collinear exponents."""
    n = len(support[0])
    rows = _newton("edge", n, (len(support), 1), (axis,), *support)
    if rows is not None:
        degree, *direction = map(_integer, rows[0].split())
        actual = degree, tuple(direction), tuple(map(_integer, rows[1:]))
        if _MODE != "check":
            return actual
    start, end = support[0], support[-1]
    difference = tuple(b-a for a, b in zip(start, end))
    degree = gcd(*difference)
    direction = tuple(v//degree for v in difference)
    reference = degree, direction, tuple((e[axis]-start[axis])//direction[axis] for e in support)
    if rows is not None and reference != actual:
        raise AssertionError("The Newton edge parameters differ from their reference.")
    return reference


def primitive_character(delta):
    vector = tuple(delta)
    axis = next((i for i, v in enumerate(vector) if v), None)
    if axis is None:
        return 0, vector
    degree, direction, _ = edge_parameters(((0,)*len(vector), vector), axis)
    return degree, direction


def projected_support(support, axes):
    """Project onto pivot coordinates and translate by their joint minimum."""
    rows = _newton("project", len(support[0]), (len(support),len(axes)), axes, *support)
    if rows is not None:
        actual = [tuple(map(_integer, row.split())) for row in rows]
        if _MODE != "check":
            return actual
    minimum = [min(e[i] for e in support) for i in axes]
    reference = [tuple(e[i]-v for i,v in zip(axes,minimum)) for e in support]
    if rows is not None and actual != reference:
        raise AssertionError("The projected Newton support differs from its reference.")
    return reference
