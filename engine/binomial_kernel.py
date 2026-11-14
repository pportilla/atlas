"""Transport the exact fan trace and generic orbit incidences from the native kernel."""
import ctypes
from functools import lru_cache
from fractions import Fraction

from .native_library import Unavailable, _kernel
from .polynomial import integer_from_decimal as _integer, integer_text as _encode


def run(operation, *rows):
    lib = _kernel()
    if lib is None or not hasattr(lib, "atlas_binomial"):
        raise Unavailable("Rebuild the native binomial kernel with make.")
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
    rows = run(f"pairings {len(forms[0])} {len(rays)} {len(forms)}", *forms, *rays)
    return tuple(tuple(map(_integer, row.split())) for row in rows)


def _newton(operation, dimension, *rows):
    return run(f"newton_{operation} {dimension}", *rows)


@lru_cache(maxsize=4096)
def edge_parameters(support, axis):
    """Length, primitive signed character and powers of collinear exponents."""
    rows = _newton("edge", len(support[0]), (len(support), 1), (axis,), *support)
    degree, *direction = map(_integer, rows[0].split())
    return degree, tuple(direction), tuple(map(_integer, rows[1:]))


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
    return [tuple(map(_integer, row.split())) for row in rows]


def separating_forms(variables, points, *, first):
    """Integer linear forms covering the finite separating-parameter bound."""
    from .affine_kernel import _polynomial
    n = len(variables)
    bound, = _newton("separating_bound", n, (points,))
    units = tuple(tuple(int(i == j) for j in range(n)) for i in range(n))
    for a in range(first, first + _integer(bound)):
        weights, = _newton("separating_coefficients", n, (a,))
        yield _polynomial(dict(zip(units, map(_integer, weights.split()))), variables)


def exceptional_data(codimension, orders):
    result, = run(f"exceptional_data {codimension} {len(orders)}", *orders)
    return tuple(map(_integer, result.split()))


def character(left, right, a, b):
    """Return the signed and primitive characters, degree, and oriented root coefficient."""
    rows = run(f"character {len(left)}", left, right, (a.numerator, a.denominator, b.numerator, b.denominator))
    degree, numerator, denominator = map(_integer, rows[0].split())
    return (tuple(map(_integer, rows[1].split())), degree,
            tuple(map(_integer, rows[2].split())), Fraction(numerator, denominator))


def minimum_faces(values):
    rows = run(f"minima {len(values[0])} {len(values)}", *values)
    orders, faces = [], []
    for row in rows:
        minimum, *axes = map(_integer, row.split())
        orders.append(minimum)
        faces.append(tuple(axes))
    return orders, faces


@lru_cache(maxsize=8192)
def signed_powers(values):
    positive, negative = run(f"signed_split {len(values)}", values)
    return tuple(map(_integer, positive.split())), tuple(map(_integer, negative.split()))
