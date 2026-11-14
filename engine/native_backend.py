"""Native one-blowup resolution for germs with smooth projectivized tangent cone."""

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import re
import sys

_LIBRARY = Path(__file__).resolve().parents[1] / "native" / "build" / (
    "libresolution.dylib" if sys.platform == "darwin" else "libresolution.so"
)
_GPU_LIBRARY = _LIBRARY.with_name("libresolution_opencl" + _LIBRARY.suffix)

_AMBIENT = "Reduced exceptional divisor of the ambient modification over the origin, over the algebraic closure of Q."
_INTRINSIC = "Reduced exceptional divisor on the resolved hypersurface over the origin, over the algebraic closure of Q."


@lru_cache(maxsize=1)
def _load(path):
    library = ctypes.CDLL(str(path))
    library.resolution_ordinary.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.resolution_ordinary.restype = ctypes.c_void_p
    library.resolution_free.argtypes = [ctypes.c_void_p]
    library.resolution_free.restype = None
    return library


@lru_cache(maxsize=1)
def _gpu_library():
    try:
        library = ctypes.CDLL(str(_GPU_LIBRARY))
    except OSError:
        return None
    library.resolution_gpu_count.argtypes = []
    library.resolution_gpu_count.restype = ctypes.c_int
    library.resolution_gpu_info.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
    library.resolution_gpu_info.restype = ctypes.c_int
    library.resolution_gpu_rank.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_int,
                                            ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    library.resolution_gpu_rank.restype = ctypes.c_int
    return library


def gpu_devices():
    """List OpenCL GPU devices visible to the installed plugin."""
    plugin = _gpu_library()
    devices = []
    if plugin:
        for index in range(plugin.resolution_gpu_count()):
            buf = ctypes.create_string_buffer(4096)
            if plugin.resolution_gpu_info(index, buf, len(buf)) == 0:
                name, vendor, version, driver = buf.value.decode(errors="replace").splitlines()
                devices.append(dict(index=index, name=name, vendor=vendor, version=version, driver=driver))
    return devices


def _gpu_config(compute=None):
    if compute not in {None, "cpu", "auto", "gpu"}:
        raise ValueError("compute must be cpu, auto, or gpu.")
    if compute is None:
        active = os.getenv("RESOLUTION_GPU", "off")
        if active not in {"auto", "off"}:
            raise ValueError("RESOLUTION_GPU must be auto or off.")
        compute = "auto" if active == "auto" else "cpu"
    if compute == "cpu":
        return None, 0, 1
    try:
        index = int(os.getenv("RESOLUTION_GPU_INDEX", 0))
        minimum_columns = 1 if compute == "gpu" else int(os.getenv("RESOLUTION_GPU_MIN_COLS", 256))
    except ValueError as exc:
        raise ValueError("GPU index and minimum column count must be integers.") from exc
    if not 0 <= index < 2**31 or not 1 <= minimum_columns < 2**31:
        raise ValueError("GPU index must be nonnegative and minimum column count must be positive.")
    plugin = _gpu_library()
    kernel = ctypes.cast(plugin.resolution_gpu_rank, ctypes.c_void_p) if plugin else None
    return kernel, index, minimum_columns


def _certificate(line, index):
    if line == "diagonal":
        return {
            "method": "diagonal gradient",
            "description": "Every variable has a nonzero pure power. The partial derivatives of the tangent form have no common projective zero.",
        }
    if line.startswith("binary "):
        _, prime, degree = line.split()
        return {
            "method": "square-free binary tangent cone",
            "prime": int(prime), "affine_degree": int(degree),
            "description": "The dehomogenized tangent form retains its degree and has unit gcd with its derivative modulo this prime. Its root at infinity, if present, is simple.",
        }
    _, prime, degree, rank, backend, micros, rows, minor = line.split()
    proof = {
        "method": "projective Jacobian rank",
        "prime": int(prime),
        "degree": int(degree),
        "rank": int(rank),
        "rows": int(rows),
        "rank_backend": backend,
        "rank_elapsed_ms": int(micros) / 1000,
        "minor_rows": [int(i) for i in minor.split(",")],
        "row_order": "Zero-based rows, grouped by partial derivative in input variable order, then monomial multiplier in ascending lexicographic exponent order.",
        "description": "Monomial multiples of the gradient span the entire recorded degree modulo this prime. A nonzero maximal minor lifts to Q.",
    }
    if backend == "opencl":
        proof["device"] = next((d for d in gpu_devices() if d["index"] == index), {"index": index})
        proof["verification"] = "CPU independently verified the reported full-rank minor modulo the recorded prime."
    return proof


def _plane_points(terms, order, tangent_polynomial):
    """Label each distinct geometric direction, including the point at infinity.

    Input binary form is certified square-free; a zero or infinity direction
    therefore contributes exactly one point. The other roots are specified
    exactly by a polynomial and index. This avoids numeric root isolation.
    """
    exponents = {tuple(powers) for powers, _ in terms if sum(powers) == order}
    zero = (order, 0) not in exponents
    infinity = (0, order) not in exponents
    directions = []
    if zero:
        directions.append(("0", None))
    for idx in range(1, order - zero - infinity + 1):
        root = f"alpha_{idx}"
        directions.append((root, f"1/{root}"))
    if infinity:
        directions.append((None, "0"))

    vertices = []
    in_chart = [[], []]
    for idx, (slope, reciprocal) in enumerate(directions, 1):
        vertex_id = f"P{idx}"
        direction = ["1", slope] if slope is not None else ["0", "1"]
        vertices.append({"id": vertex_id, "label": f"[{direction[0]}:{direction[1]}]", "direction": direction})
        for chart, value in enumerate((slope, reciprocal)):
            if value is None:
                continue
            radial = f"u{chart+1}"
            angular = f"u{2-chart}"
            ideal = [radial, angular if value == "0" else f"{angular} - {value}"]
            in_chart[chart].append({"id": vertex_id, "equation": ", ".join(ideal), "ideal": ideal})
    definition = {
        "polynomial": tangent_polynomial.replace("u2", "T"),
        "description": "alpha_j denotes the j-th distinct nonzero root in C, ordered by (real part, imaginary part), starting at 1. Directions are in the input variable order.",
    }
    return vertices, in_chart, definition


def _explain_defer(fields, variables):
    order, reason, tangent = fields[1:]
    kind, *data = reason.split()
    details = {
        "multiplicity": int(order),
        "tangent_cone": re.sub(r"\bu(\d+)\b", lambda m: variables[int(m[1])-1], tangent),
    }
    if kind == "coordinate":
        j = int(data[0])
        point = [int(i == j) for i in range(len(variables))]
        coords = ":".join(map(str, point))
        details.update(code="singular_tangent_direction", coordinate_index=j, point=point,
                       description=f"The tangent cone of degree {order} is singular at [{coords}] "
                                   f"in coordinates [{':'.join(variables)}]. All its first partial derivatives "
                                   "vanish there, so the ordinary-point resolution criterion fails.")
    elif kind == "budget":
        degree, columns, rows, limit = map(int, data)
        details.update(code="jacobian_matrix_budget", degree=degree, columns=columns, rows=rows,
                       column_limit=limit,
                       description=f"The projective Jacobian test needs a {rows} by {columns} matrix "
                                   f"in degree {degree}. Its configured limit is {limit} columns. "
                                   "Tangent-cone smoothness is undecided.")
    elif kind == "modular":
        tested = list(map(int, data))
        explanation = (f"The projective Jacobian matrix lacked full column rank modulo "
                       f"{' and '.join(map(str,tested))}." if tested else
                       "Coefficient denominators prevent reduction at both test primes.")
        details.update(code="inconclusive_modular_jacobian", tested_primes=tested,
                       description=explanation + " This leaves tangent-cone smoothness over Q undecided.")
    else:
        details.update(code="binary_tangent_uncertified",
                       description="The square-freeness test did not certify distinct projective tangent directions.")
    return details


def resolve_native(terms: list[tuple[tuple[int, ...], str]], variables: list[str], mode: str,
                   compute: str | None = None, diagnostics: dict | None = None) -> dict | None:
    """Return a proven local resolution, or None when the classification defers.

    The input must be distinct, nonzero sparse monomials over Q. Ordinary
    points use the standard n affine charts of the blowup. A smooth germ has
    a single identity chart and empty exceptional divisor. If the classifier
    defers, an optional diagnostics dictionary records the precise reason.
    """
    if diagnostics is not None:
        diagnostics.clear()
    if mode not in {"embedded", "intrinsic"}:
        raise ValueError("mode must be embedded or intrinsic")
    kernel, device_index, minimum_columns = _gpu_config(compute)
    if not _LIBRARY.is_file():
        return None
    n = len(variables)
    if any(len(exp) != n for exp, _ in terms):
        raise ValueError("monomial dimension differs from the number of variables")
    payload = [f"{n} {len(terms)}"]
    payload.extend(f"{coeff} {' '.join(map(str, powers))}" for powers, coeff in terms)
    library = _load(_LIBRARY)
    result = library.resolution_ordinary("\n".join(payload).encode("ascii"), kernel, device_index, minimum_columns)
    if result is None:
        raise RuntimeError("The native tangent-cone computation could not finish.")
    try:
        fields = ctypes.string_at(result).decode("ascii").splitlines()
    finally:
        library.resolution_free(result)
    if fields[0] == "defer":
        if diagnostics is not None and len(fields) == 4:
            diagnostics.update(_explain_defer(fields, variables))
        return None
    coordinates = [f"u{i+1}" for i in range(n)]
    dual = {"status": "complete", "convention": _AMBIENT if mode == "embedded" else _INTRINSIC,
            "vertices": [], "cells": []}
    output = {
        "engine": "native-ordinary", "status": "resolved", "charts": [], "steps": [],
        "dual_complex": dual,
    }
    if fields[0] == "smooth":
        output.update({
            "summary": "The hypersurface is smooth at the origin. The identity resolves the germ.",
            "certificate": {"method": "nonzero linear part"},
            "charts": [{"id": "U0", "coordinates": coordinates, "strict_transform": fields[1],
                        "map": dict(zip(variables, coordinates)), "exceptional": []}],
        })
        return output

    order, certificate, tangent = int(fields[1]), _certificate(fields[2], device_index), fields[3]
    output.update({
        "summary": f"One blowup at the origin resolves this ordinary point of multiplicity {order}.",
        "multiplicity": order, "tangent_cone": tangent, "tangent_coordinates": coordinates,
        "certificate": certificate,
        "steps": [{"id": "B1", "center": "origin", "center_ideal": list(variables),
                   "type": "point blowup", "exceptional": "E1", "multiplicity": order}],
    })
    vertices = [{"id": "E1", "label": "E1"}]
    plane_chart_components = None
    if mode == "intrinsic" and n == 2:
        vertices, plane_chart_components, definition = _plane_points(terms, order, fields[5])
        dual["root_definition"] = definition
    elif mode == "intrinsic":
        vertices[0].update({"projective_equation": tangent, "dimension": n-2})
    else:
        vertices[0].update({"dimension": n-1, "multiplicity": order})
    dual["vertices"] = vertices

    for k in range(n):
        radial = coordinates[k]
        strict, intersection = fields[4+2*k:6+2*k]
        blowdown = {
            name: radial if j == k else f"{radial}*{coordinates[j]}"
            for j, name in enumerate(variables)
        }
        exceptional = [{"id": "E1", "equation": radial}]
        if plane_chart_components is not None:
            exceptional = plane_chart_components[k]
        output["charts"].append({
            "id": f"U{k+1}", "coordinates": coordinates, "strict_transform": strict,
            "map": blowdown, "exceptional": exceptional,
            "pivot": variables[k], "ambient_exceptional": radial,
            "ambient_total_transform": f"{radial}^{order}*({strict})",
            "strict_transform_on_exceptional": intersection,
        })
    output["notes"] = [
        "Each U_i is the standard blowup chart for the i-th input coordinate. Smoothness and transversality hold along the entire fiber over the origin.",
    ]
    return output


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Discover OpenCL GPU devices available for exact modular rank.")
    parser.add_argument("--devices", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps({"plugin_built": _GPU_LIBRARY.is_file(), "devices": gpu_devices()}, indent=2))
