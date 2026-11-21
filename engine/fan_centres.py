"""Verify the base images of exceptional rays against the reduced source divisor."""
from functools import lru_cache

from .native_library import affine_request
from .polynomial import format_poly, integer_text, sparse_row


@lru_cache(maxsize=128)
def _reduced_equation(names, equation):
    row, = affine_request(["squarefree_part", " ".join(names), equation])
    return format_poly(sparse_row(row), names)


@lru_cache(maxsize=128)
def _in_frame(names, support, matrix, marked, modulus):
    ring = (*names, *marked)
    forms = ["+".join(f"({c})*{v}" for c, v in zip(row, names)) for row in matrix]
    row, = affine_request(["pullbacks", " ".join(ring), " ".join(names), *forms, modulus, support])
    return ring, format_poly(sparse_row(row), ring)


def singular_ray_images(equation, names, rays, linear=None, field=None):
    """Return the ray image ideals in fan coordinates when each lies in the singular locus."""
    names = tuple(names)
    targets = tuple(dict.fromkeys(tuple(j for j, value in enumerate(ray) if value) for ray in rays))
    if not targets:
        return []
    support = _reduced_equation(names, equation)
    ring = names
    # When every target is the origin its coordinates agree in both frames.
    if linear is not None and any(len(axes) < len(names) for axes in targets):
        matrix = tuple(tuple(integer_text(c) if type(c) is int else c for c in row) for row in linear)
        identity = tuple(tuple("1" if i == j else "0" for j in range(len(names))) for i in range(len(names)))
        if matrix != identity:
            marked = (field["generator"],) if field else ()
            ring, support = _in_frame(names, support, matrix, marked, field["minimal_polynomial"] if field else "0")
    # Modulo the marked minimal polynomial, the coefficients are in its
    # power basis. The native test measures only the specified geometric axes.
    found = affine_request(["singular_subspaces", " ".join(ring), support,
                            *(" ".join(map(str, axes)) for axes in targets)])
    if len(found) != len(targets):
        return None
    return [[names[i] for i in axes] for axes in targets]
