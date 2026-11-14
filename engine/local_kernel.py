"""Transport sparse QQ rows to the point-local C++ kernel."""

from . import native_polynomial as np
from .native_library import affine_request
from .polynomial import rational_text


def local_request(operation, variables, order, generators, *, point=None, known_basis=None):
    header = [operation, " ".join(v.name for v in variables), order]
    if point is not None:
        header.append(" ".join(map(rational_text, point)))
    if known_basis is not None:
        header.append("1" if known_basis else "0")
    return affine_request([*header, *(np.row(variables, f) for f in generators)])


def local_polynomials(operation, variables, order, generators, *, point=None):
    rows = local_request(operation, variables, order, generators, point=point)
    return tuple(np.from_row(variables, row) for row in rows)
