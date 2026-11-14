"""Exact compactly supported Euler characteristic of a complex affine constructible set.

Project out one coordinate. On strata of the base, polynomial Euclidean
division in that coordinate gives the greatest common divisor of the fiber
equations and its number of distinct roots. Integrating these fiber counts
gives the Euler characteristic from those of the lower-dimensional strata.
"""
import ctypes

from . import native_polynomial as np
from .chart_expressions import chart_expressions
from .polynomial import integer_from_decimal
from .reference import sp
from .reference import QQ


def _reference(variables, equations, excluded, domain):
    from tests.constructible_euler_oracle import _reference as comparison
    return comparison(variables, equations, excluded, domain)


def euler_characteristic(variables, equations, excluded=(), *, domain=QQ):
    """Compute χ_c(V(equations) minus the union of V(I) for I in excluded).

    Each excluded ideal is an iterable of generators. Coefficients are rational
    or in the supplied algebraic number field. The result concerns complex
    geometric points and ignores nilpotents.
    """
    from . import kernel
    from .groebner import CHECK, _MODE
    variables, equations = tuple(variables), tuple(equations)
    excluded = tuple(tuple(ideal) for ideal in excluded)
    def reference():
        return _reference(variables, equations, excluded, domain)

    if _MODE == "sympy":
        return reference()
    coords, closed, removed, degree = variables, equations, excluded, 1
    algebraic = getattr(domain, "is_AlgebraicField", False)
    if algebraic:
        from .candidate_field import NativeField
        from .field_polynomial import lift, modulus
        native_field = isinstance(domain, NativeField)
        theta = (np if native_field else sp).Dummy("coefficient")
        def lifted(ideal):
            return tuple(lift(f, variables, domain, theta) for f in ideal)
        coords = (*variables, theta)
        closed = (*lifted(equations), modulus(domain, theta))
        removed = tuple(lifted(z) for z in excluded)
        degree = domain.degree if native_field else domain.mod.degree()
    if domain != QQ and not algebraic:
        if _MODE == "native" or np.native_ring(variables):
            raise ValueError("Expected rational or marked number-field coefficients.")
        return reference()
    # Free complex coordinates have Euler characteristic one. Keep a free
    # coordinate for constant problems because the native ring has arity >= 1.
    support = set().union(*(getattr(f, "free_symbols", ()) for ideal in (closed, *removed) for f in ideal))
    coords = tuple(v for v in coords if v in support) or (chart_expressions(variables).Dummy("free"),)
    from .native_library import _kernel
    library = _kernel()
    if hasattr(library, "atlas_euler"):
        from .affine_kernel import _ring
        ring = _ring(coords)
        sizes = " ".join(map(str, (len(closed), *map(len, removed))))
        payload = ("\n".join([ring.header, sizes, *(ring.text(f) for ideal in (closed, *removed) for f in ideal)])
                   + "\n").encode()
        operation = library.atlas_euler
        operation.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        operation.restype = ctypes.c_void_p
        result = operation(payload, len(payload))
        if not result:
            raise ArithmeticError(library.atlas_groebner_error().decode())
        try:
            total = integer_from_decimal(ctypes.string_at(result).decode("ascii"))
        finally:
            library.atlas_groebner_release(result)
    else:
        found = kernel.strata(coords, closed, ((),), removed)
        if found is None:
            if _MODE == "native" or np.native_ring(coords):
                raise ArithmeticError("Build the native Euler characteristic kernel.")
            return reference()
        total = found.get((0,), 0)
    # The finite etale coefficient scheme splits into deg(m) conjugate
    # fibres. Each fibre projection has the same degree/gcd stratification
    # under a field embedding, hence the same compactly supported chi.
    from .chart_kernel import InconsistentSheets, strata_totals
    try:
        count = total if degree == 1 else strata_totals([(None, degree, total)]).get(None, 0)
    except InconsistentSheets as exc:
        raise ArithmeticError("Inconsistent Euler characteristic on conjugate sheets.") from exc
    if CHECK and count != reference():
        raise AssertionError("The native Euler characteristic differs from the fibre integral.")
    return count
