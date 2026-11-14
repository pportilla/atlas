"""Monic polynomial gcd over QQ in the declared coordinate and monomial order."""

from . import native_polynomial as np
from .affine_kernel import _request, _ring
from .algebra_execution import cpu_stage, current_execution, record_fallback, stage_label
from .algebra_groebner import reduced_basis
from .groebner import _MAX_DEGREE, _MAX_VARIABLES
from .ideal_operations import intersection
from .native_library import algebra_device_unavailable


def _device_basis(label, variables, generators, order):
    with stage_label(f"gcd_{label}"):
        return reduced_basis(generators, variables, order=order)


def _packed_pair(variables, left, right):
    # The intersection adds a selector, and the modular route homogenizes.
    # Count the selector's degree as well when testing the packed input limit.
    from .quotient_monomials import degree_bounds
    if len(variables) + 2 > _MAX_VARIABLES:
        return False
    support = (powers for f in (left, right) for powers in np.terms(variables, f))
    return not (left or right) or degree_bounds(len(variables), support)[1] < _MAX_DEGREE


def monic_gcd(variables, left, right, *, order="grevlex"):
    """Return the monic gcd, with gcd(0, 0) = 0, using the enclosing execution policy."""
    variables = tuple(variables)
    left, right = np.coerce(left), np.coerce(right)
    if order not in ("lex", "grevlex"):
        raise ValueError("The gcd requires lex or grevlex order.")
    if (len(set(variables)) != len(variables) or
            any(not isinstance(v, np.Expr) or not v.is_Symbol for v in variables) or
            not (left.free_symbols | right.free_symbols).issubset(variables)):
        raise ValueError("Supply polynomials in the same declared rational coordinates.")
    if not variables:
        with cpu_stage("polynomial_gcd", algorithm="rational_constants"):
            return np.S.One if left or right else np.S.Zero

    run = current_execution()
    ring = _ring(variables)
    if (run is not None and run.replay.compute == "gpu" and
            len(left._terms) > 1 and len(right._terms) > 1 and
            not left.free_symbols.isdisjoint(right.free_symbols)):
        if not _packed_pair(variables, left, right):
            record_fallback("The principal intersection exceeds the packed GPU range. FLINT computed the gcd.")
        elif algebra_device_unavailable(run.replay.backend, run.replay.index):
            record_fallback(f"No algebra GPU available for {run.replay.backend} at index {run.replay.index}. "
                            "FLINT computed the gcd.")
        else:
            # In this UFD (a) intersect (b) = (lcm(a,b)). A certified basis for
            # the contraction has precisely one generator; b / (lcm / a) is
            # the gcd up to a rational unit, by prime factor multiplicities.
            with stage_label("gcd_lcm"):
                common = intersection(variables, (left,), (right,), order="grevlex",
                                      basis=_device_basis, stage=cpu_stage)
            if len(common.generators) != 1:
                raise ArithmeticError("Expected one generator of the nonzero principal intersection.")
            lcm, = common.generators
            with cpu_stage("gcd_from_lcm", divisions=2, order=order):
                row, = _request(["gcd_from_lcm", ring.header, order, ring.text(left), ring.text(right),
                                 ring.text(lcm)])
            return np.from_row(variables, row)

    with cpu_stage("polynomial_gcd", algorithm="flint", order=order):
        row, = _request(["polynomial_gcd", ring.header, order, ring.text(left), ring.text(right)])
    return np.from_row(variables, row)
