"""Rational quotient algebras globally and at rational affine points.

The local order takes the terms of lowest total degree first, breaking ties
by the requested order on the coordinates. A homogeneous lift realizes this
order with an ordinary global Gröbner computation (Lazard homogenization).
The standard monomials give homogeneous bases of the successive pieces of
the maximal-ideal filtration. In finite length these lift to a basis of the
local algebra. A surviving pure power in every degree implies infinite length.
"""

from . import native_polynomial as np
from .algebra_execution import cpu_stage, stage_label
from .algebra_groebner import reduced_basis
from .point_germs import centered_ideal, is_smooth_at_point, nonzero_terms_at_point
from .polynomial import InputError, format_poly, integer_text, rational_text
from .quotient_monomials import quotient_monomials
from .rational_point import coordinate_fraction, normalize_point


def _basis_heads(variables, generators, order, stage, *, local=False):
    from .local_kernel import local_request
    from .polynomial import sparse_row

    with cpu_stage(stage, generators=len(generators)):
        rows = local_request("local_heads" if local else "local_global_heads", variables, order, generators)
        return [next(iter(sparse_row(row))) for row in rows]


def _initial_monomials(variables, generators, order, local):
    """Ask the kernel for the initial ideal in the specified coordinates."""
    if local:
        from .local_standard import _standard_basis
        generators = _standard_basis(variables, generators, order)
    elif len(generators) > 1 and any(len(f._terms) > 1 for f in generators):
        with stage_label("quotient_basis"):
            generators = reduced_basis(generators, variables, order=order).exprs
    return _basis_heads(variables, generators, order, "quotient_initial_monomials", local=local)


def _jacobian(variables, terms):
    from .local_kernel import local_polynomials

    with cpu_stage("jacobian_generators", variables=len(variables), terms=len(terms)):
        return local_polynomials("local_partials", variables, "lex", (np.from_terms(variables, terms),))


def _basis_text(basis, names, shifts):
    if not any(shifts):
        return [format_poly({e: 1}, names) for e in basis]
    # Return the centered monomials in the original ring. Keeping products
    # factored also avoids a binomial expansion for every basis element.
    axes = [name if not a else name + ("-" if a > 0 else "+") + rational_text(abs(a))
            for name, a in zip(names, shifts)]
    result = []
    for exponents in basis:
        factors = []
        grouped = sum(exponents) > 1
        for axis, shift, power in zip(axes, shifts, exponents):
            if power:
                factor = f"({axis})" if shift and grouped else axis
                factors.append(factor if power == 1 else factor + "^" + integer_text(power))
        result.append("*".join(factors) or "1")
    return result


def handle(operation, ring, operands, *, known_basis=False):
    """Compute validated QQ quotients, local Jacobian algebras and their lengths."""
    if ring["field"] != "QQ" or ring["order"] not in ("lex", "grevlex"):
        raise InputError("Quotient algebras require QQ and lex or grevlex order.")
    names = ring["variables"]
    variables = tuple(map(np.Symbol, names))
    local = operation not in ("quotient-basis", "multiplicity")
    number = operation in ("multiplicity", "milnor-number", "tjurina-number")
    result = {"ring": ring, "scope": "origin" if local else "global"}
    shifts = (0,) * len(names)
    if "point" in operands:
        if not local:
            raise InputError("A global quotient calculation does not take a point.")
        point = normalize_point(operands["point"], len(names))
        shifts = tuple(map(coordinate_fraction, point))
        result.update(scope="point", point=point)
    shifted = any(shifts)
    if operation in ("multiplicity", "quotient-basis", "local-quotient-basis"):
        generators = tuple(map(np.coerce, operands["ideal"]))
        result["type"] = "integer" if number else "quotient-basis"
    elif operation in ("milnor-number", "tjurina-number", "local-jacobian-algebra"):
        f = np.coerce(operands["polynomial"])
        with cpu_stage("hypersurface_first_order" if number else
                       ("hypersurface_at_point" if shifted else "hypersurface_at_origin"),
                       variables=len(names)):
            if not variables:
                raise InputError("A hypersurface germ requires at least one coordinate.")
            terms = np.terms(variables, f)
            if shifted:
                if number and is_smooth_at_point(shifts, terms):
                    return {**result, "type": "integer", "finite": True, "value": 0}
                if not number and nonzero_terms_at_point(shifts, terms):
                    raise InputError("The hypersurface must pass through the supplied point.")
            elif (0,) * len(variables) in terms:
                raise InputError("The hypersurface must pass through the origin.")
            # A linear term makes its partial derivative a unit at the origin.
            # Both local quotients then have length zero.
            if number and not shifted and is_smooth_at_point(shifts, terms):
                return {**result, "type": "integer", "finite": True, "value": 0}
        generators = _jacobian(variables, terms)
        if operation == "tjurina-number":
            generators = (*generators, f)
        result["type"] = "integer" if number else operation
        if not number:
            result["generators"] = [format_poly(np.terms(variables, g), names) for g in generators]
    else:
        raise InputError(f"Unknown quotient algebra operation {operation!r}.")

    if shifted:
        with cpu_stage("center_at_point", variables=len(names), generators=len(generators)):
            generators = centered_ideal(variables, shifts, generators)
    if operation == "multiplicity" and known_basis:
        heads = _basis_heads(variables, generators, ring["order"], "restore_global_basis")
    else:
        heads = _initial_monomials(variables, generators, ring["order"], local)
    with cpu_stage("quotient_length" if number else "standard_monomials", generators=len(heads)):
        dimension, basis = quotient_monomials(len(variables), heads, include_basis=not number)
        result["finite"] = dimension is not None
        if number:
            result["value"] = dimension
        else:
            result["dimension"] = dimension
            result["basis"] = None if basis is None else _basis_text(basis, names, shifts)
    return result
