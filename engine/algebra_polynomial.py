"""Public handler for exact polynomial arithmetic over the rationals."""

from . import native_polynomial as polynomial
from .affine_kernel import _request, _ring
from .algebra_execution import cpu_stage
from .polynomial import format_poly, rational_text
from .polynomial_gcd import monic_gcd


def handle(operation, ring, operands):
    variables = tuple(map(polynomial.Symbol, ring["variables"]))
    if operation == "hensel-factors":
        from .gpu_polynomial import compute_row
        from .kernel_json import kernel_json
        from .algebra_execution import record_fallback
        from .native_library import current_replay_scope
        from .polynomial import sparse_row

        options = dict(width=len(variables), prime=operands["modulus"])
        left, right = operands["factors"]
        inputs = dict(polynomial=polynomial.row(variables, operands["polynomial"]),
                      left=polynomial.row(variables, left), right=polynomial.row(variables, right))
        scope = current_replay_scope()
        if scope is None or scope.compute != "gpu":
            with cpu_stage("hensel_correction_and_verification"):
                output = kernel_json("atlas_hensel_factors", "compute", **options, **inputs)
        else:
            with cpu_stage("hensel_prepare"):
                data = kernel_json("atlas_hensel_factors", "prepare", **options, **inputs)
            product = compute_row("multiply", 1, 1, [data["left"], data["right"]])
            with cpu_stage("hensel_correction_and_verification"):
                output = kernel_json("atlas_hensel_factors", "lift", **options, **data, product=product)
            if output["recovered"]:
                record_fallback("The Hensel verification rejected the initial polynomial product. The CPU recovered it.")
        return {"type": operation, "ring": ring, "modulus": output["modulus"],
                **{key: format_poly(sparse_row(output[key]), ring["variables"]) for key in ("left", "right")}}
    if operation == "polynomial-arithmetic":
        from .gpu_polynomial import compute_row

        field = ring["field"]
        names = ring["variables"]
        modulus = None
        if isinstance(field, dict):
            from .polynomial import parse_expression
            names = [*names, field["generator"]]
            variables = tuple(map(polynomial.Symbol, names))
            modulus = parse_expression(field["minimal_polynomial"], names, as_row=True)
        action = operands["action"]
        inputs = [operands["polynomial"]]
        if "other_polynomial" in operands:
            inputs.append(operands["other_polynomial"])
        inputs.extend(operands.get("images", ()))
        parameter = operands.get("exponent")
        if "variable" in operands:
            parameter = str(ring["variables"].index(operands["variable"]))
        row = compute_row(action, len(variables), len(variables),
                          [polynomial.row(variables, value) for value in inputs], parameter, modulus=modulus)
        result = polynomial.from_row(variables, row)
        return {"type": "polynomial", "ring": ring,
                "polynomial": format_poly(polynomial.terms(variables, result), names)}
    if operation in ("extended-gcd", "polynomial-division", "cancel-fraction"):
        from .gpu_factor_division import factor_division_hooks
        from .kernel_json import kernel_json
        from .polynomial import sparse_row
        with factor_division_hooks(), cpu_stage(operation):
            rows = kernel_json("atlas_polynomial_division", operation, width=len(variables), order=ring["order"],
                               left=polynomial.row(variables, operands["polynomial"]),
                               right=polynomial.row(variables, operands["other_polynomial"]))
        return {"type": operation, "ring": ring,
                **{key: format_poly(sparse_row(value), ring["variables"]) for key, value in rows.items()}}
    if operation == "absolute-factor":
        from .polynomial_factorization import absolute_factor_record
        return absolute_factor_record(ring, variables, operands["polynomial"])
    if operation in ("factor", "squarefree-factor"):
        from .polynomial_factorization import factor_polynomial

        unit, factors = factor_polynomial(variables, operands["polynomial"], order=ring["order"],
                                         squarefree=operation=="squarefree-factor")
        kind = "squarefree-factorization" if operation=="squarefree-factor" else "factorization"
        return {"type": kind, "ring": ring, "unit": rational_text(unit),
                "factors": [{"polynomial": format_poly(polynomial.terms(variables, factor), ring["variables"]),
                             "multiplicity": multiplicity} for factor, multiplicity in factors]}
    if operation == "resultant":
        name, = operands["variables"]
        index = ring["variables"].index(name)
        source = _ring(variables)
        with cpu_stage("polynomial_resultant", algorithm="flint", variable=name):
            row, = _request(["resultant", source.header, str(index), source.text(operands["polynomial"]),
                             source.text(operands["other_polynomial"])])
            result = polynomial.from_row(variables, row)
        ring = {**ring, "variables": [v for v in ring["variables"] if v != name]}
        variables = variables[:index] + variables[index + 1:]
    elif operation == "gcd":
        result = monic_gcd(variables, operands["polynomial"], operands["other_polynomial"],
                           order=ring["order"])
    else:
        raise ValueError(f"Unknown polynomial operation {operation!r}.")
    return {"type": "polynomial", "ring": ring,
            "polynomial": format_poly(polynomial.terms(variables, result), ring["variables"])}
