"""Short usage cards for the callables exposed in an Atlas session."""
from inspect import Parameter, getdoc, signature
from textwrap import indent

from .algebra_cli import OPERATION_DESCRIPTIONS


_COORDINATES = 'x, y = coordinates("x", "y")'
# Algebra examples share a setup so each can be copied into a fresh session.
_RING_EXAMPLES = {
    "resolve": "R = resolve(y^2 - x^3)",
    "dual_complex": "R = resolve(y^2 - x^3)\ndual_complex(R).vertices",
    "monodromy_check": "monodromy_check(y^2 - x^3).status",
    "polynomial": 'polynomial("y^2 - x^3", variables=(x, y))',
    "ideal": "ideal(x^2, y)",
    "groebner": "groebner(ideal(x^2, x*y + y))",
    "gcd": "gcd(x*y, x^2)",
    "factor": "factor(x^2 - y^2)",
    "resultant": "resultant(y^2 - x, y - 2, y)",
    "normal_form": "normal_form(x^2 + y, ideal(x^2))",
    "membership": "membership(x*y, ideal(x)).value",
    "local_standard_basis": "local_standard_basis(ideal(x^2, y), point=(0, 0))",
    "local_normal_form": "local_normal_form(x + y, ideal(x), point=(0, 0)).remainder",
    "eliminate": "eliminate(ideal(x - y, y^2), y)",
    "intersection": "intersection(ideal(x), ideal(y))",
    "quotient_ring": "A = quotient_ring(ideal(x*y - 1))\nu, v = A.gens\nu.inverse() == v",
    "quotient": "quotient(ideal(x*y), ideal(x))",
    "saturation": "saturation(ideal(x*y), x)",
    "radical": "radical(ideal(x^2, y))",
    "dimension": "dimension(ideal(x*y)).value",
    "ideal_power": "ideal_power(ideal(x, y), 2)",
    "symbolic_power": "symbolic_power(ideal(x, y), 2)",
    "primary_decomposition": "primary_decomposition(ideal(x*y)).components",
    "localize": "localize(ideal(x*y), ideal(x))",
    "normalization": "normalization(ideal(y^2 - x^3)).components",
    "quotient_basis": "quotient_basis(ideal(x^2, y)).basis",
    "local_quotient_basis": "local_quotient_basis(ideal(x^2, y)).basis",
    "multiplicity": "multiplicity(ideal(x^2, y)).value",
    "hilbert_series": "hilbert_series(ideal(x^2, y^3))",
    "hilbert_samuel_function": "hilbert_samuel_function(ideal(y^2 - x^3), degrees=[0, 1, 2]).values",
    "hilbert_samuel_multiplicity": "hilbert_samuel_multiplicity(ideal(y^2 - x^3)).value",
    "tangent_cone": "tangent_cone(ideal(y^2 - x^3)).generators",
    "tangent_cone_hilbert_series": "tangent_cone_hilbert_series(ideal(y^2 - x^3))",
    "zariski_tangent_space": "zariski_tangent_space(ideal(y - x^2)).basis",
    "embedding_dimension": "embedding_dimension(ideal(y^2 - x^3)).value",
    "local_dimension": "local_dimension(ideal(y^2 - x^3)).value",
    "local_smoothness": "local_smoothness(ideal(y - x^2)).value",
    "local_jacobian_algebra": "local_jacobian_algebra(y^2 - x^3).basis",
    "milnor_number": "milnor_number(y^2 - x^3).value",
    "tjurina_number": "tjurina_number(y^2 - x^3).value",
}

_EXAMPLES = {
    "coordinates": _COORDINATES,
    "len": "len([2, 3, 5])",
    "print": 'print("multiplicity", 6)',
    "print_json": 'print_json(divisor="E3", multiplicity=6)',
    "algebra_request": (
        """request = '{"operation":"ideal","ring":{"variables":["x"]},"operands":{"ideal":["x^2"]}}'\n"""
        "algebra_request(request)"
    ),
}

_PURPOSES = {
    "resolve": "Resolve a hypersurface at the origin and return its charts and divisors.",
    "dual_complex": "Recompute the exceptional incidence from the charts of a saved resolution.",
    "monodromy_check": "Check the poles of the local topological zeta function for monodromy witnesses.",
    "print": "Display values at the prompt or write them to a script's output.",
}


def function_help(name, builtins):
    """Describe a registered function. Future functions can include a >>> example in their docstring."""
    function = builtins.get(name)
    if not callable(function):
        return None

    canonical = getattr(function, "__name__", name)
    doc_lines = (getdoc(function) or "").splitlines()
    summary = doc_lines[0] if doc_lines else ""
    doc_examples = [line[4:] for line in doc_lines if line.startswith((">>> ", "... "))]
    operation = canonical.replace("_", "-") if canonical != "localize" else "localization"
    purpose = _PURPOSES.get(canonical) or OPERATION_DESCRIPTIONS.get(operation) or summary
    purpose = (purpose or f"Call the registered Atlas function {name}").rstrip(".") + "."
    purpose = purpose[0].upper() + purpose[1:]

    sig = signature(function)
    # Opaque sentinels mark omitted arguments. Keep their addresses out of the UI.
    display = str(sig.replace(parameters=[
        p.replace(default=Ellipsis) if type(p.default) is object else p for p in sig.parameters.values()
    ])).replace("=Ellipsis", "=...")
    if canonical in _RING_EXAMPLES:
        example = _COORDINATES + "\n" + _RING_EXAMPLES[canonical].replace(canonical + "(", name + "(", 1)
    elif canonical in _EXAMPLES:
        example = _EXAMPLES[canonical].replace(canonical + "(", name + "(", 1)
    elif doc_examples:
        example = "\n".join(doc_examples)
    else:
        arguments = [f"{p.name}={p.name}" if p.kind == Parameter.KEYWORD_ONLY else p.name
                     for p in sig.parameters.values() if p.default is Parameter.empty
                     and p.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)]
        example = f"{name}({', '.join(arguments)})"
    return f"{name}{display}\n{purpose}\n\nExample\n{indent(example, '  ')}"
