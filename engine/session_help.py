"""Short usage cards for the callables exposed in an Atlas session."""
from inspect import Parameter, getdoc, signature
from textwrap import fill, indent
from typing import NamedTuple

from .algebra_cli import OPERATION_DESCRIPTIONS
from .session_algebra import Ideal, _Result, _ScalarResult
from .session_maps import RingHomomorphism
from .session_polynomial import Polynomial, _Coordinates, _CoordinateSpace
from .session_quotient import QuotientElement, QuotientRing


_COORDINATES = 'x, y = coordinates("x", "y")'
# Algebra examples share a setup so each can be copied into a fresh session.
_RING_EXAMPLES = {
    "resolve": "f = y^2 - x^3\nR = resolve(f)\nR.polynomial == f\nR.dual_complex",
    "recompute_dual_complex": "R = resolve(y^2 - x^3)\nD = recompute_dual_complex(R)\nD.vertices",
    "dual_complex": "R = resolve(y^2 - x^3)\ndual_complex(R).vertices",
    "monodromy_check": "monodromy_check(y^2 - x^3).status",
    "polynomial": ('f = y^2 - x^3\ng = polynomial("x^2+y", ring=f.parent)\ng.parent == f.parent\n'
                   'K = ideal(f).quotient_ring\npolynomial("y^2-x^3", ring=K) == 0'),
    "ideal": "I = ideal(x^2, y)\nJ = ideal(x, y^2)\nS = I + J\nP = I * J\nK = I^2",
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
    "quotient_ring": "I = ideal(y^2 + x^5)\nK = quotient_ring(I)\nK == I.quotient_ring\nu, v = K.coordinates\nv^2 + u^5 == 0",
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
    "coordinates": _COORDINATES + '\nf = y^2 + x\nx, y = coordinates("x", "y")\nx.parent == f.parent',
    "len": "len([2, 3, 5])",
    "print": 'print("multiplicity", 6)',
    "print_json": 'print_json(divisor="E3", multiplicity=6)',
    "algebra_request": (
        """request = '{"operation":"ideal","ring":{"variables":["x"]},"operands":{"ideal":["x^2"]}}'\n"""
        "algebra_request(request)"
    ),
}

_PURPOSES = {
    "quotient_ring": ("Return the ambient polynomial ring modulo the given ideal. "
                      "The default reuses I.quotient_ring. Use fresh=True for an independent parent. "
                      "An explicit CPU or GPU request reruns the declaration and keeps the same parent. "
                      "K.declaration_executions gives the latest execution report for each requested mode."),
    "resolve": "Resolve a hypersurface at the origin and return its charts and divisors.",
    "recompute_dual_complex": ("Rebuild exceptional incidence from the charts of a resolution. "
                               "R.dual_complex holds the complex saved by resolve."),
    "dual_complex": ("Compatibility function returning the raw recomputed record. "
                     "Use R.dual_complex to inspect a resolution or recompute_dual_complex(R) to rebuild incidence."),
    "monodromy_check": "Check the poles of the local topological zeta function for monodromy witnesses.",
    "print": "Display values at the prompt or write them to a script's output.",
}


# The order matters for computed polynomials and ideals, which are also results.
_VALUE_CARDS = {
    _Coordinates: (
        "The generators of {n}.parent in declaration order. Unpack this tuple to give its coordinates names.",
        '{n}.parent\npolynomial("1", variables={n})',
    ),
    _CoordinateSpace: (
        "A live parent ring. Its coordinates are the generators in declaration order. "
        "Pass this ring to polynomial to read an expression in it.",
        '{n}.coordinates\npolynomial("1", ring={n})',
    ),
    Polynomial: (
        "A polynomial in {n}.parent. When both parents have generators, two polynomials in distinct "
        "parents compare unequal, even if both are constant. A rational number or a polynomial "
        "in QQ is compared by its value in the other polynomial's ring.",
        '{n}.parent\n{n}.parent.coordinates\npolynomial("1", ring={n}.parent)',
    ),
    Ideal: (
        "An ideal generated by {n}.generators in the live ring {n}.parent. "
        "Use {n}.quotient_ring for the quotient by this ideal. "
        "Use + or * with another ideal in the same parent ring. Use ^ for an ordinary power. "
        "Its order is the monomial order for basis calculations.",
        '{n}.parent\n{n}.generators\n{n}.quotient_ring\n{n}^2',
    ),
    QuotientRing: (
        "A quotient of {n}.ambient by {n}.defining_ideal. Its coordinates are the residue "
        "classes of the ordered generators in {n}.ambient.coordinates. Call {n}(f) to project "
        "an ambient polynomial. Its relations give the reduced basis of the defining ideal.",
        "{n}.coordinates\n{n}.ambient.coordinates\n{n}.gens\n{n}.relations\n{n}(1)",
    ),
    QuotientElement: (
        "A residue class in {n}.parent. Its representative is a polynomial in the ambient ring. "
        "Project that polynomial to recover its class.",
        "{n}.parent\n{n}.representative\n{n}.parent({n}.representative) == {n}",
    ),
    RingHomomorphism: (
        "A ring map from {n}.domain to {n}.codomain. The images follow the order of the domain "
        "generators. Apply the map to a domain element by calling {n}.",
        "{n}.domain\n{n}.codomain\n{n}.images\n{n}(1)",
    ),
    _ScalarResult: (
        "A calculation result. Read its scalar with {n}.value and use {n}.status to inspect its status.",
        "{n}.value\n{n}.status\n{n}.result",
    ),
    _Result: (
        "A calculation result in the ambient ring {n}.ambient. Its result and status fields "
        "describe the output of the operation.",
        "{n}.operation\n{n}.ambient\n{n}.status\n{n}.result",
    ),
}


class _FieldHelp(NamedTuple):
    description: str
    example: str


_IDEAL_QUOTIENT = _FieldHelp(
    "The default quotient ring for this ideal. Repeated access returns the same live parent as quotient_ring({owner}).",
    "K = {n}\nK.defining_ideal\nK.coordinates",
)
_RESIDUE_COORDINATES = _FieldHelp(
    "The residue classes of the ambient generators in declaration order. Each class has parent {owner}.",
    "{n}\n{owner}.ambient.coordinates",
)
_AMBIENT_COORDINATES = _FieldHelp(
    "The polynomial generators of the ambient ring, in declaration order.",
    "{n}",
)
_DECLARATION_EXECUTIONS = _FieldHelp(
    "Execution reports keyed by the requested compute mode. Each entry describes the latest successful "
    "declaration in that mode. {owner}.execution describes the original declaration.",
    "{owner}.execution\n{n}",
)
_QUOTIENT_PRESENTATION = _FieldHelp(
    "The coefficient field, ordered variable names, monomial order and defining relations of the quotient.",
    "{n}",
)


def algebra_help_values(session):
    """Collect live help topics using stored parents and map endpoints."""
    values = {}
    kinds = tuple(_VALUE_CARDS)
    functions = {id(value) for value in session.builtins.values() if callable(value)}
    for name, value in session.variables.items():
        if isinstance(value, kinds) or id(value) in functions:
            values[name] = value
        if isinstance(value, (Polynomial, Ideal, QuotientElement, _Coordinates)):
            values[name + ".parent"] = value.parent
        if isinstance(value, (Ideal, _Result)):
            values[name + ".ambient"] = value.ambient
        if isinstance(value, Ideal):
            values[name + ".quotient_ring"] = _IDEAL_QUOTIENT
        if isinstance(value, QuotientRing):
            for field in ("coordinates", "gens", "generators"):
                values[name + "." + field] = _RESIDUE_COORDINATES
            values[name + ".ambient.coordinates"] = _AMBIENT_COORDINATES
            values[name + ".declaration_executions"] = _DECLARATION_EXECUTIONS
        if isinstance(value, (QuotientRing, QuotientElement)):
            values[name + ".ring"] = _QUOTIENT_PRESENTATION
        if isinstance(value, RingHomomorphism):
            values[name + ".domain"] = value.domain
            values[name + ".codomain"] = value.codomain
    return values


def algebra_value_help(name, value):
    card = value if isinstance(value, _FieldHelp) else next(
        (card for kind, card in _VALUE_CARDS.items() if isinstance(value, kind)), None)
    if card is None:
        return function_help(name, {name: value})
    names = {"n": name, "owner": name.rpartition(".")[0]}
    description, example = card
    summary = fill(description.format(**names), width=80, break_long_words=False, break_on_hyphens=False)
    return f"{name}\n{summary}\n\nExample\n{indent(example.format(**names), '  ')}"


def function_help(name, builtins):
    """Describe a registered function. Future functions can include a >>> example in their docstring."""
    function = builtins.get(name)
    if not callable(function):
        return None

    canonical = getattr(function, "__name__", name)
    doc_lines = (getdoc(function) or "").splitlines()
    doc_examples = [line[4:] for line in doc_lines if line.startswith((">>> ", "... "))]
    operation = canonical.replace("_", "-") if canonical != "localize" else "localization"
    if doc_examples:
        example_start = next(i for i, line in enumerate(doc_lines) if line.startswith((">>> ", "... ")))
        purpose = "\n".join(doc_lines[:example_start]).strip()
    else:
        summary = doc_lines[0] if doc_lines else ""
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
