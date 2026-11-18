"""Signatures, runnable examples and guide sections for the Atlas session functions."""
from inspect import getdoc, signature
from io import StringIO
import re
from tokenize import NAME, generate_tokens, untokenize

from .algebra_cli import OPERATION_DESCRIPTIONS
from .help_card import usage_card
from .session_algebra import algebra_bindings


_COORDINATES = 'x, y = coordinates("x", "y")'
# Algebra examples share a setup so each can be copied into a fresh session.
_RING_EXAMPLES = {
    "resolve": "f = y^2 - x^3\nR = resolve(f)\nR.dual_complex",
    "recompute_dual_complex": "R = resolve(y^2 - x^3)\nD = recompute_dual_complex(R)\nD.vertices",
    "dual_complex": "R = resolve(y^2 - x^3)\ndual_complex(R).vertices",
    "monodromy_check": "monodromy_check(y^2 - x^3).status",
    "ideal": "I = ideal(y^2+x^5)\nK = I.quotient_ring\nJ = ideal((y, x^5), ring=K)\nJ.generators",
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
    "quotient_ring": "I = ideal(y^2 + x^5)\nK = quotient_ring(I)\nu, v = K.coordinates\nv^2 + u^5 == 0",
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
    "coordinates": _COORDINATES + '\nf = y^2 + x\nf.parent',
    "len": "len([2, 3, 5])",
    "print": 'print("multiplicity", 6)',
    "print_json": 'print_json(divisor="E3", multiplicity=6)',
    "algebra_request": (
        """request = '{"operation":"ideal","ring":{"variables":["x"]},"operands":{"ideal":["x^2"]}}'\n"""
        "algebra_request(request)"
    ),
}


_PURPOSES = {
    "coordinates": "Return the ordered generators of a polynomial ring.",
    "ideal": "Create an ideal in its generators' parent ring, the ring you specify, or the active ring.",
    "quotient_ring": "Return the quotient of an ideal’s parent ring by that ideal.",
    "resolve": "Resolve a hypersurface at the origin and return its resolution.",
    "recompute_dual_complex": "Rebuild the exceptional dual complex from a resolution’s recorded charts.",
    "dual_complex": "Return the raw record of the exceptional dual complex rebuilt from the charts.",
    "monodromy_check": "Check the poles of the local topological zeta function for monodromy witnesses.",
    "print": "Display values at the prompt or write them to script output.",
}

# Established functions share guide sections for related operations. New
# functions can give their reference and example directly in their docstring.
_SESSIONS = "docs/sessions.md#"
_IDEALS = "examples/polynomial-ideals.md#"
_QUOTIENTS = "examples/quotient-algebras.md#"
_LOCAL = "examples/local-algebra.md#"
_GEOMETRY = "examples/local-geometry.md#"
_RECORDS = "docs/resolution-records.md#"
_GUIDES = {
    _SESSIONS + "choose-the-ring-of-a-polynomial": ("coordinates", "polynomial"),
    _SESSIONS + "create-an-ideal-in-a-quotient-ring": ("ideal",),
    _SESSIONS + "name-a-quotient-and-its-coordinates": ("quotient_ring",),
    _SESSIONS + "add-multiply-and-raise-ideals-to-powers": ("ideal_power",),
    _SESSIONS + "move-a-polynomial-between-rings": ("hom",),
    "docs/resolving-a-cusp.md#follow-the-branch-to-the-exceptional-divisor": ("resolve",),
    _SESSIONS + "run-or-save-a-calculation": ("len", "print", "print_json"),
    _RECORDS + "chart-and-complex-conventions": ("dual_complex", "recompute_dual_complex"),
    _RECORDS + "topological-zeta-function": ("monodromy_check",),
    _IDEALS + "reduce-a-polynomial-modulo-an-ideal": ("groebner", "normal_form", "membership"),
    _IDEALS + "eliminate-a-coordinate": ("eliminate", "resultant"),
    _IDEALS + "factor-and-reconstruct-a-polynomial": ("factor",),
    _IDEALS + "find-a-common-factor": ("gcd",),
    _IDEALS + "intersect-divide-and-saturate-ideals": ("intersection", "quotient", "saturation"),
    _IDEALS + "recover-the-reduced-support": ("radical",),
    _IDEALS + "reconstruct-an-ideal-from-its-primary-components": ("primary_decomposition",),
    _IDEALS + "compare-ordinary-and-symbolic-powers": ("symbolic_power",),
    _QUOTIENTS + "localize-at-a-prime": ("localize",),
    _QUOTIENTS + "normalize-the-cusp": ("normalization",),
    _QUOTIENTS + "read-the-graded-hilbert-series": ("hilbert_series",),
    _LOCAL + "count-global-and-local-lengths": ("dimension", "multiplicity", "quotient_basis", "local_quotient_basis"),
    _LOCAL + "compare-the-cusp-and-its-parametrization": ("local_jacobian_algebra", "milnor_number", "tjurina_number"),
    _LOCAL + "reduce-at-a-rational-point": ("local_standard_basis", "local_normal_form"),
    _GEOMETRY + "sample-the-hilbert-samuel-function": ("hilbert_samuel_function", "hilbert_samuel_multiplicity"),
    _GEOMETRY + "compute-the-tangent-cone": ("tangent_cone", "tangent_cone_hilbert_series"),
    _GEOMETRY + "compute-the-tangent-space": ("zariski_tangent_space", "embedding_dimension"),
    _GEOMETRY + "compare-local-dimension-and-smoothness": ("local_dimension", "local_smoothness"),
    "examples/algebra-http.md#specify-a-ring-and-its-operands": ("algebra_request",),
}
_FUNCTION_DOCS = {name: guide for guide, names in _GUIDES.items() for name in names}


def _guide_section(canonical, lines):
    trailers = [line for line in lines if line.startswith("Docs:")]
    if not trailers:
        return _FUNCTION_DOCS.get(canonical)
    if trailers == [lines[-1]]:
        ref = re.fullmatch(r"Docs: ((?:[\w-]+/)*[\w.-]+\.md#[\w-]+)", trailers[0])
        if ref:
            return ref.group(1)
    raise ValueError(f"{canonical} needs a final Docs: path/to/guide.md#section line.")


def function_documentation(operation, builtins=None):
    """Find the guide section of a registered function or stored algebra result."""
    function = (algebra_bindings() if builtins is None else builtins).get(operation)
    if function:
        return _guide_section(function.__name__, (getdoc(function) or "").splitlines())
    return _FUNCTION_DOCS.get(operation)


def _aliased_example(example, canonical, alias):
    tokens = list(generate_tokens(StringIO(example).readline))
    for i, token in enumerate(tokens[:-1]):
        if (token.type == NAME and token.string == canonical and tokens[i + 1].string == "("
                and (i == 0 or tokens[i - 1].string != ".")):
            tokens[i] = token._replace(string=alias)
    return untokenize(tokens)


def session_signature(function):
    """Read a session callable's signature, including print on CPython 3.10."""
    try:
        return signature(function)
    except ValueError:
        if function is not print:
            raise
        # CPython 3.10 implements these options but omits print.__text_signature__.
        return signature(lambda *args, sep=" ", end="\n", file=None, flush=False: None)


def function_help(name, builtins):
    """Render a signature, one-line summary, Atlas example and anchored guide link."""
    function = builtins.get(name)
    if not callable(function):
        return None

    canonical = getattr(function, "__name__", name)
    lines = (getdoc(function) or "").splitlines()
    doc_example = "\n".join(line[4:] for line in lines if line.startswith((">>> ", "... ")))
    reference = _guide_section(canonical, lines)
    if not reference:
        raise ValueError(f"{canonical} needs a final Docs: path/to/guide.md#section line.")
    operation = "localization" if canonical == "localize" else canonical.replace("_", "-")
    summary = _PURPOSES.get(canonical) or (lines[0] if lines else OPERATION_DESCRIPTIONS.get(operation))

    if doc_example:
        example = doc_example
    elif canonical in _RING_EXAMPLES:
        example = _COORDINATES + "\n" + _RING_EXAMPLES[canonical]
    else:
        example = _EXAMPLES.get(canonical)
    if not summary or not example:
        raise ValueError(f"{canonical} needs a one-line help summary and a runnable Atlas example.")
    if name != canonical:
        example = _aliased_example(example, canonical, name)

    sig = session_signature(function)
    # Some session defaults distinguish omission from None with a sentinel.
    display = str(sig.replace(parameters=[
        p.replace(default=Ellipsis) if type(p.default) is object else p for p in sig.parameters.values()
    ])).replace("=Ellipsis", "=...")
    summary = summary[0].upper() + summary[1:].rstrip(".") + "."
    return usage_card(summary, example, reference, signature=name + display)
