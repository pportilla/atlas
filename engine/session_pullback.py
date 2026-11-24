"""Transport polynomials between a resolution's input and chart coordinates."""

from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cached_property
import re

from . import native_polynomial as native
from .marked_field import flint_literal, record_field
from .plane_backend import ordered_roots
from .polynomial import InputError, parse_expression, rational_text
from .point_geometry import request as point_request, support_profile
from .session_polynomial import Polynomial, QQ, _CoordinateSpace, _NumberField


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


@dataclass(frozen=True, eq=False, repr=False)
class _ChartSpace(_CoordinateSpace):
    # Record names (e.g. a3 and a6) can represent elements of the same field.
    aliases: dict[str, str] = field(default_factory=dict)
    ambient_equations: tuple[str, ...] = ()

    @cached_property
    def ambient(self):
        return replace(self, has_relations=False, ambient_equations=()) if self.has_relations else self

    @cached_property
    def presentation_relations(self):
        return tuple(chart_polynomial(equation, self.ambient) for equation in self.ambient_equations)

    @cached_property
    def relation_basis(self):
        from .algebra_groebner import reduced_basis

        relations = [g._value for g in self.presentation_relations]
        if self.coeffs is not QQ:
            relations.append(self.coeffs.minimal_polynomial)
        return reduced_basis(relations, self.polynomial_generators, order="lex")


def _expressions(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _expressions(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _expressions(item)


def _local_field(data):
    name = data["generator"]
    terms = parse_expression(data["minimal_polynomial"], (name,))
    _, degree, _ = support_profile(terms)
    if degree < 1 or data["degree"] != degree:
        raise ValueError("The chart's coefficient field has an inconsistent defining degree.")
    coefficients = tuple(terms.get((j,), 0) for j in range(degree, -1, -1))
    mark = data["embedding"]
    numbered = re.fullmatch(r"root (\d+) \(FLINT order\)", mark)
    if numbered:
        integers = point_request("marked_coefficients", coefficients=list(map(rational_text, coefficients)))
        mark = flint_literal(integers, int(numbered[1]))
    return dict(generator=name, exact_root=mark, minimal_polynomial_coefficients=coefficients)


def chart_space(resolution_record, chart_record):
    names = chart_record["coordinates"]
    equations = tuple(chart_record.get("ambient_ideal", ()))
    declared = resolution_record.get("coefficient_field", {})
    local_field = chart_record.get("coefficient_field")
    if local_field:
        declared = _local_field(local_field)
        names = tuple(name for name in names if name != declared["generator"])
    constants = resolution_record.get("algebraic_numbers", {})
    weights = chart_record.get("weight_field")
    dual = resolution_record.get("dual_complex", {})
    families = dict(resolution_record.get("root_families", dual.get("root_families", {})))
    if "root_definition" in dual:
        families["alpha"] = dual["root_definition"]
    if not (declared.get("generator") or constants or weights or families):
        return _ChartSpace(tuple(native.Dummy(name) for name in names),
                           has_relations=bool(equations), ambient_equations=equations)

    # Recover exactly the constants used in polynomial fields of this chart.
    texts = [expr for key in ("map", "strict_transform", "exceptional", "ambient_ideal",
                              "ambient_components", "ambient_exceptional", "ambient_total_transform", "pullback_factor",
                              "reduced_strict_transform", "strict_transform_on_exceptional", "map_to_parent",
                              "reduced_strict_transform_on_exceptional", "pivot_generator", "projective_coordinates",
                              "strict_transform_components")
             for expr in _expressions(chart_record.get(key))]
    words = " ".join(texts)
    used_names = set(_NAME.findall(words))
    constants = dict(constants)
    for name in sorted(used_names):
        individual = families.get(name)
        if individual and "exact" in individual:
            constants[name] = {"exact": individual["exact"]}
            continue
        family, _, index = name.rpartition("_")
        if family not in families or not index.isdigit():
            continue
        definition = families[family]
        equation = definition.get("polynomial")
        if not equation:
            equation = f"T^{definition['degree']}-({definition['power_value']})"
        roots = ordered_roots(equation)
        if family == "alpha":  # Ordinary tangent-point labels omit the zero slope.
            roots = tuple(root for root in roots if root != "0")
        if not 1 <= int(index) <= len(roots):
            raise ValueError(f"Root label {name} exceeds the distinct roots of {equation}.")
        constants[name] = {"exact": roots[int(index)-1]}
    if weights:
        constants = {**constants, weights["generator"]: {"exact": weights["exact"]}}
    if not local_field and resolution_record.get("engine") == "native-a-series" and declared.get("generator"):
        # The A-series normalization allows either root. The solver's marked
        # chart convention takes the first FLINT root of the quadratic.
        generator = declared["generator"]
        match = re.fullmatch(re.escape(generator) + r"\^2-\((-?\d+(?:/\d+)?)\)", declared["minimal_polynomial"])
        if not match:
            raise ValueError(f"Unrecognized A-series coefficient relation in chart {chart_record['id']}.")
        declared = dict(declared, exact_root=f"sqrt({match[1]})")

    source_parameter = declared.get("generator")
    parameter = source_parameter or "theta"
    # The primitive parameter may reuse the declared generator's name. Other
    # record names must be explicitly decoded, never captured by accident.
    while parameter in names or (parameter != source_parameter and parameter in used_names):
        parameter += "_"
    field, aliases = record_field(declared, constants, texts, parameter)
    if not field:
        return _ChartSpace(tuple(native.Dummy(name) for name in names),
                           has_relations=bool(equations), ambient_equations=equations)
    if field["degree"] > 1 and parameter in aliases and aliases[parameter] != parameter:
        # The compositum has a different primitive element from the declared
        # generator. Its printed name must round-trip independently of that alias.
        while parameter in names or parameter in aliases or parameter in used_names:
            parameter += "_"
        field, aliases = record_field(declared, constants, texts, parameter)
    if any(name in names for name in aliases):
        raise ValueError(f"A coefficient in chart {chart_record['id']} shares a coordinate name.")
    coeffs = QQ
    if field["degree"] > 1:
        theta = native.Dummy(parameter)
        modulus = native.from_terms((theta,), parse_expression(field["minimal_polynomial"], (parameter,)))
        coeffs = _NumberField(theta, modulus, field["embedding"])
    space = _ChartSpace(tuple(native.Dummy(name) for name in names), coeffs=coeffs, aliases=aliases)
    if local_field and equations:
        # In field-cover coordinates the minimal polynomial occurs as an
        # ambient equation. Moving its generator into the coefficient field
        # leaves only the other, genuinely geometric relations.
        equations = tuple(g for g in equations if chart_polynomial(g, space))
    return replace(space, has_relations=True, ambient_equations=equations) if equations else space


def chart_source(expression, local_space):
    """Express recorded names in this chart's canonical coefficient parameter."""
    if not isinstance(local_space, _ChartSpace) or not local_space.aliases:
        return expression
    aliases = local_space.aliases

    def substitute(match):
        name = match[0]
        return "(" + aliases[name] + ")" if name in aliases else name

    return _NAME.sub(substitute, expression)


def chart_polynomial(expression, local_space):
    """Decode a recorded equation in this chart's generators and marked coefficients."""
    source = chart_source(expression, local_space)
    generators = local_space.polynomial_generators
    modulus = str(local_space.coeffs.minimal_polynomial) if local_space.coeffs is not QQ and "/" in source else None
    try:
        terms = parse_expression(source, tuple(g.name for g in generators), coefficient_modulus=modulus)
    except InputError as error:
        raise ValueError(
            f"Cannot read {expression!r} as a polynomial over {local_space.coeffs} in "
            f"{', '.join(local_space.names)}. {error}"
        ) from error
    return Polynomial(local_space, native.from_terms(generators, terms))


def pullback(polynomial, source_space, chart_record, local_space):
    """Substitute the recorded projection to the original input simultaneously."""
    if type(polynomial) is int or isinstance(polynomial, Fraction):
        return Polynomial(local_space, native.coerce(polynomial))
    if not isinstance(polynomial, Polynomial):
        raise ValueError("Pass a polynomial in the original resolution coordinates to pullback.")
    if polynomial._space is not source_space and polynomial._space is not QQ:
        raise ValueError("This polynomial belongs to a different coordinate space from the resolution input.")

    # Each map in the solver record is already composed through the parents.
    # Only the images used by the operand enter its pullback.
    projection = chart_record.get("map", {})
    substitutions = {}
    for generator in polynomial._value.free_symbols:
        if generator.name not in projection:
            raise ValueError(f"Chart {chart_record['id']} has no recorded image for {generator.name}.")
        image = chart_polynomial(projection[generator.name], local_space)
        substitutions[generator] = image._value
    return Polynomial(local_space, polynomial._value.xreplace(substitutions))
