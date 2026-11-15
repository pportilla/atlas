"""Contextual help for saved session values, using their live parents."""

from fractions import Fraction
from typing import NamedTuple

from .help_card import usage_card
from .session_algebra import Ideal, _NormalizedComponent, _Result, _ScalarResult
from .session_function_help import function_documentation, function_help
from .session_maps import RingHomomorphism
from .session_panel import workspace
from .session_polynomial import QQ, Polynomial, _Coordinates, _CoordinateSpace, _NumberField
from .session_quotient import QuotientElement, QuotientIdeal, QuotientRing


POLYNOMIALS = "docs/sessions.md#choose-the-ring-of-a-polynomial"
QUOTIENTS = "docs/sessions.md#name-a-quotient-and-its-coordinates"
IDEALS = "docs/sessions.md#add-multiply-and-raise-ideals-to-powers"
QUOTIENT_IDEALS = "docs/sessions.md#create-an-ideal-in-a-quotient-ring"
MAPS = "docs/sessions.md#move-a-polynomial-between-rings"
SAVED_VALUES = "docs/sessions.md#inspect-a-saved-value"
CHARTS = "docs/sessions.md#look-inside-a-chart"


class _FieldHelp(NamedTuple):
    description: str
    examples: str
    reference: str


_IDEAL_QUOTIENT = _FieldHelp("{n} is the quotient of {owner}.parent by {owner}.",
                             "{n}.defining_ideal\n{n}.coordinates", QUOTIENTS)
_FURTHER_QUOTIENT = _IDEAL_QUOTIENT._replace(reference=QUOTIENT_IDEALS)
_RESIDUE_COORDINATES = _FieldHelp("{n} contains the ordered residue classes of the generators in {owner}.",
                                  "{n}\n{owner}.ambient.coordinates", QUOTIENTS)
_AMBIENT_COORDINATES = _FieldHelp("{n} contains the ordered polynomial generators of {owner}.", "{n}", POLYNOMIALS)
_DECLARATION_EXECUTIONS = _FieldHelp("{n} records the latest quotient declaration for each compute mode.",
                                     "{n}\n{owner}.execution", QUOTIENTS)
_QUOTIENT_PRESENTATION = _FieldHelp("{n} records the variables, monomial order and relations of the quotient.",
                                    "{n}.variables\n{n}.relations", QUOTIENTS)
_PROJECTION = _FieldHelp("{n} maps the parent of {owner}.defining_ideal to {owner}.",
                         "{n}.domain\n{n}.images", MAPS)
_RELATIONS = _FieldHelp("{n} contains the defining relations in {owner}.ambient.",
                        "{n}\n{owner}.defining_ideal.generators", QUOTIENTS)
_VARIABLE_NAMES = _FieldHelp("{n} gives the ordered variable names of {owner}.", "{n}", POLYNOMIALS)
_MULTIPLICITY = _FieldHelp("{n}(factor) gives the largest power of a nonconstant factor that divides {owner}.",
                           "{n}({owner}.parent.coordinates[0])", CHARTS)
_GENERATORS = _FieldHelp("{n} contains the generators of the ideal {owner}.",
                         "{n}\n{owner}.parent", IDEALS)
_QUOTIENT_GENERATORS = _FieldHelp("{n} contains residue classes generating {owner} in its quotient ring.",
                                  "{n}\n{owner}.parent", QUOTIENT_IDEALS)

_SIMPLE_TYPES = {bool: "a boolean", int: "an integer", Fraction: "a rational number", str: "a string",
                 list: "a list", tuple: "a tuple", dict: "a record", type(None): "None"}
_RING_VALUES = (Polynomial, Ideal, _Result, _NormalizedComponent, _Coordinates, _CoordinateSpace, RingHomomorphism)


def algebra_help_values(session):
    """Index help topics without evaluating any mathematical fields."""
    values = {}
    functions = {id(value) for value in session.builtins.values() if callable(value)}
    for name, value in session.variables.items():
        if (isinstance(value, (*_RING_VALUES, _NumberField, tuple))
                or type(value) in _SIMPLE_TYPES or id(value) in functions):
            values[name] = value
        if isinstance(value, (Polynomial, Ideal, QuotientElement, _Coordinates)):
            values[name + ".parent"] = value.parent
        if isinstance(value, (Polynomial, _CoordinateSpace, Ideal, QuotientRing)):
            values[name + ".variables"] = _VARIABLE_NAMES
            values[name + ".coeffs"] = value.coeffs
        if isinstance(value, Polynomial):
            method = _MULTIPLICITY
            if value.parent.has_relations:
                method = _FieldHelp("{n} requires a polynomial ring without chart relations.", "{owner}.parent", CHARTS)
            elif not value._value._terms:
                method = _FieldHelp("{owner} is zero and has no finite factor multiplicity.", "{owner}", CHARTS)
            elif not value.parent.generators:
                method = _FieldHelp("{n} requires a nonconstant factor in the parent ring.", "{owner}.parent", CHARTS)
            values[name + ".multiplicity"] = method
        if isinstance(value, (Ideal, _Result)):
            values[name + ".ambient"] = value.ambient
        if isinstance(value, Ideal):
            values[name + ".quotient_ring"] = _FURTHER_QUOTIENT if isinstance(value, QuotientIdeal) else _IDEAL_QUOTIENT
            values[name + ".generators"] = _QUOTIENT_GENERATORS if isinstance(value, QuotientIdeal) else _GENERATORS
        if isinstance(value, _CoordinateSpace):
            values[name + ".coordinates"] = _AMBIENT_COORDINATES
        if isinstance(value, QuotientElement):
            values[name + ".representative"] = value.representative
        if isinstance(value, QuotientRing):
            for field in ("coordinates", "gens", "generators"):
                values[name + "." + field] = _RESIDUE_COORDINATES
            values[name + ".ambient.coordinates"] = _AMBIENT_COORDINATES
            values[name + ".defining_ideal"] = value.defining_ideal
            values[name + ".declaration_executions"] = _DECLARATION_EXECUTIONS
            values[name + ".relations"] = _RELATIONS
            values[name + ".projection"] = _PROJECTION
        if isinstance(value, (QuotientRing, QuotientElement, QuotientIdeal)):
            values[name + ".ring"] = _QUOTIENT_PRESENTATION
        if isinstance(value, RingHomomorphism):
            values[name + ".domain"] = value.domain
            values[name + ".codomain"] = value.codomain
    return values


def _basic_card(name, value):
    kind = type(value)
    fields = getattr(kind, "_fields", ()) if isinstance(value, tuple) else ()
    examples = [name]
    if fields and isinstance(fields, tuple):
        examples = [f"{name}.{field}" for field in fields[:2]]
        label = "a record"
    else:
        label = _SIMPLE_TYPES[kind]
        if isinstance(value, (tuple, list, dict)):
            examples = [f"len({name})"]
            if value:
                key = repr(next(iter(value))) if isinstance(value, dict) else "0"
                examples.append(f"{name}[{key}]")
        elif kind in (int, Fraction):
            examples.append(f"{name}^2")
        elif kind is bool:
            examples.append(f"not {name}")
        elif kind is str:
            examples.append(f"len({name})")
    return usage_card(f"{name} is {label}.", "\n".join(examples), SAVED_VALUES)


def algebra_value_help(name, value, variables=None):
    if isinstance(value, _FieldHelp):
        context = {"n": name, "owner": name.rpartition(".")[0]}
        return usage_card(value.description.format(**context), value.examples.format(**context), value.reference)
    if isinstance(value, _NumberField):
        return usage_card(f"{name} is the coefficient field {value}.", name, CHARTS)
    if not isinstance(value, _RING_VALUES):
        if callable(value):
            return function_help(name, {name: value})
        return _basic_card(name, value)

    # Use the workspace's ring names so independent copies of QQ[x, y] remain distinguishable.
    inventory = workspace(variables if variables is not None else {name: value})

    def ring_name(parent):
        if parent in inventory.references:
            return inventory.ring_name(parent)
        # A field can expose a parent whose only reference is through another value.
        return workspace({name: parent}).presentations[parent]

    if isinstance(value, (QuotientRing, _CoordinateSpace)):
        presentation = inventory.presentations.get(value) or ring_name(value)
        if isinstance(value, QuotientRing):
            return usage_card(f"{name} is the quotient ring {presentation}.",
                              f"{name}.coordinates\n{name}.ambient.coordinates\n{name}.defining_ideal", QUOTIENTS)
        if value is QQ:
            return usage_card(f"{name} is the rational field QQ.", f'polynomial("1/2", ring={name})', POLYNOMIALS)
        kind = "coordinate ring" if value.has_relations else "polynomial ring"
        return usage_card(f"{name} is the {kind} {presentation}.",
                          f'{name}.coordinates\npolynomial("1", ring={name})', POLYNOMIALS)
    if isinstance(value, _Coordinates):
        reference = QUOTIENTS if isinstance(value.parent, QuotientRing) else POLYNOMIALS
        return usage_card(f"{name} contains the ordered coordinates of {ring_name(value.parent)}.",
                          f'{name}.parent\npolynomial("1", variables={name})', reference)
    if isinstance(value, Polynomial):
        variable = value.variables[0] + "+1" if value.variables else "1/2"
        return usage_card(f"{name} is a polynomial in {ring_name(value.parent)}.",
                          f'{name}.parent\n{name}^2\npolynomial("{variable}", ring={name}.parent)', POLYNOMIALS)
    if isinstance(value, Ideal):
        if isinstance(value, _Result) and value.operation == "groebner":
            return usage_card(f"{name} is a Gröbner basis in {ring_name(value.parent)}.",
                              f"{name}.generators\n{name}.order", function_documentation("groebner"))
        reference = QUOTIENT_IDEALS if isinstance(value, QuotientIdeal) else IDEALS
        return usage_card(f"{name} is an ideal in {ring_name(value.parent)}.",
                          f"{name}.generators\n{name}^2\n{name}.quotient_ring", reference)
    if isinstance(value, QuotientElement):
        return usage_card(f"{name} is a residue class in {ring_name(value.parent)}.",
                          f"{name}.representative\n{name}.parent\n{name}^2", QUOTIENTS)
    if isinstance(value, RingHomomorphism):
        return usage_card(f"{name} is a ring map from {ring_name(value.domain)} to {ring_name(value.codomain)}.",
                          f"{name}.images\n{name}(1)", MAPS)
    if isinstance(value, _Result):
        operation = value.operation.replace("-", "_")
        fields = ("value", "finite", "point") if isinstance(value, _ScalarResult) else (
            "factors", "basis", "remainder", "components", "values", "numerator", "generators", "dimension", "point")
        examples = [f"{name}.{field}" for field in fields if field in value._fields][:2]
        return usage_card(f"{name} holds the result of {operation} in {ring_name(value.ambient)}.",
                          "\n".join(examples or [f"{name}.result"]), function_documentation(operation) or SAVED_VALUES)
    return usage_card(f"{name} is a normalization component over {ring_name(value.ambient)}.",
                      f"{name}.map\n{name}.relations", function_documentation("normalization"))
