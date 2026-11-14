"""Transport chart ideals and exact splitting fields for geometric incidence."""
from .affine_kernel import _polynomial, decode, sparse_terms
from .atlas_finite import _extend_chart, _field_mark, _integer_polynomial
from .chart_expressions import chart_expressions
from .kernel_json import kernel_json
from .marked_field import flint_literal
from .plane_backend import coefficient_field
from .point_geometry import decode_terms, encode_terms


class ExtendSectionField(Exception):
    """The certified smooth fibre supplies these exact algebraic roots."""


def absolute_primes(chart, equations, dimension):
    field = chart.coefficient_field
    axes = chart.variables
    if field:
        axes = tuple(v for v in axes if v != field[0]) + (field[0],)
        equations = (*equations, field[1])
    rows = [encode_terms(dict(sparse_terms(axes, g))) for g in dict.fromkeys(equations)]
    result = kernel_json("atlas_absolute_sections", "decompose", width=len(axes),
                         marked=bool(field), ideal=rows, dimension=dimension)
    if result is None:
        return None
    if result["splitting_polynomials"]:
        t = chart_expressions(axes).Dummy("section_root")
        roots = []
        for row in result["splitting_polynomials"]:
            for coeffs in _integer_polynomial(t, _polynomial(decode_terms(row), (t,))):
                roots.extend(flint_literal(coeffs, i) for i in range(1, len(coeffs)))
        raise ExtendSectionField(tuple(dict.fromkeys(roots)))
    return [tuple(_polynomial(decode_terms(row), axes) for row in prime) for prime in result["primes"]]


def extend_section_field(charts, roots):
    fields = tuple(dict.fromkeys(c.coefficient_field for c in charts if c.coefficient_field))
    marks = tuple(_field_mark(field) for field in fields)
    if None in marks:
        return None
    generators = tuple(dict.fromkeys((*marks, *roots)))
    common = coefficient_field(generators, exhaustive=True)
    if common["degree"] <= max((field[2] for field in fields), default=1):
        raise ArithmeticError("The certified splitting field did not enlarge the chart field.")
    api = chart_expressions(charts[0].variables)
    theta = api.Dummy("section_field")
    T = api.Symbol("T")

    def read(value):
        return decode((T,), value).xreplace({T: theta})

    field = theta, read(common["minimal_polynomial"]), common["degree"], common["exact"]
    images = {mark: read(value) for mark, value in zip(generators, common["values"])}
    substitutions = {old: images[mark] for old, mark in zip(fields, marks)}
    return [_extend_chart(c, field, substitutions.get(c.coefficient_field)) for c in charts]
