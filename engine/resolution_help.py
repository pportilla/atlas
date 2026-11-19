"""Short prompt help for linked views of a saved resolution."""
import json

from .help_card import usage_card
from .session_values import Branch, Chart, Divisor, DualComplex, Resolution


_CHART_DOC = "docs/tutorial/resolving-a-cusp.md#look-inside-a-chart"
_CUSP_DOC = "docs/tutorial/resolving-a-cusp.md#follow-the-branch-to-the-exceptional-divisor"
_COMPLEX_DOC = "docs/thematic/resolution-records.md#chart-and-complex-conventions"
_MODE_DOC = "docs/thematic/resolution-records.md#resolution-modes"

# Examples use the requested binding. The saved record selects IDs and supplies
# alternatives for empty lists and partial charts without computing equations.
_PAGES = {
    Resolution: {
        "": ("A resolution records the charts and exceptional divisors above the origin.",
             "{n}.polynomial\n{n}.charts\n{n}.dual_complex", _MODE_DOC),
        "polynomial": ("{n}.polynomial is the input equation in {ring}.",
                       "{n}.polynomial\n{n}.polynomial.parent", _CUSP_DOC),
        "coordinates": ("{n}.coordinates gives the ordered generators of the input ring {ring}.",
                        "{n}.coordinates\n{n}.coordinates.parent", _CUSP_DOC),
        "mode": ("{n}.mode gives the chosen resolution mode.", "{n}.mode", _MODE_DOC),
        "status": ("{n}.status gives the recorded outcome of the resolution.", "{n}.status",
                   "docs/thematic/resolution-pipeline.md#what-the-record-certifies"),
        "charts": ("{n}.charts lists the local charts of this resolution.",
                   "{n}.charts[0].strict_transform\n{n}.chart({chart_id}).coordinates", _CHART_DOC),
        "chart": ("{n}.chart(id) retrieves a chart by its recorded ID.",
                  "{n}.charts\n{n}.chart({chart_id}).coordinates", _CHART_DOC),
        "branches": ("{n}.branches lists the recorded branches of the strict transform.",
                     "{n}.branches\n{n}.branches[0].chart\n{n}.branches[0].meets", _CUSP_DOC),
        "divisors": ("{n}.divisors lists the exceptional divisors of this resolution.",
                     "{n}.divisors\n{n}.divisors[0].charts", _COMPLEX_DOC),
        "divisor": ("{n}.divisor(id) retrieves an exceptional divisor by its recorded ID.",
                    "{n}.divisors\n{n}.divisor({divisor_id}).charts", _COMPLEX_DOC),
        "record": ("{n}.record contains the saved resolution data.",
                   "{n}.record.polynomial\n{n}.record.mode", _MODE_DOC),
    },
    Chart: {
        "": ("A chart is a local coordinate patch of a resolution.",
             "{n}.coordinates\n{n}.strict_transform\n{n}.map\n{n}.divisors", _CHART_DOC),
        "id": ("{n}.id is the chart's ID in its resolution.", "{n}.id", _CHART_DOC),
        "coordinates": ("{n}.coordinates gives the ordered generators of this chart's ring.",
                        "{n}.coordinates\n{n}.coordinates.parent", _CHART_DOC),
        "strict_transform": ("{n}.strict_transform gives the local equation or equations of the strict transform.",
                             "{n}.strict_transform", _CHART_DOC),
        "map": ("{n}.map gives each recorded original coordinate as a polynomial on this chart.",
                "{n}.map", _CHART_DOC),
        "pullback_map": ("{n}.pullback_map is the ring map from the input ring to this chart.",
                         "{n}.pullback_map.domain\n{n}.pullback_map.codomain\n{n}.pullback_map.images", _CHART_DOC),
        "pullback": ("{n}.pullback(f) gives the image of an input polynomial in the chart.",
                     "{n}.pullback({source})", _CHART_DOC),
        "parent": ("{n}.parent is the ring of functions on the chart.",
                   "{n}.parent\n{n}.parent.coordinates", _CHART_DOC),
        "ambient": ("{n}.ambient is the polynomial ring in which the chart relations are written.",
                    "{n}.ambient.coordinates", _CHART_DOC),
        "divisors": ("{n}.divisors lists the exceptional divisors recorded in this chart.",
                     "{n}.divisors\nfor E in {n}.divisors: print(E.id, E.ideal_in({n}))", _CHART_DOC),
    },
    Divisor: {
        "": ("A divisor is a geometric exceptional component in the resolution.",
             "{n}.id\n{n}.charts", _COMPLEX_DOC),
        "charts": ("{n}.charts lists the charts with local equations for this divisor.",
                   "{n}.charts", _CHART_DOC),
        "ideal_in": ("{n}.ideal_in(chart) gives the generators defining the divisor on that chart.",
                     "{n}.ideal_in({local_chart})", _CHART_DOC),
        "equation_in": ("{n}.equation_in(chart) gives the local equation in a chart with one recorded generator.",
                        "{n}.equation_in({local_chart})", _CHART_DOC),
        "id": ("{n}.id is the divisor's label in this resolution.", "{n}.id", _COMPLEX_DOC),
    },
    Branch: {
        "": ("A branch records a component of the strict transform over the origin.",
             "{n}.chart\n{n}.meets", _CUSP_DOC),
        "id": ("{n}.id is the branch's label in this resolution.", "{n}.id", _CUSP_DOC),
        "chart": ("{n}.chart is the chart containing the recorded branch point.",
                  "{n}.chart.id\n{n}.chart.strict_transform", _CHART_DOC),
        "meets": ("{n}.meets gives the exceptional divisors meeting the strict transform along this branch.",
                  "{n}.meets", _CUSP_DOC),
        "point": ("{n}.point gives the recorded coordinates of the branch point.", "{n}.point", _CUSP_DOC),
    },
    DualComplex: {
        "": ("The exceptional dual complex records intersections over the algebraic closure of QQ.",
             "{n}.vertices\n{n}.cells", _COMPLEX_DOC),
        "vertices": ("{n}.vertices lists the exceptional components as linked divisors.",
                     "{n}.vertices", _COMPLEX_DOC),
        "cells": ("{n}.cells records the vertices and boundary faces of each positive-dimensional cell.",
                  "{n}.cells", _COMPLEX_DOC),
        "status": ("{n}.status reports the completeness of the incidence data.",
                   "{n}.status", _COMPLEX_DOC),
        "record": ("{n}.record contains the saved incidence data.", "{n}.record.vertices\n{n}.record.cells", _COMPLEX_DOC),
    },
}
_GENERIC = {"resolution": Resolution, "chart": Chart, "divisor": Divisor, "branch": Branch}
_SETUP = {Resolution: ("R", ""), Chart: ("C", 'C = R.chart("B3.u")\n'),
          Divisor: ("E", 'E = R.divisor("E3")\n'), Branch: ("b", "b = R.branches[0]\n")}


def resolution_topics(variables):
    """List documented topics using types alone, including the saved complex of a resolution."""
    owners = {name: kind for name, kind in _GENERIC.items() if name not in variables}
    owners.update((name, type(value)) for name, value in variables.items() if type(value) in _PAGES)
    topics = set()
    for name, kind in owners.items():
        topics.update(f"{name}.{field}" if field else name for field in _PAGES[kind])
        if kind is Resolution:
            topics.update(f"{name}.dual_complex" + (f".{field}" if field else "") for field in _PAGES[DualComplex])
    return topics


def _saved_name(value, variables):
    return next((name for name, saved in variables.items() if saved is value), None)


def _source_label(resolution, variables):
    source = vars(resolution).get("polynomial")
    if source is not None and (name := _saved_name(source, variables)):
        return name
    equation = resolution._record.get("polynomial", "")
    return equation if equation and len(equation) <= 72 else "the input polynomial"


def _divisor_chart(divisor, name, field, variables):
    """Find a chart that supports the method without reading its local polynomials."""
    for index, chart in enumerate(divisor.charts):
        component = chart._exceptional[divisor.id]
        generators = component.get("ideal")
        if generators is None:
            if "equation" not in component:
                continue
        elif not generators or (field == "equation_in" and len(generators) != 1):
            continue
        return _saved_name(chart, variables) or f"{name}.charts[{index}]"
    return None


def _resolution_card(value, field, summary, example, context, variables):
    if not field:
        context.update(mode=value._record.get("mode", ""), equation=_source_label(value, variables))
        if value._record.get("engine") == "weighted":
            # The weighted engine records its blowups, charts and divisor data, without cells.
            return ("{n} is the embedded resolution of {equation} in {ring} by weighted blowups.",
                    "{n}.steps\n{n}.divisors\n{n}.certificate")
        return "{n} is the {mode} resolution record for {equation} in {ring}.", example
    if field in ("chart", "charts"):
        if not value._charts:
            return "{n} has no recorded charts.", "{n}.charts"
        context["chart_id"] = json.dumps(next(iter(value._charts)), ensure_ascii=False)
        if field == "charts":
            count = len(value._charts)
            summary = f"{{n}}.charts contains {count} recorded chart{'s' if count != 1 else ''}."
    if field in ("divisor", "divisors"):
        if not value._divisors:
            return "{n} has no recorded exceptional divisors.", "{n}.divisors"
        context["divisor_id"] = json.dumps(next(iter(value._divisors)), ensure_ascii=False)
        if field == "divisors":
            count = len(value._divisors)
            summary = f"{{n}}.divisors contains {count} exceptional divisor{'s' if count != 1 else ''}."
    if field == "branches":
        if not value._branches:
            return "{n} has no recorded branches.", "{n}.branches" if "branches" in value._record else "{n}.charts"
        count = len(value._branches)
        summary = f"{{n}}.branches contains {count} recorded {'branch' if count == 1 else 'branches'} of the strict transform."
    return summary, example


def _complex_card(value, field, summary, example, context, resolution, owner):
    if not field:
        if value is not None:
            context["status"] = value.status
            rebuilt = "recomputed " if value is not resolution.dual_complex else ""
            summary = "{n} is the {status} " + rebuilt + "exceptional dual complex"
            if owner:
                summary += " of {owner}"
            field_name = resolution._record.get("geometric_field")
            if field_name:
                context["geometric_field"] = ("the algebraic closure of QQ"
                                               if field_name in ("algebraic closure of Q", "Qbar") else field_name)
                summary += " over {geometric_field}"
            summary += "."
        if owner and (value is None or resolution._branches):
            example += "\n{owner}.branches[0].meets  # strict transform attachment"
    elif field == "vertices" and (value is None or value._vertices):
        example += "\n{n}.vertices[0].charts"
    return summary, example


def _component_card(value, field, summary, example, context, variables):
    if not field:
        context["id"] = value.id
        context["kind"] = {Chart: "chart", Branch: "branch", Divisor: "exceptional divisor"}[type(value)]
        summary = "{n} is {kind} {id} of {origin}."
    if isinstance(value, Divisor) and field in ("", "ideal_in", "equation_in"):
        chart = _divisor_chart(value, context["n"], field, variables)
        if chart:
            context["local_chart"] = chart
            if not field:
                example += "\n{n}.ideal_in({local_chart})"
        elif field:
            summary = ("No chart of {n} records a single local equation." if field == "equation_in"
                       else "No chart of {n} records a local ideal.")
            example = "{n}.charts"
    elif isinstance(value, Chart) and field == "pullback_map":
        if not set(value._resolution._source_space.names).issubset(value._record.get("map", {})):
            summary, example = "{n} records a partial chart map.", "{n}.map"
    elif isinstance(value, Branch) and field == "point" and "point" not in value._record:
        summary, example = "{n} has no recorded point coordinates.", "{n}.chart"
    return summary, example


def resolution_help(topic, variables):
    root, _, field = topic.partition(".")
    value = variables.get(root)
    kind = type(value) if root in variables else _GENERIC.get(root)
    if kind not in _PAGES:
        return None
    if value is None:
        name, binding = _SETUP[kind]
        resolution, owner = None, "R"
        setup = 'x, y = coordinates("x", "y")\nR = resolve(y^2-x^3)\n' + binding
    else:
        name, setup = root, ""
        resolution = value if kind is Resolution else value._resolution
        owner = root if kind is Resolution else _saved_name(resolution, variables)
    if kind is Resolution and (field == "dual_complex" or field.startswith("dual_complex.")):
        field = field.partition(".")[2]
        name += ".dual_complex"
        kind = DualComplex
        value = resolution.dual_complex if resolution else None
    page = _PAGES[kind].get(field)
    if page is None:
        return None
    summary, example, reference = page
    if kind is Resolution and not field and value is not None and value._record.get("engine") == "weighted":
        reference = "docs/thematic/weighted-resolution.md#resolve-with-method-weighted"
    context = dict(n=name, owner=owner, ring=resolution._source_space if resolution else "QQ[x, y]",
                   source=f"{owner}.polynomial" if owner else "1", local_chart=name + ".charts[0]",
                   chart_id='"B3.u"', divisor_id='"E3"')
    if kind is DualComplex:
        summary, example = _complex_card(value, field, summary, example, context, resolution, owner)
    elif value is not None and kind is Resolution:
        summary, example = _resolution_card(value, field, summary, example, context, variables)
    elif value is not None:
        context["origin"] = owner or f"the resolution of {_source_label(resolution, variables)}"
        summary, example = _component_card(value, field, summary, example, context, variables)
    return usage_card(summary.format(**context), setup + example.format(**context), reference)
