"""Prompt help for inspecting saved resolutions and charts."""
import json
from textwrap import fill, indent

from .session_values import Chart, Resolution


_CHART_LOOKUP = "{name}.charts[0].strict_transform\n{name}.chart({chart_id}).coordinates"
_RESOLUTION = {
    "": (
        "A resolution holds the original polynomial, its charts and its exceptional dual complex. "
        "Type help {name}.charts or help {name}.dual_complex to inspect either one.",
        "{name}.polynomial\n{name}.charts\n{name}.dual_complex",
    ),
    "dual_complex": (
        "The complex records intersections of the exceptional divisors in the chosen mode. "
        "For an embedded plane curve, each vertex represents an exceptional curve in the blown-up "
        "plane and each edge an intersection point of two such curves.\nThe strict transform maps onto "
        "the original curve. When branch data are available, {name}.branches records their attachments. "
        "An arrowhead in a decorated resolution graph marks the divisor a branch meets. "
        "For the cusp y^2-x^3, that divisor is E3.\n"
        "In intrinsic mode the complex describes the exceptional locus on the resolved variety. "
        "For the intrinsic cusp its vertex represents a point on the smooth curve.\n"
        "Qbar is the algebraic closure of QQ. Atlas distinguishes components over this field. "
        "Check {name}.dual_complex.status for completeness.",
        "{name}.dual_complex.vertices\n{name}.dual_complex.cells",
    ),
    "dual_complex.vertices": (
        "The vertices are the exceptional divisors in the recorded complex. Each is a linked object. "
        "Its .id gives its label and .charts gives the charts where it has recorded equations.",
        "{name}.dual_complex.vertices",
    ),
    "dual_complex.cells": (
        "Each positive-dimensional cell records a component of an intersection. Its dimension, "
        "vertices and faces give the dimension, incident divisors and boundary. For an embedded "
        "plane curve the cells are edges. The zero-dimensional cells are listed in .vertices.",
        "{name}.dual_complex.cells",
    ),
    "charts": (
        "{name}.charts lists chart objects. Each prints its ID. Select one by index or pass a "
        'displayed ID to {name}.chart, for example {name}.chart("B3.u") for the cusp. '
        "Save a chart as C and type help C for its coordinates, map and equations.",
        _CHART_LOOKUP,
    ),
    "chart": (
        "{name}.chart(id) opens the chart with that ID in this resolution. "
        'For the cusp, {name}.chart("B3.u") is the chart recorded for its branch. '
        "Use {name}.charts to see the available IDs.",
        _CHART_LOOKUP,
    ),
    "branches": (
        "When branch data are available, {name}.branches lists the recorded branches of the strict "
        "transform over the origin. Each .chart locates a branch, and .meets gives the exceptional "
        "divisors it meets there. For the embedded cusp the branch meets E3 in B3.u.",
        "{name}.branches\n{name}.branches[0].chart\n{name}.branches[0].meets",
    ),
}
_CHART = {
    "": (
        "A chart is a local coordinate patch of the resolution. Its ID is its display name. "
        "Use its fields to inspect the local equations and the map to the original space.",
        "{name}.coordinates\n{name}.strict_transform\n{name}.map\n{name}.divisors",
    ),
    "id": (
        "This is the chart's ID in its resolution. Pass it to the resolution's chart(id) method "
        "to retrieve the same chart.",
        "{name}.id",
    ),
    "coordinates": (
        "The generators of the chart's coordinate ring. Use them to write a polynomial in this chart.",
        "{name}.coordinates\n{name}.coordinates[0].parent",
    ),
    "strict_transform": (
        "The local equation of the strict transform in this chart, or a list of equations when "
        "several are needed. For the plane cusp in B3.u it is v-1, so the strict transform is v=1.",
        "{name}.strict_transform",
    ),
    "map": (
        "The map to the original coordinates. Each key is an original coordinate name and its "
        "value is the corresponding polynomial in this chart. The ring map {name}.pullback_map "
        "carries its domain, codomain, and ordered images.",
        "{name}.map",
    ),
    "pullback_map": (
        "The ring homomorphism from the original polynomial ring to this chart. "
        "Apply it to a polynomial in its domain to substitute the chart coordinates.",
        "{name}.pullback_map.domain\n{name}.pullback_map.codomain\n{name}.pullback_map.images",
    ),
    "parent": (
        "The ring of functions on the chart. The defining equations of its ambient presentation "
        "vanish in this ring.",
        "{name}.parent\n{name}.parent.coordinates",
    ),
    "ambient": (
        "The polynomial ring in which the chart's defining relations are written. "
        "Chart functions belong to {name}.parent.",
        "{name}.ambient.coordinates",
    ),
    "divisors": (
        "The exceptional divisors with a recorded local equation or ideal in this chart. "
        "For a listed divisor E, E.ideal_in({name}) gives its local generators.",
        "{name}.divisors\nfor E in {name}.divisors: print(E.id, E.ideal_in({name}))",
    ),
    "pullback": (
        "{name}.pullback(f) substitutes this chart's coordinate map into a polynomial f in the "
        "original ring. Pulling back the original equation gives the total transform.",
        "{name}.pullback({source})",
    ),
}
_PAGES = {Resolution: _RESOLUTION, Chart: _CHART}
_GENERIC = {"resolution": Resolution, "chart": Chart}


def resolution_topics(variables):
    """List documented topics without reading any chart equations."""
    owners = {**_GENERIC, **{name: type(value) for name, value in variables.items() if type(value) in _PAGES}}
    return {f"{name}.{field}" if field else name for name, kind in owners.items() for field in _PAGES[kind]}


def _resolution_example(field, value, description, example):
    # With no saved value, the example setup resolves the embedded cusp.
    if field == "dual_complex" and (value is None or value._branches):
        example += "\n{name}.branches[0].meets"
    elif field == "dual_complex.vertices" and (value is None or value._divisors):
        example += "\n{name}.dual_complex.vertices[0].charts"
    elif field == "branches" and value is not None and not value._branches:
        if "branches" in value._record:
            return "The recorded branch list for this resolution is empty.", "{name}.branches"
        return ("This resolution has no recorded branch data. Inspect its recorded charts for local "
                "strict-transform equations.", "{name}.charts")
    elif field in ("chart", "charts") and value is not None and not value._charts:
        return "The recorded chart list for this resolution is empty.", "{name}.charts"
    return description, example


def resolution_help(topic, variables):
    root, _, field = topic.partition(".")
    value = variables.get(root)
    kind = type(value) if type(value) in _PAGES else _GENERIC.get(root)
    entry = _PAGES.get(kind, {}).get(field)
    if entry is None:
        return None
    live = type(value) is kind
    if kind is Resolution:
        entry = _resolution_example(field, value if live else None, *entry)
    description, example = entry
    name, setup, source, chart_id = root, "", "1", "B3.u"
    if not live:
        name = "R" if kind is Resolution else "C"
        setup = 'x, y = coordinates("x", "y")\nR = resolve(y^2-x^3)\n'
        if kind is Chart:
            setup += 'C = R.chart("B3.u")\n'
            source = "R.polynomial"
    elif kind is Chart and field == "pullback":
        owner = next((n for n, obj in variables.items() if obj is value._resolution), None)
        if owner:
            source = f"{owner}.polynomial"
    elif kind is Resolution and field in ("chart", "charts") and value._charts:
        chart_id = next(iter(value._charts))
    commands = setup + example.format(name=name, source=source, chart_id=json.dumps(chart_id, ensure_ascii=False))
    paragraphs = "\n".join(fill(line, width=80, break_long_words=False, break_on_hyphens=False)
                           for line in description.format(name=name).splitlines())
    return f"{topic}\n{paragraphs}\n\nExample\n{indent(commands, '  ')}"
