"""Linked, read-only views of the solver's resolution record for Atlas scripts."""
from functools import cached_property

from .session_polynomial import _CoordinateSpace
from .session_pullback import chart_polynomial, chart_space, pullback


class _RecordView:
    # Only these operations and fields actually present in the public result
    # are available to scripts. Python implementation attributes stay private.
    _members = ()

    def __init__(self, record):
        self._record = record

    def session_attribute(self, name):
        if name in self._members:
            return getattr(self, name)
        if not name.startswith("_") and name in self._record:
            return self._record[name]
        raise ValueError(f"{type(self).__name__} has no recorded field {name!r}.")


class _Component(_RecordView):
    def __init__(self, resolution, record):
        super().__init__(record)
        self._resolution = resolution

    @property
    def id(self):
        return self._record["id"]

    def __repr__(self):
        return self.id


class Divisor(_Component):
    _members = ("charts", "ideal_in", "equation_in")

    @property
    def charts(self):
        return [chart for chart in self._resolution.charts if self.id in chart._exceptional]

    def ideal_in(self, chart):
        """The generators supplied by this chart, in its own coordinates."""
        if not isinstance(chart, Chart):
            raise ValueError("Pass a chart of this resolution.")
        if chart._resolution is not self._resolution:
            raise ValueError("The divisor and chart belong to different resolutions.")
        component = chart._exceptional.get(self.id)
        if component is None:
            raise ValueError(f"{self.id} has no recorded local equation in {chart.id}.")
        if "ideal" in component:
            return [chart._polynomial(g) for g in component["ideal"]]
        if "equation" in component:
            return [chart._polynomial(component["equation"])]
        raise ValueError(f"No local ideal was recorded for {self.id} in {chart.id}.")

    def equation_in(self, chart):
        equations = self.ideal_in(chart)
        if len(equations) != 1:
            raise ValueError(f"{self.id} has {len(equations)} recorded generators in {chart.id}. "
                             "Use ideal_in(chart) to inspect them.")
        return equations[0]


class Chart(_Component):
    _members = ("coordinates", "divisors", "pullback", "strict_transform", "map")

    def __init__(self, resolution, record):
        super().__init__(resolution, record)
        self._exceptional = {item["id"]: item for item in record.get("exceptional", ())}

    @cached_property
    def _space(self):
        return chart_space(self._resolution._record, self._record)

    def _polynomial(self, expression):
        return chart_polynomial(expression, self._space)

    @property
    def coordinates(self):
        return self._space.coordinates

    def _equations(self, value):
        if isinstance(value, list):
            return [self._polynomial(g) for g in value]
        return self._polynomial(value)

    @property
    def strict_transform(self):
        equations = self._equations(self._record["strict_transform"])
        if isinstance(equations, list) and len(equations) == 1:
            return equations[0]
        return equations

    @property
    def map(self):
        return {name: self._polynomial(image) for name, image in self._record["map"].items()}

    def pullback(self, polynomial):
        return pullback(polynomial, self._resolution._source_space, self._record, self._space)

    def _local_component(self, record):
        component = dict(record)
        for name in ("ideal", "ambient_ideal", "root_equation"):
            if name in record:
                component[name] = self._equations(record[name])
        if "equation" in record:
            # A displayed multi-generator equation joins the ideal by commas.
            ideal = component.get("ideal")
            if ideal is None:
                component["equation"] = self._polynomial(record["equation"])
            else:
                component["equation"] = ideal[0] if len(ideal) == 1 else ideal
        return component

    def session_attribute(self, name):
        if name in self._record:
            if name in ("ambient_ideal", "ambient_total_transform", "ambient_exceptional", "pullback_factor",
                        "strict_transform_on_exceptional", "reduced_strict_transform",
                        "reduced_strict_transform_on_exceptional", "pivot_generator", "projective_coordinates"):
                return self._equations(self._record[name])
            if name == "map_to_parent":
                return {key: self._polynomial(image) for key, image in self._record[name].items()}
            if name in ("ambient_components", "exceptional", "components", "strict_transform_components"):
                return [self._local_component(component) for component in self._record[name]]
        return super().session_attribute(name)

    @property
    def divisors(self):
        # Match IDs from the chosen mode's exceptional loci. In intrinsic
        # mode the ambient components can differ from the actual divisors.
        return [self._resolution.divisor(name) for name in self._exceptional]


class Branch(_Component):
    _members = ("chart", "meets")

    @property
    def chart(self):
        return self._resolution.chart(self._record["chart"])

    @property
    def meets(self):
        resolution = self._resolution
        if resolution._record["mode"] == "intrinsic":
            # For a smooth strict curve the exceptional locus consists of
            # points. Its recorded point for this branch has the branch ID.
            if self.id not in self.chart._exceptional or self.id not in resolution._divisors:
                raise ValueError(f"No intrinsic divisor attachment was identified for {self.id}.")
            return [resolution.divisor(self.id)]
        return [resolution.divisor(name) for name in self._record["ambient_components"]]


class Resolution(_RecordView):
    _members = ("coordinates", "divisors", "charts", "branches", "divisor", "chart")

    def __init__(self, record, source_space=None):
        super().__init__(record)
        self._source_space = source_space or _CoordinateSpace.named(record["variables"])
        self._charts = {c["id"]: Chart(self, c) for c in record["charts"]}
        vertices = record["dual_complex"]["vertices"]
        self._divisors = {v["id"]: Divisor(self, v) for v in vertices}
        self._branches = [Branch(self, b) for b in record.get("branches", ())]

    def __repr__(self):
        record = self._record
        return (f"Resolution({record['polynomial']}, mode={record['mode']}, "
                f"status={record['status']}, charts={len(self._charts)}, divisors={len(self._divisors)})")

    @property
    def coordinates(self):
        return self._source_space.coordinates

    @property
    def charts(self):
        return list(self._charts.values())

    @property
    def divisors(self):
        return list(self._divisors.values())

    @property
    def branches(self):
        if "branches" not in self._record:
            raise ValueError("This resolution has no recorded branch data.")
        return list(self._branches)

    def chart(self, id):
        if not isinstance(id, str) or id not in self._charts:
            raise ValueError(f"No chart with ID {id!r} in this resolution.")
        return self._charts[id]

    def divisor(self, id):
        if not isinstance(id, str) or id not in self._divisors:
            raise ValueError(f"No divisor with ID {id!r} in this resolution's dual complex.")
        return self._divisors[id]
