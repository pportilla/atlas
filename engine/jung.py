"""Jung's method for surface singularities: transport to the native resolver.

The native program ``native/build/jung`` computes everything: the projection,
the discriminant, its embedded resolution, the quasi-ordinary normalization
over every crossing, the Hirzebruch-Jung strings, the resolution graph, the
canonical cycle and the invariants. This module parses the input, runs the
program in a forked helper and exposes its record as a read-only value.
"""
from copy import deepcopy
from fractions import Fraction
import json
import os
import subprocess

from .polynomial import InputError, LimitError, parse_expression, rational_text


class JungDeclined(InputError):
    """The surface is outside the method's scope, for example a non-isolated singularity."""


def jung_record(expression, variables, *, timeout=None):
    """Run the native Jung resolver on a polynomial in three variables and return its record."""
    variables = tuple(variables)
    if len(variables) != 3:
        raise InputError("Jung's method resolves surfaces: give exactly three variables.")
    poly = parse_expression(expression, variables)
    payload = {"terms": [[*e, rational_text(c)] for e, c in poly.items()], "variables": list(variables)}
    from . import native_process
    try:
        done = native_process.run("jung", input=json.dumps(payload), timeout=timeout,
                                  env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    except FileNotFoundError:
        raise RuntimeError("The native Jung resolver is not built. Run make.") from None
    except subprocess.TimeoutExpired:
        raise LimitError(f"Jung's method did not finish within {timeout} seconds.") from None
    if done.returncode:
        raise LimitError(f"The native Jung resolver stopped with status {done.returncode}.")
    try:
        record = json.loads(done.stdout)
    except ValueError:
        raise RuntimeError("The native Jung resolver did not return JSON.") from None
    if "error" in record:
        kind = record.get("kind")
        if kind == "declined":
            raise JungDeclined(record["error"])
        if kind == "input":
            raise InputError(record["error"])
        raise LimitError(record["error"])
    record["variables"] = list(variables)
    return record


class JungResolution:
    """A resolution of a surface singularity computed by Jung's method.

    The fields describe the minimal good resolution reached from Jung's
    resolution by contracting smooth rational (-1)-curves. ``jung_graph``
    keeps the graph before the contractions.
    """
    _members = ("curves", "edges", "K2", "euler_characteristic", "geometric_genus", "canonical_cycle",
                "milnor_number_from_laufer", "jung_graph", "projection", "discriminant", "status", "summary",
                "record")
    _fields = _members     # completion lists these without evaluating anything

    def __init__(self, record, source_space=None):
        self._record = record
        self._source_space = source_space

    def session_attribute(self, name):
        if name in self._members:
            return getattr(self, name)
        raise ValueError(f"JungResolution has no field {name!r}; available: {', '.join(self._members)}.")

    @property
    def _minimal(self):
        return self._record["minimal_good_resolution"]

    @property
    def curves(self):
        return [{"id": v["id"], "genus": v["genus"], "self_intersection": v["self_intersection"],
                 "kind": v["kind"], "over": v["over"]} for v in self._minimal["vertices"]]

    @property
    def edges(self):
        return [(*e["vertices"], e["points"]) for e in self._minimal["edges"]]

    @property
    def K2(self):
        return Fraction(self._minimal["invariants"]["K2"])

    @property
    def euler_characteristic(self):
        return self._minimal["invariants"]["euler_characteristic"]

    @property
    def geometric_genus(self):
        return self._record["invariants"].get("geometric_genus")

    @property
    def milnor_number_from_laufer(self):
        return self._record["invariants"].get("milnor_number_from_laufer")

    @property
    def canonical_cycle(self):
        ids = [v["id"] for v in self._minimal["vertices"]]
        return {i: Fraction(k) for i, k in zip(ids, self._minimal["invariants"]["canonical_cycle"])}

    @property
    def jung_graph(self):
        return deepcopy(self._record["graph"])

    @property
    def projection(self):
        return deepcopy(self._record.get("projection"))

    @property
    def discriminant(self):
        return deepcopy(self._record.get("discriminant"))

    @property
    def status(self):
        return self._record["status"]

    @property
    def summary(self):
        return self._record["summary"]

    @property
    def record(self):
        return deepcopy(self._record)

    def __repr__(self):
        lines = [f"Jung resolution of {self._record['input']}: {self.summary}"]
        for curve in self.curves:
            lines.append(f"  {curve['id']}: genus {curve['genus']}, self-intersection {curve['self_intersection']}")
        if self.edges:
            lines.append("  meets: " + ", ".join(f"{a}—{b}" + (f" ({n} points)" if n > 1 else "")
                                                 for a, b, n in self.edges))
        return "\n".join(lines)


_DOC = "docs/jung-method.md#resolve-a-surface-with-jungs-method"
_FIELDS_DOC = "docs/jung-method.md#fields-of-a-jung-resolution"
_CARDS = {
    "": ("{n} is a Jung resolution of {source} in {ring}.", "{n}.curves\n{n}.edges\n{n}.geometric_genus", _DOC),
    "curves": ("{n}.curves lists the exceptional curves of the minimal good resolution with genus and self-intersection.",
               "{n}.curves\nlen({n}.curves)", _FIELDS_DOC),
    "edges": ("{n}.edges lists pairs of curves that meet and the number of points where they meet.",
              "{n}.edges", _FIELDS_DOC),
    "K2": ("{n}.K2 is the self-intersection of the canonical cycle on the minimal good resolution.", "{n}.K2", _FIELDS_DOC),
    "euler_characteristic": ("{n}.euler_characteristic is the topological Euler characteristic of the exceptional set.",
                             "{n}.euler_characteristic", _FIELDS_DOC),
    "geometric_genus": ("{n}.geometric_genus is p_g = dim O_X/I for the adjoint ideal I of the resolution.",
                        "{n}.geometric_genus", _FIELDS_DOC),
    "canonical_cycle": ("{n}.canonical_cycle gives the coefficient of each curve in K = sum k_i E_i.",
                        "{n}.canonical_cycle", _FIELDS_DOC),
    "milnor_number_from_laufer": ("{n}.milnor_number_from_laufer is 12 p_g + K^2 + chi - 1 (Laufer's formula).",
                                  "{n}.milnor_number_from_laufer", _FIELDS_DOC),
    "jung_graph": ("{n}.jung_graph is the graph from Jung's method before (-1)-curves are contracted.",
                   "{n}.jung_graph", _FIELDS_DOC),
    "projection": ("{n}.projection records the linear change of coordinates and the fibre coordinate.",
                   "{n}.projection", _FIELDS_DOC),
    "discriminant": ("{n}.discriminant lists the factors of the discriminant through the origin.",
                     "{n}.discriminant", _FIELDS_DOC),
    "status": ("{n}.status is the outcome of the calculation.", "{n}.status", _DOC),
    "summary": ("{n}.summary describes the resolution in one sentence.", "{n}.summary", _DOC),
    "record": ("{n}.record is the complete native record.", "{n}.record", _FIELDS_DOC),
}


def jung_help(topic, variables, ring_name=None):
    """Usage card for a saved Jung resolution or one of its fields, from the saved record only."""
    from .help_card import usage_card
    root, _, field = topic.partition(".")
    value = variables.get(root)
    if not isinstance(value, JungResolution) or field not in _CARDS:
        return None
    summary, examples, reference = _CARDS[field]
    space = value._source_space
    ring = (ring_name(space) if ring_name and space is not None else
            "QQ[" + ", ".join(value._record.get("variables", ())) + "]")
    context = dict(n=root, source=value._record.get("input", "its input"), ring=ring)
    return usage_card(summary.format(**context), examples.format(**context), reference)
