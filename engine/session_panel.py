"""The interactive workspace header and its expanded :workspace view."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from inspect import isroutine
import os
import shutil
import signal
import sys
import threading
from unicodedata import combining, east_asian_width

from .cli_highlight import expression_pieces
from .cli_style import paint, use_color
from .session_algebra import Ideal, _RingValue, _Result
from .session_maps import RingHomomorphism
from .session_polynomial import Polynomial, _Coordinates, _CoordinateSpace
from .session_quotient import QuotientElement, QuotientRing
from .session_values import Branch, Chart, Divisor, DualComplex, Resolution


def _ring_label(space, references, ideal_names, paths):
    if isinstance(space, QuotientRing):
        ambient = references[space.ambient]
        ideal = ideal_names.get(space.defining_ideal)
        if ideal and isinstance(space.defining_ideal.parent, QuotientRing):
            return f"{paths[space.defining_ideal.parent]}/{ideal}"
        relations = space._record["result"]["relations"]
        if len(relations) > 3 or sum(map(len, relations)) > 36:
            return f"{ambient}/{ideal}" if ideal else f"{ambient} (quotient)"
        return f"{ambient}/({', '.join(relations) or '0'})"
    if isinstance(space, _CoordinateSpace):
        return str(space)
    chart = space._record
    source = space._resolution._record
    dual = source.get("dual_complex", {})
    marked = (chart.get("coefficient_field") or chart.get("weight_field")
              or source.get("coefficient_field", {}).get("generator") or source.get("algebraic_numbers")
              or source.get("root_families") or dual.get("root_families") or dual.get("root_definition"))
    field = f"chart {space.id}" if marked else "QQ"
    names, quotient = chart["coordinates"], bool(chart.get("ambient_ideal"))
    ring = f"{field}[{', '.join(names)}]" if names else field
    return ring + (" (quotient)" if quotient else "")


def _space(value):
    if isinstance(value, (_CoordinateSpace, QuotientRing)):
        return value
    if isinstance(value, (Polynomial, QuotientElement, _Coordinates)):
        return value.parent
    if isinstance(value, _RingValue):
        return value._space
    if isinstance(value, Resolution):
        return value._source_space
    if isinstance(value, Chart):
        # A chart's cached space is authoritative. Its construction can involve
        # a number-field calculation, so use only recorded data until then.
        if "_space" in vars(value):
            return value._space
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], (Polynomial, QuotientElement)):
        space = value[0].parent
        if all(isinstance(item, (Polynomial, QuotientElement)) and item.parent is space for item in value):
            return space
    return None


def _coordinate(value):
    """Recognize the actual saved generators, even if their reduced forms coincide."""
    if isinstance(value, QuotientElement):
        for index, coordinate in enumerate(vars(value.parent).get("gens", ())):
            if coordinate is value:
                return f"[{value.parent.ambient.generators[index].name}]"
    if isinstance(value, Polynomial):
        for index, coordinate in enumerate(vars(value.parent).get("coordinates", ())):
            if coordinate is value:
                return value.parent.generators[index].name
    return None


def _describe(value):
    coordinate = _coordinate(value)
    if coordinate is not None:
        return "Coordinates", coordinate
    if isinstance(value, QuotientElement):
        representative = str(value)
        return "Residue classes", f"[{representative}]" if len(representative) <= 36 else "residue class"
    if isinstance(value, Polynomial):
        terms = value._value._terms
        if len(terms) <= 3 and all(len(p) <= 6 and all(k.bit_length() < 30 for k in p)
                                   and c.numerator.bit_length() < 80 and c.denominator.bit_length() < 80
                                   for p, c in terms):
            return "Polynomials", str(value)
        return "Polynomials", "polynomial"
    if isinstance(value, Ideal):
        return "Ideals", value.operation if isinstance(value, _Result) else "ideal"
    if isinstance(value, _Result):
        return "Results", value.operation
    if isinstance(value, Chart):
        return "Charts", value.id
    if isinstance(value, Resolution):
        return "Resolutions", "resolution"
    if isinstance(value, RingHomomorphism):
        return "Maps", "ring map"
    if isinstance(value, _Coordinates):
        names = value.parent.variables
        if len(names) > 6:
            return "Coordinate tuples", f"tuple[{len(names)}]"
        if isinstance(value.parent, QuotientRing):
            names = [f"[{name}]" for name in names]
        return "Coordinate tuples", "(" + ", ".join(names) + ("," if len(names) == 1 else "") + ")"
    if isinstance(value, (Branch, Divisor)):
        return ("Branches" if isinstance(value, Branch) else "Divisors"), value.id
    if isinstance(value, DualComplex):
        return "Dual complexes", "dual complex"
    if isroutine(value):
        return "Functions", value.__name__
    if isinstance(value, (bool, int, Fraction)):
        preview = str(value) if max(value.numerator.bit_length(), value.denominator.bit_length()) < 160 else "number"
        return "Scalars", preview
    if isinstance(value, str):
        return "Text", repr(value[:28]) + ("…" if len(value) > 28 else "")
    if isinstance(value, (list, tuple, dict)):
        return "Collections", f"{type(value).__name__}[{len(value)}]"
    return "Values", "None" if value is None else type(value).__name__.lower()


def _parent_reference(name, value):
    if isinstance(value, (Polynomial, QuotientElement, _Coordinates, Ideal, Chart)):
        return f"{name}.parent"
    if isinstance(value, Resolution):
        return f"{name}.polynomial.parent"
    if isinstance(value, (list, tuple)):
        return f"{name}[0].parent"
    return f"{name}.ambient"


@dataclass
class _Binding:
    name: str
    role: str
    preview: str
    parent: object

    def text(self, *, qualifier=None, expanded=False):
        if self.name == self.preview or (expanded and (self.role, self.preview) in
                                         (("Ideals", "ideal"), ("Resolutions", "resolution"))):
            return self.name
        if self.role == "Coordinates" and not isinstance(self.parent, QuotientRing):
            return f"{self.name} (generator {self.preview})"
        suffix = f" in {qualifier}" if qualifier is not None else ""
        return f"{self.name} = {self.preview}{suffix}"


@dataclass
class _RingGroup:
    aliases: list[str] = field(default_factory=list)
    members: list[_Binding] = field(default_factory=list)
    title: str = ""


@dataclass
class _Workspace:
    rings: dict[object, _RingGroup]
    bindings: list[_Binding]
    other: list[_Binding]
    references: dict[object, str]

    def header(self):
        rings = [ring.title + (" → " + ", ".join(b.name for b in ring.members) if ring.members else "")
                 for ring in self.rings.values()]
        bindings = []
        for binding in self.bindings:
            qualifier = None
            if isinstance(binding.parent, QuotientRing):
                reference = self.references[binding.parent]
                if reference != self.rings[binding.parent].title:
                    qualifier = reference
            bindings.append(binding.text(qualifier=qualifier))
        return rings, bindings


def workspace(variables):
    """Inventory the live bindings once. Looking at a ring must never force its construction."""
    rings, ideal_names, paths = defaultdict(_RingGroup), {}, {}
    bindings, other, maps = [], [], []
    for name, value in variables.items():
        parent = _space(value)
        group = rings[parent] if parent is not None else None
        if isinstance(value, (_CoordinateSpace, QuotientRing)):
            group.aliases.append(name)
            paths[value] = group.aliases[0]
            continue
        if group is not None:
            paths.setdefault(parent, _parent_reference(name, value))
        if isinstance(value, Ideal):
            ideal_names.setdefault(value, name)
        role, preview = _describe(value)
        binding = _Binding(name, role, preview, parent)
        (group.members if group is not None else other).append(binding)
        bindings.append(binding)
        if isinstance(value, RingHomomorphism):
            paths.setdefault(value.domain, name + ".domain")
            paths.setdefault(value.codomain, name + ".codomain")
            maps.append((binding, value))

    # Endpoints and ambient rings may be reachable only through a map or a
    # quotient. Give them usable references without adding synthetic bindings.
    for parent, path in tuple(paths.items()):
        if isinstance(parent, QuotientRing):
            paths.setdefault(parent.ambient, path + ".ambient")
    references, presentations = {}, {}
    # First name the ambient rings, then use those names to describe their quotients.
    for quotient in (False, True):
        labels = {parent: _ring_label(parent, references, ideal_names, paths) for parent in paths
                  if isinstance(parent, QuotientRing) == quotient}
        presentations.update(labels)
        counts = Counter(labels.values())
        for parent, label in labels.items():
            group = rings.get(parent)
            aliases = group.aliases if group is not None else ()
            names = aliases or ([paths[parent]] if counts[label] > 1 else [])
            references[parent] = names[0] if names else label
            if group is not None:
                group.title = " = ".join((*names, label))

    def endpoint(parent):
        reference = references[parent]
        if parent not in rings and reference != presentations[parent]:
            return f"{presentations[parent]} ({reference})"
        return reference

    for binding, mapping in maps:
        binding.preview = f"{endpoint(mapping.domain)} → {endpoint(mapping.codomain)}"
    return _Workspace(rings, bindings, other, references)


def _binding(entry, role, names=()):
    """Split "name = preview" so that the name can carry its own colour."""
    name, equals, preview = entry.partition(" = ")
    if not equals and " (generator " in name:
        name, _, suffix = name.partition(" ")
        return [(name, role), (" " + suffix, "muted")]
    return [(name, role), (equals, "muted"), *expression_pieces(preview, names=names, verdicts=True)]


def _ring(entry, names=()):
    """Split "QQ[x, y] → x, f" into the ring, its arrow and the names it holds."""
    ring, arrow, members = entry.partition(" → ")
    return [*expression_pieces(ring, names=names, functions={"QQ"}), (arrow, "muted"),
            *expression_pieces(members, names=names)]


def _names(variables):
    return {name for name in variables if name.isidentifier()}


def print_workspace(variables):
    inventory = workspace(variables)
    color = use_color(sys.stdout)
    known = _names(variables)
    width = shutil.get_terminal_size().columns

    def show(pieces):
        return "".join(paint(text, role, color) for text, role in pieces)

    def rows(members):
        roles = {}
        for binding in members:
            roles.setdefault(binding.role, []).append(binding.text(expanded=True))
        label_width = max(15, max(map(len, roles), default=0))
        for role, entries in roles.items():
            if width < 40:
                print("  " + paint(role, "label", color))
                prefix = "    "
            else:
                prefix = "  " + role.ljust(label_width) + " "
            indent = " " * len(prefix)
            line = paint(prefix, "label", color)
            used = len(prefix)
            for entry in entries:
                size = _width(entry)
                if used > len(prefix) and used + 2 + size > width:
                    print(line)
                    line, used = indent, len(indent)
                separator = ", " if used > len(prefix) else ""
                line += separator + show(_binding(entry, "name", known))
                used += len(separator) + size
            print(line)

    sections = [(group.title, group.members) for group in inventory.rings.values()]
    if inventory.other:
        sections.append(("Other values", inventory.other))
    for index, (title, members) in enumerate(sections):
        if index:
            print()
        print(show(_ring(title, known)) if members is not inventory.other else paint(title, "label", color))
        rows(members)
    if not sections:
        print("Workspace is empty.")


def _cells(char):
    return 0 if combining(char) else 2 if east_asian_width(char) in ("F", "W") else 1


def _width(text):
    return sum(map(_cells, text))


def _fit(text, width):
    cells = 0
    for i, char in enumerate(text):
        cells += _cells(char)
        if cells > width:
            return text[:i]
    return text


def _take(pieces, width):
    """Keep the leading (text, role) pieces that fit in width cells and report any cut."""
    kept = []
    for text, role in pieces:
        part = _fit(text, width)
        if part:
            kept.append((part, role))
        if part != text:
            return kept, True
        width -= _width(part)
    return kept, False


def _header_row(label, entries, width, role=None, names=()):
    """Fit a labelled row into width cells as (text, role) pieces."""
    narrow = width < 26
    prefix = label + " " if narrow else f"{label:<10}"
    if not entries:
        return [(prefix, "label"), ("—", "muted")]
    separator = " " if narrow else "  ·  "
    # Fill from the right so the newest bindings survive overflow.
    remaining = width - len(prefix)
    first = len(entries)
    for i in range(first - 1, -1, -1):
        remaining -= _width(entries[i]) + (len(separator) if first < len(entries) else 0)
        if remaining < (len(str(i)) + 1 + len(separator) if i else 0):
            break
        first = i
    first = min(first, len(entries) - 1)
    if first and len(prefix) + len(str(first)) + 1 + len(separator) + 1 >= width:
        return _take([(prefix, "label"), (f"+{len(entries)}", "muted")], width)[0]
    pieces = [(prefix, "label")] + ([(f"+{first}", "muted")] if first else [])
    for entry in entries[first:]:
        if len(pieces) > 1:
            pieces.append((separator, "muted"))
        pieces += _binding(entry, role, names) if role else _ring(entry, names)
    kept, cut = _take(pieces, width)
    return _take(pieces, width - 1)[0] + [("…", "muted")] if cut else kept


HEADER_ROWS = 4


class SessionPanel:
    """Pin a framed ring and variable summary above readline and command output."""

    top = HEADER_ROWS + 1  # The first row that scrolls.

    def __init__(self, stream):
        self.stream = stream
        self.size = None
        self.rows = None
        self._display_data = None
        self._last_assignment = None
        self._quotient_displayed = False
        self.variables = {}
        self.enabled = stream.isatty() and os.environ.get("TERM", "") not in ("", "dumb", "unknown")
        self.color = not os.environ.get("NO_COLOR")
        self._previous_resize = None

    def __enter__(self):
        self.refresh({})
        self._follow_resizes()
        return self

    def _follow_resizes(self):
        """Redraw the frame as soon as the window changes size, as zooming the font does.

        Without this the frame keeps its old width until the next prompt, and a
        narrower window wraps it. The handler goes in before readline loads, so
        readline's own handler calls it and still rewraps the line being edited.
        """
        if (not self.enabled or not hasattr(signal, "SIGWINCH") or "readline" in sys.modules
                or threading.current_thread() is not threading.main_thread()):
            return
        previous = signal.signal(signal.SIGWINCH, self._resized)
        # None stands for a handler set outside Python. Window changes are ignored by default.
        self._previous_resize = signal.SIG_DFL if previous is None else previous

    def _resized(self, signum, frame):
        try:
            self.refresh(self.variables)
        except RuntimeError:
            pass  # Output was being written. The next prompt redraws the frame.

    def refresh(self, variables, assigned=None):
        if not self.enabled:
            return
        self.variables = variables
        try:
            size = os.get_terminal_size(self.stream.fileno())
        except (OSError, ValueError):
            self.enabled = False
            return
        if size.columns < 16 or size.lines < HEADER_ROWS + 4:
            self.close()
            return
        resized = size != self.size
        if self.size is None:
            self.stream.write(f"\033[?1049h\033[2J\033[{self.top};r\033[{self.top};1H")
        elif resized:
            self.stream.write(f"\0337\033[{self.top};r\0338")
        self.size = size
        # A simple assignment changes only one binding. An updated value
        # with the same ring and preview leaves its entry on screen intact.
        if assigned is None:
            self._last_assignment = None
        else:
            value = variables[assigned]
            space, preview = _space(value), _describe(value)
            last = self._last_assignment
            # A named ideal can also be the denominator of a quotient. Its
            # identity matters there even when its ordinary preview stays "ideal".
            identity = value if (value is space or isinstance(value, RingHomomorphism)
                                 or (self._quotient_displayed and isinstance(value, Ideal))) else None
            self._last_assignment = assigned, space, preview, identity
            if (not resized and last and last[0] == assigned and last[1] is space
                    and last[2] == preview and last[3] is identity):
                return
        inventory = workspace(variables)
        rings, bindings = inventory.header()
        self._quotient_displayed = any(isinstance(parent, QuotientRing) for parent in inventory.references)
        # Assigning another ideal in the same ring often leaves both header
        # rows identical. Avoid laying out and coloring the same text again.
        if not resized and self._display_data == (rings, bindings):
            return
        self._display_data = rings, bindings
        inner = size.columns - 4
        known = _names(variables)
        rows = (_header_row("Rings", rings, inner, names=known),
                _header_row("Variables", bindings, inner, "name", known))
        plain = tuple("".join(text for text, _ in row) for row in rows)
        if resized or plain != self.rows:
            self.rows = plain
            frame = "".join(f"\033[{y};1H\033[2K{line}" for y, line in enumerate(self._frame(size.columns, rows), 1))
            self.stream.write("\0337" + frame + "\0338")
            self.stream.flush()

    def _frame(self, columns, rows):
        def ink(text, role):
            return paint(text, role, self.color)

        # The panel needs 16 columns, enough for the title in plain text.
        title = [("─ ", "frame"), ("Workspace", None), (" ", None)]
        fill = columns - 2 - sum(len(text) for text, _ in title)
        top = [("╭", "frame"), *title, ("─" * fill, "frame"), ("╮", "frame")]
        lines = ["".join(ink(text, role) for text, role in top)]
        side = ink("│", "frame")
        for row in rows:
            text = "".join(ink(text, role) for text, role in row)
            lines.append(f"{side} {text}{' ' * (columns - 4 - sum(_width(text) for text, _ in row))} {side}")
        lines.append(ink("╰" + "─" * (columns - 2) + "╯", "frame"))
        return lines

    def close(self):
        if self.size is not None:
            self.stream.write("\033[r\033[?1049l")
            self.stream.flush()
            self.size = self.rows = self._display_data = self._last_assignment = None

    def __exit__(self, *exc):
        if self._previous_resize is not None:
            signal.signal(signal.SIGWINCH, self._previous_resize)
            self._previous_resize = None
        self.close()
