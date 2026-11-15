"""The interactive workspace header and its expanded :workspace view."""

from fractions import Fraction
import os
import signal
import sys
import threading
from unicodedata import combining, east_asian_width

from .cli_highlight import expression_pieces
from .cli_style import paint, use_color
from .session_algebra import Ideal, _RingValue, _Result
from .session_polynomial import Polynomial, QQ, _CoordinateSpace
from .session_quotient import QuotientElement, QuotientRing
from .session_values import Chart, Resolution


def _ring_label(space):
    if isinstance(space, QuotientRing):
        relations = space._record["result"]["relations"]
        if len(relations) > 3 or sum(map(len, relations)) > 36:
            return f"{space._space} (quotient)"
        return f"{space._space}/({', '.join(relations) or '0'})"
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
    if isinstance(value, (Polynomial, QuotientElement)):
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


def _preview(value):
    if isinstance(value, _CoordinateSpace):
        return "ring"
    if isinstance(value, QuotientElement):
        return _preview(value.representative)
    if isinstance(value, Polynomial):
        if value._value.is_Symbol:
            return value._value.name
        terms = value._value._terms
        if len(terms) <= 3 and all(len(p) <= 6 and all(k.bit_length() < 30 for k in p)
                                   and c.numerator.bit_length() < 80 and c.denominator.bit_length() < 80
                                   for p, c in terms):
            return str(value)
        return "polynomial"
    if isinstance(value, _Result):
        return value.operation
    if isinstance(value, Ideal):
        return "ideal"
    if isinstance(value, (bool, int, Fraction)):
        return str(value) if max(value.numerator.bit_length(), value.denominator.bit_length()) < 160 else "number"
    if isinstance(value, str):
        return repr(value[:28]) + ("…" if len(value) > 28 else "")
    if isinstance(value, (list, tuple, dict)):
        return f"{type(value).__name__}[{len(value)}]"
    if value is QQ or value is None:
        return str(value)
    return type(value).__name__.lower()


def workspace(variables):
    spaces = {}
    bindings = []
    for name, value in variables.items():
        space = _space(value)
        if space is not None:
            spaces.setdefault(space, []).append(name)
        preview = _preview(value)
        bindings.append(name if preview == name else f"{name} = {preview}")

    rings = [f"{_ring_label(space)} → {', '.join(names)}" for space, names in spaces.items()]
    return rings, bindings


def _binding(entry, role, names=()):
    """Split "name = preview" so that the name can carry its own colour."""
    name, equals, preview = entry.partition(" = ")
    return [(name, role), (equals, "muted"), *expression_pieces(preview, names=names, verdicts=True)]


def _ring(entry, names=()):
    """Split "QQ[x, y] → x, f" into the ring, its arrow and the names it holds."""
    ring, arrow, members = entry.partition(" → ")
    return [*expression_pieces(ring, names=names, functions={"QQ"}), (arrow, "muted"),
            *expression_pieces(members, names=names)]


def _names(variables):
    return {name for name in variables if name.isidentifier()}


def print_workspace(variables):
    rings, bindings = workspace(variables)
    color = use_color(sys.stdout)
    known = _names(variables)

    def ink(text, role):
        return paint(text, role, color)

    names = ["".join(ink(text, role) for text, role in _binding(entry, "name", known)) for entry in bindings]
    rings = ["".join(ink(text, role) for text, role in _ring(entry, known)) for entry in rings]
    print(ink("Rings", "label") + "     " + ("\n          ".join(rings) if rings else ink("—", "muted")))
    print(ink("Variables", "label") + " " + ("\n          ".join(names) if names else ink("—", "muted")))


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
_HINT = (("help", "field"), ("  ·  ", "frame"), (":workspace", "field"), ("  ·  ", "frame"), ("quit", "field"))


class SessionPanel:
    """Pin a framed ring and variable summary above readline and command output."""

    top = HEADER_ROWS + 1  # The first row that scrolls.

    def __init__(self, stream):
        self.stream = stream
        self.size = None
        self.rows = None
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

    def refresh(self, variables):
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
        rings, bindings = workspace(variables)
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

        # The standing commands, then the title, give way as the terminal narrows.
        title = [("─ ", "frame"), ("Atlas", "brand"), (" ", None)] if columns >= 24 else []
        hint = [(" ", None), *_HINT, (" ─", "frame")] if columns >= 60 else []
        fill = columns - 2 - sum(len(text) for text, _ in title + hint)
        top = [("╭", "frame"), *title, ("─" * fill, "frame"), *hint, ("╮", "frame")]
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
            self.size = self.rows = None

    def __exit__(self, *exc):
        if self._previous_resize is not None:
            signal.signal(signal.SIGWINCH, self._previous_resize)
            self._previous_resize = None
        self.close()
