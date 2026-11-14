"""The interactive workspace header and its expanded :workspace view."""

from collections import Counter
from fractions import Fraction
import os
from unicodedata import combining, east_asian_width

from .session_algebra import Ideal, _RingValue, _Result
from .session_polynomial import Polynomial, QQ
from .session_values import Chart, Resolution


def _ring_label(space):
    if isinstance(space, Chart):
        chart = space._record
        source = space._resolution._record
        dual = source.get("dual_complex", {})
        marked = (chart.get("coefficient_field") or chart.get("weight_field")
                  or source.get("coefficient_field", {}).get("generator") or source.get("algebraic_numbers")
                  or source.get("root_families") or dual.get("root_families") or dual.get("root_definition"))
        field = f"chart {space.id}" if marked else "QQ"
        names, quotient = chart["coordinates"], bool(chart.get("ambient_ideal"))
    else:
        field = "QQ" if space.coeffs is QQ else f"QQ[{space.coeffs.parameter.name}]"
        names, quotient = space.names, space.has_relations
    ring = f"{field}[{', '.join(names)}]" if names else field
    return ring + (" (quotient)" if quotient else "")


def _space(value):
    if isinstance(value, (Polynomial, _RingValue)):
        return value._space
    if isinstance(value, Resolution):
        return value._source_space
    if isinstance(value, Chart):
        # A chart's cached space is authoritative. Its construction can involve
        # a number-field calculation, so use only recorded data until then.
        if "_space" in vars(value):
            return value._space
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], Polynomial):
        space = value[0]._space
        if all(isinstance(item, Polynomial) and item._space is space for item in value):
            return space
    return None


def _preview(value):
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
            spaces.setdefault(space, name)
        preview = _preview(value)
        bindings.append(name if preview == name else f"{name} = {preview}")

    rings = [_ring_label(space) for space in spaces]
    counts = Counter(rings)
    rings = [f"{ring} ({name})" if counts[ring] > 1 else ring for ring, name in zip(rings, spaces.values())]
    return rings, bindings


def print_workspace(variables):
    rings, bindings = workspace(variables)
    print("Rings     " + ("  ·  ".join(rings) if rings else "—"))
    print("Variables " + ("\n          ".join(bindings) if bindings else "—"))


def _cells(char):
    return 0 if combining(char) else 2 if east_asian_width(char) in ("F", "W") else 1


def _fit(text, width):
    cells = 0
    for i, char in enumerate(text):
        cells += _cells(char)
        if cells > width:
            return text[:i]
    return text


def _header_row(label, entries, width):
    prefix = label + " " if width < 26 else f"{label:<10}"
    if not entries:
        return prefix + "—"
    separator = " " if width < 26 else "  ·  "
    # Fill from the right so the newest bindings survive overflow.
    remaining = width - len(prefix)
    first = len(entries)
    for i in range(first - 1, -1, -1):
        remaining -= sum(map(_cells, entries[i])) + (len(separator) if first < len(entries) else 0)
        if remaining < (len(str(i)) + 1 + len(separator) if i else 0):
            break
        first = i
    first = min(first, len(entries) - 1)
    body = (f"+{first}{separator}" if first else "") + separator.join(entries[first:])
    row = prefix + body
    if first and len(prefix) + len(str(first)) + 1 + len(separator) + 1 >= width:
        return _fit(prefix + f"+{len(entries)}", width)
    return row if _fit(row, width) == row else _fit(row, width - 1) + "…"


class SessionPanel:
    """Reserve two terminal rows while keeping readline and command output below them."""

    def __init__(self, stream):
        self.stream = stream
        self.size = None
        self.rows = None
        self.enabled = stream.isatty() and os.environ.get("TERM", "") not in ("", "dumb", "unknown")
        self.color = "NO_COLOR" not in os.environ

    def __enter__(self):
        self.refresh({})
        return self

    def refresh(self, variables):
        if not self.enabled:
            return
        try:
            size = os.get_terminal_size(self.stream.fileno())
        except (OSError, ValueError):
            self.enabled = False
            return
        if size.columns < 16 or size.lines < 6:
            self.close()
            return
        resized = size != self.size
        if self.size is None:
            self.stream.write("\033[?1049h\033[2J\033[3;r\033[3;1H")
        elif resized:
            self.stream.write("\0337\033[3;r\0338")
        self.size = size
        rings, bindings = workspace(variables)
        rows = (_header_row("Rings", rings, size.columns - 1),
                _header_row("Variables", bindings, size.columns - 1))
        if resized or rows != self.rows:
            self.rows = rows
            first, second = rows
            if self.color:
                first = first.replace("Rings", "\033[1;36mRings\033[0m", 1)
                second = second.replace("Variables", "\033[1;36mVariables\033[0m", 1)
            self.stream.write("\0337\033[1;1H\033[2K" + first + "\033[2;1H\033[2K" + second + "\0338")
            self.stream.flush()

    def close(self):
        if self.size is not None:
            self.stream.write("\033[r\033[?1049l")
            self.stream.flush()
            self.size = self.rows = None

    def __exit__(self, *exc):
        self.close()
