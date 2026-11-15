"""Colour prompt input, displayed values, help cards and hints by meaning.

Each word takes a role from engine/cli_style.py. A registered function, a
saved name or a prompt command is coloured once Atlas recognises it, so a
correct name stands apart from a typo. Highlighting runs only in a live
terminal; scripts, pipes and NO_COLOR sessions keep plain text.
"""
import keyword
import re
import sys

from .cli_style import paint

_EXPRESSION = re.compile(r"""
    (?P<comment>\#.*)
  | (?P<string>[rRbBfFuU]{0,2}(?:"(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?))
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<word>[A-Za-z_]\w*)
  | (?P<other>\s+|.)
""", re.VERBOSE)
# Shell commands also accept options and bare words such as file names.
_COMMAND = re.compile(r"""
    (?P<comment>\#.*)
  | (?P<string>"(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?)
  | (?P<option>(?<!\S)--?[A-Za-z][\w-]*)
  | (?P<word>:?[A-Za-z_][\w.-]*)
  | (?P<number>\d+(?:\.\d*)?)
  | (?P<other>\s+|.)
""", re.VERBOSE)
_TARGETS = re.compile(r"\s*[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*\s*=(?!=)")
_OPEN, _CLOSE = "([{", ")]}"
_LONGEST_VALUE = 20000  # Longer values print as they are.

# Words in displayed values that report a verdict.
_SETTLED = frozenset({"True", "resolved", "complete", "verified", "certified", "finite"})
_UNSETTLED = frozenset({"False", "failed", "incomplete", "unresolved", "inconclusive", "infinite"})


def live_terminal():
    """True when the prompt reads from and echoes to this process's own terminal."""
    return (sys.stdin is sys.__stdin__ and sys.stdout is sys.__stdout__
            and sys.stdin.isatty() and sys.stdout.isatty())


def expression_pieces(source, *, names=(), functions=(), verdicts=False):
    """Split Atlas source or a displayed value into (text, role) pieces."""
    pieces = []
    targets = _TARGETS.match(source)
    targets_end = targets.end() - 1 if targets else 0
    depth = 0
    previous = ""
    for match in _EXPRESSION.finditer(source):
        kind, text = match.lastgroup, match.group()
        role = None
        if kind in ("comment", "string", "number"):
            role = kind
        elif kind == "word":
            following = source[match.end():].lstrip()
            if verdicts and text in _SETTLED:
                role = "ok"
            elif verdicts and text in _UNSETTLED:
                role = "false"
            elif text == "True":
                role = "ok"
            elif text == "False":
                role = "false"
            elif text == "None":
                role = "muted"
            elif keyword.iskeyword(text):
                role = "keyword"
            elif previous == ".":
                role = "field"
            elif match.start() < targets_end or text in names:
                role = "name"
            elif depth and following.startswith("=") and not following.startswith("=="):
                role = "field"  # a keyword argument or a recorded attribute
            elif text in functions:
                role = "function"
        else:
            if text in _OPEN:
                depth += 1
            elif text in _CLOSE:
                depth = max(0, depth - 1)
        pieces.append((text, role))
        if not text.isspace():
            previous = text
    return pieces


def command_pieces(source, *, commands=(), topics=()):
    """Split a shell command such as help resolve or :save run.atlas into pieces."""
    pieces = []
    words = 0
    head = ""
    for match in _COMMAND.finditer(source):
        kind, text = match.lastgroup, match.group()
        role = kind if kind in ("comment", "string", "number", "option") else None
        if kind == "word":
            words += 1
            if words == 1 and text in commands:
                role, head = "command", text
            elif words == 2 and head == "help":
                role = "function" if text in topics else None
            elif words == 2 and head == "algebra":
                role = "command"
        pieces.append((text, role))
    return pieces


def paint_pieces(pieces, color=True):
    return "".join(paint(text, role, color) for text, role in pieces)


def repaint_entry(stream, prompt_pieces, line, pieces, columns):
    """Redraw the line the terminal just echoed, with its words in colour.

    The cursor sits at the start of the row below the entry. A line that wraps
    or holds tabs or wide characters is left as the terminal drew it."""
    width = sum(len(text) for text, _ in prompt_pieces) + len(line)
    if not line.strip() or width >= columns or not (line.isascii() and line.isprintable()):
        return
    stream.write("\0337\033[1A\r" + paint_pieces(prompt_pieces) + paint_pieces(pieces) + "\033[K\0338")
    stream.flush()


def paint_value(text, *, names=(), functions=(), color=True):
    """Colour a displayed value, keeping verdicts such as resolved or False visible."""
    if not color or len(text) > _LONGEST_VALUE:
        return text
    return "\n".join(paint_pieces(expression_pieces(row, names=names, functions=functions, verdicts=True))
                     for row in text.split("\n"))


def assigned_names(source):
    """The names a statement such as x, y = coordinates("x", "y") saves."""
    targets = _TARGETS.match(source)
    return set(re.findall(r"[A-Za-z_]\w*", targets.group()[:-1])) if targets else set()


def _signature_pieces(row):
    name, rest = row.split("(", 1)
    pieces = [(name, "function")]
    for text, role in expression_pieces("(" + rest):
        pieces.append((text, "field" if role is None and text.isidentifier() else role))
    return pieces


def paint_card(card, *, names=(), functions=(), commands=(), color=True):
    """Colour a help card: its signature, its prose and its example."""
    if not color:
        return card
    known = set(names)
    lines = []
    for i, row in enumerate(card.split("\n")):
        if i == 0 and "(" in row and row.split("(", 1)[0].isidentifier():
            lines.append(paint_pieces(_signature_pieces(row)))
        elif row in ("Example", "Prompt commands", "Session functions", "Command form"):
            lines.append(paint(row, "label"))
        elif row.startswith("  "):
            statement = row.strip()
            first = statement.split(maxsplit=1)[0] if statement else ""
            if first in commands and not statement.startswith(first + "("):
                pieces = command_pieces(row, commands=commands)
            else:
                known |= assigned_names(statement)
                pieces = expression_pieces(row, names=known, functions=functions)
            lines.append(paint_pieces(pieces))
        else:
            lines.append(row)
    return "\n".join(lines)


_QUOTED = re.compile(r"'([^']+)'")
_HELP = re.compile(r"\b(help(?: [A-Za-z_:.][\w:.-]*)?|[\w-]+ --help)\b")


def paint_hint(text, *, commands=(), topics=(), names=(), functions=(), color=True, rest="muted"):
    """Set a hint in muted text with the commands and names it suggests in colour."""
    if not color:
        return text
    out = []
    last = 0
    for match in sorted([*_QUOTED.finditer(text), *_HELP.finditer(text)], key=lambda m: m.start()):
        if match.start() < last:
            continue
        out.append(paint(text[last:match.start()], rest))
        inner = match.group(1)
        first = inner.split(maxsplit=1)[0] if inner.strip() else ""
        if first in commands:
            pieces = command_pieces(inner, commands=commands, topics=topics)
        else:
            pieces = expression_pieces(inner, names=names, functions=functions)
        quote = paint("'", "muted") if match.group().startswith("'") else ""
        out.append(quote + paint_pieces(pieces) + quote)
        last = match.end()
    out.append(paint(text[last:], rest))
    return "".join(out)
