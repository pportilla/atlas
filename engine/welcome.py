"""The cartographic wordmark shown when opening Atlas from a terminal."""
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap

from .welcome_graph import graph_rows
from .welcome_curve import resolve_welcome_curve


WORDMARK = (
    "     ___   ______  __      ___    _____",
    "    /   | /_  __/ / /     /   |  / ___/",
    "   / /| |  / /   / /     / /| |  \\__ \\",
    "  / ___ | / /   / /___  / ___ | ___/ /",
    " /_/  |_|/_/   /_____/ /_/  |_|/____/",
)

SMALL_WORDMARK = (
    r"   /\   ___ |      /\    __",
    r"  /__\   |  |     /__\  (_",
    r" /    \  |  |___ /    \ __)",
)

GREETING = "Welcome to Atlas."


def folded_map(result, inside):
    """Fit the computed tree on the sheet, with clearance around the fold."""
    drawing = graph_rows(result, inside)
    fold = 3 if inside >= 20 else 2
    edge = inside - fold
    # A tall tree gets as much paper as it needs. Short trees sit in the
    # middle of the wide banner beside the wordmark and greeting.
    height = max(8 if inside >= 20 else 4, len(drawing) + 2)
    block_width = max(map(len, drawing))
    # Put the branch path in the lower middle of the page when space permits.
    top = (height - len(drawing)) // 2
    for i, row in enumerate(drawing):
        if ">" in row:
            top = max(1, min(2 * height // 3 - i, height - len(drawing) - 1))
            break
    left = (inside - block_width) // 2
    if left + block_width > edge:
        top = max(top, fold)
    height = max(height, top + len(drawing) + 1)

    rows = [" ." + "-" * edge + "."]
    for i in range(height):
        body = ""
        if top <= i < top + len(drawing):
            body = " " * left + drawing[i - top]
        if i < fold:
            crease = "_" if i == fold - 1 else " "
            rows.append(" |" + body.ljust(edge) + "|" + crease * i + "\\")
        else:
            rows.append(" |" + body.ljust(inside) + "|")
    rows.append(" '" + "-" * inside + "'")
    return rows


def render_welcome(columns, *, color, interactive, example=None):
    margin = "  " if columns >= 24 else ""
    width = max(1, columns - len(margin))
    lines = []

    def ink(text, code):
        return f"\033[{code}m{text}\033[0m" if color and code and text else text

    def paragraph(text, code=None):
        for row in textwrap.wrap(text, width, break_on_hyphens=False):
            lines.append(ink(row, code))

    def map_ink(row):
        if "|" not in row:
            return ink(row, "2")
        indent, _, sheet = row.partition("|")
        drawing, _, fold = sheet.rpartition("|")
        if color:
            drawing = "".join(ink(run, "1;33" if run[0] in "<>^v" else "36")
                              for run in re.findall(r"[<>^v]|[^<>^v]+", drawing))
        return ink(indent + "|", "2") + drawing + ink("|" + fold, "2")

    error = None
    try:
        polynomial, result = resolve_welcome_curve() if example is None else example
    except (ArithmeticError, OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        result = None
        error = str(exc)

    full_title_width = max(map(len, WORDMARK))
    gap = 3
    full = result is not None and width >= full_title_width + gap + 32
    inside = 29 if full else min(13, width - 3)
    # Even a portrait layout needs five columns to show the largest sampled tree.
    paper = folded_map(result, inside) if result is not None and inside >= 5 else []
    lettering = next((title for title in (WORDMARK, SMALL_WORDMARK)
                      if max(map(len, title)) <= width), ("ATLAS",))
    side_by_side = False
    if paper:
        for title in (WORDMARK, SMALL_WORDMARK):
            if max(map(len, title)) + gap + inside + 3 <= width:
                lettering = title
                side_by_side = True
                break

    if side_by_side:
        column = max(map(len, lettering)) + gap
        labels = ["", *lettering, "", GREETING] if full else list(lettering)
        for i, row in enumerate(paper):
            label = labels[i] if i < len(labels) else ""
            lines.append(ink(label, "1") + " " * (column - len(label)) + map_ink(row))
    else:
        if paper:
            lines.extend(map_ink(row) for row in paper)
            lines.append("")
        for row in lettering:
            paragraph(row, "1")

    if not full:
        lines.append("")
        paragraph(GREETING, "1")
    lines.append("")
    if result is not None:
        command = shlex.join(["resolve", polynomial, "--vars", "x,y"])
        if not interactive:
            command = "./atlas " + command
        if len("Try  " + command) <= width:
            lines.append("Try  " + ink(command, "36"))
        else:
            paragraph(command, "36")
    else:
        paragraph("Startup resolution unavailable. " + error)
    paragraph("Type help for commands or quit to leave." if interactive
              else "Run ./atlas --help for all commands.")
    return "\n" + "\n".join(margin + line if line else "" for line in lines) + "\n"


def print_welcome(*, interactive):
    color = sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and not os.environ.get("NO_COLOR")
    print(render_welcome(shutil.get_terminal_size().columns, color=color, interactive=interactive))
