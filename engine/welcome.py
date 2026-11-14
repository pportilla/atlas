"""The cartographic wordmark shown when opening Atlas from a terminal."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

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


# ASCII redraws of web/atlas-mark.svg. The colons mark the two folds.
LOGO = (
    "       .-+-.             .-.",
    "   _.-'  :  `-._     _.-'  |",
    ".-'      :      `-.-'      |",
    "|        :        :        |",
    "|        .-o-._   :      o |",
    "|     .-'      `-._   .-'  |",
    "|  o-'   :        :o-'     |",
    "|        :        :        |",
    "|      .-+-.      :      .-'",
    "|  _.-'     `-._  :  _.-'",
    ".-'             `-.-'",
)

SMALL_LOGO = (
    "   .-+.      .-.",
    ".-'  : `-.--'  |",
    "|    :   :   o |",
    "|   .o-. : .-' |",
    "| o' :  `o'    |",
    "|  .-+.  :   .-'",
    ".-'    `-.--'",
)


def atlas_version():
    root = Path(__file__).resolve().parents[1]
    if (root / ".git").exists():
        try:
            return subprocess.check_output(
                ["git", "describe", "--tags", "--always", "HEAD"], cwd=root,
                stderr=subprocess.DEVNULL, text=True, timeout=1,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return "development"


def render_welcome(columns, *, color, interactive, example=None):
    margin = "  " if columns >= 24 else ""
    width = max(1, columns - len(margin))
    lines = []

    def ink(text, code):
        return f"\033[{code}m{text}\033[0m" if color and code and text else text

    def paragraph(text, code=None):
        for row in textwrap.wrap(text, width, break_on_hyphens=False):
            lines.append(ink(row, code))

    paragraph(f"Atlas version {atlas_version()}", "1")
    paragraph("Developer: pportilla <p.portilla@upm.es>")
    lines.append("")

    # Pick the largest pair that fits on one line, in terminal character cells.
    layout = next(((mark, title) for mark in (LOGO, SMALL_LOGO)
                   for title in (WORDMARK, SMALL_WORDMARK)
                   if max(map(len, title)) + 3 + max(map(len, mark)) <= width), None)
    if layout:
        logo, lettering = layout
        labels = (*lettering, "", GREETING)
        top = (len(logo) - len(labels)) // 2
        column = max(map(len, lettering)) + 3
        for i, row in enumerate(logo):
            label = labels[i - top] if top <= i < top + len(labels) else ""
            lines.append(ink(label, "1") + " " * (column - len(label)) + ink(row, "32"))
    else:
        if max(map(len, SMALL_LOGO)) <= width:
            lines.extend(ink(row, "32") for row in SMALL_LOGO)
            lines.append("")
        title = SMALL_WORDMARK if max(map(len, SMALL_WORDMARK)) <= width else ("ATLAS",)
        for row in title:
            paragraph(row, "1")
        lines.append("")
        paragraph(GREETING, "1")
    lines.append("")

    try:
        polynomial, _ = resolve_welcome_curve() if example is None else example
    except (ArithmeticError, OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        paragraph("Startup resolution unavailable. " + str(exc))
    else:
        paragraph("Resolved curve  " + polynomial, "36")
    paragraph("Type help for commands or quit to leave." if interactive
              else "Run ./atlas --help for all commands.")
    return "\n" + "\n".join(margin + line if line else "" for line in lines) + "\n"


def print_welcome(*, interactive):
    color = sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and not os.environ.get("NO_COLOR")
    print(render_welcome(shutil.get_terminal_size().columns, color=color, interactive=interactive))
