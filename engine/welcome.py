"""The cartographic wordmark shown when opening Atlas from a terminal."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

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


# The folds show the cusp, tangency after the first blowup, and the resolution graph.
LOGO = (
    "        .-+-.                 _.-.",
    "    _.-'  :  `-._         _.-'   |",
    ".--'      :      `-.+._.-'       |",
    r"|       / :\       /:     ^      |",
    r"|    .-'  : \     / :     |      |",
    r"| .--     :  \   /  :  o--o---o  |",
    r"|    `-.  :___\_/___:            |",
    r"|       \ :         :            |",
    "|       .-+-.       :        _.-'",
    "|   _.-'     `-._   :    _.-'",
    ".--'             `-.+.-'",
)

SMALL_LOGO = (
    "    .+.        .-.",
    "_.-' : `-.+_.-'  |",
    "|   /:    :      |",
    "|  .':|  |:  ^   |",
    r"|.-- :\__/:  |   |",
    "|  `.:----:o-o-o |",
    r"|   \:    :      |",
    "|   .+.   :   _.-'",
    "`.-'   `-.+.-'",
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


def render_welcome(columns, *, color, interactive):
    margin = "  " if columns >= 24 else ""
    width = max(1, columns - len(margin))
    lines = []

    def ink(text, code):
        return f"\033[{code}m{text}\033[0m" if color and code and text else text

    def paragraph(text, code=None):
        for row in textwrap.wrap(text, width, break_on_hyphens=False):
            lines.append(ink(row, code))

    version = f"version {atlas_version()}"
    contact = "pportilla <p.portilla@upm.es>"

    # Pick the largest pair that fits on one line, in terminal character cells.
    layout = next(((mark, title) for mark in (LOGO, SMALL_LOGO)
                   for title in (WORDMARK, SMALL_WORDMARK)
                   if max(map(len, title)) + 3 + max(map(len, mark)) <= width), None)
    if layout:
        logo, lettering = layout
        column = width - max(map(len, logo)) - 3
        column = min(column, max(max(map(len, lettering)), len(contact)))
        labels = [(row, "1") for row in lettering]
        labels.append(("", None))
        for text, style in ((version, "2"), (contact, None)):
            labels.extend((row, style) for row in textwrap.wrap(text, column, break_on_hyphens=False))
        height = max(len(labels), len(logo))
        title_top = (height - len(labels)) // 2
        map_top = (height - len(logo)) // 2
        for i in range(height):
            label, style = labels[i - title_top] if title_top <= i < title_top + len(labels) else ("", None)
            line = ink(label, style)
            if map_top <= i < map_top + len(logo):
                line += " " * (column + 3 - len(label)) + ink(logo[i - map_top], "32")
            lines.append(line)
    else:
        title = SMALL_WORDMARK if max(map(len, SMALL_WORDMARK)) <= width else ("ATLAS",)
        for row in title:
            paragraph(row, "1")
        lines.append("")
        paragraph(version, "2")
        paragraph(contact)
        if max(map(len, SMALL_LOGO)) <= width:
            lines.append("")
            lines.extend(ink(row, "32") for row in SMALL_LOGO)
    lines.append("")
    paragraph(GREETING, "1")
    paragraph("Type help for commands or quit to leave." if interactive
              else "Run ./atlas --help for all commands.")
    return "\n" + "\n".join(margin + line if line else "" for line in lines) + "\n"


def print_welcome(*, interactive):
    color = sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and not os.environ.get("NO_COLOR")
    print(render_welcome(shutil.get_terminal_size().columns, color=color, interactive=interactive))
