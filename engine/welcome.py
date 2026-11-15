"""The cartographic wordmark shown when opening Atlas from a terminal."""
from contextlib import contextmanager
from itertools import islice
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
from threading import Event, Thread

from .cli_style import paint, use_color

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
_COMMANDS = re.compile(r"\bTab\b|\bhelp( NAME)?\b|\bquit\b|\./atlas --help")

FOLDS = (
    (
        " .-+.",
        " |::|",
        " |::|",
        " |::|",
        " |::|",
        " `-+'",
    ),
    (
        "   +     +     .",
        ".-':`-+-':`-+-'|",
        "|  :  :  :  :  |",
        "|  :  :  :  :  |",
        "|  :  :  :  :  |",
        "|  :  :  :  :  |",
        "|  +  :  +  :  .",
        ".-' `-+-' `-+-'",
    ),
    (
        "   -+-     -+-     -.",
        ".-' : `-+-' : `-+-' |",
        "|   :   :   :   :   |",
        "|   :   :   :   :   |",
        "|   :   :   :   :   |",
        "|   :   :   :   :   |",
        "|  -+-  :  -+-  :  -.",
        ".-'   `-+-'   `-+-'",
    ),
    (
        "      +           +           .",
        "   _.':`._     _.':`._     _.'|",
        ".-'   :   `-+-'   :   `-+-'   |",
        "|     :     :     :     :     |",
        "|     :     :     :     :     |",
        "|     :     :     :     :     |",
        "|     +     :     +     :     .",
        "|  _.' `._  :  _.' `._  :  _.'",
        ".-'       `-+-'       `-+-'",
    ),
    (
        "         _.-+-._                 _.-+-._                 _.-.",
        "    _.--'   :   `--._       _.--'   :   `--._       _.--'   |",
        ".--'        :        `--+--'        :        `--+--'        |",
        "|           :           :           :           :           |",
        "|           :           :           :           :           |",
        "|           :           :           :           :           |",
        "|           :           :           :           :           |",
        "|           :           :           :           :           |",
        "|        _.-+-._        :        _.-+-._        :        _.-'",
        "|   _.--'       `--._   :   _.--'       `--._   :   _.--'",
        ".--'                 `--+--'                 `--+--'",
    ),
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


def render_welcome(columns, *, color, interactive, art_rows=None):
    margin = "  " if columns >= 24 else ""
    width = max(1, columns - len(margin))
    lines = []

    def paragraph(text, role=None):
        for row in textwrap.wrap(text, width, break_on_hyphens=False):
            lines.append(paint(row, role, color))

    version = f"version {atlas_version()}"
    contact = "pportilla <p.portilla@upm.es>"

    wordmark = next((art for art in (WORDMARK, SMALL_WORDMARK) if max(map(len, art)) <= width), None)
    if wordmark:
        if art_rows is None:
            lines.extend(paint(row, "brand", color) for row in wordmark)
        else:
            lines.extend([""] * art_rows)
    else:
        paragraph("ATLAS", "brand")

    lines.append("")
    paragraph(version, "muted")
    paragraph(contact, "muted")
    lines.append("")
    paragraph(GREETING, "strong")
    hint = len(lines)
    paragraph("Tab completes names. Type help for topics, help NAME for an example, or quit." if interactive
              else "Run ./atlas --help for all commands.")
    lines[hint:] = [_COMMANDS.sub(lambda word: paint(word[0], "name" if word[0] == "Tab" else "command", color), row)
                    for row in lines[hint:]]
    return "\n" + "\n".join(margin + line if line else "" for line in lines) + "\n"


def _animation_frames(color, rest=0, painter=paint):
    """Yield the rows of each frame and its delay. The wordmark settles rest rows down.

    The painter colours each run of text by its role. The web workspace and the
    documentation pass one that writes HTML spans instead of terminal codes."""
    # The slanted T and L share columns on different rows. Two blank
    # cells before the L's top keep its strokes on their original diagonal.
    starts = (1, 9, 15, 23, 31)
    glyphs = list(zip(*(
        (row[1:9], row[9:cut], " " * (cut - 15) + row[cut:22], row[23:31], row[31:])
        for row, cut in zip(WORDMARK, (17, 17, 16, 15, 15))
    )))

    sheet = FOLDS[-1]
    width = max(map(len, sheet))
    pitch = (width - 1) // len(glyphs)
    for frame in FOLDS:
        indent = " " * (2 + (width - max(map(len, frame))) // 2)
        yield [painter(indent + row, "paper", color) for row in frame], 0.4 if frame is sheet else 0.25

    positions = [i * pitch + (pitch + 1 - max(len(row.rstrip()) for row in glyph)) // 2
                 for i, glyph in enumerate(glyphs)]
    body = next(i for i, row in enumerate(sheet) if row.count(":") == 4)
    seam = painter(":", "paper", color)
    side = painter("|", "paper", color)
    # Start the next letter as the preceding one receives its last stroke.
    stagger = len(WORDMARK) - 1
    last_stroke = (len(glyphs) - 1) * stagger + len(WORDMARK) - 1
    for stroke in range(last_stroke + 1):
        rows = ["  " + painter(row, "paper", color) for row in sheet]
        for y in range(len(WORDMARK)):
            cells = []
            for i, glyph in enumerate(glyphs):
                left = positions[i] - i * pitch - 1
                ink = glyph[y] if stroke >= i * stagger + y else ""
                cells.append((" " * left + ink).ljust(pitch - 1))
            rows[body + y] = "  " + side + seam.join(painter(cell, "brand", color) for cell in cells) + side
        yield rows, 0.4 if stroke == last_stroke else 0.085

    for remaining in (1, 0.78, 0.54, 0.32, 0.15, 0):
        offsets = [end + round((start - end) * remaining) for start, end in zip(positions, starts)]
        rows = [""] * (rest + round((body - rest) * remaining))
        for y in range(len(WORDMARK)):
            line = [" "] * width
            for x, glyph in zip(offsets, glyphs):
                for dx, stroke in enumerate(glyph[y]):
                    if stroke != " ":
                        line[x + dx] = stroke
            rows.append("  " + painter("".join(line).rstrip(), "brand", color))
        yield rows, 0.20 if remaining == 1 else 0.095


def lettered_map(painter=paint, color=True):
    """Rows of the unfolded map once each letter of the wordmark is drawn in its panel."""
    letters, stagger = 5, len(WORDMARK) - 1
    last_stroke = (letters - 1) * stagger + len(WORDMARK) - 1
    frames = _animation_frames(color, painter=painter)
    return next(islice(frames, len(FOLDS) + last_stroke, None))[0]


@contextmanager
def interactive_welcome(panel):
    """Play above the editable prompt. Finish early to give completion the full workspace."""
    size, top = panel.size, panel.top
    height = max(map(len, FOLDS))
    banner = ""
    if size and size.columns >= max(map(len, FOLDS[-1])) + 3:
        # A blank row above and below sets the map apart. A short terminal
        # gives both up, and the wordmark settles one row down instead.
        for margin in (1, 0):
            spare = height - len(WORDMARK) - 2 * (1 - margin)
            banner = render_welcome(size.columns, color=panel.color, interactive=True,
                                    art_rows=len(WORDMARK) + spare)
            # Leave room for a wrapped input line and completion choices below the welcome.
            prompt_row = top + len(banner.splitlines()) + 1
            if prompt_row + 2 <= size.lines:
                break
        else:
            banner = ""
    if not banner:
        print_welcome(interactive=True)
        yield lambda: None
        return

    stream = panel.stream
    print(banner, file=stream)
    stream.write(f"\033[{prompt_row};r\033[{prompt_row};1H")
    stream.flush()
    stop = Event()

    def paint_map(rows):
        # One write saves the live editing cursor, updates just the
        # map's rows, and restores the cursor (including its column).
        commands = ["\0337"]
        for i in range(height):
            commands.append(f"\033[{top + margin + i};1H\033[2K")
            if i < len(rows):
                commands.append(rows[i])
        commands.append("\0338")
        stream.write("".join(commands))
        stream.flush()

    def close_gap():
        # The wordmark needs fewer rows than the map did. Delete the spare
        # ones, one at a time unless the prompt is waiting, to draw the text
        # and the prompt up to the wordmark.
        closed = 0
        while closed < spare:
            step = spare - closed if stop.is_set() else 1
            closed += step
            # Deleting rows needs them inside the scrolling region. Return to
            # the prompt's own region so its output cannot move the banner,
            # and move the editing cursor up with its line.
            stream.write(f"\0337\033[{top};r\033[{top + 1 + len(WORDMARK)};1H\033[{step}M"
                         f"\033[{prompt_row - closed};r\0338\033[{step}A")
            stream.flush()
            stop.wait(0.03)

    def play():
        try:
            for rows, delay in _animation_frames(panel.color, 1 - margin):
                if os.get_terminal_size(stream.fileno()) != size:
                    return
                paint_map(rows)
                if stop.wait(delay):
                    break
            if os.get_terminal_size(stream.fileno()) == size:
                paint_map([""] * (1 - margin) + ["  " + paint(row, "brand", panel.color) for row in WORDMARK])
                close_gap()
        finally:
            # DECSTBM also homes the cursor. Restore readline's
            # position as well as the session panel's scroll region.
            stream.write(f"\0337\033[{top};r\0338")
            stream.flush()

    thread = Thread(target=play, name="atlas-welcome")

    def finish():
        stop.set()
        if thread.ident is not None:
            thread.join()

    try:
        thread.start()
        yield finish
    finally:
        finish()


def print_welcome(*, interactive):
    stream = sys.stdout
    print(render_welcome(shutil.get_terminal_size().columns, color=use_color(stream), interactive=interactive))
