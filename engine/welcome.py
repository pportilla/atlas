"""The cartographic wordmark shown when opening Atlas from a terminal."""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from threading import Event, Thread

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


def _ink(text, style, color):
    return f"\033[{style}m{text}\033[0m" if color and style and text else text


def render_welcome(columns, *, color, interactive, art_rows=None):
    margin = "  " if columns >= 24 else ""
    width = max(1, columns - len(margin))
    lines = []

    def paragraph(text, code=None):
        for row in textwrap.wrap(text, width, break_on_hyphens=False):
            lines.append(_ink(row, code, color))

    version = f"version {atlas_version()}"
    contact = "pportilla <p.portilla@upm.es>"

    wordmark = next((art for art in (WORDMARK, SMALL_WORDMARK) if max(map(len, art)) <= width), None)
    if wordmark:
        if art_rows is None:
            lines.extend(_ink(row, "1", color) for row in wordmark)
        else:
            lines.extend([""] * art_rows)
    else:
        paragraph("ATLAS", "1")

    lines.append("")
    paragraph(version, "2")
    paragraph(contact)
    lines.append("")
    paragraph(GREETING, "1")
    paragraph("Tab completes names. Type help for topics, help NAME for an example, or quit." if interactive
              else "Run ./atlas --help for all commands.")
    return "\n" + "\n".join(margin + line if line else "" for line in lines) + "\n"


def _animation_frames(color):
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
        yield [_ink(indent + row, "32", color) for row in frame], 0.4 if frame is sheet else 0.25

    positions = [i * pitch + (pitch + 1 - max(len(row.rstrip()) for row in glyph)) // 2
                 for i, glyph in enumerate(glyphs)]
    body = next(i for i, row in enumerate(sheet) if row.count(":") == 4)
    seam = _ink(":", "32", color)
    side = _ink("|", "32", color)
    # Start the next letter as the preceding one receives its last stroke.
    stagger = len(WORDMARK) - 1
    last_stroke = (len(glyphs) - 1) * stagger + len(WORDMARK) - 1
    for stroke in range(last_stroke + 1):
        rows = ["  " + _ink(row, "32", color) for row in sheet]
        for y in range(len(WORDMARK)):
            cells = []
            for i, glyph in enumerate(glyphs):
                left = positions[i] - i * pitch - 1
                ink = glyph[y] if stroke >= i * stagger + y else ""
                cells.append((" " * left + ink).ljust(pitch - 1))
            rows[body + y] = "  " + side + seam.join(_ink(cell, "1", color) for cell in cells) + side
        yield rows, 0.4 if stroke == last_stroke else 0.085

    for remaining in (1, 0.78, 0.54, 0.32, 0.15, 0):
        offsets = [end + round((start - end) * remaining) for start, end in zip(positions, starts)]
        rows = [""] * round(body * remaining)
        for y in range(len(WORDMARK)):
            line = [" "] * width
            for x, glyph in zip(offsets, glyphs):
                for dx, stroke in enumerate(glyph[y]):
                    if stroke != " ":
                        line[x + dx] = stroke
            rows.append("  " + _ink("".join(line).rstrip(), "1", color))
        yield rows, 0.20 if remaining == 1 else 0.095


@contextmanager
def interactive_welcome(panel):
    """Play above the editable prompt. Finish early to give completion the full workspace."""
    size = panel.size
    height = max(map(len, FOLDS))
    banner = ""
    if size and size.columns >= max(map(len, FOLDS[-1])) + 3:
        banner = render_welcome(size.columns, color=panel.color, interactive=True, art_rows=height)
    # The session panel starts the welcome at row 3. Leave room for a
    # wrapped input line and completion choices below the welcome.
    prompt_row = 3 + len(banner.splitlines()) + 1
    if not banner or prompt_row + 2 > size.lines:
        print_welcome(interactive=True)
        yield lambda: None
        return

    stream = panel.stream
    print(banner, file=stream)
    stream.write(f"\033[{prompt_row};r\033[{prompt_row};1H")
    stream.flush()
    stop = Event()

    def paint(rows):
        # One write saves the live editing cursor, updates just the
        # map's rows, and restores the cursor (including its column).
        commands = ["\0337"]
        for i in range(height):
            commands.append(f"\033[{4 + i};1H\033[2K")
            if i < len(rows):
                commands.append(rows[i])
        commands.append("\0338")
        stream.write("".join(commands))
        stream.flush()

    def play():
        try:
            for rows, delay in _animation_frames(panel.color):
                if os.get_terminal_size(stream.fileno()) != size:
                    return
                paint(rows)
                if stop.wait(delay):
                    break
            if os.get_terminal_size(stream.fileno()) == size:
                paint(["  " + _ink(row, "1", panel.color) for row in WORDMARK])
        finally:
            # DECSTBM also homes the cursor. Restore readline's
            # position as well as the session panel's scroll region.
            stream.write("\0337\033[3;r\0338")
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
    terminal = stream.isatty() and os.environ.get("TERM", "") not in ("", "dumb", "unknown")
    color = terminal and not os.environ.get("NO_COLOR")
    print(render_welcome(shutil.get_terminal_size().columns, color=color, interactive=interactive))
