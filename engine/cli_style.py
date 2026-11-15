"""The colour palette shared by the interactive prompt, its header, and the welcome."""
from contextlib import contextmanager
import os
import re
import sys

# Each role lists its SGR codes for 8-colour and for 256-colour terminals.
# The 256-colour midtones stay legible on dark and on light backgrounds.
# Teal marks Atlas itself and what it recognises, amber what you saved,
# green settled results; each family has a few tones.
_ROLES = {
    "brand": ("1;36", "1;38;5;37"),   # wordmark and prompt
    "label": ("36", "38;5;37"),       # row labels and help headings
    "command": ("1;36", "1;38;5;73"),  # prompt commands such as help, :save and quit
    "function": ("36", "38;5;73"),    # registered functions and QQ
    "field": ("36", "38;5;109"),      # fields, keyword arguments and recorded attributes
    "option": ("36", "38;5;109"),     # command-line options
    "keyword": ("1", "1;38;5;66"),    # for, in, if and the other Python keywords
    "name": ("33", "38;5;172"),       # saved names
    "number": ("33", "38;5;137"),     # numerals
    "string": ("32", "38;5;107"),     # quoted text
    "ok": ("32", "38;5;72"),          # True and settled verdicts
    "false": ("31", "38;5;173"),      # False and open verdicts
    "paper": ("32", "38;5;71"),       # the unfolding map
    "frame": ("2", "38;5;240"),       # box lines
    "muted": ("2", "38;5;245"),       # separators, hints and secondary text
    "comment": ("2", "3;38;5;244"),   # comments
    "strong": ("1", "1"),
    "error": ("31", "38;5;167"),
}
_SGR = re.compile(r"(\033\[[0-9;]*m)")


def use_color(stream):
    return (stream.isatty() and os.environ.get("TERM", "") not in ("", "dumb", "unknown")
            and not os.environ.get("NO_COLOR"))


def paint(text, role, color=True):
    if not (color and role and text):
        return text
    rich = "256color" in os.environ.get("TERM", "") or os.environ.get("COLORTERM") in ("truecolor", "24bit")
    return f"\033[{_ROLES[role][rich]}m{text}\033[0m"


def readline_prompt(*pieces, color):
    """Join (text, role) pieces, marking colour codes as zero-width for line editing."""
    return "".join(_SGR.sub("\001\\1\002", paint(text, role, color)) for text, role in pieces)


class _Tinted:
    def __init__(self, stream, role):
        self._stream, self._role = stream, role

    def write(self, text):
        # Text that brings its own colours, such as argparse usage, keeps them.
        self._stream.write(paint(text, self._role) if text.strip() and "\033" not in text else text)
        return len(text)

    def __getattr__(self, name):
        return getattr(self._stream, name)


@contextmanager
def tinted_errors(color):
    """Colour what the prompt reports on standard error."""
    if not (color and use_color(sys.stderr)):
        yield
        return
    previous, sys.stderr = sys.stderr, _Tinted(sys.stderr, "error")
    try:
        yield
    finally:
        sys.stderr = previous
