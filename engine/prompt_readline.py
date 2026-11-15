"""Readline bindings for the interactive prompt and its continuation lines."""

from contextlib import contextmanager
import shutil

from .prompt_help import topic_names
from .session_completion import candidates


@contextmanager
def prompt_completion(session, parser, commands, *, on_list=None):
    """Complete names with Tab. on_list receives the rows a list of choices will take."""
    try:
        import readline
    except ImportError:  # Some Python builds omit the optional line editor.
        yield
        return

    previous = readline.get_completer()
    delimiters = readline.get_completer_delims()
    matches = []

    def complete(text, state):
        nonlocal matches
        if state == 0:
            line = readline.get_line_buffer()
            start, end = readline.get_begidx(), readline.get_endidx()
            head = line[:end]
            words = head.split()
            prefix = words.pop() if words and not head[-1].isspace() else ""
            if words == ["help"]:
                names = topic_names(session, parser)
            elif words in (["algebra"], ["help", "algebra"]):
                names = parser.command_names["algebra"].command_names
            elif not words and "-" in prefix and any(name.startswith(prefix) for name in commands):
                names = commands
            else:
                names = None
            if names is not None:
                # Keep the part of a hyphenated command before readline's word break.
                offset = len(prefix) - len(text)
                matches = sorted(name[offset:] for name in names if name.startswith(prefix))
            else:
                matches = candidates(text, session, commands if not line[:start].strip() else ())
            if line[end:].startswith("("):
                matches = [name.removesuffix("(") for name in matches]
            if on_list and len(matches) > 1:
                # Readline lists choices in columns two cells wider than the longest.
                per_row = max(1, shutil.get_terminal_size().columns // (max(map(len, matches)) + 2))
                on_list(-(-len(matches) // per_row))
        return matches[state] if state < len(matches) else None

    try:
        readline.set_completer_delims(delimiters.replace(".", "").replace(":", ""))
        readline.set_completer(complete)
        if getattr(readline, "backend", None) == "editline" or "libedit" in (readline.__doc__ or ""):
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind("set show-all-if-ambiguous on")
        yield
    finally:
        readline.set_completer(previous)
        readline.set_completer_delims(delimiters)
