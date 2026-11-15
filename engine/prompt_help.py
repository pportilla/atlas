"""Help lookup and command discovery at the interactive Atlas prompt."""

import sys
import textwrap

from .cli_highlight import paint_card, paint_hint, paint_value
from .cli_style import paint
from .resolution_help import resolution_help, resolution_topics
from .session_help import function_help
from .session_hints import closest_name


_TOPICS = {
    "variables": (
        "Use coordinates to declare polynomial variables and = to save a value under a name.\n"
        "Repeating the same coordinate names in the same order reuses their ring.\n"
        "Use fresh=True for an independent copy. The Rings row groups saved names by ring.\n"
        "Use ring=f.parent to write another polynomial in the ring of f.\n\n"
        'Example\n  x, y = coordinates("x", "y")\n  f = y^2 - x^3\n  :workspace'
    ),
    ":workspace": "Show which saved names belong to each ring and list their values.\n\nExample\n  :workspace",
    ":save": "Write this session's successful statements to an Atlas script.\n\nExample\n  :save cusp.atlas",
    "QQ": 'The rational coefficient field. Coordinates use QQ by default.\n\nExample\n  x = coordinates("x", coeffs=QQ)',
    "quit": "Leave the Atlas prompt.\n\nExample\n  quit",
}
_ALIASES = {"save": ":save", "workspace": ":workspace", "exit": "quit"}
_FLAGS = {"-h", "--help", "help"}


def help_request(line):
    """Match standalone help commands, leaving expressions and quoted strings alone."""
    words = line.split()
    if words and words[0] == "help":
        return words[1:]
    if words in (["-h"], ["--help"]):
        return []
    if (len(words) == 2 or (len(words) == 3 and words[0] == "algebra")) and words[-1] in _FLAGS:
        if words[0] == "resolve" and words[-1] == "--help":
            return None  # Preserve the complete command-line option reference.
        names = words[:-1]
        if all(all(part.isidentifier() for part in name.lstrip(":").replace("-", "_").split("."))
               for name in names):
            return names
    return None


def topic_names(session, parser):
    return {*session.builtins, *parser.command_names, *_TOPICS, *_ALIASES, *resolution_topics(session.variables),
            "functions", "script", "help"}


def _functions(session, color=False):
    names = sorted(name for name, value in session.builtins.items() if callable(value))
    listing = paint_value(textwrap.fill(", ".join(names), width=76), functions=names, color=color)
    print(paint("Session functions", "label", color) + "\n" + listing)
    print(paint_hint("\nUse help NAME for a signature and an example.", commands={"help"}, color=color, rest=None))


def _unknown(topic, choices, prefix="", color=False):
    print(f"Unknown help topic {topic!r}.", file=sys.stderr)
    match = closest_name(topic, choices)
    if match:
        hint = f"Did you mean {match!r}? Type help {prefix}{match}."
    else:
        hint = "Type help algebra for operations." if prefix else "Type help for a list of topics."
    print(paint_hint(hint, commands={"help"}, topics=choices, functions=choices, color=color), file=sys.stderr)


def show_help(words, session, parser, *, color=False):
    functions = set(session.builtins)
    commands = {*parser.command_names, ":workspace", ":save", "quit", "help"}

    def painted(text):
        return paint_card(text, names=session.variables, functions=functions, commands=commands, color=color)

    if not words or words == ["help"]:
        for line in ("Use help NAME, NAME help, or NAME -h for a short explanation and an example.",
                     "Start with help variables, or type help script for a scripting walkthrough.",
                     "Use help resolution or help chart to explore a resolution and its local equations."):
            print(paint_hint(line, commands=commands, topics=topic_names(session, parser), color=color, rest=None))
        listing = ", ".join(paint(name, "command", color) for name in [*parser.command_names, ":workspace", ":save", "quit"])
        print(paint("Prompt commands", "label", color) + "\n" + listing)
        print()
        _functions(session, color)
        return
    if len(words) == 2 and words[0] == "algebra":
        operations = parser.command_names["algebra"].command_names
        if words[1] in operations:
            operations[words[1]].print_help()
        else:
            _unknown(words[1], operations, "algebra ", color)
        return
    if len(words) != 1:
        print("Use help NAME or help algebra OPERATION.", file=sys.stderr)
        return

    name = _ALIASES.get(words[0], words[0])
    if name == "functions":
        _functions(session, color)
    elif name == "script":
        from .session import HELP
        print(HELP)
    elif name in _TOPICS:
        print(painted(_TOPICS[name]))
    elif card := resolution_help(name, session.variables):
        print(painted(card))
    else:
        text = function_help(name, session.builtins)
        if text:
            print(painted(text))
            if name == "resolve":
                print(paint_hint("\nUse help resolution.charts to inspect a chart or help resolution.dual_complex "
                                 "to read the exceptional complex and find the branch attachments.",
                                 commands=commands, topics=topic_names(session, parser), color=color, rest=None))
                print(painted('\nCommand form\n  resolve "y^2-x^3" --vars x,y\n  resolve --help lists all command options.'))
        elif name in parser.command_names:
            parser.command_names[name].print_help()
        else:
            _unknown(name, topic_names(session, parser), color=color)


def command_hint(argv, parser):
    """Point parser and shell-input errors to the relevant registered command."""
    name = argv[0]
    if name == "algebra" and len(argv) > 1:
        operations = parser.command_names[name].command_names
        if argv[1] in operations:
            return f"Type help algebra {argv[1]} for the command options."
        match = closest_name(argv[1], operations)
        if match:
            return f"Did you mean {match!r}? Type help algebra {match}."
    return f"Type {name} --help for usage."
