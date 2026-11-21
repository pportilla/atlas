"""Help lookup and command discovery at the interactive Atlas prompt."""

import sys
import textwrap

from .cli_highlight import paint_card, paint_hint, paint_value
from .cli_style import paint
from .help_card import usage_card
from .resolution_help import resolution_help, resolution_topics
from .session_help import algebra_help_values, algebra_value_help, function_help
from .session_hints import closest_name


_PROMPT_GUIDE = "docs/tutorial/prompt-and-scripts.md#"
_RINGS_GUIDE = "docs/tutorial/rings.md#"

_TOPICS = {
    "variables": (
        "Declare coordinates with coordinates(), then assign an expression to a name.",
        'x, y = coordinates("x", "y")\nf = y^2 - x^3\n:workspace',
        _RINGS_GUIDE + "reuse-and-rebind-coordinate-names",
    ),
    ":workspace": (
        "Show saved values grouped by their parent ring, and mark the active ring.",
        'x, y = coordinates("x", "y")\nf = y^2 - x^3\n:workspace',
        _RINGS_GUIDE + "choose-the-ring-of-a-polynomial",
    ),
    "rings": (
        "Name a ring when you create it with ring(), or Atlas names it R1, R2, … until you name it.",
        'R = ring("x", "y")\nf = y^2 + x^5\nuse_ring(R)\nx2, z = coordinates("x", "z")\nR1\ndelete_ring(R1)',
        _RINGS_GUIDE + "name-select-and-delete-rings",
    ),
    "QQ": (
        "QQ is the field of rational numbers.",
        'q = polynomial("1/2", ring=QQ)\nq.parent == QQ',
        _RINGS_GUIDE + "move-a-polynomial-between-rings",
    ),
    ":save": (
        "Write the successful statements from this session to an Atlas script.",
        'x = coordinates("x")\nprint(x^2 + 1)\n:save square.atlas',
        _PROMPT_GUIDE + "run-or-save-a-calculation",
    ),
    "script": (
        "Save these lines in square.atlas and run ./atlas run square.atlas from the shell.",
        'x = coordinates("x")\nf = x^2 + 1\nprint(f)',
        _PROMPT_GUIDE + "run-or-save-a-calculation",
    ),
    "quit": (
        "Leave the Atlas prompt.",
        "quit",
        _PROMPT_GUIDE + "run-or-save-a-calculation",
    ),
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
            *algebra_help_values(session), "functions", "help"}


def _functions(session, color=False):
    names = sorted(name for name, value in session.builtins.items() if callable(value))
    listing = paint_value(textwrap.fill(", ".join(names), width=76), functions=names, color=color)
    print(paint("Session functions", "label", color) + "\n" + listing)
    card = usage_card("Use help NAME to see a function’s signature and examples.",
                      "help coordinates", _PROMPT_GUIDE + "inspect-a-saved-value")
    print("\n" + paint_card(card, commands={"help"}, functions=names, color=color))


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
        print(paint_hint("Enter help NAME or NAME -h to inspect a saved value, function or command.",
                         commands=commands, color=color, rest=None))
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

    name = words[0]
    values = algebra_help_values(session)
    if name in values:
        print(painted(algebra_value_help(name, values[name], session.variables, session.rings.active)))
        return
    if card := resolution_help(name, session.variables):
        print(painted(card))
        return

    name = _ALIASES.get(name, name)
    if name == "functions":
        _functions(session, color)
    elif name in _TOPICS:
        print(painted(usage_card(*_TOPICS[name])))
    else:
        text = function_help(name, session.builtins)
        if text:
            print(painted(text))
            if name == "resolve":
                print(painted('\nCommand form\n  resolve "y^2-x^3" --vars x,y\n  resolve --help'))
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
