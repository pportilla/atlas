"""Short recovery suggestions for errors at the interactive prompt."""

from difflib import get_close_matches
from inspect import Parameter
import re

from .session_function_help import session_signature


def closest_name(name, candidates):
    """Return one similar public name, preserving its spelling and case."""
    names = {item.casefold(): item for item in sorted(candidates)}
    key = name.casefold()
    if key in names:
        return names[key]
    matches = get_close_matches(key, names, n=1, cutoff=0.73)
    return names[matches[0]] if matches else None


def _usage(topic):
    return f"Type 'help {topic}' for usage and an example."


def error_hint(error, statement, session, topics):
    """Use the failed lookup or call to suggest a next step without re-evaluating it."""
    unknown = error.unknown_name
    function = error.function
    if unknown:
        if function == "coordinates" and unknown.casefold() != "qq":
            return 'Quote coordinate names, for example x, y = coordinates("x", "y").'
        match = closest_name(unknown, set(topics) | session.variables.keys())
        if match in session.variables:
            return f"Did you mean '{match}'? Type ':workspace' to see your variables."
        if match:
            return _usage(match) if match == unknown else f"Did you mean '{match}'? {_usage(match)}"
        hint = _usage(function) if function in topics else "Type 'help coordinates' to define coordinates."
        return f"Type ':workspace' to see your variables. {hint}"

    # A callable can be saved in a session variable. Show its registered help
    # under the original name rather than suggesting help for the variable.
    if function in session.variables:
        value = session.variables[function]
        function = next((name for name, op in session.builtins.items() if op is value), None)
        if function is None:
            return "Type ':workspace' to inspect the saved value."
    if error.method is not None:
        if callable(error.method):
            try:
                return f"Usage: {function}{session_signature(error.method)}."
            except (TypeError, ValueError):
                return "Type 'help' to browse commands and functions."
        return f"Read {function} without parentheses."
    if function in topics:
        keyword = re.search(r"unexpected keyword argument '([A-Za-z_]\w*)'", str(error))
        if keyword:
            try:
                parameters = session_signature(session.builtins[function]).parameters.values()
            except (KeyError, ValueError, TypeError):
                parameters = ()
            match = closest_name(keyword[1], (p.name for p in parameters
                                              if p.kind in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD)))
            if match and f"Did you mean '{match}'" not in str(error):
                return f"Did you mean '{match}='? {_usage(function)}"
        return _usage(function)

    # Syntax errors have no expression tree. Recognize a named operation at
    # the start of a statement or its assignment's right-hand side.
    head = re.match(r"\s*(?:[\w, ]+\s*=\s*)?([A-Za-z_]\w*)", statement)
    if head:
        name = head[1]
        if name in topics:
            return _usage(name)
        match = closest_name(name, topics)
        if match and name not in session.variables:
            return f"Did you mean '{match}'? {_usage(match)}"
    return "Type 'help' to browse commands and functions."
