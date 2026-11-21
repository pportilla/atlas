"""Name completion for live Atlas sessions, without evaluating the input."""
from inspect import getattr_static, isroutine
from keyword import iskeyword

from .session_values import _RecordView


def _public_name(name):
    return isinstance(name, str) and name.isidentifier() and not name.startswith("_") and not iskeyword(name)


def _members(value):
    if isinstance(value, dict):
        return {name: isroutine(item) for name, item in value.items() if _public_name(name)}

    cls = type(value)
    if isinstance(value, _RecordView):
        names = {name: False for name in value._record if _public_name(name)}
        declared = value._members
    elif getattr_static(cls, "session_attribute", None) is not None:
        names = {}
        declared = getattr_static(cls, "_fields", ())
    else:
        return {}
    for name in declared:
        if _public_name(name):
            names[name] = callable(getattr_static(cls, name, None))
    return names


def _saved_value(path, session):
    parts = path.split(".")
    if not all(_public_name(part) for part in parts):
        return None
    value = session.variables.get(parts[0])
    for name in parts[1:]:
        if isinstance(value, dict):
            value = value.get(name)
        elif isinstance(value, _RecordView) and type(value).session_attribute is _RecordView.session_attribute:
            # Follow an already stored public view, or a plain record entry.
            # Reading the instance dictionary avoids evaluating any property.
            if name in value._members and name in vars(value):
                value = vars(value)[name]
            else:
                value = value._record.get(name)
        else:
            # Linked fields and methods can invoke the solver. Complete a saved
            # result directly once the user has chosen to evaluate that field.
            return None
    return value


def candidates(text, session, commands=()):
    """Return sorted replacement tokens, appending '(' to functions and methods."""
    path, dot, prefix = text.rpartition(".")
    if dot:
        names = _members(_saved_value(path, session))
    else:
        names = dict.fromkeys(commands, False)
        names.update((name, callable(value)) for name, value in session.builtins.items())
        names.update((name, isroutine(value)) for name, value in session.variables.items())
    return sorted(path + dot + name + ("(" if function else "")
                  for name, function in names.items() if name.startswith(prefix))
