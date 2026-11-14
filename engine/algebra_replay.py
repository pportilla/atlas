"""Render a validated algebra request as a shell command and an Atlas script."""

import json
from keyword import iskeyword
import shlex

from .polynomial import integer_text


_MAX_SCRIPT_LINE = 640


def _arguments_too_long(polynomials):
    remaining = _MAX_SCRIPT_LINE
    for poly in polynomials:
        # Each nonzero term needs at least one character, plus a sign between
        # terms. The zero polynomial prints as one character as well.
        remaining -= max(1, 2 * len(poly._terms) - 1)
        if remaining < 0:
            return True
        remaining -= 2  # ", " before the next argument, if present
    return False


def _command(request):
    ring, operands = request["ring"], request["operands"]
    args = ["./atlas", "algebra", request["operation"], "--vars", ",".join(ring["variables"]),
            "--field", ring["field"], "--order", ring["order"], "--compute", request["compute"]]
    for key, flag in (("ideal", "--gen"), ("other_ideal", "--other-gen"), ("prime", "--prime"),
                      ("numerator", "--numerator"), ("denominator", "--denominator"),
                      ("polynomial", "--poly"), ("other_polynomial", "--other-poly")):
        if key not in operands:
            continue
        values = operands[key]
        if isinstance(values, str):
            values = (values,)
        for value in values:
            if value.startswith("-"):
                args.append(flag + "=" + value)
            else:
                args.extend([flag, value])
    if "exponent" in operands:
        args.extend(["--exponent", operands["exponent"]])
    if "variables" in operands:
        args.extend(["--eliminate", ",".join(operands["variables"])])
    if "point" in operands:
        args.extend(["--point", ",".join(operands["point"])])
    if "degrees" in operands:
        args.extend(["--degrees", ",".join(map(integer_text, operands["degrees"]))])
    args.append("--json")
    return shlex.join(args)


def _script(request, prepared):
    from .session import new_session

    if request["operation"] == "ideal":
        # The public record of this declaration carries execution details and
        # the original generator list; the console's ideal() is a constructor.
        return None
    # Each ideal declaration has its own line. The polynomial operands share
    # the operation line. Check all three before formatting any polynomial.
    polynomial_args = (prepared[key] for key in ("polynomial", "other_polynomial", "numerator", "denominator")
                       if key in prepared)
    if any(_arguments_too_long(group) for group in
           (prepared.get("ideal", ()), prepared.get("other_ideal", ()), prepared.get("prime", ()), polynomial_args)):
        return None
    names = request["ring"]["variables"]
    session = new_session(None)
    reserved = {"I", "J", "result", "__debug__", *session.builtins, *session.reserved}
    if any(not name.isidentifier() or iskeyword(name) or name in reserved for name in names):
        return None

    variables = ", ".join(names)
    lines = []
    if names:
        labels = ", ".join(map(json.dumps, names))
        lines.append(f"{variables} = coordinates({labels}, coeffs=QQ)")
    for var, key in (("I", "ideal"), ("J", "other_ideal"), ("J", "prime")):
        if key not in prepared:
            continue
        generators = prepared[key]
        args = [str(poly) for poly in generators]
        if all(poly.is_number for poly in generators):
            args.append(f"variables=[{variables}]")
        args.append(f"order={request['ring']['order']!r}")
        lines.append(f"{var} = ideal({', '.join(args)})")
    op = request["operation"]
    args = ["I"]
    if "ideal" not in prepared:
        args = [str(prepared[key]) for key in ("polynomial", "other_polynomial") if key in prepared]
        if op == "resultant":
            args.extend(prepared["variables"])
        args.extend([f"variables=[{variables}]", f"order={request['ring']['order']!r}"])
    elif op in ("normal-form", "membership", "local-normal-form"):
        args.insert(0, str(prepared["polynomial"]))
    elif op in ("ideal-power", "symbolic-power"):
        args.append(prepared["exponent"])
    elif op == "saturation":
        args.append(str(prepared["polynomial"]))
    elif op == "localization":
        args.append("J")
        args.extend(f"{key}={prepared[key]}" for key in ("numerator", "denominator") if key in prepared)
    elif op in ("intersection", "quotient"):
        args.append("J")
    elif op == "eliminate":
        args.extend(prepared["variables"])
    if "point" in prepared:
        args.append(f"point={prepared['point']!r}")
    if "degrees" in prepared:
        args.append("degrees=[" + ", ".join(map(integer_text, prepared["degrees"])) + "]")
    args.append(f"compute={request['compute']!r}")
    lines.append(f"result = {op.replace('-', '_')}({', '.join(args)})")
    # Long, expanded polynomials are easier to replay from their input strings.
    # This also avoids the evaluator's recursion and decimal-literal limits.
    if any(len(line) > _MAX_SCRIPT_LINE for line in lines):
        return None
    return "\n".join(lines)


def make_replay(request, prepared):
    """Render original operands at the shell and parsed exact values in a script."""
    script = _script(request, prepared)
    if script is None:
        replay_request = request
        if "degrees" in request["operands"]:
            # Quoted decimal integers replay even in a digit-limited host.
            operands = dict(request["operands"], degrees=list(map(integer_text, prepared["degrees"])))
            replay_request = dict(request, operands=operands)
        source = json.dumps(replay_request, ensure_ascii=False)
        script = f"result = algebra_request({source!r})"
    return {"command": _command(request), "script": script + "\nprint(result.json)\n"}
