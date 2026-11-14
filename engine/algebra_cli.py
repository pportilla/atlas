"""Shell commands for exact polynomial and ideal arithmetic."""

import argparse
import json
import sys


_DESCRIPTIONS = {
    "gcd": "compute the monic greatest common divisor of two polynomials",
    "ideal": "declare an ideal from ordered generators",
    "groebner": "compute a reduced Gröbner basis",
    "normal-form": "reduce a polynomial modulo an ideal",
    "membership": "test whether a polynomial belongs to an ideal",
    "eliminate": "eliminate variables from an ideal",
    "intersection": "intersect two ideals",
    "quotient": "compute the quotient of two ideals",
    "saturation": "saturate an ideal by a polynomial",
    "dimension": "compute the Krull dimension of the quotient ring",
}


class _OperandParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        # argparse treats a quoted polynomial such as '-x+1' as another option.
        # Attach it to the preceding operand flag just as in '--gen=-x+1'.
        if args is not None:
            tokens = []
            for token in args:
                if (tokens and tokens[-1] in ("--gen", "--other-gen", "--poly", "--other-poly")
                        and token.startswith("-") and token != "--"
                        and (token == "-h" or token.partition("=")[0] not in self._option_string_actions)):
                    tokens[-1] += "=" + token
                else:
                    tokens.append(token)
            args = tokens
        return super().parse_known_args(args, namespace)


def _names(value):
    return [name.strip() for name in value.split(",")] if value.strip() else []


def register_parser(subparsers):
    from .algebra import OPERATION_OPERANDS

    command = subparsers.add_parser("algebra", help="compute with polynomials and ideals over QQ")
    operations = command.add_subparsers(dest="operation", required=True, parser_class=_OperandParser)
    for name, keys in OPERATION_OPERANDS.items():
        parser = operations.add_parser(name, help=_DESCRIPTIONS[name], description=_DESCRIPTIONS[name])
        parser.add_argument("--vars", type=_names, required=True, metavar="x,y",
                            help="ordered variables, highest priority first")
        parser.add_argument("--field", metavar="QQ", help="coefficient field (default: QQ)")
        parser.add_argument("--order", metavar="{lex,grevlex}", help="monomial order (default: grevlex)")
        if "ideal" in keys:
            parser.add_argument("--gen", dest="ideal", action="append", default=[], metavar="POLY",
                                help="ideal generator (repeatable; omit for the zero ideal)")
        if "polynomial" in keys:
            parser.add_argument("--poly", dest="polynomial", metavar="POLY", required=True,
                                help="polynomial operand")
        if "other_polynomial" in keys:
            parser.add_argument("--other-poly", dest="other_polynomial", metavar="POLY", required=True,
                                help="second polynomial operand")
        if "other_ideal" in keys:
            parser.add_argument("--other-gen", dest="other_ideal", action="append", default=[],
                                metavar="POLY", help="second ideal generator (repeatable; omit for zero)")
        if "variables" in keys:
            parser.add_argument("--eliminate", dest="variables", type=_names, metavar="x[,y,...]",
                                required=True, help="variables to eliminate, separated by commas")
        compute = parser.add_mutually_exclusive_group()
        compute.add_argument("--compute", metavar="{cpu,auto,gpu}", help="execution mode (default: auto)")
        compute.add_argument("--gpu", dest="compute", action="store_const", const="gpu",
                             help="request GPU acceleration")
        parser.add_argument("--json", action="store_true", help="print the complete result as JSON")


def _show(record):
    result = record["result"]
    ring = result["ring"]
    names = ", ".join(ring["variables"])
    ring_name = f"{ring['field']}[{names}]" if names else ring["field"]
    print(f"{ring_name}   {ring['order']}")
    if result["type"] in ("basis", "ideal"):
        print("Generators:", ", ".join(result["generators"]) or "(zero ideal)")
    elif result["type"] == "polynomial":
        label = "GCD" if record["operation"] == "gcd" else "Remainder"
        print(f"{label}:", result["polynomial"])
    elif result["type"] == "boolean":
        print("Member:", "yes" if result["value"] else "no")
    else:
        print("Dimension:", result["value"])
    execution = record["execution"]
    print(f"Device: {execution['device']} (requested {execution['requested']})")
    if execution.get("note"):
        print(execution["note"])


def run(args):
    from .algebra import OPERATION_OPERANDS, calculate_algebra

    ring = {"variables": args.vars}
    if args.field is not None:
        ring["field"] = args.field
    if args.order is not None:
        ring["order"] = args.order
    request = {
        "operation": args.operation,
        "ring": ring,
        "operands": {key: getattr(args, key) for key in OPERATION_OPERANDS[args.operation]},
    }
    if args.compute is not None:
        request["compute"] = args.compute
    try:
        status, record = calculate_algebra(request)
    except KeyboardInterrupt:
        print("Algebra calculation interrupted.", file=sys.stderr)
        return 130
    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
    elif status == 200:
        _show(record)
    else:
        print(record["error"]["message"], file=sys.stderr)
    return 0 if status == 200 else 2
