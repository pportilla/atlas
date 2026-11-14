"""Shell commands for exact polynomial and ideal arithmetic."""

import argparse
import sys

from .exact_json import result_json
from .polynomial import integer_text


_DESCRIPTIONS = {
    "gcd": "compute the monic greatest common divisor of two polynomials",
    "resultant": "compute the resultant of two polynomials in one variable over QQ",
    "factor": "factor a polynomial over QQ into monic irreducible factors",
    "ideal": "declare an ideal from ordered generators",
    "groebner": "compute a reduced Gröbner basis",
    "normal-form": "reduce a polynomial modulo an ideal",
    "membership": "test whether a polynomial belongs to an ideal",
    "eliminate": "eliminate variables from an ideal",
    "intersection": "intersect two ideals",
    "quotient": "compute the quotient of two ideals",
    "saturation": "saturate an ideal by a polynomial",
    "dimension": "compute the Krull dimension of the quotient ring",
    "hilbert-samuel-function": "compute local quotient lengths and graded dimensions at the requested degrees",
    "hilbert-samuel-multiplicity": "compute the Hilbert–Samuel multiplicity of the local quotient",
    "quotient-basis": "compute a vector space basis of the global quotient algebra over QQ",
    "local-quotient-basis": "compute a vector space basis of the quotient algebra at a rational point",
    "local-standard-basis": "compute a standard basis of an ideal at a rational point",
    "local-normal-form": "reduce a polynomial modulo an ideal at a rational point",
    "local-jacobian-algebra": "compute the Jacobian generators and quotient basis at a rational point",
    "milnor-number": "compute the dimension over QQ of the Jacobian algebra at a rational point",
    "tjurina-number": "compute the quotient dimension over QQ by f and its partial derivatives at a rational point",
}


class _OperandParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        # argparse treats a quoted polynomial such as '-x+1' as another option.
        # Attach it to the preceding operand flag just as in '--gen=-x+1'.
        if args is not None:
            tokens = []
            for token in args:
                if (tokens and tokens[-1] in ("--gen", "--other-gen", "--poly", "--other-poly", "--point", "--degrees")
                        and token.startswith("-") and token != "--"
                        and (token == "-h" or token.partition("=")[0] not in self._option_string_actions)):
                    tokens[-1] += "=" + token
                else:
                    tokens.append(token)
            args = tokens
        return super().parse_known_args(args, namespace)


def _comma_list(value):
    return [name.strip() for name in value.split(",")] if value.strip() else []


def register_parser(subparsers):
    from .algebra import OPERATION_OPERANDS, POINT_OPERATIONS

    command = subparsers.add_parser("algebra", help="compute with polynomials and ideals over QQ")
    operations = command.add_subparsers(dest="operation", required=True, parser_class=_OperandParser)
    for name, keys in OPERATION_OPERANDS.items():
        parser = operations.add_parser(name, help=_DESCRIPTIONS[name], description=_DESCRIPTIONS[name])
        parser.add_argument("--vars", type=_comma_list, required=True, metavar="x,y",
                            help="ordered variables, highest priority first")
        parser.add_argument("--field", metavar="QQ", help="coefficient field (default: QQ)")
        order_help = ("tie-breaker among terms of lowest degree in translated coordinates (default: grevlex)"
                      if name in POINT_OPERATIONS else "monomial order (default: grevlex)")
        parser.add_argument("--order", metavar="{lex,grevlex}", help=order_help)
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
            single = name == "resultant"
            parser.add_argument("--eliminate", dest="variables", type=_comma_list,
                                metavar="VAR" if single else "x[,y,...]", required=True,
                                help="variable to eliminate" if single else "variables to eliminate, separated by commas")
        if name in POINT_OPERATIONS:
            parser.add_argument("--point", type=_comma_list, metavar="q1,q2,...",
                                help="rational coordinates in --vars order (default: origin)")
        if name == "hilbert-samuel-function":
            parser.add_argument("--degrees", type=_comma_list, metavar="n[,n,...]",
                                help="ordered nonnegative degrees (default: 0 through 5; use '' for none)")
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
    if "scope" in result:
        print("Scope:", result["scope"])
    if "point" in result:
        print("Point: (" + ", ".join(result["point"]) + ")")
    if result["type"] in ("basis", "ideal", "local-standard-basis"):
        print("Generators:", ", ".join(result["generators"]) or "(zero ideal)")
    elif result["type"] == "local-normal-form":
        print("Reduction:", result["reduction"])
        print("Remainder:", result["remainder"])
        print("Unit:", result["unit"])
    elif result["type"] == "factorization":
        product = [result["unit"]]
        for factor in result["factors"]:
            term = "(" + factor["polynomial"] + ")"
            if factor["multiplicity"] != 1:
                term += f"^{factor['multiplicity']}"
            product.append(term)
        print("Factorization:", " * ".join(product))
    elif result["type"] == "polynomial":
        label = {"gcd": "GCD", "resultant": "Resultant"}.get(record["operation"], "Remainder")
        print(f"{label}:", result["polynomial"])
    elif result["type"] == "boolean":
        print("Member:", "yes" if result["value"] else "no")
    elif record["operation"] in ("hilbert-samuel-function", "hilbert-samuel-multiplicity"):
        print("Dimension:", integer_text(result["dimension"]))
        function = result["type"] == "hilbert-samuel-function"
        print("Hilbert–Samuel multiplicity:", integer_text(result["multiplicity"] if function else result["value"]))
        if function:
            print("Degree\tLength of A/m^(n+1)\tDimension of m^n/m^(n+1)")
            for row in zip(result["degrees"], result["values"], result["graded_values"]):
                print(*map(integer_text, row), sep="\t")
    elif result["type"] in ("quotient-basis", "local-jacobian-algebra"):
        if result["type"] == "local-jacobian-algebra":
            print("Jacobian generators:", ", ".join(result["generators"]) or "(none)")
        print(f"Vector space dimension over {ring['field']}:",
              result["dimension"] if result["finite"] else "infinite")
        if result["finite"]:
            print("Basis:", ", ".join(result["basis"]) or "(empty)")
    elif record["operation"] in ("milnor-number", "tjurina-number"):
        label = "Milnor number" if record["operation"] == "milnor-number" else "Tjurina number"
        print(f"{label}:", result["value"] if result["finite"] else "infinite")
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
    for key in ("point", "degrees"):
        value = getattr(args, key, None)
        if value is not None:
            request["operands"][key] = value
    if args.compute is not None:
        request["compute"] = args.compute
    try:
        status, record = calculate_algebra(request)
    except KeyboardInterrupt:
        print("Algebra calculation interrupted.", file=sys.stderr)
        return 130
    if args.json:
        print(result_json(record, indent=2))
    elif status == 200:
        _show(record)
    else:
        print(record["error"]["message"], file=sys.stderr)
    return 0 if status == 200 else 2
