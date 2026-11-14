"""Shell commands for exact polynomial and ideal arithmetic."""

import argparse
import re
import sys

from .exact_json import result_json
from .polynomial import integer_text
from .series_display import format_hilbert_series


OPERATION_DESCRIPTIONS = {
    "gcd": "compute the monic greatest common divisor of two polynomials",
    "resultant": "compute the resultant of two polynomials in one variable over QQ",
    "factor": "factor a polynomial over QQ into monic irreducible factors",
    "ideal": "declare an ideal from ordered generators",
    "groebner": "compute a reduced Gröbner basis",
    "normal-form": "reduce a polynomial modulo an ideal",
    "membership": "test whether a polynomial belongs to an ideal",
    "eliminate": "eliminate variables from an ideal",
    "radical": "compute the radical of a polynomial ideal over QQ",
    "ideal-power": "compute an ordinary power of a polynomial ideal over QQ",
    "symbolic-power": "compute a symbolic power of a prime or radical ideal over QQ",
    "localization": "localize an ideal at a prime and optionally test a fraction",
    "normalization": "normalize an affine algebra over QQ and return its component maps",
    "primary-decomposition": "compute primary components and associated primes of an ideal over QQ",
    "intersection": "intersect two ideals",
    "quotient": "compute the quotient of two ideals",
    "saturation": "saturate an ideal by a polynomial",
    "dimension": "compute the Krull dimension of the quotient ring",
    "multiplicity": "compute the dimension over QQ of a finite-colength quotient across all its points",
    "hilbert-series": "compute the standard graded Hilbert series of a quotient by a homogeneous ideal over QQ",
    "tangent-cone": "compute the homogeneous ideal of the tangent cone at a rational point",
    "zariski-tangent-space": "compute the linear equations and a basis of tangent directions at a rational point",
    "embedding-dimension": "compute the dimension of the Zariski tangent space at a rational point",
    "local-dimension": "compute the Krull dimension of the local ring at a rational point",
    "local-smoothness": "test smoothness at a rational point and show its local and embedding dimensions",
    "tangent-cone-hilbert-series": "compute the graded Hilbert series of the tangent cone at a rational point",
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
                if (tokens and tokens[-1] in ("--gen", "--other-gen", "--prime", "--numerator", "--denominator", "--poly", "--other-poly", "--point", "--degrees")
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
    command.command_names = operations.choices
    for name, keys in OPERATION_OPERANDS.items():
        parser = operations.add_parser(name, aliases=["localize"] if name == "localization" else [],
                                       help=OPERATION_DESCRIPTIONS[name], description=OPERATION_DESCRIPTIONS[name])
        parser.set_defaults(operation=name)
        parser.add_argument("--vars", type=_comma_list, required=True, metavar="x,y",
                            help="ordered variables, highest priority first")
        parser.add_argument("--field", metavar="QQ", help="coefficient field (default: QQ)")
        order_help = ("tie-breaker among terms of lowest degree in translated coordinates (default: grevlex)"
                      if name in POINT_OPERATIONS else "monomial order (default: grevlex)")
        parser.add_argument("--order", metavar="{lex,grevlex}", help=order_help)
        if "ideal" in keys:
            parser.add_argument("--gen", dest="ideal", action="append", default=[], metavar="POLY",
                                help="ideal generator (repeatable; omit for the zero ideal)")
        if "exponent" in keys:
            parser.add_argument("--exponent", required=True, metavar="N", help="nonnegative integer exponent")
        if "polynomial" in keys:
            parser.add_argument("--poly", dest="polynomial", metavar="POLY", required=True,
                                help="polynomial operand")
        if "other_polynomial" in keys:
            parser.add_argument("--other-poly", dest="other_polynomial", metavar="POLY", required=True,
                                help="second polynomial operand")
        if "prime" in keys:
            parser.add_argument("--prime", action="append", default=[], metavar="POLY",
                                help="generator of the prime ideal (repeatable; omit for zero)")
            parser.add_argument("--numerator", metavar="POLY", help="fraction numerator (default: 1 if queried)")
            parser.add_argument("--denominator", metavar="POLY", help="denominator outside the prime (default: 1)")
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


def _show_direction_equations(result, equations, label, empty):
    # The wire record uses the original coordinate names for directions.
    names = result["ring"]["variables"]
    if names:
        point = result.get("point", ["0"] * len(names))
        print("Tangent coordinates:", ", ".join(f"Δ{v} = {v} - ({p})" for v, p in zip(names, point)))
    directions = [re.sub(r"[A-Za-z][A-Za-z0-9_]*", r"Δ\g<0>", f) for f in equations]
    print(label, ", ".join(directions) or empty)


def localization_lines(result):
    yield "Prime ideal: (" + (", ".join(result["prime"]) or "0") + ")"
    yield "Contracted ideal: (" + (", ".join(result["generators"]) or "0") + ")"
    if "element" in result:
        element = result["element"]
        yield f"Fraction: ({element['numerator']})/({element['denominator']})"
        yield "In the localized ideal: " + ("yes" if element["member"] else "no")
        yield "Unit in the local ring: " + ("yes" if element["unit"] else "no")


def normalization_lines(result):
    if not result["input_reduced"]:
        yield "The input has nilpotents. Showing the normalization of its reduction."
    if not result["components"]:
        yield "The zero algebra has no components."
    for index, component in enumerate(result["components"], start=1):
        yield f"Component {index}, prime ideal (" + (", ".join(component["prime"]) or "0") + ")"
        target = component["ring"]["variables"]
        yield "Normalized algebra: QQ[" + ", ".join(target) + "]/(" + (", ".join(component["relations"]) or "0") + ")"
        for var, image in zip(result["ring"]["variables"], component["images"]):
            yield f"  {var} -> {image}"
        for var, fraction in zip(target, component["fractions"]):
            yield f"  {var} = ({fraction['numerator']})/({fraction['denominator']})"


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
    if result["type"] in ("hilbert-series", "tangent-cone-hilbert-series", "tangent-cone"):
        if result["type"] == "tangent-cone":
            _show_direction_equations(result, result["generators"], "Tangent cone generators:", "(zero ideal)")
        else:
            print("H(t) =", format_hilbert_series(result["numerator"], result["denominator_power"]))
        print("Dimension:", integer_text(result["dimension"]))
        print("Multiplicity:", integer_text(result["multiplicity"]))
    elif record["operation"] in ("zariski-tangent-space", "embedding-dimension"):
        space = result["type"] == "zariski-tangent-space"
        if space:
            _show_direction_equations(result, result["equations"], "Tangent equations:", "(no constraints)")
            vectors = ["(" + ", ".join(v) + ")" for v in result["basis"]]
            print("Basis of tangent directions:", ", ".join(vectors) or "(zero space)")
        print("Ambient dimension:", result["ambient_dimension"])
        print("Jacobian rank:", result["jacobian_rank"])
        print("Embedding dimension:", result["dimension"] if space else result["value"])
    elif record["operation"] in ("local-dimension", "local-smoothness"):
        print("Local dimension:", result["local_dimension"])
        if record["operation"] == "local-smoothness":
            print("Embedding dimension:", result["embedding_dimension"])
            print("Jacobian rank:", result["jacobian_rank"])
            print("Smooth at point:", "yes" if result["value"] else "no")
    elif result["type"] in ("basis", "ideal", "local-standard-basis"):
        print("Generators:", ", ".join(result["generators"]) or "(zero ideal)")
    elif result["type"] == "localization":
        print("\n".join(localization_lines(result)))
    elif result["type"] == "normalization":
        print("\n".join(normalization_lines(result)))
    elif result["type"] == "primary-decomposition":
        if not result["components"]:
            print("The unit ideal has no primary components.")
        for index, component in enumerate(result["components"], start=1):
            print(f"Primary component {index}: (" + (", ".join(component["primary"]) or "0") + ")")
            print("Associated prime: (" + (", ".join(component["prime"]) or "0") + ")")
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
    elif record["operation"] == "multiplicity":
        if result["finite"]:
            print("Multiplicity:", integer_text(result["value"]))
        else:
            print("The ideal has infinite colength.")
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
    for key in ("point", "degrees", "numerator", "denominator"):
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
