"""Compute the Resolution Zoo catalogue with the public Atlas solver."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.polynomial import integer_from_decimal, parse_expression
from engine.quick_budget import checked_seconds
from engine.quotient_monomials import degree_bounds
from engine.resolution_zetas import replay_zetas
from engine.zoo_data import catalogue_directory


class BudgetExpired(RuntimeError):
    """A step of a record was not finished within its time budget.

    This is not a failure of the germ or of Atlas: the same step with a
    larger budget, or without one, can still finish. `step` names the step,
    `seed` the record when it is known, and `budget` is the `budget` field
    of the answer: the seconds given and used, and the stage and the number
    of blowups that the search had reported last.
    """

    def __init__(self, step, budget, seed=None):
        self.step, self.budget, self.seed = step, budget, seed
        where = f"{seed}: " if seed else ""
        doing = f" while {budget['stage'][:1].lower()}{budget['stage'][1:]}" if budget.get("stage") else ""
        super().__init__(f"{where}the {step} was not finished within {budget['seconds']:g} seconds{doing}.")


def atlas(*arguments, step=None):
    """One Atlas command and its JSON answer. `step` names a resolution whose budget may end."""
    run = subprocess.run([str(ROOT / "atlas"), *arguments, "--json"], capture_output=True,
                         cwd=ROOT, encoding="utf-8")
    if run.returncode:
        try:
            diagnostic = json.loads(run.stdout)
            detail = diagnostic.get("diagnostic") or diagnostic.get("error") or diagnostic.get("summary")
        except ValueError:
            diagnostic, detail = None, None
        if step and isinstance(diagnostic, dict) and (diagnostic.get("budget") or {}).get("expired"):
            raise BudgetExpired(step, diagnostic["budget"])
        raise RuntimeError(detail or run.stderr.strip() or f"Atlas exited with code {run.returncode}")
    return json.loads(run.stdout)


def select(source, fields):
    return {key: source[key] for key in fields if key in source}


def resolution_summary(raw):
    if raw["status"] != "resolved" or raw["dual_complex"]["status"] != "complete":
        raise ValueError(f"Incomplete {raw['mode']} geometry for {raw['polynomial']}")
    record = select(raw, ("status", "engine", "summary", "multiplicity", "steps", "dual_complex",
                          "coefficient_field", "algebraic_numbers", "root_families", "tangent_cone",
                          "tangent_coordinates", "branches", "input_divisor"))
    record["chart_count"] = len(raw["atlas"])
    # Keep coordinates, loci and the fan describing the modification. Chart-batch
    # timings, row-reduction traces and repeated copies of zeta strata do not belong here.
    certificate = raw["certificate"]
    record["certificate"] = select(certificate, ("method", "description", "atlas_verification",
        "components", "normalized_chart", "linear_map_to_input", "coordinate_translation", "adapted_polynomial", "face_coordinates",
        "fan_coordinate_frame", "adapted_boundary_image_coordinate_ideals", "boundary_image_ideals",
        "compact_faces", "weights", "weighted_degree", "principal_exponents", "primitive_difference",
        "character_degree", "fan_ideal_exponents", "boundary_image_coordinate_ideals"))
    if "fan" in raw:
        record["fan"] = select(raw["fan"], ("rays", "cones", "boundary_rays", "isomorphism_off_origin"))
    return record


def local_number(operation, equation, variables):
    result = atlas("algebra", operation, "--poly=" + equation, "--vars", ",".join(variables),
                   "--compute", "cpu")["result"]
    return select(result, ("finite", "value"))


def origin_zeta(output):
    """A zeta output without its optional search for nearby strata."""
    return {key: value for key, value in output.items() if key not in ("nearby_strata", "nearby_search")}


def build_record(seed, dimension, variables, seconds=None):
    """The catalogue entry and the record of one seed.

    A quick search has a time budget: `seconds` of the seed, else the
    `seconds` given here, else the default of Atlas. Both resolutions get
    it. A step that is not finished within it raises BudgetExpired.
    """
    equation = seed["equation"]
    native_dimension = atlas("algebra", "local-dimension", "--gen=" + equation, "--vars", ",".join(variables),
                             "--compute", "cpu")["result"]["local_dimension"]
    if native_dimension != dimension:
        raise ValueError(f"The requested collection has dimension {dimension}; the germ has dimension {native_dimension}")
    policy = seed.get("search", "quick")
    args = ("--vars", ",".join(variables), "--compute", "cpu", "--" + policy)
    budget = seed.get("seconds", seconds)
    if budget is not None:
        if policy != "quick":
            raise ValueError("A time budget in seconds belongs to a quick search")
        args += ("--seconds", str(checked_seconds(budget)))
    embedded = atlas("resolve", equation, *args, "--mode", "embedded", "--monodromy",
                     step="embedded resolution with zetas")
    intrinsic = atlas("resolve", equation, *args, "--mode", "intrinsic", step="intrinsic resolution")
    for key in ("polynomial", "variables", "base_field", "geometric_field", "origin"):
        if embedded[key] != intrinsic[key]:
            raise ValueError(f"The intrinsic and embedded records disagree on {key}")
    zeta = {name: embedded[name + "_zeta"] for name in ("monodromy", "topological")}
    if any(output["status"] != "computed" for output in zeta.values()):
        raise ValueError("The embedded certificate did not produce both zeta functions")
    zeta["pole_check"] = embedded["monodromy_check"]
    source = {"solver": "Atlas", "record_version": embedded["schema_version"], "search": policy,
              "compute": "cpu", "method": "auto"}
    if zeta["pole_check"]["status"] == "unresolved":
        replay = replay_zetas(dict(record=embedded, zetas=["monodromy", "topological"],
                                   compute="cpu", search="exhaustive"))
        # A quick request may stop its optional search for nearby strata, and
        # the replay carries that search to its end. The zeta functions at the
        # origin must agree; the replayed outputs hold the complete search.
        for name in ("monodromy", "topological"):
            if origin_zeta(replay[name + "_zeta"]) != origin_zeta(zeta[name]):
                raise ValueError("Replaying the embedded certificate changed a computed zeta function")
            zeta[name] = replay[name + "_zeta"]
        zeta["pole_check"] = replay["monodromy_check"]
        source["pole_check_search"] = "exhaustive"
    # The sparse row and degree bounds are native operations. Transfer their
    # exact decimal strings directly into the searchable representation.
    terms = []
    for row in parse_expression(equation, variables, as_row=True).split(";"):
        coefficient, *exponents = row.split()
        terms.append({"coefficient": coefficient, "exponents": exponents})
    exponent_rows = (tuple(map(integer_from_decimal, term["exponents"])) for term in terms)
    order, degree = degree_bounds(len(variables), exponent_rows)
    if order < 2:
        raise ValueError("A Zoo seed must be singular at the origin")
    mu = local_number("milnor-number", equation, variables)
    tau = local_number("tjurina-number", equation, variables)
    entry = {"id": seed["id"], "name": seed.get("name", ""), "dimension": dimension,
             "labels": [*seed["labels"], "Isolated" if mu["finite"] else "Non-isolated"],
             "equation": equation, "variables": variables, "terms": terms}
    detail = {"schema_version": 1, **entry,
              **select(embedded, ("origin", "base_field", "geometric_field")),
              "invariants": {"multiplicity": order, "degree": degree, "milnor_number": mu, "tjurina_number": tau},
              "intrinsic": resolution_summary(intrinsic), "embedded": resolution_summary(embedded), "zeta": zeta,
              "source": source}
    return entry, detail


def json_text(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"


def seeds(source):
    groups = json.loads(source.read_text(encoding="utf-8"))
    seen = set()
    for group in groups:
        for seed in group["entries"]:
            ident = seed["id"]
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", ident) or ident in seen:
                raise ValueError(f"Invalid or repeated record id {ident!r}")
            seen.add(ident)
            yield seed, group["dimension"], group["variables"]


def compute(item, seconds=None):
    seed, dimension, variables = item
    try:
        return build_record(seed, dimension, variables, seconds)
    except BudgetExpired as error:
        # Not finished in time: the caller can tell this from a failure.
        raise BudgetExpired(error.step, error.budget, seed["id"]) from None
    except (ValueError, RuntimeError, KeyError) as error:
        raise RuntimeError(f"Could not generate {seed['id']}: {error}") from error


def generate(source, destination, jobs=1, seconds=None):
    """Compute every record of the seed file. `seconds` is the budget of each quick resolution."""
    if jobs < 1:
        raise ValueError("Use at least one job.")
    if (destination / "seeds.json").is_file() and source.resolve() != (destination / "seeds.json").resolve():
        raise ValueError("Use the checkout's seeds.json to regenerate it. For custom seeds choose a separate --out.")
    work = list(seeds(source))
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    # Publish the index after all its records exist. A failed native computation
    # cannot expose a half-built record or overwrite the last complete catalogue.
    # Each record runs its own solver processes, so threads only wait on them;
    # results are consumed in seed order.
    with tempfile.TemporaryDirectory(prefix=".zoo-", dir=destination) as staged, \
            ThreadPoolExecutor(max_workers=jobs) as pool:
        staging = Path(staged)
        try:
            for entry, detail in pool.map(lambda item: compute(item, seconds), work):
                print(entry["id"], flush=True)
                entries.append(entry)
                (staging / (entry["id"] + ".json")).write_text(json_text(detail), encoding="utf-8")
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
        records = destination / "records"
        records.mkdir(exist_ok=True)
        index = {"schema_version": 1, "entries": entries}
        for entry in entries:
            name = entry["id"] + ".json"
            (staging / name).replace(records / name)
        new_index = staging / "index.json"
        new_index.write_text(json_text(index), encoding="utf-8")
        new_index.replace(destination / "index.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, help="seed file (default: seeds.json in the catalogue checkout)")
    parser.add_argument("--out", type=Path, help="output directory (default: the catalogue checkout)")
    parser.add_argument("--jobs", type=int, default=1, help="number of records to compute at once")
    parser.add_argument("--seconds", type=float, help="time budget of each quick resolution (default: that of Atlas); "
                        "a seed can name its own")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    directory = catalogue_directory()
    if args.seeds and not args.out and args.seeds.resolve() != (directory / "seeds.json").resolve():
        parser.error("For a separate seed file, pass --out to choose the output directory.")
    try:
        generate(args.seeds or directory / "seeds.json", args.out or directory, args.jobs, args.seconds)
    except BudgetExpired as error:
        # Not finished: the same exit code as an Atlas search whose budget ended.
        print(error, file=sys.stderr)
        sys.exit(3)
