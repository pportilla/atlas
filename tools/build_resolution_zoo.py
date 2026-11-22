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
from engine.quotient_monomials import degree_bounds

DEFAULT_SEEDS = ROOT / "examples" / "resolution-zoo-seeds.json"
DEFAULT_OUT = ROOT / "web" / "zoo-data"


def atlas(*arguments):
    run = subprocess.run([str(ROOT / "atlas"), *arguments, "--json"], capture_output=True,
                         cwd=ROOT, encoding="utf-8")
    if run.returncode:
        try:
            diagnostic = json.loads(run.stdout)
            detail = diagnostic.get("diagnostic") or diagnostic.get("error") or diagnostic.get("summary")
        except ValueError:
            detail = None
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


def build_record(seed, dimension, variables):
    equation = seed["equation"]
    native_dimension = atlas("algebra", "local-dimension", "--gen=" + equation, "--vars", ",".join(variables),
                             "--compute", "cpu")["result"]["local_dimension"]
    if native_dimension != dimension:
        raise ValueError(f"The requested collection has dimension {dimension}; the germ has dimension {native_dimension}")
    policy = seed.get("search", "quick")
    args = ("--vars", ",".join(variables), "--compute", "cpu", "--" + policy)
    embedded = atlas("resolve", equation, *args, "--mode", "embedded", "--monodromy")
    intrinsic = atlas("resolve", equation, *args, "--mode", "intrinsic")
    for key in ("polynomial", "variables", "base_field", "geometric_field", "origin"):
        if embedded[key] != intrinsic[key]:
            raise ValueError(f"The intrinsic and embedded records disagree on {key}")
    zeta = {name: embedded[name + "_zeta"] for name in ("monodromy", "topological")}
    if any(output["status"] != "computed" for output in zeta.values()):
        raise ValueError("The embedded certificate did not produce both zeta functions")
    zeta["pole_check"] = embedded["monodromy_check"]
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
              "source": {"solver": "Atlas", "record_version": embedded["schema_version"], "search": policy,
                         "compute": "cpu", "method": "auto"}}
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


def compute(item):
    seed, dimension, variables = item
    try:
        return build_record(seed, dimension, variables)
    except (ValueError, RuntimeError, KeyError) as error:
        raise RuntimeError(f"Could not generate {seed['id']}: {error}") from error


def generate(source, destination, jobs=1):
    if jobs < 1:
        raise ValueError("Use at least one job.")
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
            for entry, detail in pool.map(compute, work):
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
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=1, help="number of records to compute at once")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    generate(args.seeds, args.out, args.jobs)
