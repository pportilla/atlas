"""Cross-check an installed Resolution Zoo checkout against its native sources."""
import argparse
from collections import Counter
import contextlib
from copy import deepcopy
from fractions import Fraction
from functools import partial
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator

from engine.polynomial import parse_expression
from engine import newton_fan
from engine.resolve_cli import add_resolution_options, calculate_resolution, resolution_arguments, run_resolution
from engine.worker import evaluate
from engine.zoo_data import catalogue_directory
from tests.test_newton_fan import check_fan
from tools import build_resolution_zoo
from tools.build_resolution_zoo import build_record, generate, seeds

DATA = catalogue_directory()
SEEDS = DATA / "seeds.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def subdivision_independent_zetas(zetas):
    """Omit exceptional-divisor labels, strata and candidate poles of this subdivision."""
    results = deepcopy(zetas)
    del results["monodromy"]["contributions"]
    del results["topological"]["strata"], results["topological"]["numerical_data"]
    del results["pole_check"]["candidate_poles"]
    for kind in ("topological", "pole_check"):
        for pole in results[kind]["poles"]:
            del pole["components"]
    return results


class ZooGenerationTests(unittest.TestCase):
    def test_custom_seed_file_cannot_replace_the_installed_index(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "seeds.json").write_text("[]")
            (root / "index.json").write_text("previous index")
            custom = root / "custom.json"
            custom.write_text("[]")
            with self.assertRaisesRegex(ValueError, "separate --out"):
                generate(custom, root)
            self.assertEqual((root / "index.json").read_text(), "previous index")


class ResolutionZooTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (DATA / "index.json").is_file():
            raise unittest.SkipTest("Install the separate Resolution Zoo to validate its full catalogue")
        cls.index = read_json(DATA / "index.json")
        cls.records = {e["id"]: read_json(DATA / "records" / (e["id"] + ".json"))
                       for e in cls.index["entries"]}

    def test_schema_accepts_every_record_and_explains_unavailable_results(self):
        schema = read_json(DATA / "schema.json")
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        validator.validate(self.index)
        for label in ({}, {"name": ""}):
            unnamed = {key: value for key, value in self.records["plane-cusp"].items() if key != "name"}
            validator.validate({**unnamed, **label})
        for record in self.records.values():
            with self.subTest(record=record["id"]):
                validator.validate(record)
        partial = {**self.records["plane-cusp"], "zeta": {
            "monodromy": {"status": "partial", "reason": "Open strata still under study."},
            "topological": {"status": "unavailable", "reason": "No Euler integral yet."}}}
        self.assertTrue(validator.is_valid(partial))
        del partial["zeta"]["topological"]["reason"]
        self.assertFalse(validator.is_valid(partial))

    def test_index_and_exact_equations(self):
        entries = self.index["entries"]
        self.assertEqual(self.index["schema_version"], 1)
        self.assertEqual(len(entries), len(self.records))
        self.assertGreaterEqual(Counter(e["dimension"] for e in entries)[2], 50)
        self.assertTrue({1, 2, 3, 4, 5} <= {e["dimension"] for e in entries})
        self.assertEqual({f.stem for f in (DATA / "records").glob("*.json")}, set(self.records))
        self.assertEqual(set(self.records), {s["id"] for s, _, _ in seeds(SEEDS)})
        for entry in entries:
            with self.subTest(entry=entry["id"]):
                record = self.records[entry["id"]]
                self.assertEqual(record["schema_version"], 1)
                self.assertEqual({field: record[field] for field in entry}, entry)
                variables = record["variables"]
                self.assertEqual(record["dimension"], len(variables) - 1)
                self.assertEqual(record["origin"], [0] * len(variables))
                terms = [" ".join((t["coefficient"], *t["exponents"])) for t in entry["terms"]]
                self.assertEqual(";".join(terms), parse_expression(record["equation"], variables, as_row=True))

    def test_complex_attachments_and_zeta_strata(self):
        for ident, record in self.records.items():
            with self.subTest(record=ident):
                for mode in ("intrinsic", "embedded"):
                    resolution = record[mode]
                    self.assertEqual(resolution["status"], "resolved")
                    complex_ = resolution["dual_complex"]
                    self.assertEqual(complex_["status"], "complete")
                    dimensions = {v["id"]: 0 for v in complex_["vertices"]}
                    dimensions.update({cell["id"]: cell["dimension"] for cell in complex_["cells"]})
                    self.assertEqual(len(dimensions), len(complex_["vertices"]) + len(complex_["cells"]))
                    for cell in complex_["cells"]:
                        self.assertTrue(all(dimensions[v] == 0 for v in cell["vertices"]))
                        self.assertTrue(all(dimensions[f] == cell["dimension"] - 1 for f in cell["faces"]))

                mono, top = (record["zeta"][kind] for kind in ("monodromy", "topological"))
                self.assertEqual([mono["status"], top["status"]], ["computed"] * 2)
                check = record["zeta"]["pole_check"]
                for pole in check["poles"]:
                    for match in pole["nearby_matches"]:
                        self.assertIn(match["stratum_index"], range(len(mono["nearby_strata"])))

                # A'Campo uses only the singleton strata, while the local
                # topological zeta also includes higher intersections.
                powers = Counter()
                for row in mono["contributions"]:
                    powers[row["multiplicity"]] -= row["euler_characteristic"]
                self.assertEqual({n: e for n, e in powers.items() if e},
                                 {f["multiplicity"]: f["exponent"] for f in mono["factors"]})
                euler = -sum(n * e for n, e in powers.items())
                self.assertEqual(euler, mono["milnor_fiber_euler_characteristic"])
                mu = record["invariants"]["milnor_number"]
                if mu["finite"]:
                    self.assertEqual(euler, 1 + (-1)**record["dimension"] * mu["value"])
                    self.assertIn("Isolated", record["labels"])
                else:
                    self.assertIn("Non-isolated", record["labels"])
                    self.assertIsNone(mu["value"])

                data = {d["component"]: d for d in top["numerical_data"]}
                for s in (0, 1, 2):
                    total = Fraction()
                    for row in top["strata"]:
                        value = Fraction(row["euler_characteristic"])
                        for name in row["components"]:
                            value /= data[name]["multiplicity"] * s + data[name]["nu"]
                        total += value
                    def at(source):
                        poly = parse_expression(source, ["s"])
                        return sum(c * s**e[0] for e, c in poly.items())
                    self.assertEqual(total, at(top["numerator"]) / at(top["denominator"]))

    @unittest.skipUnless(shutil.which("Singular"), "Singular is required for the independent local algebra check")
    def test_local_invariants_agree_with_singular(self):
        # Local std can spend its whole deadline on an infinite quotient.
        # Let Singular first exhibit a positive-dimensional germ contained
        # in both Jacobian loci. Every restriction and factorization is over
        # QQ and independent of the invariant stored in the catalogue.
        dimensions = """
ideal T=J,f;
int i,j,positive,branches;
ideal L;
for(i=1; i<=nvars(r); i++) {
    L=T;
    for(j=1; j<=nvars(r); j++) { if(j!=i) { L=subst(L,var(j),0); } }
    if(size(L)==0) { positive=1; break; }
}
if(!positive && nvars(r)>=3) {
    list fac=factorize(f);
    for(i=1; i<=size(fac[1]); i++) {
        if(jet(fac[1][i],0)==0) { branches=branches+fac[2][i]; }
    }
    // Two factors through zero meet in dimension at least n-2.
    // Both f and all its partials vanish on that intersection.
    if(branches>=2) { positive=1; }
}
if(positive) { print(-1); print(-1); }
else { print(vdim(std(J))); print(vdim(std(T))); }
quit;
"""
        def singular(source):
            run = subprocess.run(["Singular", "-q", "--no-rc"], input=source, text=True,
                                 capture_output=True, check=True, timeout=10)
            return list(map(int, run.stdout.splitlines()))

        for ident, record in self.records.items():
            with self.subTest(record=ident):
                setup = (f"ring r=0,({','.join(record['variables'])}),ds;\n"
                         f"poly f={record['equation']}; ideal J=jacob(f);\n")
                try:
                    actual = singular(setup + dimensions)
                except subprocess.TimeoutExpired:
                    # Singular's locstd homogenizes in an extra variable.
                    # A global standard basis then dehomogenizes to a local
                    # degree standard basis, giving the same exact lengths.
                    actual = singular('LIB "sing.lib";\n' + setup
                                      + "print(vdim(locstd(J))); print(vdim(locstd(J+ideal(f)))); quit;")
                expected = [record["invariants"][key] for key in ("milnor_number", "tjurina_number")]
                self.assertEqual(actual, [n["value"] if n["finite"] else -1 for n in expected])

    def test_parallel_generation_keeps_seed_order(self):
        chosen = ("a2", "plane-cusp", "whitney-umbrella")
        groups = [{**group, "entries": [s for s in group["entries"] if s["id"] in chosen][::-1]}
                  for group in read_json(SEEDS)]
        groups = [group for group in groups if group["entries"]]
        with tempfile.TemporaryDirectory() as scratch:
            source, target = Path(scratch) / "seeds.json", Path(scratch) / "zoo"
            source.write_text(json.dumps(groups), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                generate(source, target, jobs=3)
            index = read_json(target / "index.json")
            self.assertEqual([e["id"] for e in index["entries"]], ["whitney-umbrella", "a2", "plane-cusp"])
            for ident in chosen:
                self.assertEqual(read_json(target / "records" / (ident + ".json")), self.records[ident])
            with self.assertRaises(ValueError):
                generate(source, target, jobs=0)

    def test_native_recomputation_across_families(self):
        fixtures = {"plane-cusp", "a3", "whitney-umbrella", "cusp-4-4-4", "mixed-quartic", "fivefold-cubic",
                    "chain-3-2-4", "cone-braid-arrangement", "hidden-bp-2-5-5-quadric", "laufer-1",
                    "cone-d4-quartic", "quartic-face-plus", "hesse-triangle-quartic",
                    "audit-hyperplane-e7-cylinder", "m-s-varying-cusp-slope"}
        for seed, dimension, variables in seeds(SEEDS):
            if seed["id"] in fixtures:
                with self.subTest(record=seed["id"]):
                    _, detail = build_record(seed, dimension, variables)
                    archived = self.records[seed["id"]]
                    changed = {mode for mode in ("intrinsic", "embedded")
                               if detail[mode].get("fan") != archived[mode].get("fan")}
                    if not changed:
                        self.assertEqual(detail, archived)
                        continue

                    # A new regular subdivision can change every exceptional divisor ID.
                    # Check both covers independently and reproduce every archived field
                    # with the archived subdivision as the only input from that record.
                    supports = {}
                    for mode in changed:
                        before, after = archived[mode], detail[mode]
                        dependent = {"fan", "summary", "chart_count", "dual_complex"}
                        self.assertEqual({k: v for k, v in before.items() if k not in dependent},
                                         {k: v for k, v in after.items() if k not in dependent})
                        for key in ("status", "convention"):
                            self.assertEqual(before["dual_complex"][key], after["dual_complex"][key])
                        proof = after["certificate"]
                        if "fan_ideal_exponents" in proof:
                            support = proof["fan_ideal_exponents"]
                        else:
                            support = list(parse_expression(proof["adapted_polynomial"], proof["face_coordinates"]))
                        supports[mode] = frozenset(map(tuple, support))
                        for result in (before, after):
                            fan = result["fan"]
                            self.assertEqual(result["chart_count"], len(fan["cones"]))
                            self.assertEqual({i for cone in fan["cones"] for i in cone}, set(range(len(fan["rays"]))))
                            check_fan(self, fan, support)

                    # Compare everything independent of those exceptional divisors directly.
                    dependent = changed | ({"zeta"} if "embedded" in changed else set())
                    self.assertEqual({k: v for k, v in detail.items() if k not in dependent},
                                     {k: v for k, v in archived.items() if k not in dependent})
                    self.assertEqual(subdivision_independent_zetas(detail["zeta"]),
                                     subdivision_independent_zetas(archived["zeta"]))
                    for expected in (archived, detail):
                        self.assertEqual(self._rebuild_with_fans(seed, dimension, variables, expected, supports),
                                         expected)

    def _rebuild_with_fans(self, seed, dimension, variables, expected, supports):
        ordinary_atlas = build_resolution_zoo.atlas
        ordinary_fan = newton_fan.bounded_fan
        parser = argparse.ArgumentParser()
        add_resolution_options(parser)

        def atlas(operation, *arguments, **options):
            if operation != "resolve":
                return ordinary_atlas(operation, *arguments, **options)
            args = parser.parse_args(resolution_arguments([*arguments, "--json"]))
            if args.mode not in supports:
                return ordinary_atlas(operation, *arguments, **options)
            used = False

            def subdivision(exponents, *, trials=False):
                nonlocal used
                if not trials and frozenset(map(tuple, exponents)) == supports[args.mode]:
                    used = True
                    return deepcopy(expected[args.mode]["fan"])
                return ordinary_fan(exponents, trials=trials)

            # Use the CLI parser, worker and JSON boundary in-process so only fan
            # selection changes. Chart construction and verification use the germ.
            calculate = partial(calculate_resolution, job=lambda request, timeout: evaluate(request))
            with patch.object(newton_fan, "bounded_fan", subdivision), contextlib.redirect_stdout(io.StringIO()) as out:
                status = run_resolution(args, calculate)
            self.assertEqual(status, 0, out.getvalue()[:500])
            self.assertTrue(used)
            return json.loads(out.getvalue())

        with patch.object(build_resolution_zoo, "atlas", atlas):
            _, detail = build_record(seed, dimension, variables)
        return detail
