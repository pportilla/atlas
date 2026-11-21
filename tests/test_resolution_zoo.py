"""Cross-check the checked-in Resolution Zoo against its native sources."""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from jsonschema import Draft202012Validator

from engine.polynomial import parse_expression
from tools.build_resolution_zoo import DEFAULT_SEEDS, build_record, seeds

DATA = Path(__file__).resolve().parents[1] / "web" / "zoo-data"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class ResolutionZooTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = read_json(DATA / "index.json")
        cls.records = {e["id"]: read_json(DATA / "records" / (e["id"] + ".json"))
                       for e in cls.index["entries"]}

    def test_schema_accepts_every_record_and_explains_unavailable_results(self):
        schema = read_json(DATA / "schema.json")
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        validator.validate(self.index)
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
        self.assertEqual(set(self.records), {s["id"] for s, _, _ in seeds(DEFAULT_SEEDS)})
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
        for ident, record in self.records.items():
            with self.subTest(record=ident):
                source = (f"ring r=0,({','.join(record['variables'])}),ds;\n"
                          f"poly f={record['equation']}; ideal J=jacob(f);\n"
                          "print(vdim(std(J))); print(vdim(std(J+ideal(f)))); quit;")
                done = subprocess.run(["Singular", "-q", "--no-rc"], input=source, text=True,
                                      capture_output=True, check=True, timeout=10)
                actual = list(map(int, done.stdout.splitlines()))
                expected = [record["invariants"][key] for key in ("milnor_number", "tjurina_number")]
                self.assertEqual(actual, [n["value"] if n["finite"] else -1 for n in expected])

    def test_native_recomputation_across_families(self):
        fixtures = {"plane-cusp", "a3", "whitney-umbrella", "cusp-4-4-4", "mixed-quartic", "fivefold-cubic"}
        for seed, dimension, variables in seeds(DEFAULT_SEEDS):
            if seed["id"] in fixtures:
                with self.subTest(record=seed["id"]):
                    _, detail = build_record(seed, dimension, variables)
                    self.assertEqual(detail, self.records[seed["id"]])
