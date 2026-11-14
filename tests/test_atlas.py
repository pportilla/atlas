import unittest

from engine import atlas
from engine.monodromy import monodromy_zeta
from engine.polynomial import format_poly, parse
from engine.resolve import resolve
from engine.topological import topological_zeta

NAMES = ["x", "y", "z"]


def general(text, record):
    """Both zeta functions from the atlas of the record alone."""
    p = parse(text, NAMES)
    charts = atlas.from_record(record, p)
    if charts is None:
        return None
    found = atlas.outputs(atlas.from_json(atlas.to_json(charts)), len(NAMES), complex=False)
    shell = {"status": "resolved", "engine": "canonical-blowups", "mode": "embedded", "variables": NAMES,
             "polynomial": format_poly(p, NAMES), "origin": [0, 0, 0], "charts": record["charts"],
             "certificate": {"numerical_data": found["numerical_data"], "strata": found["strata"]}}
    zeta = monodromy_zeta(shell, polynomial=p)
    return zeta["expression"], topological_zeta(shell, polynomial=p, monodromy=zeta)["expression"]


class OneMachineForEveryEngine(unittest.TestCase):
    """The outputs computed from the atlas, by the same code for every
    engine, agree with what each engine computes in its own way."""

    def test_zeta_functions_from_the_atlas(self):
        cases = {"x^2+y^3+z^5": "native-toric", "x^2*y-z^3": "binomial-coordinate-blowups",
                 "x^4+y^4+z^4": "native-ordinary", "x*y*z+x^4+y^4+z^4": "newton-polyhedron",
                 "x^3+y^3+z^4": "native-toric"}
        for text, engine in cases.items():
            with self.subTest(germ=text):
                record = resolve({"polynomial": text, "variables": NAMES, "mode": "embedded", "monodromy": True})
                self.assertEqual(record["engine"], engine)
                own = record["monodromy_zeta"]["expression"], record["topological_zeta"]["expression"]
                self.assertEqual(general(text, record), own)

    def test_the_charts_partition_the_fibre(self):
        # The Euler characteristics of all the strata add up to that of the
        # fibre over the origin, counted once.
        p = parse("x^4+y^4+z^4", NAMES)
        record = resolve({"polynomial": "x^4+y^4+z^4", "variables": NAMES, "mode": "embedded"})
        found = atlas.outputs(atlas.from_record(record, p), 3, complex=False)
        self.assertEqual(sum(row["euler_characteristic"] for row in found["strata"]), 3)  # the projective plane


if __name__ == "__main__":
    unittest.main()
