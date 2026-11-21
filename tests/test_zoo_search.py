import itertools
import json
import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

from engine.zoo_search import _Catalogue, zoo_search
from tests.cli_support import run_prompt, run_script


def entry(name, *terms):
    return {"id": name, "terms": [{"coefficient": str(c), "exponents": [str(x) for x in e]} for c, e in terms]}


class ZooSearchTests(unittest.TestCase):
    def test_exact_coefficients_sign_and_consistent_renaming(self):
        def row(name, a, index, b):
            powers = [0, 0, 0]
            powers[index] = 2
            return entry(name, (a, (0, 3, 1)), (b, powers))
        catalogue = _Catalogue([row("yes", "1/2", 2, "-1"), row("opposite", "-1/2", 2, "1"),
                                row("sign-mismatch", "1/2", 2, "1"), row("separate-renamings", "1/2", 1, "-1"),
                                row("scale", "3/2", 2, "-3")])
        for query in ("1/2*x*y^3, -x^2", "x y**3 / 2 − x^2", ["2/4*u*v^3", "-u*u"]):
            self.assertEqual(catalogue.search(query)["ids"], ["yes", "opposite"])
        self.assertEqual(catalogue.search("2/4*x*y^3")["ids"],
                         ["yes", "opposite", "sign-mismatch", "separate-renamings"])
        self.assertEqual(catalogue.search("")["ids"], catalogue.search([])["ids"])
        self.assertEqual(catalogue.search("xy^3")["ids"], [])

    def test_wide_exponents_and_rationals(self):
        exponent = 2 ** 260
        numerator = 5 ** 120
        denominator = 2 ** 160
        c = _Catalogue([entry("wide", (f"{numerator}/{denominator}", (exponent, 1)))])
        query = f"-{3 * numerator}/{3 * denominator}*u*v^{exponent - 1}*v"
        self.assertEqual(c.search(query)["ids"], ["wide"])
        self.assertEqual(c.search(query + "*v")["ids"], [])
        self.assertEqual(c.search(query.replace(str(3 * numerator), str(3 * numerator + 1)))["ids"], [])
        digits = "1" + "0" * 9000
        catalogue = _Catalogue([entry("large-power", ("1", (digits,)))])
        self.assertEqual(catalogue.search("a^" + digits)["ids"], ["large-power"])

    def test_input_errors_do_not_damage_the_index(self):
        c = _Catalogue([entry("example", ("1", (2, 0)))])
        self.assertEqual(c.search([])["ids"], ["example"])
        for query in ("x*", "1/0*x", "x^-1", "0*x", "x,", "x\0z", {}, ["x", ""], "x/y"):
            with self.subTest(query=query), self.assertRaises(ValueError):
                c.search(query)
        self.assertEqual(c.search("x^2")["ids"], ["example"])

    def test_independent_exhaustive_oracle(self):
        rng = random.Random(169)
        rows = []
        for i in range(60):
            n = rng.randint(2, 5)
            data = {tuple(rng.randrange(3) for _ in range(n)): rng.choice([-2, -1, 1, 2]) for _ in range(9)}
            rows.append(data)
        catalogue = _Catalogue([entry(str(i), *((c, e) for e, c in row.items())) for i, row in enumerate(rows)])
        for _ in range(50):
            source = rng.choice(rows)
            terms = rng.sample(list(source.items()), rng.randint(1, min(3, len(source))))
            # Use variables with a positive exponent in at least one query term.
            axes = [j for j in range(len(terms[0][0])) if any(e[j] for e, _ in terms)]
            query = ",".join("*".join([str(c)] + [f"u{j}^{e[j]}" for j in axes if e[j]]) for e, c in terms)
            expected = []
            for i, row in enumerate(rows):
                width = len(next(iter(row)))
                for images in itertools.permutations(range(width), len(axes)):
                    coefficients = []
                    for e, c in terms:
                        target = [0] * width
                        for axis, coordinate in zip(axes, images):
                            target[coordinate] = e[axis]
                        coefficients.append(Fraction(row.get(tuple(target), 0), c))
                    if all(v == 1 for v in coefficients) or all(v == -1 for v in coefficients):
                        expected.append(str(i))
                        break
            self.assertEqual(catalogue.search(query)["ids"], expected, query)

    def test_immutable_index_across_concurrent_queries(self):
        c = _Catalogue([entry("a", ("3", (1, 2))), entry("b", ("1/2", (3, 1)))])
        queries = ["3*y*x^2", "u^3*v/2", "z^24", "x^"] * 20

        def run(q):
            try:
                return c.search(q)["ids"]
            except ValueError as error:
                return str(error)
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(list(pool.map(run, queries)), [["a"], ["b"], [], "Expected a nonnegative integer."] * 20)

    def test_prompt_and_script(self):
        query = f"123456789012345678901234567890*x^{2 ** 200}"
        source = f'hits = zoo_search(["x^2", "y^2"])\nprint_json(ids=hits.ids, wide=zoo_search("{query}").ids)'
        ids = zoo_search("x^2 + y^2")["ids"]
        self.assertTrue(ids)
        expected = {"ids": ids, "wide": []}
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(json.loads(script.stdout), expected)
        status, output, error = run_prompt(source + "\nquit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(json.loads(output.split("atlas> ")[2]), expected)

    def test_polynomial_query_uses_live_parent_at_both_entry_points(self):
        setup = ('x,y = coordinates("x","y",coeffs=QQ)\nq = x*y^2\n'
                 'first = zoo_search([q,-y^3])\n'
                 'x,y = coordinates("x","y",fresh=True)\n')
        check = 'print(first.ids == zoo_search("u*v^2 - v^3").ids)'
        valid = run_script(setup + check)
        self.assertEqual((valid.returncode, valid.stdout), (0, "True\n"), valid.stderr)
        failed = run_script(setup + 'first = zoo_search([q,y^2])')
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("different coordinate spaces", failed.stderr)
        status, output, error = run_prompt(setup + 'first = zoo_search([q,y^2])\n' + check + '\nquit\n')
        self.assertEqual(status, 0)
        self.assertIn("different coordinate spaces", output + error)
        self.assertIn("atlas> True", output)
